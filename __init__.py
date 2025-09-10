import os
import json
import re
import traceback
import pandas as pd
from io import BytesIO
from fuzzywuzzy import fuzz
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def log_step(logs, message):
    print(f"[LANGCHAIN FUNC] {message}", flush=True)
    logs.append(message)

# normalize columns
def normalize_dataframe(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]  # keep original casing but strip whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # strip strings
    logs.append("[DEBUG] DataFrame normalized (trimmed whitespace).")
    return df

# stage 1:  pandas w/ regex matching w/ levenshtein distance formula
def fuzzy_match(query: str, targets: list, threshold: int = 80):
    for t in targets:
        if fuzz.ratio(query.lower(), t.lower()) >= threshold:
            return t
    return None

def try_pandas_eval(question: str, df: pd.DataFrame, logs: list):
    q = question.lower()

    # Count
    m = re.search(r"how many times does ['\"]?(.+?)['\"]? appear in (?:the column )?(\w+)", q)
    if m:
        val, col = m.groups()
        match_col = fuzzy_match(col, df.columns)
        if match_col:
            count = (df[match_col].astype(str).str.strip().str.lower() == val.strip().lower()).sum()
            logs.append(f"[Pandas] Count of '{val}' in '{match_col}': {count}")
            return {"answer": f"'{val}' appears {count} times in '{match_col}'."}

    # Sum
    m = re.search(r"(?:total|sum) of (\w+)", q)
    if m:
        col = m.group(1)
        match_col = fuzzy_match(col, df.columns)
        if match_col and pd.api.types.is_numeric_dtype(df[match_col]):
            total = df[match_col].sum()
            logs.append(f"[Pandas] Sum of '{match_col}': {total}")
            return {"answer": f"The total of '{match_col}' is {total:,.2f}."}

    # Average
    m = re.search(r"(?:average|mean) of (\w+)", q)
    if m:
        col = m.group(1)
        match_col = fuzzy_match(col, df.columns)
        if match_col and pd.api.types.is_numeric_dtype(df[match_col]):
            avg = df[match_col].mean()
            logs.append(f"[Pandas] Average of '{match_col}': {avg}")
            return {"answer": f"The average of '{match_col}' is {avg:,.2f}."}

    # Min
    m = re.search(r"(?:minimum|min|lowest) of (\w+)", q)
    if m:
        col = m.group(1)
        match_col = fuzzy_match(col, df.columns)
        if match_col and pd.api.types.is_numeric_dtype(df[match_col]):
            minimum = df[match_col].min()
            logs.append(f"[Pandas] Minimum of '{match_col}': {minimum}")
            return {"answer": f"The minimum of '{match_col}' is {minimum:,.2f}."}

    # Max
    m = re.search(r"(?:maximum|max|highest) of (\w+)", q)
    if m:
        col = m.group(1)
        match_col = fuzzy_match(col, df.columns)
        if match_col and pd.api.types.is_numeric_dtype(df[match_col]):
            maximum = df[match_col].max()
            logs.append(f"[Pandas] Maximum of '{match_col}': {maximum}")
            return {"answer": f"The maximum of '{match_col}' is {maximum:,.2f}."}

    # Distinct values
    m = re.search(r"(?:list|show|what are) all (?:the )?distinct values in (?:the )?(\w+)", q)
    if m:
        col = m.group(1)
        match_col = fuzzy_match(col, df.columns)
        if match_col:
            unique_vals = df[match_col].dropna().unique().tolist()
            logs.append(f"[Pandas] Distinct values in '{match_col}': {len(unique_vals)} found.")
            return {"answer": f"{len(unique_vals)} distinct values in '{match_col}'.", "values": unique_vals}

    # Row count
    if re.search(r"(?:how many rows|total rows|number of rows|count rows|total records)", q):
        total_rows = len(df)
        logs.append(f"[Pandas] Total rows: {total_rows}")
        return {"answer": f"The dataset contains {total_rows:,} rows."}

    return None  # No regex match

# stage 2: let ai run its own pandas code
def ai_generate_and_run_pandas(question: str, df: pd.DataFrame, llm: AzureChatOpenAI, logs: list):
    prompt = f"""
You are a Python data analysis assistant.
Given this pandas DataFrame (columns: {list(df.columns)}), write Python code using pandas to answer:
'{question}'.

Rules:
- Only output executable Python code.
- Use the variable 'df' (already defined).
- End with a single variable named 'result' that holds the final answer.
- Do NOT print or explain.
"""
    try:
        code_response = llm.invoke(prompt).content.strip()
        logs.append("[AI-Pandas] Generated code:\n" + code_response)
        # Extract Python code block if wrapped in ```python
        code = re.sub(r"^```(?:python)?|```$", "", code_response.strip(), flags=re.MULTILINE).strip()
        local_vars = {"df": df}
        exec(code, {}, local_vars)
        result = local_vars.get("result", "No result produced.")
        return {"answer": str(result)}
    except Exception as e:
        logs.append(f"[AI-Pandas ERROR] {str(e)}")
        logs.append(traceback.format_exc())
        return None

# main
def main(req: func.HttpRequest) -> func.HttpResponse:
    logs = []
    try:
        log_step(logs, "Starting Function Execution")

        # Validate env
        required_env_vars = [
            "AZURE_STORAGE_CONNECTION_STRING",
            "BLOB_CONTAINER_NAME",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_API_VERSION"
        ]
        env = {}
        for var in required_env_vars:
            val = os.getenv(var)
        #if no vals then return success along with error code
            if not val:
                return func.HttpResponse(json.dumps({"error": f"Missing env var: {var}", "logs": logs}),
                                         status_code=200, mimetype="application/json")
            env[var] = val

        # Parse request
        body = req.get_json()
        file_name, question = body.get("file_name"), body.get("question")
        if not file_name or not question:
            return func.HttpResponse(json.dumps({"error": "file_name and question are required", "logs": logs}),
                                     status_code=200, mimetype="application/json")

        # Load file
        blob_service_client = BlobServiceClient.from_connection_string(env["AZURE_STORAGE_CONNECTION_STRING"])
        blob_data = blob_service_client.get_blob_client(container=env["BLOB_CONTAINER_NAME"], blob=file_name).download_blob().readall()
        df = pd.read_csv(BytesIO(blob_data)) if file_name.endswith(".csv") else pd.read_excel(BytesIO(blob_data))
        df = normalize_dataframe(df, logs)

        llm = AzureChatOpenAI(
            azure_deployment=env["AZURE_OPENAI_DEPLOYMENT"],
            api_key=env["AZURE_OPENAI_API_KEY"],
            api_version=env["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
            temperature=0
            #using temp 0
        )

        # stage 1
        pandas_result = try_pandas_eval(question, df, logs)
        if pandas_result:
            return func.HttpResponse(json.dumps({"file_name": file_name, "row_count": len(df), "answer": pandas_result}),
                                     status_code=200, mimetype="application/json")

        # stage 2
        ai_result = ai_generate_and_run_pandas(question, df, llm, logs)
        if ai_result:
            return func.HttpResponse(json.dumps({"file_name": file_name, "row_count": len(df), "answer": ai_result}),
                                     status_code=200, mimetype="application/json")

        # stage 3: langchain narrative response (worst)
        columns_str = ", ".join(df.columns.tolist())
        df_head_str = df.head(5).to_string(index=False)
        prompt_text = """
You are a cloud FinOps assistant. 
Answer clearly in JSON with either:
{ "answer": "..." } OR { "table": {...} }.

Columns: {columns}
Sample data:
{df_head}

Question: {question}
"""
        chain = LLMChain(llm=llm, prompt=PromptTemplate(template=prompt_text, input_variables=["columns", "df_head", "question"]))
        answer = chain.run({"columns": columns_str, "df_head": df_head_str, "question": question})

        return func.HttpResponse(json.dumps({"file_name": file_name, "row_count": len(df), "answer": answer}),
                                 status_code=200, mimetype="application/json")

    except Exception as e:
        logs.append(f"[ERROR] {str(e)}")
        logs.append(traceback.format_exc())
        return func.HttpResponse(json.dumps({"error": str(e), "logs": logs}), status_code=200, mimetype="application/json")
