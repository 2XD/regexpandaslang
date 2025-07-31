# regexpandaslang

Three tier approach to answering questions about a preloaded csv file in azure blob.

Tier 1: It first tries to parse and answer straightforward questions using regex and fuzzy matching with pandas — like counts, sums, averages, distinct values, and more. It can do these type of questions really fast and accurate
Tier 2: If Tier 1 doesn’t cover the question, it generates and executes custom pandas code using Azure OpenAI to dynamically analyze the data. This covers complex questions that are math related.
Tier 3: For anything more complex, it falls back to LangChain’s prompt-based narrative generation using a few sample rows of the dataset.
Also nothing is hardcoded in terms of reading the .csv file. This type of approach will work for all .csv and .xlsx files hopefully


Here’s some questions that I have tested that have worked:

{ "file_name": "sample.csv",
  "question": "How many times does Storage appear in the column serviceFamily?" }
 
{ "file_name": "sample.csv", "question": "max value in the unitprice column" }
 
{ "file_name": "sample.csv", "question": "what is the metercategory of the entry with the max unitprice?" }
