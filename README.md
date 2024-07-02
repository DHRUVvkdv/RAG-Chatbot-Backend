# RAG-Chatbot-Backend

making the backend server for AI assisted chatbot

docker build --platform linux/amd64 -t aws_rag_app .
docker build -t aws_rag_app .
docker run --rm -p 8000:8000 --entrypoint python --env-file .env aws_rag_app app_api_handler.py

cdk init app --language=typescript

cdk bootstrap --region us-east-1
cdk deploy

RagCdkInfraStack

Enable Bedrcok services to be used - Error Faced
thanks to : https://www.reddit.com/r/aws/comments/17f8rk6/how_long_does_it_take_to_gain_access_to_a_model/

# Note

Deleted the Data folder, if needed in future, this is the commit ID on branch 'lancedb' : 1e96f4f9fa23d67ebe0221cab12c34cca8689b37
When deploying dont specify any build platform, lanceDB doesnt work otherwise.

# Ask:

Query id to be random ?

# TO DO:

make get_query in handler like this:
@app.get("/query/{query_id}")

Have user authentication.
Make a new project for LEWAS?
Vector Database online? - Project will always have to be deployed again in order to work.
Link to Google Drive.
Figure out Chunk size, and word count. and include links to drive.
Front End?
Size - repsonse size; 10 query;
have text overlap and include file source link in metadata

Optimize code, hardcore values to be pushed in Environment
Testing:
ask a llm, model and compare.

Later Phase:
Assign Query Id as a human redable query ID. (three words)
WebSocket (very alter)
Maybe use log instead of storing in DB?

<!-- linux, apcahe, node.js, sql, postegresql, -->

Jul 1, Mon:
save the queries to DynamoDB
dont make the embedding run again for documents

- test with 10 instances
- docker ps -a
- docker kill
- give link of the data source - METADATA
- have a script which takes in argument on which documents to run it - maybe sometimes it fail so you can reset.
- have a processed file tracker
- fully functional with DynamoDB
- delete embedding
- edit a embedding (add metadata)
- pretty slow
- Time:
  -- 5 seconds for Resume (1 page) - 131.7 kB - 9 chunks
  -- 2min 50 sec for Lewas Systems Document () - 600kb - 98
  -- 7 - 9 min - three files - 1.7 MB - 322
  -- 3 min 8 Sec - three files - 1.7 MB - 322 - Pinecone

  Time taken with two documents in DB - 15 sec
  Time taken with three documents in DB - 30 sec

  Time taken with three documents in DB - 18 sec - Pinecone
  Time taken with three documents in DB - 5 sec - Pinecone
  Time taken with three documents in DB - 4 sec - Pinecone

3 seconds

Questions tested:

- OK ->
  "How components are connected with each other?"
  "What language is the API written in?"
  "Tell me about Lohani research project"
  "What research work has been conducted in LEWAS"
  What is Dr Lohani's contribution to the lab?
  what is the IJEE research project about?

-Not OK ->
"query_text": "Explain how the different components are connected in LEWAS"

Feature:
Summaries this document

Warning:
[2024-07-01T19:33:27Z WARN lance_table::io::commit] Using unsafe commit handler. Concurrent writes may result in data loss. Consider providing a commit handler that prevents conflicting writes.

Final:
Clean the code, better organize.

# Dessciption and how it works

This is the RAG application backend for our chatbot. RAG stands for Retrieval-Augmented Generation.
The working:
The data we need to read is placed in the image/src/data/source. Initially I have only one pdf over here.
We have the populate_database.py file which is run initilaly to create the vector database.

# populate_database.py

This utility script populates a Chroma vector database with text chunks extracted from PDF documents. It's designed for a RAG (Retrieval-Augmented Generation) application.

## Features:

- Loads PDF documents from a specified directory
- Splits documents into manageable text chunks
- Generates unique IDs for each chunk
- Adds new documents to the database, avoiding duplicates
- Supports database reset with a command-line flag

## Usage:

- python3 populate_database.py [--reset]
  The `--reset` flag clears the existing database before populating it with new data.
