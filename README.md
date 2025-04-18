# RAG-Chatbot-Backend

making the backend server for AI assisted chatbot

docker build --platform linux/amd64 -t aws_rag_app .
docker build -t aws_rag_app .
docker run --rm -p 8000:8000 --add-host=host.docker.internal:host-gateway --entrypoint python --env-file .env aws_rag_app main.py
docker run --rm -p 8000:8000 --entrypoint python --env-file .env aws_rag_app main.py

docker build -t lewas_chatbot .
docker run --rm -p 8000:8000 --add-host=host.docker.internal:host-gateway --entrypoint python --env-file .env lewas_chatbot main.py
docker run --rm -p 8000:8000 --entrypoint python --env-file .env lewas_chatbot main.py

cdk init app --language=typescript

cdk bootstrap --region us-east-1
cdk deploy

RagCdkInfraStack

Enable Bedrcok services to be used - Error Faced
thanks to : https://www.reddit.com/r/aws/comments/17f8rk6/how_long_does_it_take_to_gain_access_to_a_model/

Errors faced:
you get 502 error - Pinecone API not provided go to Lambda Config and add key over there or use AWS Secret Manager

# Note

Deleted the Data folder, if needed in future, this is the commit ID on branch 'lancedb' : 1e96f4f9fa23d67ebe0221cab12c34cca8689b37
When deploying dont specify any build platform, lanceDB doesnt work otherwise.

# Future

- Caching:
  Implement DAX (DynamoDB Accelerator) or ElastiCache to cache frequent queries and reduce database load.
  - User Analytics:
    Develop a background process to analyze user queries and generate insights (e.g., most common topics, peak usage times).
- Tagging System: Add a 'Tags' field to the Queries table to categorize queries, enabling better organization and search capabilities.
  - Collaboration Feature:
    Add a 'SharedWith' field in the Queries table to allow users to share specific queries with others.
  - Pagination:
    Implement pagination in your query history API to handle large volumes of data efficiently.
    - User management (who can access the API)
    - Use Pinecone Inference

# Ask:

Query id to be random - YES
Meta data to be stored with every chunk?

# TO DO:

make get_query in handler like this:
@app.get("/query/{query_id}")

Have user authentication.
Make a new project for LEWAS?
Vector Database online? - Project will always have to be deployed again in order to work - DONE
Link to Google Drive.
Figure out Chunk size, and word count, Size - repsonse size; 10 query, have text overlap - Research Part
and include links to drive - DONE
Front End? - DONE (Streamlit)
and include file source link in metadata - Done

Optimize code, hardcore values to be pushed in Environment
Testing:
ask a llm, model and compare.

Next Phase:

- have the top k, model be determined by user or stored in a different file?

Later Phase:
Assign Query Id as a human redable query ID. (three words)
WebSocket (very alter)
Maybe use log instead of storing in DB?

Potential:

## Load Testing

Corrective RAG using Tally - https://github.com/run-llama/llama-agents/blob/main/examples/corrective_rag.ipynb

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
  How_sys_network_works.pdf - 1.0 MB - 7 pages - 22 chunks - 20 seconds

  Time taken with two documents in DB - 15 sec
  Time taken with three documents in DB - 30 sec

  Time taken with three documents in DB - 18 sec - Pinecone
  Time taken with three documents in DB - 5 sec - Pinecone
  Time taken with three documents in DB - 4 sec - Pinecone

3 seconds

- Jul 2, Mon:
  -- make dynamodb utils
  -- parralely convert to embeddings
  -- Delete certain embeddings
  -- what is the same document existed before
  -- modify metadata
  -- upload to DynamoDB - DONE
  -- return or store what chunks were used
  -- What happens when mutliple files with same name in different parts?
  -- output size (from model)
  -- update_drive_link_for_file - for manual updation
  -- List number fo vectors of a given document

  Done:
  Made feature to keep track of processed files, deployed to AWS, source as a google link set up,
  setup local api to download all ids and name, made it usable by local chatbot to store in the memory.
  feat: added drive metadata to the sources, added API adding drive links, for missing driveslinks, for corrupted drive links, modified to use SERVICE URL for Docker

  - July 3, Wed:
    -- meeting
    -- researching about user authentication
    -- store every query to DB
    -- Organize DB
    -- Store chunks of text to DB for now - for later maybe have an API to get the specific chunk
    -- Figure out chunk ID
    -- implement retry for pinecone db

  - July 4, Wed:
    -- Organised the project with using config and env files, also

  - July 5, Wed:
    -- Store the user queries along with sources in DB.
    -- Use variables from the config file.
    -- The chat doesn't show older messages' sources.

    - July 11, Thursday:
      -- chunk size, overlap top k all in the config file
      -- use config file
      -- make a new file for: get_google_drive_link_pdf() in pinecone_service
      -- efficiency: You can upsert data in batches for improved efficiency (recommended limit is 100 vectors per request).
      -- can we remove this in pinecone_service: metadata = chunk.metadata.copy() # Start with the existing metadata
      -- do we need this: "text": chunk.page_content, in metadata?
      -- upsert function has the following:
      --- batch_size (optional): This specifies the number of vectors to upsert in each batch. If not specified, all vectors will be upserted in a single batch.
      ---show_progress (optional): If set to True, this shows a progress bar using tqdm.
      --- create UI for uploading PDFs, showcase this in the Poster
      -- Feedback from user......

      - July 12, Friday:
        -- will check later but this is it i believe: results = index.query(
        vector=query_embedding, top_k=top_k, include_metadata=True
        ), when top_k (my provided value) is less than the available related approximate neighbours
        - July 13:
          dont revel personal adress, credit card, - should mask PII for people
      - like addressess and credit card infor etc
      - July 14
        -- What happens in drive when files with the same name ?

- Research part:

  - User Analytics:
    -- Develop a background process to analyze user queries and generate insights (e.g., most common topics, peak usage times).

  Questions tested:

- OK ->
  "How components are connected with each other?"
  "What language is the API written in?"
  "Tell me about Lohani research project"
  "What research work has been conducted in LEWAS"
  What is Dr Lohani's contribution to the lab?
  what is the IJEE research project about?
  "What different departments LEWAS comprise of?"
  list different sensors in LEWAS Lab.

  jULY 16, 4:02 AM (4 documents)
  explain about Acute Chloride Toxicity in LEWAS Lab
  explain effect of stormwater on LEWAS
  explain LEWAS Lab
  explain all Dr Lohani's work

-Not OK ->
"query_text": "Explain how the different components are connected in LEWAS" (correct as of Tue Jul 16 04:21:20 2024)
how are different components connected in LEWAS Lab?

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

# Storage:

- PineconeDB - Stores the vector embeddings
- S3 - stores all the PDFs
- DynamoDB ->
  -- one table stores all the file names which are processed
  -- other table stores all the queries ->
  -- query-id, answer text (response to show to user), create_time, is_complete, query_text (Question), sources (list all of the resources used).
  - THoughts:
    -- User stored somewhere where I store how many times/tokens the used so far and tokens/times for the day - not clear how to implement this

# User Authentication :

- Thoughts:
  -- Restrict to only vt.edu OR College (.edu)
  -- DynamoDB Streams
  -- Optimizing Data Retrieval: Use a GSI (Global Secondary Index) on user_id and create_time to efficiently retrieve queries for a specific user sorted by time, Use GSIs (Global Secondary Indexes):
  Create a GSI on the Queries table with CreateTime as the partition key to enable time-based queries across all users.
  -- Query Statistics: Track the number of queries each user makes and their completion status to generate usage statistics.
  -- Implement TTL (Time to Live):
  Use TTL on the DailyUsage table to automatically delete old daily records, saving storage and reducing costs.
  -- Add a 'UserFeedback' field to the Queries table to store user ratings or comments on responses.
  -- add two new tables:
  --- User Table (new):

          UserID (from Cognito)
          TotalTokensUsed
          TotalInteractions
          LastUsedDate
          LastQueryID (for quick access to the most recent query)

--- DailyUsage Table (new):

      UserID
      Date
      TokensUsedToday
      InteractionsToday

- Streamlit authenticator
- AWS Cognito with Streamlit Authentication Libraries
  will go ahead with AWS Cognito with link to DynamoDB.

# Challenges:

- if you dont have get error from streamlit / aws about not giving api key, go their environment and provide the key there.

## Features:

- Loads PDF documents from a specified directory
- Splits documents into manageable text chunks
- Generates unique IDs for each chunk
- Adds new documents to the database, avoiding duplicates
- Supports database reset with a command-line flag

## Usage:

- python3 populate_database.py [--reset]
  The `--reset` flag clears the existing database before populating it with new data.
