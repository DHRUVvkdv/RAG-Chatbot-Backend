# RAG-Chatbot-Backend

making the backend server for AI assisted chatbot

have text overlap and include file source link in metadata

docker build --platform linux/amd64 -t aws_rag_app .
docker run --rm -p 8000:8000 --entrypoint python --env-file .env aws_rag_app app_api_handler.py

cdk init app --language=typescript

cdk bootstrap --region us-east-1
cdk deploy

RagCdkInfraStack

# Ask:

Query id to be random ?

# TO DO:

make get_query in handler like this:
@app.get("/query/{query_id}")
