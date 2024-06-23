# RAG-Chatbot-Backend

making the backend server for AI assisted chatbot

have text overlap and include file source link in metadata

docker build --platform linux/amd64 -t aws_rag_app .
docker run --rm -p 8000:8000 --entrypoint python --env-file .env aws_rag_app app_api_handler.py

cdk init app --language=typescript

cdk deploy
