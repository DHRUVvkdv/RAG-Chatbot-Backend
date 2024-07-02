import os
import uvicorn
from fastapi import FastAPI, HTTPException
from mangum import Mangum
from pydantic import BaseModel
from query_model import QueryModel
from pinecone_db import initialize_pinecone, create_embeddings, query_pinecone, process_resume_pdf, process_all_pdfs, list_processed_files
from s3_utils import get_s3_buckets
import logging
import uuid
import time

WORKER_LAMBDA_NAME = os.environ.get("WORKER_LAMBDA_NAME", None)

app = FastAPI()
handler = Mangum(app)  # Entry point for AWS Lambda.

class SubmitQueryRequest(BaseModel):
    query_text: str

class EmbeddingRequest(BaseModel):
    sentences: list[str]

@app.get("/")
def index():
    return {"Hello": "World"}

@app.get("/get_query")
def get_query_endpoint(query_id: str) -> QueryModel:
    query = QueryModel.get_item(query_id)
    return query

@app.get("/get_s3")
def get_s3_endpoint():
    return get_s3_buckets()

@app.post("/initialize_pinecone")
def initialize_pinecone_endpoint():
    return initialize_pinecone()

@app.post("/create_embeddings")
def create_embeddings_endpoint(request: EmbeddingRequest):
    return create_embeddings(request.sentences)

@app.post("/query_pinecone")
def query_pinecone_endpoint(request: SubmitQueryRequest) -> QueryModel:
    return query_pinecone(request.query_text)

@app.post("/process_resume_pdf")
def process_resume_pdf_endpoint():
    return process_resume_pdf()

@app.post("/process_all_pdfs")
def process_all_pdfs_endpoint():
    return process_all_pdfs()

@app.post("/query_documents")
def query_documents_endpoint(request: SubmitQueryRequest) -> QueryModel:
    return query_pinecone(request.query_text)

@app.get("/list_processed_files")
def list_processed_files_endpoint():
    try:
        processed_files = list_processed_files()
        return {
            "status": "success",
            "processed_files": processed_files,
            "total_processed_files": len(processed_files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = 8000
    print(f"Running the FastAPI server on port {port}.")
    uvicorn.run("app_api_handler:app", host="0.0.0.0", port=port)