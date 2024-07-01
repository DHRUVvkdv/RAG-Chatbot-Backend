import os
import uvicorn
import boto3
import json
import lancedb
import pyarrow as pa
from fastapi import FastAPI, HTTPException
from mangum import Mangum
from pydantic import BaseModel
from query_model import QueryModel
from rag_app.get_embedding_function import get_embedding_function
from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
import pandas as pd
import logging
import uuid
import time
from botocore.exceptions import ClientError
from io import BytesIO
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


WORKER_LAMBDA_NAME = os.environ.get("WORKER_LAMBDA_NAME", None)
# BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
BEDROCK_MODEL_ID = "meta.llama3-8b-instruct-v1:0"
	


app = FastAPI()
handler = Mangum(app)  # Entry point for AWS Lambda.

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


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
    s3 = boto3.client("s3")
    response = s3.list_buckets()
    return response


@app.post("/create_lance_db")
def create_lance_db_endpoint():
    try:
        S3_BUCKET = os.environ.get("S3_BUCKET", "lewas-chatbot")
        LANCEDB_URI = f"s3://{S3_BUCKET}/data"
        db = lancedb.connect(LANCEDB_URI)
        
        # Optionally, create a table here if needed
        # For example:
        db.create_table("try_table", data=[{"id": 1, "name": "test"}])
        
        return {"status": "success", "message": "Database connected successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create database: {str(e)}")

@app.get("/check_lance_db")
def check_lance_db_endpoint():
    try:
        S3_BUCKET = os.environ.get("S3_BUCKET", "lewas-chatbot")
        LANCEDB_URI = f"s3://{S3_BUCKET}/data"
        db = lancedb.connect(LANCEDB_URI)
        
        # Check if the table exists
        if "try_table" in db.table_names():
            table = db.open_table("try_table")
            row_count = len(table)
            return {"status": "success", "message": f"Table 'try_table' exists with {row_count} rows"}
        else:
            return {"status": "failure", "message": "Table 'try_table' does not exist"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check database: {str(e)}")
    
@app.post("/create_embeddings")
def create_embeddings_endpoint(request: EmbeddingRequest):
    try:
        # Use the same embedding function as in your Chroma setup
        embedding_function = get_embedding_function()
        
        # Generate embeddings for the provided sentences
        embedding_vectors = embedding_function.embed_documents(request.sentences)
        
        # Connect to the LanceDB
        S3_BUCKET = os.environ.get("S3_BUCKET", "lewas-chatbot")
        LANCEDB_URI = f"s3://{S3_BUCKET}/data"
        db = lancedb.connect(LANCEDB_URI)
        
        # Define the schema using pyarrow
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), len(embedding_vectors[0])))
        ])
        
        # Open or create the 'try_model' table
        if "try_model" not in db.table_names():
            table = db.create_table("try_model", schema=schema)
        else:
            table = db.open_table("try_model")
        
        # Get the current count of rows in the table
        current_count = table.to_pandas().shape[0]
        
        # Create a DataFrame with the text and vector data
        df = pd.DataFrame({
            "id": range(current_count, current_count + len(request.sentences)),
            "text": request.sentences,
            "vector": embedding_vectors
        })
        
        # Add the new embeddings to the table
        table.add(df)
        
        return {"status": "success", "message": f"Added {len(request.sentences)} embeddings to try_model"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {str(e)}")
    
@app.post("/query_lance")
def query_lance_endpoint(request: SubmitQueryRequest) -> QueryModel:
    try:
        # Connect to LanceDB
        S3_BUCKET = os.environ.get("S3_BUCKET", "lewas-chatbot")
        LANCEDB_URI = f"s3://{S3_BUCKET}/data"
        db = lancedb.connect(LANCEDB_URI)
        
        # Open the 'try_model' table
        table = db.open_table("try_model")
        
        # Get the embedding function
        embedding_function = get_embedding_function()
        
        # Generate embedding for the query
        query_embedding = embedding_function.embed_query(request.query_text)
        
        # Perform similarity search
        results = table.search(query_embedding).limit(3).to_df()
        
        # Prepare context from search results
        context_text = "\n\n---\n\n".join(results['text'].tolist())
        
        # Prepare prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=request.query_text)
        
        # Generate response using ChatBedrock
        model = ChatBedrock(model_id=BEDROCK_MODEL_ID)
        response = model.invoke(prompt)
        response_text = response.content
        
        # # Prepare sources
        # sources = results['id'].tolist()
        # Prepare sources - convert integer IDs to strings
        sources = [str(id) for id in results['id'].tolist()]
        
        # Create QueryModel instance (without saving to DynamoDB)
        new_query = QueryModel(
            query_id=uuid.uuid4().hex,
            create_time=int(time.time()),
            query_text=request.query_text,
            answer_text=response_text,
            sources=sources,
            is_complete=True
        )
        
        logging.info(f"Query processed successfully: {new_query.query_id}")
        
        return new_query
    except Exception as e:
        logging.error(f"Failed to query LanceDB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to query LanceDB: {str(e)}")

@app.get("/list_pdfs")
def list_pdfs_endpoint():
    try:
        # Get the S3 bucket name
        S3_BUCKET = os.environ.get("S3_BUCKET", "lewas-chatbot")
        
        # Create an S3 client
        s3_client = boto3.client('s3')
        
        # List objects in the bucket with the specified prefix
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix='data/pdfs/'
        )
        
        # Extract file names from the response
        pdf_files = []
        if 'Contents' in response:
            for item in response['Contents']:
                # Extract just the filename from the full path
                filename = os.path.basename(item['Key'])
                if filename.lower().endswith('.pdf'):
                    pdf_files.append(filename)
        
        return {"status": "success", "pdf_files": pdf_files}
    except ClientError as e:
        error_message = f"Failed to list PDF files: {str(e)}"
        logging.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logging.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/submit_query")
def submit_query_endpoint(request: SubmitQueryRequest) -> QueryModel:
    # Create the query item, and put it into the data-base.
    new_query = QueryModel(query_text=request.query_text)

    if WORKER_LAMBDA_NAME:
        # Make an async call to the worker (the RAG/AI app).
        new_query.put_item()
        invoke_worker(new_query)
    else:
        # Make a synchronous call to the worker (the RAG/AI app).
        query_response = query_rag(request.query_text)
        new_query.answer_text = query_response.response_text
        new_query.sources = query_response.sources
        new_query.is_complete = True
        new_query.put_item()

    return new_query


def invoke_worker(query: QueryModel):
    # Initialize the Lambda client
    lambda_client = boto3.client("lambda")

    # Get the QueryModel as a dictionary.
    payload = query.model_dump()

    # Invoke another Lambda function asynchronously
    response = lambda_client.invoke(
        FunctionName=WORKER_LAMBDA_NAME,
        InvocationType="Event",
        Payload=json.dumps(payload),
    )

    print(f"✅ Worker Lambda invoked: {response}")

@app.post("/process_resume_pdf")
async def process_resume_pdf_endpoint():
    try:
        # S3 bucket and file details
        S3_BUCKET = os.environ.get("S3_BUCKET", "lewas-chatbot")
        PDF_KEY = "data/pdfs/resume.pdf"

        # Connect to S3
        s3 = boto3.client('s3')

        # Download the PDF file from S3
        response = s3.get_object(Bucket=S3_BUCKET, Key=PDF_KEY)
        pdf_content = response['Body'].read()

        # Use BytesIO to create a file-like object
        pdf_file = BytesIO(pdf_content)

        # Read PDF using PyPDF2
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from each page
        documents = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            documents.append(Document(page_content=text, metadata={"page": page_num + 1}))

        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=120,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)

        # Calculate chunk IDs
        chunks_with_ids = calculate_chunk_ids(chunks, "resume.pdf")

        # Connect to LanceDB
        LANCEDB_URI = f"s3://{S3_BUCKET}/data"
        db = lancedb.connect(LANCEDB_URI)

        # Open or create the 'documents' table
        if "documents" not in db.table_names():
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("source", pa.string()),
                pa.field("page", pa.int32()),
                pa.field("vector", pa.list_(pa.float32(), 1536))  # Adjust dimension as needed
            ])
            table = db.create_table("documents", schema=schema)
        else:
            table = db.open_table("documents")

        # Get embedding function
        embedding_function = get_embedding_function()

        # Add chunks to LanceDB
        for chunk in chunks_with_ids:
            embedding = embedding_function.embed_query(chunk.page_content)
            table.add([{
                "id": chunk.metadata["id"],
                "text": chunk.page_content,
                "source": chunk.metadata["source"],
                "page": chunk.metadata.get("page", 0),
                "vector": embedding
            }])

        return {"status": "success", "message": f"Processed and added {len(chunks)} chunks from resume.pdf to LanceDB"}

    except Exception as e:
        logging.error(f"Failed to process resume PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process resume PDF: {str(e)}")

def calculate_chunk_ids(chunks, filename):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = filename
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
        chunk.metadata["source"] = source

    return chunks  

@app.post("/query_resume")
async def query_resume_endpoint(request: SubmitQueryRequest) -> QueryModel:
    try:
        # Connect to LanceDB
        S3_BUCKET = os.environ.get("S3_BUCKET", "lewas-chatbot")
        LANCEDB_URI = f"s3://{S3_BUCKET}/data"
        db = lancedb.connect(LANCEDB_URI)
        
        # Open the 'documents' table where resume chunks are stored
        table = db.open_table("documents")
        
        # Get the embedding function
        embedding_function = get_embedding_function()
        
        # Generate embedding for the query
        query_embedding = embedding_function.embed_query(request.query_text)
        
        # Perform similarity search
        results = table.search(query_embedding).limit(3).to_df()
        
        # Prepare context from search results
        context_text = "\n\n---\n\n".join(results['text'].tolist())
        
        # Prepare prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=request.query_text)
        
        # Generate response using ChatBedrock
        model = ChatBedrock(model_id=BEDROCK_MODEL_ID)
        response = model.invoke(prompt)
        response_text = response.content
        
        # Prepare sources
        sources = results['id'].tolist()
        
        # Create QueryModel instance
        new_query = QueryModel(
            query_id=uuid.uuid4().hex,
            create_time=int(time.time()),
            query_text=request.query_text,
            answer_text=response_text,
            sources=sources,
            is_complete=True
        )
        
        logging.info(f"Resume query processed successfully: {new_query.query_id}")
        
        return new_query
    except Exception as e:
        logging.error(f"Failed to query resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to query resume: {str(e)}")


if __name__ == "__main__":
    # Run this as a server directly.
    port = 8000
    print(f"Running the FastAPI server on port {port}.")
    uvicorn.run("app_api_handler:app", host="0.0.0.0", port=port)