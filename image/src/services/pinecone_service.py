import os
from fastapi import HTTPException
from pinecone import Pinecone, ServerlessSpec
from models.query import QueryModel
from utils.embeddings import get_embedding_function
from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
import logging
import uuid
import time
from utils.s3_handler import get_pdf_from_s3, list_pdfs_in_s3
from utils.pdf_processor import process_pdf
import boto3
from botocore.exceptions import ClientError
import requests
from typing import Dict, Any
from pinecone import PineconeException
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from typing import List

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
# Activated this on 2025-04-18 on LEWAS AWS account
# BEDROCK_MODEL_ID = "meta.llama3-1-8b-instruct-v1:0"
# doesnt work ->
# BEDROCK_MODEL_ID = "meta.llama3-8b-instruct-v1:0"
# doesnt work because of this error:
# ValueError: Error raised by bedrock service: An error occurred (ValidationException) when calling the InvokeModel operation: Invocation of model ID meta.llama3-2-3b-instruct-v1:0 with on-demand throughput isn’t supported. Retry your request with the ID or ARN of an inference profile that contains this model.
# ERROR:models.query:ClientError in put_item: Requested resource not found
# BEDROCK_MODEL_ID = "meta.llama3-2-3b-instruct-v1:0"
# BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
# BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0" - potential error
topk = 7

# Currently working models:
# BEDROCK_MODEL_ID = "meta.llama3-8b-instruct-v1:0"
BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"


# Use host.docker.internal to refer to the host machine from within the container
FILE_ID_SERVICE_URL = "http://host.docker.internal:8001"

MIN_QUERY_LENGTH = 10  # Minimum number of characters for a query
MAX_RETRIES = 3
EMBEDDING_DIMENSION = 1024


dynamodb = boto3.resource("dynamodb")
processedFilesTable = dynamodb.Table("lewas-chatbot-processed-files")
queriesTable = dynamodb.Table("RagCdkInfraStack-QueriesTable7395E8FA-17BGT2YQ1QX1F")

# PROMPT_TEMPLATE = """
# Context information:
# {context}

# Human query: {question}

# AI Assistant: You are an AI-assisted chatbot developed for the Learning Enhanced Watershed Assessment System (LEWAS) lab at Virginia Tech. Your purpose is to provide efficient access to LEWAS information, assisting with onboarding new students and answering queries about the lab's vast amount of data and research publications. Use the following guidelines to formulate your response:

# 1. Directly address the user's query using the provided context and your knowledge about LEWAS.
# 2. Focus on providing information about LEWAS operations, research, environmental data, and related topics.
# 3. If asked about the lab's history or scope, mention that LEWAS has accumulated data and research publications from undergraduate and graduate students across various departments for over a decade.
# 4. When discussing technical aspects of LEWAS research or data, aim for clarity and accessibility in your explanations.
# 5. If the query relates to onboarding or learning about LEWAS, emphasize how you can help provide interactive and efficient access to information.
# 6. For questions about environmental monitoring or watershed assessment, provide relevant information from the LEWAS context.
# 7. If asked about your own capabilities, you can briefly mention that you use natural language processing, AWS services, and a vector database to process queries and retrieve information.
# 8. If the query is unclear or outside the scope of LEWAS information, politely ask for clarification.
# 9. Maintain a helpful and educational tone, suitable for assisting students and researchers at various levels of familiarity with LEWAS.
# 10. If you're unsure about any specific details, it's okay to say so and suggest where the user might find more information within the LEWAS resources.

# Now, please provide a clear, informative response to the query:
# """

# PROMPT_TEMPLATE = """
# Context: {context}

# Query: {question}

# You are an AI chatbot for the LEWAS (Learning Enhanced Watershed Assessment System) lab at Virginia Tech. Provide accurate answers about LEWAS research, data, and operations. Use the context provided. If unsure, say so. Keep responses clear and suitable for students and researchers.

# Response:
# """

PROMPT_TEMPLATE = """
Context: {context}

Query: {question}

You are an AI chatbot for the LEWAS (Learning Enhanced Watershed Assessment System) lab at Virginia Tech. Your role:

1. Provide accurate information about LEWAS research, data, and operations.
2. Use ONLY the provided context to answer queries.
4. Keep responses informative.
5. If unsure, admit it and suggest where to find more information.
6. Maintain a tone suitable for students and researchers.

Answer the query based on these guidelines:
"""

pc = Pinecone(api_key=PINECONE_API_KEY)

# Add this function to pinecone_db.py


def list_processed_files():
    try:
        index = pc.Index(PINECONE_INDEX_NAME)

        # Get index stats
        stats = index.describe_index_stats()
        total_vector_count = stats["total_vector_count"]

        logging.info(f"Total vectors in index: {total_vector_count}")

        if total_vector_count == 0:
            return {
                "status": "success",
                "message": "No vectors found in the index.",
                "processed_files": [],
                "total_processed_files": 0,
            }

        # Fetch all vectors in batches of 1000
        batch_size = 1000
        processed_files = set()

        for i in range(0, total_vector_count, batch_size):
            batch_ids = [
                str(j) for j in range(i, min(i + batch_size, total_vector_count))
            ]
            vectors = index.fetch(ids=batch_ids)

            for _, vector in vectors["vectors"].items():
                metadata = vector.get("metadata", {})
                if "source" in metadata:
                    processed_files.add(
                        metadata["source"].split("/")[-1]
                    )  # Add just the filename

            logging.info(
                f"Processed batch {i//batch_size + 1}, total files found: {len(processed_files)}"
            )

        processed_files_list = list(processed_files)

        return processed_files_list  # Return just the list of files

    except Exception as e:
        logging.error(f"Failed to list processed files: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list processed files: {str(e)}"
        )


def initialize_pinecone():
    try:
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east1-free"),
            )
        return {"status": "success", "message": "Pinecone initialized successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize Pinecone: {str(e)}"
        )


def create_embeddings(sentences):
    try:
        embedding_function = get_embedding_function()
        embedding_vectors = embedding_function.embed_documents(sentences)

        index = pc.Index(PINECONE_INDEX_NAME)

        vectors_to_upsert = []
        for i, (text, vector) in enumerate(zip(sentences, embedding_vectors)):
            vectors_to_upsert.append((f"vec_{i}", vector, {"text": text}))

        index.upsert(vectors=vectors_to_upsert)

        return {
            "status": "success",
            "message": f"Added {len(sentences)} embeddings to Pinecone",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create embeddings: {str(e)}"
        )


# Helper function to get embedding
@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
def get_embedding(query_text: str) -> List[float]:
    embedding_function = get_embedding_function()
    embedding = embedding_function.embed_query(query_text)
    if not embedding or len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError("Failed to generate valid embedding")
    return embedding


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
def perform_pinecone_query(vector: List[float], top_k: int) -> List[dict]:
    index = pc.Index(PINECONE_INDEX_NAME)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    logging.info(f"Pinecone query results: {results}")  # Log the entire results object
    if not results.matches:
        logging.info("No matches found in Pinecone query results")
        return []  # Return an empty list instead of raising an exception
    return results.matches


# Main query function
def query_pinecone(query_text: str, top_k: int = topk) -> QueryModel:
    query_model = QueryModel(query_text=query_text)

    try:
        # Input validation
        if len(query_text.strip()) < MIN_QUERY_LENGTH:
            query_model.answer_text = f"Your query is too short. Please provide a query with at least {MIN_QUERY_LENGTH} characters."
            query_model.is_complete = False
            return query_model

        # Get embedding
        try:
            logging.info(f"Generating embedding for query: {query_text}")
            query_embedding = get_embedding(query_text)
            logging.info(
                f"Embedding generated successfully. Dimension: {len(query_embedding)}"
            )
        except Exception as e:
            logging.error(f"Embedding creation failed: {str(e)}", exc_info=True)
            query_model.answer_text = (
                "Unable to process your query. Please try rephrasing it."
            )
            query_model.is_complete = False
            return query_model

        # Query Pinecone
        try:
            logging.info(f"Querying Pinecone with top_k={top_k}")
            matches = perform_pinecone_query(query_embedding, top_k)
            logging.info(f"Received {len(matches)} matches from Pinecone")
        except Exception as e:
            logging.error(f"Pinecone query failed: {str(e)}", exc_info=True)
            query_model.answer_text = "An error occurred while searching for relevant information. Please try again later."
            query_model.is_complete = False
            return query_model

        # Process results
        if not matches:
            logging.info(f"No matches found for query: {query_text}")
            query_model.answer_text = "No relevant results found. Please try rephrasing your query or using different keywords."
            query_model.is_complete = False
            return query_model

        context_text = "\n\n---\n\n".join(
            [
                match.metadata.get("text", "")
                for match in matches
                if "text" in match.metadata
            ]
        )
        query_model.sources = [
            f"{match.metadata.get('source', 'Unknown')} (Page {match.metadata.get('page', 'N/A')}) - {match.metadata.get('google_drive_link', 'No link available')}"
            for match in matches
        ]

        # Generate response using AI model
        try:
            logging.info("Generating AI response")
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)
            model = ChatBedrock(model_id=BEDROCK_MODEL_ID)
            response = model.invoke(prompt)
            query_model.answer_text = response.content
            query_model.is_complete = True
            logging.info("AI response generated successfully")
        except Exception as e:
            logging.error(f"Error generating AI response: {str(e)}", exc_info=True)
            query_model.answer_text = "An error occurred while generating the response. Please try again later."
            query_model.is_complete = False

        # Save the successful query to the database
        try:
            save_query_result(query_model)
            logging.info(f"Query saved successfully: {query_text}")
        except Exception as e:
            logging.error(f"Failed to save query to database: {str(e)}")
            # Note: We're not returning here because the query itself was successful

        return query_model

    except Exception as e:
        logging.error(f"Unexpected error in query_pinecone: {str(e)}", exc_info=True)
        query_model.answer_text = (
            "An unexpected error occurred. Please try again later."
        )
        query_model.is_complete = False
        return query_model


# Function to save query result to database
def save_query_result(query_model: QueryModel):
    try:
        query_model.put_item()
    except Exception as e:
        logging.error(f"Error saving query result to database: {str(e)}")


# Main handler function
def handle_query(query_text: str) -> dict:
    query_model = query_pinecone(query_text)
    save_query_result(query_model)
    return query_model.dict()


def process_resume_pdf():
    try:
        pdf_content = get_pdf_from_s3("resume.pdf")
        chunks = process_pdf(pdf_content, "resume.pdf")

        index = pc.Index(PINECONE_INDEX_NAME)
        embedding_function = get_embedding_function()

        vectors_to_upsert = []
        for chunk in chunks:
            embedding = embedding_function.embed_query(chunk.page_content)
            vectors_to_upsert.append(
                (
                    chunk.metadata["id"],
                    embedding,
                    {
                        "text": chunk.page_content,
                        "source": chunk.metadata["source"],
                        "page": chunk.metadata.get("page", 0),
                    },
                )
            )

        index.upsert(vectors=vectors_to_upsert)

        return {
            "status": "success",
            "message": f"Processed and added {len(chunks)} chunks from resume.pdf to Pinecone",
        }

    except Exception as e:
        logging.error(f"Failed to process resume PDF: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process resume PDF: {str(e)}"
        )


def process_pdf_file(pdf_key):
    try:
        pdf_content = get_pdf_from_s3(pdf_key)
        chunks = process_pdf(pdf_content, pdf_key)

        index = pc.Index(PINECONE_INDEX_NAME)
        embedding_function = get_embedding_function()

        google_drive_link = get_google_drive_link_pdf(pdf_key)

        vectors_to_upsert = []
        for chunk in chunks:
            embedding = embedding_function.embed_query(chunk.page_content)
            metadata = chunk.metadata.copy()  # Start with the existing metadata
            metadata.update(
                {
                    "text": chunk.page_content,
                    "google_drive_link": google_drive_link,
                    "processed_date": int(time.time()),
                }
            )
            vectors_to_upsert.append((metadata["id"], embedding, metadata))

        index.upsert(vectors=vectors_to_upsert)
        # Add the processed file to DynamoDB
        add_processed_file_dynamodb(pdf_key)
        return len(chunks)
    except Exception as e:
        logging.error(f"Failed to process PDF {pdf_key}: {str(e)}")
        return 0


def process_all_pdfs():
    try:
        available_pdfs = list_pdfs_in_s3()
        processed_files = list_processed_files()

        total_chunks_added = 0
        newly_processed_files = 0
        already_processed_files = len(processed_files)
        newly_processed_file_details = []
        failed_files = []

        for pdf_key in available_pdfs:
            pdf_name = pdf_key.split("/")[-1]  # Extract filename from the full path
            if pdf_name not in processed_files:
                chunks_added = process_pdf_file(pdf_name)
                if chunks_added > 0:
                    total_chunks_added += chunks_added
                    newly_processed_files += 1
                    newly_processed_file_details.append(
                        f"{pdf_name}: {chunks_added} chunks"
                    )
                else:
                    failed_files.append(pdf_name)

        return {
            "status": "success",
            "message": f"Processed {newly_processed_files} new PDF files and added {total_chunks_added} chunks to Pinecone.",
            "details": {
                "already_processed_files": already_processed_files,
                "newly_processed_files": newly_processed_files,
                "total_chunks_added": total_chunks_added,
                "newly_processed_file_details": newly_processed_file_details,
                "failed_files": failed_files,
            },
        }

    except Exception as e:
        logging.error(f"Failed to process PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDFs: {str(e)}")


def list_processed_files():
    try:
        response = processedFilesTable.scan()
        processed_files = [item["filename"] for item in response["Items"]]

        return processed_files
    except ClientError as e:
        logging.error(f"Failed to list processed files from DynamoDB: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list processed files: {str(e)}"
        )


def add_processed_file_dynamodb(filename):
    try:
        processedFilesTable.put_item(Item={"filename": filename})
    except ClientError as e:
        logging.error(f"Failed to add processed file to DynamoDB: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to add processed file: {str(e)}"
        )


def get_google_drive_link_pdf(pdf_name):
    try:
        response = requests.get(
            f"{FILE_ID_SERVICE_URL}/file_id",
            params={"file_type": "pdf", "file_name": pdf_name},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        file_id = data.get("id", "")
        if file_id:
            return f"https://drive.google.com/file/d/{file_id}/view"
        else:
            logging.warning(f"No file_id found for {pdf_name}")
            return ""
    except requests.RequestException as e:
        logging.error(f"Failed to fetch Google Drive link for {pdf_name}: {str(e)}")
        return ""
    except Exception as e:
        logging.error(
            f"Unexpected error fetching Google Drive link for {pdf_name}: {str(e)}"
        )
        return ""


def update_missing_drive_links():
    index = pc.Index(PINECONE_INDEX_NAME)

    batch_size = 1000
    total_updated = 0
    cursor = None

    while True:
        try:
            query_response = index.query(
                vector=[0] * 1024,
                top_k=batch_size,
                include_metadata=True,
                include_values=True,
                filter={},
                cursor=cursor,
            )

            if not query_response.matches:
                break

            updates = []
            for match in query_response.matches:
                should_update = False
                new_metadata = dict(match.metadata)

                if (
                    "google_drive_link" not in new_metadata
                    or not new_metadata["google_drive_link"]
                ):
                    should_update = True

                if should_update:
                    pdf_name = new_metadata.get("source", "").split("/")[-1]
                    if pdf_name:
                        drive_link = get_google_drive_link_pdf(pdf_name)
                        if drive_link:
                            new_metadata["google_drive_link"] = drive_link
                            updates.append(
                                {
                                    "id": match.id,
                                    "values": match.values,
                                    "metadata": new_metadata,
                                }
                            )

            if updates:
                upsert_response = index.upsert(vectors=updates)
                total_updated += len(updates)
                logging.info(f"Upsert response: {upsert_response}")

            logging.info(
                f"Updated {len(updates)} vectors in this batch. Total updated: {total_updated}"
            )

            cursor = query_response.cursor
            if cursor is None:
                break

        except PineconeException as e:
            logging.error(f"Pinecone error during batch update: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during batch update: {str(e)}")
            raise

    return total_updated


def update_drive_link_for_file(file_name: str, drive_link: str):
    try:
        index = pc.Index(PINECONE_INDEX_NAME)

        # Query for vectors with the given file name
        query_response = index.query(
            vector=[0] * 1024,  # Use a zero vector for querying
            top_k=10000,
            include_metadata=True,
            include_values=True,  # Include vector values in the response
            filter={"source": {"$eq": file_name}},
        )

        logging.info(f"Query response for {file_name}: {query_response}")

        if not query_response.matches:
            logging.warning(f"No matches found for file: {file_name}")
            return 0

        updates = []
        for match in query_response.matches:
            # Create a new metadata dictionary
            new_metadata = dict(match.metadata)
            new_metadata["google_drive_link"] = drive_link
            updates.append(
                {
                    "id": match.id,
                    "values": match.values,  # Include the original vector values
                    "metadata": new_metadata,
                }
            )

        if updates:
            # Use upsert with the correct format
            upsert_response = index.upsert(vectors=updates)
            logging.info(f"Upsert response: {upsert_response}")
            return len(updates)
        else:
            return 0
    except PineconeException as e:
        logging.error(f"Pinecone error updating drive link for {file_name}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Failed to update drive link for {file_name}: {str(e)}")
        raise


def delete_all_vectors() -> Dict[str, Any]:
    """
    Delete all vectors in the Pinecone index.

    Returns:
        Dict[str, Any]: Response from Pinecone delete operation
    """
    try:
        # Initialize Pinecone index
        index = pc.Index(PINECONE_INDEX_NAME)

        # Get all vector IDs from the index
        # Using a small batch size for the query to retrieve IDs
        query_response = index.query(
            vector=[0] * 1024,  # Dummy vector with the correct dimension
            top_k=10000,  # Maximum number of results
            include_metadata=False,
            include_values=False,
        )

        # Extract all IDs
        ids_to_delete = [match.id for match in query_response.matches]

        # If there are no vectors, return early
        if not ids_to_delete:
            return {"status": "success", "message": "No vectors found to delete"}

        # Delete all vectors by their IDs
        delete_response = index.delete(ids=ids_to_delete)

        return {
            "status": "success",
            "message": f"Successfully deleted {len(ids_to_delete)} vectors",
            "response": delete_response,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def classify_query(query_text: str) -> str:
    """
    Use LLM to classify the query and determine which handler to use.
    Returns:
    - "LIVE_DATA:medium:metric" for live data queries
    - "VISUALIZATION:medium:metric" for visualization queries
    - "RAG" for document-based queries
    """
    CLASSIFICATION_PROMPT = """
    You are an AI assistant for the LEWAS lab at Virginia Tech. 
    Determine if this query requires:
    
    1. Live weather station data (asking about current temperature, humidity, pressure, etc.)
    2. Visualization of weather data (asking for graphs, charts, trends)
    3. Document-based information (asking about LEWAS lab, research, operations, etc.)
    
    Query: {question}
    
    Respond ONLY with one of these formats:
    - LIVE_DATA:medium:metric (e.g., LIVE_DATA:air:temperature)
    - VISUALIZATION:medium:metric (e.g., VISUALIZATION:air:humidity)
    - RAG
    """

    try:
        prompt_template = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT)
        prompt = prompt_template.format(question=query_text)
        model = ChatBedrock(model_id=BEDROCK_MODEL_ID)
        response = model.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logging.error(f"Error classifying query: {str(e)}")
        return "RAG"  # Default to RAG on error
