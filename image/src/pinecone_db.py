import os
from fastapi import HTTPException
from pinecone import Pinecone, ServerlessSpec
from query_model import QueryModel
from rag_app.get_embedding_function import get_embedding_function
from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
import logging
import uuid
import time
from s3_utils import get_pdf_from_s3, list_pdfs_in_s3
from pdf_utils import process_pdf
import json

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "chatbot-index")
# BEDROCK_MODEL_ID = "meta.llama3-8b-instruct-v1:0"
BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

pc = Pinecone(api_key=PINECONE_API_KEY)

# Add this function to pinecone_db.py

def list_processed_files():
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Get index stats
        stats = index.describe_index_stats()
        total_vector_count = stats['total_vector_count']
        
        logging.info(f"Total vectors in index: {total_vector_count}")
        
        if total_vector_count == 0:
            return {
                "status": "success",
                "message": "No vectors found in the index.",
                "processed_files": [],
                "total_processed_files": 0
            }

        # Fetch all vectors in batches of 1000
        batch_size = 1000
        processed_files = set()
        
        for i in range(0, total_vector_count, batch_size):
            batch_ids = [str(j) for j in range(i, min(i+batch_size, total_vector_count))]
            vectors = index.fetch(ids=batch_ids)
            
            for _, vector in vectors['vectors'].items():
                metadata = vector.get('metadata', {})
                if 'source' in metadata:
                    processed_files.add(metadata['source'].split('/')[-1])  # Add just the filename
            
            logging.info(f"Processed batch {i//batch_size + 1}, total files found: {len(processed_files)}")

        processed_files_list = list(processed_files)
        
        return processed_files_list  # Return just the list of files

    except Exception as e:
        logging.error(f"Failed to list processed files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list processed files: {str(e)}")

def initialize_pinecone():
    try:
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536,  # Adjust this to match your embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east1-free"
                )
            )
        return {"status": "success", "message": "Pinecone initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Pinecone: {str(e)}")

def create_embeddings(sentences):
    try:
        embedding_function = get_embedding_function()
        embedding_vectors = embedding_function.embed_documents(sentences)
        
        index = pc.Index(PINECONE_INDEX_NAME)
        
        vectors_to_upsert = []
        for i, (text, vector) in enumerate(zip(sentences, embedding_vectors)):
            vectors_to_upsert.append((f"vec_{i}", vector, {"text": text}))
        
        index.upsert(vectors=vectors_to_upsert)
        
        return {"status": "success", "message": f"Added {len(sentences)} embeddings to Pinecone"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {str(e)}")

def query_pinecone(query_text, top_k=5):
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        
        embedding_function = get_embedding_function()
        query_embedding = embedding_function.embed_query(query_text)
        
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        
        context_text = "\n\n---\n\n".join([match.metadata['text'] for match in results.matches])
        
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        model = ChatBedrock(model_id=BEDROCK_MODEL_ID)
        response = model.invoke(prompt)
        response_text = response.content
        
        sources = [match.id for match in results.matches]
        
        new_query = QueryModel(
            query_id=uuid.uuid4().hex,
            create_time=int(time.time()),
            query_text=query_text,
            answer_text=response_text,
            sources=sources,
            is_complete=True
        )
        
        logging.info(f"Query processed successfully: {new_query.query_id}")
        
        return new_query
    except Exception as e:
        logging.error(f"Failed to query Pinecone: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to query Pinecone: {str(e)}")

def process_resume_pdf():
    try:
        pdf_content = get_pdf_from_s3("resume.pdf")
        chunks = process_pdf(pdf_content, "resume.pdf")

        index = pc.Index(PINECONE_INDEX_NAME)
        embedding_function = get_embedding_function()

        vectors_to_upsert = []
        for chunk in chunks:
            embedding = embedding_function.embed_query(chunk.page_content)
            vectors_to_upsert.append((
                chunk.metadata["id"],
                embedding,
                {
                    "text": chunk.page_content,
                    "source": chunk.metadata["source"],
                    "page": chunk.metadata.get("page", 0)
                }
            ))

        index.upsert(vectors=vectors_to_upsert)

        return {"status": "success", "message": f"Processed and added {len(chunks)} chunks from resume.pdf to Pinecone"}

    except Exception as e:
        logging.error(f"Failed to process resume PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process resume PDF: {str(e)}")

def process_pdf_file(pdf_key):
    try:
        pdf_content = get_pdf_from_s3(pdf_key)
        chunks = process_pdf(pdf_content, pdf_key)

        index = pc.Index(PINECONE_INDEX_NAME)
        embedding_function = get_embedding_function()

        vectors_to_upsert = []
        for chunk in chunks:
            embedding = embedding_function.embed_query(chunk.page_content)
            vectors_to_upsert.append((
                chunk.metadata["id"],
                embedding,
                {
                    "text": chunk.page_content,
                    "source": chunk.metadata["source"],
                    "page": chunk.metadata.get("page", 0)
                }
            ))

        index.upsert(vectors=vectors_to_upsert)
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
            pdf_name = pdf_key.split('/')[-1]  # Extract filename from the full path
            if pdf_name not in processed_files:
                try:
                    pdf_content = get_pdf_from_s3(pdf_name)  # Note: get_pdf_from_s3 expects just the filename
                    chunks = process_pdf(pdf_content, pdf_name)

                    index = pc.Index(PINECONE_INDEX_NAME)
                    embedding_function = get_embedding_function()

                    vectors_to_upsert = []
                    for chunk in chunks:
                        embedding = embedding_function.embed_query(chunk.page_content)
                        vectors_to_upsert.append((
                            chunk.metadata["id"],
                            embedding,
                            {
                                "text": chunk.page_content,
                                "source": chunk.metadata["source"],
                                "page": chunk.metadata.get("page", 0)
                            }
                        ))

                    index.upsert(vectors=vectors_to_upsert)
                    
                    total_chunks_added += len(chunks)
                    newly_processed_files += 1
                    newly_processed_file_details.append(f"{pdf_name}: {len(chunks)} chunks")
                except Exception as e:
                    logging.error(f"Failed to process PDF {pdf_name}: {str(e)}")
                    failed_files.append(pdf_name)

        return {
            "status": "success", 
            "message": f"Processed {newly_processed_files} new PDF files and added {total_chunks_added} chunks to Pinecone.",
            "details": {
                "already_processed_files": already_processed_files,
                "newly_processed_files": newly_processed_files,
                "total_chunks_added": total_chunks_added,
                "newly_processed_file_details": newly_processed_file_details,
                "failed_files": failed_files
            }
        }

    except Exception as e:
        logging.error(f"Failed to process PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDFs: {str(e)}")
