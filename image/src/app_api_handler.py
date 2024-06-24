import uvicorn
from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel
from query_model import QueryModel
from rag_app.query_rag import query_rag
import logging

app = FastAPI()
handler = Mangum(app)  # Entry point for AWS Lambda.


class SubmitQueryRequest(BaseModel):
    query_text: str


@app.get("/")
def index():
    return {"Hello": "World"}


@app.get("/get_query")
def get_query_endpoint(query_id: str) -> QueryModel:
    query = QueryModel.get_item(query_id)
    return query


@app.post("/submit_query")
def submit_query_endpoint(request: SubmitQueryRequest) -> QueryModel:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    try:
        query_response = query_rag(request.query_text)
        
        # Create the query item, and put it into the database.
        new_query = QueryModel(
            query_text=request.query_text,
            answer_text=query_response.response_text,
            sources=query_response.sources,
            is_complete=True,
        )
        new_query.put_item()
        logger.info(f"Successfully created and stored new query: {new_query.query_id}")
        return new_query
    except Exception as e:
        logger.error(f"Error in submit_query_endpoint: {str(e)}")
        # Return a default response instead of None
        return QueryModel(
            query_text=request.query_text,
            answer_text="An error occurred while processing the query.",
            sources=[],
            is_complete=False,
        )

if __name__ == "__main__":
    # Run this as a server directly.
    port = 8000
    print(f"Running the FastAPI server on port {port}.")
    uvicorn.run("app_api_handler:app", host="0.0.0.0", port=port)