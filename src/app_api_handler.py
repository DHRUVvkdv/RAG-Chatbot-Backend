from fastapi import FastAPI
from pydantic import BaseModel
from rag_app.query_rag import QueryResponse, query_rag
import uvicorn

app = FastAPI()

class SubmitQueryRequest(BaseModel):
    query_text: str

@app.get("/")
def index():
    return {"data": "Hello World"}

@app.post("/submit_query")
def submit_query(request: SubmitQueryRequest) -> QueryResponse:
    response = query_rag(request.query_text)
    return response

if __name__ == "__main__":
    port = 8000
    print(f"🚀 Starting server on port {port}")
    uvicorn.run("app_api_handler:app", host="0.0.0.0", port=port)
