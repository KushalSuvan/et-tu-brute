from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pipeline import Pipeline

pipe = Pipeline()

app = FastAPI(
    title="Shakespeare Query",
    description="Query on the Shaky stories",
    version="1.0.0"
)
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]



def run_rag_pipeline(query: str) -> QueryResponse:
    response, docs = pipe(query)

    return QueryResponse(
        answer=response,
        sources=docs
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        response = run_rag_pipeline(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

