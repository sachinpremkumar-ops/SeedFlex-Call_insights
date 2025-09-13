from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from src.graph import graph
from typing import Optional

app= FastAPI()

class Process(BaseModel):
    messages: str
    audio_file_key: Optional[str] = None  # Optional: specify a specific audio file

@app.get("/")
def index():
    return {"message": "Hello World"}

@app.post("/process")
async def process(request: Process):
    # Create initial state with the message
    initial_state = {
        "messages": [HumanMessage(content=request.messages)]
    }
    
    # If a specific audio file is provided, add it to the state
    if request.audio_file_key:
        initial_state["audio_file_key"] = request.audio_file_key
    
    # If skip_ingestion is True, we could potentially start from a different node
    # For now, the workflow will still go through Ingestion_Agent but it will handle existing files
    
    # Use ainvoke for async parallel processing
    response = await graph.ainvoke(initial_state)
    return response




    