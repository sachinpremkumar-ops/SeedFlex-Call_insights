from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from src.graph import graph
from src.async_processor import processor, ProcessingJob, ProcessingStatus
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Processing API", version="1.0.0")

class ProcessRequest(BaseModel):
    messages: str
    audio_file_key: Optional[str] = None
    skip_ingestion: Optional[bool] = False
    async_processing: Optional[bool] = False  # New: enable async processing

class ProcessResponse(BaseModel):
    job_id: Optional[str] = None
    status: str
    message: str
    result: Optional[Dict[str, Any]] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

@app.get("/")
def index():
    return {"message": "Audio Processing API", "version": "1.0.0"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "active_jobs": len(processor.active_jobs)}

@app.post("/process", response_model=ProcessResponse)
async def process(request: ProcessRequest):
    """
    Process audio file synchronously or asynchronously
    """
    try:
        if request.async_processing:
            # Async processing - return job ID immediately
            if not request.audio_file_key:
                raise HTTPException(status_code=400, detail="audio_file_key required for async processing")
            
            job_id = await processor.submit_job(
                file_key=request.audio_file_key,
                messages=request.messages
            )
            
            return ProcessResponse(
                job_id=job_id,
                status="submitted",
                message=f"Job {job_id} submitted for async processing"
            )
        else:
            # Synchronous processing - return result immediately
            initial_state = {
                "messages": [HumanMessage(content=request.messages)]
            }
            
            if request.audio_file_key:
                initial_state["audio_files"] = {
                    "key": request.audio_file_key,
                    "bucket": "experiment2407",
                    "original_key": request.audio_file_key
                }
            
            response = graph.invoke(initial_state)
            
            return ProcessResponse(
                status="completed",
                message="Processing completed successfully",
                result=response
            )
            
    except Exception as e:
        logger.error(f"Error in process endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Get the status of a specific job"""
    job = processor.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error_message=job.error_message,
        result=job.result
    )

@app.get("/jobs")
def get_all_jobs():
    """Get all jobs (active and completed)"""
    jobs = processor.get_all_jobs()
    return {
        "total_jobs": len(jobs),
        "active_jobs": len(processor.active_jobs),
        "completed_jobs": len(processor.completed_jobs),
        "jobs": [
            {
                "job_id": job.job_id,
                "file_key": job.file_key,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error_message": job.error_message
            }
            for job in jobs.values()
        ]
    }

@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    """Cancel a pending or processing job"""
    success = processor.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
    
    return {"message": f"Job {job_id} cancelled successfully"}

@app.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    """Get the result of a completed job"""
    job = processor.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    return job.result




    