import asyncio
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
import json
import logging
from enum import Enum
from pydantic import BaseModel
from src.graph import graph
from src.utils.s3_utils import s3_get_audio_file
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize S3 client
s3_client = boto3.client('s3')
BUCKET_NAME = "experiment2407"

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProcessingJob(BaseModel):
    job_id: str
    file_key: str
    status: ProcessingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    progress: int = 0  # 0-100

class AsyncProcessor:
    def __init__(self):
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.completed_jobs: Dict[str, ProcessingJob] = {}
        self.max_concurrent_jobs = 5  # Configurable
        
    async def submit_job(self, file_key: str, messages: str = "Process audio file") -> str:
        """Submit a new processing job and return job ID"""
        job_id = str(uuid.uuid4())
        
        job = ProcessingJob(
            job_id=job_id,
            file_key=file_key,
            status=ProcessingStatus.PENDING,
            created_at=datetime.utcnow()
        )
        
        self.active_jobs[job_id] = job
        
        # Start processing asynchronously
        asyncio.create_task(self._process_job(job_id, file_key, messages))
        
        logger.info(f"Job {job_id} submitted for file {file_key}")
        return job_id
    
    async def _process_job(self, job_id: str, file_key: str, messages: str):
        """Process a single job asynchronously"""
        try:
            job = self.active_jobs[job_id]
            job.status = ProcessingStatus.PROCESSING
            job.started_at = datetime.utcnow()
            job.progress = 10
            
            logger.info(f"Starting processing for job {job_id}")
            
            # Check file size to determine processing strategy
            file_size = await self._get_file_size(file_key)
            
            if file_size > 1_000_000:  # 1MB+
                result = await self._process_large_file(job_id, file_key, messages)
            else:
                result = await self._process_standard_file(job_id, file_key, messages)
            
            # Mark as completed
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result = result
            job.progress = 100
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job
            del self.active_jobs[job_id]
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            job = self.active_jobs.get(job_id)
            if job:
                job.status = ProcessingStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                
                # Move to completed jobs (failed)
                self.completed_jobs[job_id] = job
                del self.active_jobs[job_id]
    
    async def _get_file_size(self, file_key: str) -> int:
        """Get file size from S3"""
        try:
            response = s3_client.head_object(Bucket=BUCKET_NAME, Key=file_key)
            return response['ContentLength']
        except Exception as e:
            logger.error(f"Error getting file size for {file_key}: {e}")
            return 0
    
    async def _process_standard_file(self, job_id: str, file_key: str, messages: str) -> Dict[str, Any]:
        """Process standard files (< 1MB) synchronously"""
        job = self.active_jobs[job_id]
        job.progress = 20
        
        # Create initial state
        initial_state = {
            "messages": [{"role": "user", "content": messages}],
            "audio_file_key": file_key
        }
        
        job.progress = 40
        
        # Run the graph synchronously (it's already fast for small files)
        result = graph.invoke(initial_state)
        
        job.progress = 80
        
        return result
    
    async def _process_large_file(self, job_id: str, file_key: str, messages: str) -> Dict[str, Any]:
        """Process large files (> 1MB) with chunking and async operations"""
        job = self.active_jobs[job_id]
        
        try:
            # Step 1: Download and chunk the audio file
            job.progress = 20
            audio_chunks = await self._chunk_audio_file(file_key)
            
            # Step 2: Process each chunk
            job.progress = 30
            chunk_results = []
            
            for i, chunk in enumerate(audio_chunks):
                chunk_progress = 30 + (i / len(audio_chunks)) * 50
                job.progress = int(chunk_progress)
                
                chunk_result = await self._process_audio_chunk(chunk, f"Process chunk {i+1}")
                chunk_results.append(chunk_result)
            
            # Step 3: Combine results
            job.progress = 80
            combined_result = await self._combine_chunk_results(chunk_results)
            
            job.progress = 90
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error processing large file {file_key}: {e}")
            raise
    
    async def _chunk_audio_file(self, file_key: str) -> list:
        """Split large audio file into chunks"""
        try:
            # Download the file
            audio_data = s3_get_audio_file(file_key)
            
            # For now, return the full file as a single chunk
            # In production, you'd implement actual audio chunking
            return [audio_data]
            
        except Exception as e:
            logger.error(f"Error chunking audio file {file_key}: {e}")
            raise
    
    async def _process_audio_chunk(self, chunk_data: bytes, messages: str) -> Dict[str, Any]:
        """Process a single audio chunk"""
        # This would process each chunk through the graph
        # For now, return a placeholder
        return {"chunk_processed": True, "data": chunk_data[:100]}  # Truncated for demo
    
    async def _combine_chunk_results(self, chunk_results: list) -> Dict[str, Any]:
        """Combine results from multiple chunks"""
        # This would combine transcription, analysis, etc.
        return {
            "combined_result": True,
            "chunks_processed": len(chunk_results),
            "final_analysis": "Combined analysis from all chunks"
        }
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get the status of a specific job"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        return None
    
    def get_all_jobs(self) -> Dict[str, ProcessingJob]:
        """Get all jobs (active and completed)"""
        return {**self.active_jobs, **self.completed_jobs}
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or processing job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            if job.status in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]:
                job.status = ProcessingStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                
                # Move to completed jobs
                self.completed_jobs[job_id] = job
                del self.active_jobs[job_id]
                
                logger.info(f"Job {job_id} cancelled")
                return True
        return False

# Global processor instance
processor = AsyncProcessor()
