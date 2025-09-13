"""
Example client for async audio processing API
"""
import asyncio
import httpx
import time
from typing import Optional

class AsyncAudioProcessor:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def process_audio_sync(self, file_key: str, messages: str = "Process audio file") -> dict:
        """Process audio file synchronously (for small files)"""
        response = await self.client.post(
            f"{self.base_url}/process",
            json={
                "messages": messages,
                "audio_file_key": file_key,
                "async_processing": False
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def process_audio_async(self, file_key: str, messages: str = "Process audio file") -> str:
        """Process audio file asynchronously (for large files) - returns job ID"""
        response = await self.client.post(
            f"{self.base_url}/process",
            json={
                "messages": messages,
                "audio_file_key": file_key,
                "async_processing": True
            }
        )
        response.raise_for_status()
        result = response.json()
        return result["job_id"]
    
    async def get_job_status(self, job_id: str) -> dict:
        """Get the status of a job"""
        response = await self.client.get(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_job_result(self, job_id: str) -> dict:
        """Get the result of a completed job"""
        response = await self.client.get(f"{self.base_url}/jobs/{job_id}/result")
        response.raise_for_status()
        return response.json()
    
    async def wait_for_completion(self, job_id: str, poll_interval: int = 5) -> dict:
        """Wait for a job to complete and return the result"""
        while True:
            status = await self.get_job_status(job_id)
            
            if status["status"] == "completed":
                return await self.get_job_result(job_id)
            elif status["status"] == "failed":
                raise Exception(f"Job failed: {status.get('error_message', 'Unknown error')}")
            elif status["status"] == "cancelled":
                raise Exception("Job was cancelled")
            
            print(f"Job {job_id} status: {status['status']} ({status['progress']}%)")
            await asyncio.sleep(poll_interval)
    
    async def get_all_jobs(self) -> dict:
        """Get all jobs"""
        response = await self.client.get(f"{self.base_url}/jobs")
        response.raise_for_status()
        return response.json()
    
    async def cancel_job(self, job_id: str) -> dict:
        """Cancel a job"""
        response = await self.client.delete(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    async def health_check(self) -> dict:
        """Check API health"""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

async def example_usage():
    """Example of how to use the async audio processor"""
    processor = AsyncAudioProcessor()
    
    try:
        # Check API health
        health = await processor.health_check()
        print(f"API Health: {health}")
        
        # Example 1: Process small file synchronously
        print("\n=== Processing Small File Synchronously ===")
        small_file_result = await processor.process_audio_sync(
            file_key="small_file.mp3",
            messages="Process this small audio file"
        )
        print(f"Small file result: {small_file_result}")
        
        # Example 2: Process large file asynchronously
        print("\n=== Processing Large File Asynchronously ===")
        job_id = await processor.process_audio_async(
            file_key="large_file.mp3",
            messages="Process this large audio file"
        )
        print(f"Job submitted: {job_id}")
        
        # Wait for completion
        result = await processor.wait_for_completion(job_id)
        print(f"Large file result: {result}")
        
        # Example 3: Check all jobs
        print("\n=== All Jobs ===")
        all_jobs = await processor.get_all_jobs()
        print(f"Total jobs: {all_jobs['total_jobs']}")
        print(f"Active jobs: {all_jobs['active_jobs']}")
        print(f"Completed jobs: {all_jobs['completed_jobs']}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await processor.close()

async def batch_processing_example():
    """Example of batch processing multiple files"""
    processor = AsyncAudioProcessor()
    
    try:
        # Submit multiple jobs
        file_keys = ["file1.mp3", "file2.mp3", "file3.mp3"]
        job_ids = []
        
        print("=== Submitting Batch Jobs ===")
        for file_key in file_keys:
            job_id = await processor.process_audio_async(
                file_key=file_key,
                messages=f"Process {file_key}"
            )
            job_ids.append(job_id)
            print(f"Submitted job {job_id} for {file_key}")
        
        # Wait for all jobs to complete
        print("\n=== Waiting for Completion ===")
        results = []
        for job_id in job_ids:
            try:
                result = await processor.wait_for_completion(job_id)
                results.append(result)
                print(f"Job {job_id} completed successfully")
            except Exception as e:
                print(f"Job {job_id} failed: {e}")
        
        print(f"\nBatch processing completed. {len(results)} jobs successful.")
        
    except Exception as e:
        print(f"Batch processing error: {e}")
    finally:
        await processor.close()

if __name__ == "__main__":
    print("Async Audio Processing Client Examples")
    print("=" * 50)
    
    # Run examples
    asyncio.run(example_usage())
    print("\n" + "=" * 50)
    asyncio.run(batch_processing_example())

