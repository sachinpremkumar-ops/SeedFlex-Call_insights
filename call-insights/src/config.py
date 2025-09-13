import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for the audio processing system"""
    
    # AWS Configuration
    AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
    S3_BUCKET = os.getenv("S3_BUCKET", "experiment2407")
    S3_PROCESSING_PREFIX = os.getenv("S3_PROCESSING_PREFIX", "processing/")
    S3_PROCESSED_PREFIX = os.getenv("S3_PROCESSED_PREFIX", "processed_latest/")
    
    # Database Configuration
    DB_SECRET_NAME = os.getenv("DB_SECRET_NAME", "rds/sachin")
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    
    # Async Processing Configuration
    MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "5"))
    LARGE_FILE_THRESHOLD = int(os.getenv("LARGE_FILE_THRESHOLD", "1000000"))  # 1MB
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10000000"))  # 10MB
    
    # Processing Timeouts
    LAMBDA_TIMEOUT = int(os.getenv("LAMBDA_TIMEOUT", "900"))  # 15 minutes
    PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", "1800"))  # 30 minutes
    
    # Memory Configuration
    LAMBDA_MEMORY_SMALL = int(os.getenv("LAMBDA_MEMORY_SMALL", "512"))  # MB
    LAMBDA_MEMORY_LARGE = int(os.getenv("LAMBDA_MEMORY_LARGE", "1024"))  # MB
    
    # Retry Configuration
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))  # seconds
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_WORKERS = int(os.getenv("API_WORKERS", "1"))
    
    @classmethod
    def get_lambda_config(cls, file_size: int) -> Dict[str, Any]:
        """Get Lambda configuration based on file size"""
        if file_size > cls.LARGE_FILE_THRESHOLD:
            return {
                "memory_size": cls.LAMBDA_MEMORY_LARGE,
                "timeout": cls.LAMBDA_TIMEOUT,
                "max_concurrent_executions": cls.MAX_CONCURRENT_JOBS
            }
        else:
            return {
                "memory_size": cls.LAMBDA_MEMORY_SMALL,
                "timeout": cls.LAMBDA_TIMEOUT // 2,  # Shorter timeout for small files
                "max_concurrent_executions": cls.MAX_CONCURRENT_JOBS * 2
            }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        return True

# Validate configuration on import
Config.validate_config()
