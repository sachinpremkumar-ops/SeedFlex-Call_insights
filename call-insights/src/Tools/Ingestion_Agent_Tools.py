from langchain_core.tools import tool, InjectedToolCallId
from utils.s3_utils import is_supported_format, check_if_file_is_processed, move_s3_object, get_original_key_from_processed
from botocore.exceptions import ClientError
from typing import Optional, Annotated
import json
from langchain_core.messages import  ToolMessage
from langgraph.types import Command
import boto3
import logging

s3_client = boto3.client('s3')

BUCKET_NAME = "experiment2407"
PROCESSING_PREFIX = "processing/"
PROCESSED_PREFIX = "processed_latest/"


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@tool
def get_single_audio_file_from_s3() -> str:
    """Fetch ONLY ONE audio file from S3 bucket for single-file processing mode"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME
        )
        
        if "Contents" not in response:
            logger.info("No files found in S3 bucket")
            return json.dumps({"status": "no_files", "file": None})
        
        selected_file = None
        
        # Find the FIRST unprocessed audio file
        for obj in response['Contents']:
            if is_supported_format(obj['Key']):
                
                # Skip files in processing, processed, or processed_latest folders
                if obj['Key'].startswith('processing/') or obj['Key'].startswith('processed_latest/') or obj['Key'].startswith('processed/'):
                    continue
                    
                if not check_if_file_is_processed(obj['Key']):
                    if selected_file is None:  # Select only the FIRST unprocessed file
                        selected_file = {
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                            'bucket': BUCKET_NAME,
                            'original_key': obj['Key'],
                            'current_state': 'original'
                        }
                        logger.info(f"🎯 SINGLE FILE MODE: Selected {obj['Key']} for processing")
                        break
                
        
        if selected_file:
            result = {"status": "file_selected", "file": selected_file}
            return json.dumps(result)
        else:
            logger.info("✅ No unprocessed files found.")
            result = {"status": "all_processed", "file": None}
            return json.dumps(result)
    
    except ClientError as e:
        logger.error(f"S3 client error fetching files: {e}")
        result = {"status": "error", "file": None, "error": str(e)}
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Unexpected error fetching files from S3: {e}")
        result = {"status": "error", "file": None, "error": str(e)}
        return json.dumps(result) 


@tool 
def move_file_to_processing(file_key: str, tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None) -> str:
    """Move audio file to processing folder in S3 with automatic cleanup registration"""
    try:
        file_name = file_key.split('/')[-1]
        processing_key = PROCESSING_PREFIX + file_name
        success = move_s3_object(file_key=file_key, destination_key=processing_key)
        
        if success:
            logger.info(f"Successfully moved file {file_key} to processing ")
            update_dict = {
                "messages": [
                    ToolMessage(
                        content=f"Successfully moved file {file_key} to processing folder",
                        tool_call_id=tool_call_id or "move_to_processing_success",
                    )
                ]
            }
            return Command(update=update_dict)
        else:
            logger.error(f"Failed to move file {file_key} to processing")
            update_dict = {
                "messages": [
                    ToolMessage(
                        content=f"Failed to move file {file_key} to processing folder",
                        tool_call_id=tool_call_id or "move_to_processing_failed",
                    )
                ]
            }
            return Command(update=update_dict)
            
    except Exception as e:
        logger.error(f"Unexpected error moving file {file_key} to processing: {e}")
        update_dict = {
            "messages": [
                ToolMessage(
                    content=f"Error moving file {file_key} to processing: {str(e)}",
                    tool_call_id=tool_call_id or "move_to_processing_error",
                )
            ]
        }
        return Command(update=update_dict)


@tool 
def roll_back_file_from_processing(file_key: str, tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None) -> str:
    """Roll back file from processing folder to original folder in S3"""
    try:
        # If file_key doesn't include processing/, add it
        if not file_key.startswith(PROCESSING_PREFIX):
            file_name = file_key.split('/')[-1]
            processing_file_key = PROCESSING_PREFIX + file_name
        else:
            processing_file_key = file_key
            
        original_key = get_original_key_from_processed(processing_file_key)
        success = move_s3_object(file_key=processing_file_key, destination_key=original_key)
        
        if success:
            logger.info(f"Successfully rolled back file {processing_file_key} from processing")
            update_dict = {
                "messages": [
                    ToolMessage(
                        content=f"Successfully rolled back file {processing_file_key} from processing folder",
                        tool_call_id=tool_call_id or "rollback_success",
                    )
                ]
            }
            return Command(update=update_dict)
        else:
            logger.error(f"Failed to roll back file {processing_file_key} from processing")
            update_dict = {
                "messages": [
                    ToolMessage(
                        content=f"Failed to roll back file {processing_file_key} from processing folder",
                        tool_call_id=tool_call_id or "rollback_failed",
                    )
                ]
            }
            return Command(update=update_dict)
            
    except Exception as e:
        logger.error(f"Unexpected error rolling back file {file_key} from processing: {e}")
        update_dict = {
            "messages": [
                ToolMessage(
                    content=f"Error rolling back file {file_key} from processing: {str(e)}",
                    tool_call_id=tool_call_id or "rollback_error",
                )
            ]
        }
        return Command(update=update_dict)



@tool
def update_state_Ingestion_Agent(
    processing_status: Optional[str] = None,
    processing_complete: Optional[bool] = None,
    tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None
) -> str:
    """
    Update the state of the agent
    Args:
        processing_status: Status of the processing
        processing_complete: Whether processing is complete
    Returns:
        Confirmation message that the state has been updated.
    """
    update_dict = {}
    if processing_status is not None:
        update_dict['processing_status'] = processing_status
    if processing_complete is not None:
        update_dict['processing_complete'] = processing_complete

    update_dict["messages"] = [
        ToolMessage(
            content="State updated successfully.",
            tool_call_id=tool_call_id or "ingestion_state_updated",
        )
    ]
    # Return a Command to update the state
    return Command(update=update_dict)

