from typing import Optional, Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from utils.s3_utils import move_s3_object
from utils.rds_utils import connect_to_rds
from sql.tables_sql import create_tables
import boto3
import logging
import psycopg2.extras
import psycopg2
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import logging
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

s3_client = boto3.client('s3')

BUCKET_NAME = "experiment2407"
PROCESSING_PREFIX = "processing/"
PROCESSED_PREFIX = "processed_latest/"


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@tool
def insert_data_all(file_key, file_size, uploaded_at,
                    transcription=None, translation=None,
                    topic=None, summary=None, key_points=None,
                    action_items=None, sentiment_label=None,
                    sentiment_scores=None, embeddings=None, tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None):
    """
    Insert the data into the database
    Args:
        file_key: the file key
        file_size: the file size
        uploaded_at: the uploaded at
        transcription: the transcription
        translation: the translation
        topic: the topic
        summary: the summary
        key_points: the key points
        action_items: the action items
        sentiment_label: the sentiment label
        sentiment_scores: the sentiment scores
        embeddings: the embeddings
    Returns:
        Confirmation message that the data has been inserted.
    """

    connection = None
    try:
        connection = connect_to_rds()  # Make sure this returns a psycopg2 connection
        with connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:

            # Insert into calls and get call_id
            cursor.execute("""
                INSERT INTO calls(file_name, file_size, uploaded_at)
                VALUES (%s, %s, %s)
                RETURNING call_id
            """, (file_key, file_size, uploaded_at))

            result = cursor.fetchone()
            if not result:
                print("ERROR: No call_id returned from INSERT")
                connection.rollback()
                return

            call_id = result['call_id']

            # Insert transcript if available
            if transcription or translation:
                cursor.execute("""
                    INSERT INTO transcripts(call_id, transcript_text, translated_text, created_at)
                    VALUES (%s, %s, %s, NOW())
                """, (call_id, transcription, translation))

            # Insert analysis if any field is provided
            if any([topic, summary, key_points, action_items, sentiment_label, sentiment_scores, embeddings]):
                # Ensure embeddings is a list of floats or None
                if embeddings:
                    if not isinstance(embeddings, (list, tuple)):
                        raise ValueError("Embeddings must be a list or tuple of floats")
                    if len(embeddings) != 1536:
                        raise ValueError("Embeddings must have length 1536")

                cursor.execute("""
                    INSERT INTO analyses(
                        call_id, topic, abstract_summary, key_points, action_items,
                        sentiment_label, sentiment_scores, embeddings
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                """, (call_id, topic, summary, key_points, action_items, sentiment_label, sentiment_scores, embeddings))

            # Commit once at the end
            connection.commit()
            print("All data inserted successfully!")

    except Exception as e:
        print(f"Error inserting data: {e}")
        if 'connection' in locals():
            connection.rollback()
        # Return error message as ToolMessage
        update_dict = {
            "messages": [
                ToolMessage(
                    content=f"Error inserting data: {str(e)}",
                    tool_call_id=tool_call_id,
                )
            ]
        }
        return Command(update=update_dict)
    finally:
        if 'connection' in locals():
            connection.close()

    update_dict = {
        "messages": [
            ToolMessage(
                content="Data inserted successfully!",
                tool_call_id=tool_call_id,
            )
        ]
    }
    return Command(update=update_dict)  


@tool
def update_state_Storage_Agent(
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
            tool_call_id=tool_call_id or "storage_state_updated",
        )
    ]
    # Return a Command to update the state
    return Command(update=update_dict)

@tool
def move_file_to_processed(file_key: str, tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None) -> str:
    """Move audio file to processed_latest folder from processing folder in S3 and update the state of the agent
    Args:
        file_key: the file key
    Returns:
        Confirmation message that the file has been moved to the processed_latest folder.
    """
    try:
        file_name = file_key.split('/')[-1]
        processing_key = PROCESSING_PREFIX + file_name
        processed_key = PROCESSED_PREFIX + file_name
        success = move_s3_object(file_key=processing_key, destination_key=processed_key)
        
        if success:
            # Unregister from cleanup manager since processing is complete
            logger.info(f"Successfully moved file {file_key} to processed_latest ")
            update_dict = {
                "messages": [
                    ToolMessage(
                        content=f"Successfully moved file {file_key} to processed_latest folder",
                        tool_call_id=tool_call_id or "move_success",
                    )
                ]
            }
            return Command(update=update_dict)
        else:
            logger.error(f"Failed to move file {file_key} to processed_latest")
            update_dict = {
                "messages": [
                    ToolMessage(
                        content=f"Failed to move file {file_key} to processed_latest folder",
                        tool_call_id=tool_call_id or "move_failed",
                    )
                ]
            }
            return Command(update=update_dict)
            
    except Exception as e:
        logger.error(f"Unexpected error moving file {file_key} to processed_latest: {e}")
        update_dict = {
            "messages": [
                ToolMessage(
                    content=f"Error moving file {file_key} to processed_latest: {str(e)}",
                    tool_call_id=tool_call_id or "move_error",
                )
            ]
        }
        return Command(update=update_dict)

@tool
def make_embeddings_of_transcription(
    transcription: str = None,
    translation: str = None,
    action_items: str = None,
    key_points: str = None,
    summary: str = None,
    topic: str = None,
    sentiment_label: str = None,
    sentiment_scores: float = None,
    tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None
):
    """Generate embeddings and store them in the agent state without adding to messages"""
    try:
        # Combine text
        text = ""
        for t in [translation, transcription, action_items, key_points, summary, topic, sentiment_label, sentiment_scores]:
            if t:
                text += t + " "
        if not text.strip():
            raise ValueError("No text provided for embeddings")
        
        # Use OpenAI embeddings instead of Google (synchronous)
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-ada-002", 
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        embeddings = embeddings_model.embed_query(text.strip())
        logger.info(f"Generated embeddings for text length {len(text.strip())}")

        update_dict = {
            "embeddings": embeddings,
            "messages": [
                ToolMessage(
                    content="Embeddings generated successfully.",
                    tool_call_id=tool_call_id or "embeddings_success"
                )
            ]
        }
        # Return a Command to update only the embeddings in state
        return Command(update=update_dict)
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return Command(update={
            "embeddings": [0.0]*1536,  # OpenAI embeddings are 1536-dimensional
            "messages": [
                ToolMessage(
                    content=f"Error generating embeddings: {str(e)}",
                    tool_call_id=tool_call_id or "embeddings_error"
                )
            ]
        })