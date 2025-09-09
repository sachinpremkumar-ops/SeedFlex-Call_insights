from typing import Literal, TypedDict, Sequence, Annotated, Optional, Union, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.types import Command  # Added missing import
from pydantic import BaseModel 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import boto3
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from botocore.exceptions import ClientError
import os
import io
import logging
from openai import OpenAI
from utils.s3_utils import s3_get_audio_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Constants
BUCKET_NAME = "experiment2407"
PROCESSING_PREFIX = "processing/"
PROCESSED_PREFIX = "processed_latest/"

@tool
def transcribe_audio(file_name: str):
    """Transcribe audio file"""
    try:
        file_name = file_name.split('/')[-1]
        processing_key = f'processing/{file_name}'
        audio_bytes = s3_get_audio_file(processing_key)
        
        if audio_bytes is None:
            logger.error(f"Failed to fetch audio file: {processing_key}")
            return "Error: Audio file not found in processing folder"
        
        # Create a file-like object with proper filename for OpenAI API
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = file_name  # Set the filename for format detection
        
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        logger.info(f"Successfully transcribed {file_name}")
        print(transcription.text)
        return transcription.text
    except Exception as e:
        logger.error(f"Error transcribing {file_name}: {e}")
        print(f"Error transcribing {file_name}: {e}")
        return f"Error: Audio file not transcribed - {str(e)}"

@tool
def translate_audio(file_name:str):
    """ Translate Audio files into English if the original is not in English"""
    try:
        file_name = file_name.split('/')[-1]
        processing_key=PROCESSING_PREFIX + file_name
        audio_bytes = s3_get_audio_file(processing_key)
        if audio_bytes is None:
            logger.error(f"Failed to fetch audio file: {processing_key}")
            return None

        audio_file= io.BytesIO(audio_bytes)
        audio_file.name = file_name

        translatedtranscript = client.audio.translations.create(
            model="whisper-1",
            file=audio_file
        )
        logger.info(f"Successfully translated {file_name}")
        print(translatedtranscript.text)
        return translatedtranscript.text
    except Exception as e:
        logger.error(f"Error translating audio file: {e}")
        return None

@tool
def update_state_Speech_Agent(
    processing_status: Optional[str] = None,
    processing_complete: Optional[bool] = None,
    transcription: Optional[str] = None,
    translation: Optional[str] = None,
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
    if transcription is not None:
        update_dict['transcription'] = transcription
    if translation is not None:
        update_dict['translation'] = translation

    update_dict["messages"] = [
        ToolMessage(
            content="State updated successfully.",
            tool_call_id=tool_call_id,
        )
    ]
    # Return a Command to update the state
    return Command(update=update_dict)