from typing import Literal, TypedDict, Sequence, Annotated, Optional, Union, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import ToolNode
from langgraph.types import Command  # Added missing import
from pydantic import BaseModel 
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from dotenv import load_dotenv
import boto3
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from botocore.exceptions import ClientError
import os
import io
from utils.prompt_templates import Ingestion_Model_Template, Speech_Model_Template, Summarization_Model_Template, Topic_Classification_Model_Template, Key_Points_Model_Template, Action_Items_Model_Template, Storage_Model_Template, Sentiment_Analysis_Model_Template
import logging
from Tools.Ingestion_Agent_Tools import get_single_audio_file_from_s3, move_file_to_processing, roll_back_file_from_processing, update_state_Ingestion_Agent
from openai import OpenAI
from Tools.Speech_Agent_Tools import transcribe_audio, translate_audio, update_state_Speech_Agent
from langgraph.graph.state import LastValue, LastValueAfterFinish, BinaryOperatorAggregate
from Tools.Summarization_Agent_Tools import update_state_Summarization_Agent
from Tools.Topic_Classification_Agent_Tools import update_state_Topic_Classification_Agent, classify_conversation
from Tools.Key_Points_Agent_Tools import update_state_Key_Points_Agent, extract_key_points
from Tools.Action_Items_Agent_Tools import update_state_Action_Items_Agent, extract_action_items
from Tools.Storage_Agent_Tools import update_state_Storage_Agent, insert_data_all, move_file_to_processed, make_embeddings_of_transcription
from utils.rds_utils import connect_to_rds, get_secret
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from Tools.Sentiment_Analysis_Agent import sentiment_analysis, update_state_Sentiment_Analysis_Agent
from utils.openai_utils import safe_model_invoke

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = ["OPENAI_API_KEY", "GEMINI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {missing_vars}")

# Constants
BUCKET_NAME = "experiment2407"
PROCESSING_PREFIX = "processing/"
PROCESSED_PREFIX = "processed_latest/"

# Initialize OpenAI model with error handling
try:
    model = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    # ollama_model= ChatOllama(model="llama3.2:1b", temperature=0, base_url="http://localhost:11434")
    # ollama_model= ChatOllama(model="llama3.2", temperature=0, base_url="http://localhost:11434")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))
    # embeddings_model = GoogleGenerativeAIEmbeddings(model="embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
    client = OpenAI()
except Exception as e:
    logger.error(f"Failed to initialize OpenAI models: {e}")
    raise e

# Initialize S3 client with error handling
try:
    s3_client = boto3.client('s3')  
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {e}")
    raise e

class AudioFile(BaseModel):
    key: str
    size: int
    bucket: str = "experiment2407"
    original_key: str = ""

class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[Union[HumanMessage, AIMessage, SystemMessage]], add_messages]
    audio_files: AudioFile
    transcription: Annotated[str, LastValueAfterFinish]
    translation: Annotated[str, LastValueAfterFinish]
    summary: Annotated[str, LastValueAfterFinish]
    action_items: Annotated[str, LastValueAfterFinish]
    key_points: Annotated[str, LastValueAfterFinish]
    topic: Annotated[str, LastValueAfterFinish]
    sentiment: Annotated[str, LastValueAfterFinish]
    processing_status: Annotated[str, LastValueAfterFinish]
    processing_complete: Annotated[bool, LastValueAfterFinish]
    error_message: Annotated[Optional[str], LastValue]
    embeddings: Annotated[List[float], LastValueAfterFinish]
    sentiment_label: Annotated[str, LastValueAfterFinish]
    sentiment_scores: Annotated[float, LastValueAfterFinish]

Ingestion_model_tools = [
    get_single_audio_file_from_s3,
    move_file_to_processing,
    roll_back_file_from_processing,
    update_state_Ingestion_Agent,
]

def Ingestion_Model(state: AgentState) -> AgentState:
    """Create an agent for audio file ingestion workflow"""
    messages = state.get('messages', [])
    messages = messages + [SystemMessage(content=Ingestion_Model_Template)]
    Ingestion_Model_With_Tools = model.bind_tools(Ingestion_model_tools)
    
    response = safe_model_invoke(Ingestion_Model_With_Tools, messages)
    
    if isinstance(response, Command):
        # For Command responses, merge the updates and add any ToolMessages from the update
        new_state = {**state, "messages": messages}
        if "messages" in response.update:
            new_state["messages"] = messages + response.update["messages"]
        new_state.update(response.update)  # merge other state updates
        return new_state
    else:
        # fallback if response is not a Command
        return {
            **state,
            "messages": state.get("messages", []) + [response],
        }

def Ingestion_Agent_Should_Continue(state: AgentState):
    """Check if the last message contains tool calls"""
    messages = state.get('messages', [])
    if not messages:
        return END
        
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return 'Ingestion_Agent_Tools'
    return END

Ingestion_Agent=(StateGraph(AgentState)
            .add_node("Ingestion_Agent", Ingestion_Model)
            .add_node("Ingestion_Agent_Tools", ToolNode(Ingestion_model_tools))
            .add_conditional_edges(
                'Ingestion_Agent',
                Ingestion_Agent_Should_Continue,
                {
                    'Ingestion_Agent_Tools': 'Ingestion_Agent_Tools',
                    END: END
                }
            )
            .add_edge(START, "Ingestion_Agent")
            .add_edge("Ingestion_Agent_Tools", "Ingestion_Agent")
            ).compile()
            
Speech_Model_Tools = [transcribe_audio, update_state_Speech_Agent, translate_audio]

def Speech_Model(state:AgentState) -> AgentState:
    messages=state['messages']
    messages=state['messages'] + [SystemMessage(content=Speech_Model_Template)]
    Speech_Model_With_Tools=model.bind_tools(Speech_Model_Tools)
    response=safe_model_invoke(Speech_Model_With_Tools, messages)
    
    if isinstance(response, Command):
        # For Command responses, merge the updates and add any ToolMessages from the update
        new_state = {**state, "messages": messages}
        if "messages" in response.update:
            new_state["messages"] = messages + response.update["messages"]
        new_state.update(response.update)  # merge other state updates
        return new_state
    else:
        # fallback if response is not a Command
        return {**state, "messages" : messages + [response]}
    
def Speech_model_Should_Continue(state:AgentState):
    """check if the last message contains tool calls"""
    messages=state['messages']
    last_message=messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return 'Speech_Model_Tools'
    return END

Speech_Agent=(StateGraph(AgentState)
            .add_node("Speech_Model", Speech_Model)
            .add_node("Speech_Model_Tools", ToolNode(Speech_Model_Tools))
            .add_conditional_edges(
                'Speech_Model',
                Speech_model_Should_Continue,
                {
                    'Speech_Model_Tools': 'Speech_Model_Tools',
                    END: END
                }
            )
            .add_edge(START, "Speech_Model")
            .add_edge("Speech_Model_Tools", "Speech_Model")
            ).compile()




Summarization_Model_Tools = [update_state_Summarization_Agent]

def Summarization_Model(state:AgentState):
    messages = state.get('messages', [])
    messages = messages + [SystemMessage(content=Summarization_Model_Template)]
    Summarization_Model_With_Tools = model.bind_tools(Summarization_Model_Tools)
    response = safe_model_invoke(Summarization_Model_With_Tools, messages)
    if isinstance(response, Command):
        # For Command responses, merge the updates and add any ToolMessages from the update
        new_state = {**state, "messages": messages}
        if "messages" in response.update:
            new_state["messages"] = messages + response.update["messages"]
        new_state.update(response.update)  # merge other state updates (summary, embeddings, etc.)
        return new_state
    else:
        # fallback if response is not a Command
        return {**state, "messages": messages + [response]}

def Summarization_Model_Should_Continue(state:AgentState):
    """check if the last message contains tool calls"""
    messages = state.get('messages', [])
    if not messages:
        return END
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return 'Summarization_Model_Tools'
    return END

Summarization_Agent=(StateGraph(AgentState)
            .add_node("Summarization_Model", Summarization_Model)
            .add_node("Summarization_Model_Tools", ToolNode(Summarization_Model_Tools))
            .add_conditional_edges(
                'Summarization_Model',
                Summarization_Model_Should_Continue,
                {
                    'Summarization_Model_Tools': 'Summarization_Model_Tools',
                    END: END
                }
            )
            .add_edge(START, "Summarization_Model")
            .add_edge("Summarization_Model_Tools", "Summarization_Model")
            ).compile()

Topic_Classification_Model_Tools = [update_state_Topic_Classification_Agent, classify_conversation]

def Topic_Classification_Model(state:AgentState):
    messages = state.get('messages', [])
    messages = messages + [SystemMessage(content=Topic_Classification_Model_Template)]
    Topic_Classification_Model_With_Tools = model.bind_tools(Topic_Classification_Model_Tools)
    response = safe_model_invoke(Topic_Classification_Model_With_Tools, messages)
    
    if isinstance(response, Command):
        # For Command responses, merge the updates and add any ToolMessages from the update
        new_state = {**state, "messages": messages}
        if "messages" in response.update:
            new_state["messages"] = messages + response.update["messages"]
        new_state.update(response.update)  # merge other state updates
        return new_state
    else:
        # fallback if response is not a Command
        return {**state, "messages" : messages + [response]}

def Topic_Classification_Model_Should_Continue(state:AgentState):
    """check if the last message contains tool calls"""
    messages = state.get('messages', [])
    if not messages:
        return END
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return 'Topic_Classification_Model_Tools'
    return END

Topic_Classification_Agent=(StateGraph(AgentState)
            .add_node("Topic_Classification_Model", Topic_Classification_Model)
            .add_node("Topic_Classification_Model_Tools", ToolNode(Topic_Classification_Model_Tools))
            .add_conditional_edges(
                'Topic_Classification_Model',
                Topic_Classification_Model_Should_Continue,
                {
                    'Topic_Classification_Model_Tools': 'Topic_Classification_Model_Tools',
                    END: END
                }
            )
            .add_edge(START, "Topic_Classification_Model")
            .add_edge("Topic_Classification_Model_Tools", "Topic_Classification_Model")
            ).compile()

Key_Points_Model_Tools = [update_state_Key_Points_Agent, extract_key_points]

def Key_Points_Model(state:AgentState):
    messages=state['messages']
    messages=messages + [SystemMessage(content=Key_Points_Model_Template)]
    Key_Points_Model_With_Tools=model.bind_tools(Key_Points_Model_Tools)
    response=safe_model_invoke(Key_Points_Model_With_Tools, messages)
    
    if isinstance(response, Command):
        # For Command responses, merge the updates and add any ToolMessages from the update
        new_state = {**state, "messages": messages}
        if "messages" in response.update:
            new_state["messages"] = messages + response.update["messages"]
        new_state.update(response.update)  # merge other state updates
        return new_state
    else:
        # fallback if response is not a Command
        return {**state, "messages" : messages + [response]}

def Key_Points_Model_Should_Continue(state:AgentState):
    """check if the last message contains tool calls"""
    messages=state['messages']
    last_message=messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return 'Key_Points_Model_Tools'
    return END

Key_Points_Agent=(StateGraph(AgentState)
            .add_node("Key_Points_Model", Key_Points_Model)
            .add_node("Key_Points_Model_Tools", ToolNode(Key_Points_Model_Tools))
            .add_conditional_edges(
                'Key_Points_Model',
                Key_Points_Model_Should_Continue,
                {
                    'Key_Points_Model_Tools': 'Key_Points_Model_Tools',
                    END: END
                }
            )
            .add_edge(START, "Key_Points_Model")
            .add_edge("Key_Points_Model_Tools", "Key_Points_Model")
            ).compile()

Action_Items_Model_Tools = [update_state_Action_Items_Agent, extract_action_items]

def Action_Items_Model(state:AgentState):
    messages=state['messages']
    messages=messages + [SystemMessage(content=Action_Items_Model_Template)]
    Action_Items_Model_With_Tools=model.bind_tools(Action_Items_Model_Tools)
    response=safe_model_invoke(Action_Items_Model_With_Tools, messages)
    
    if isinstance(response, Command):
        # For Command responses, merge the updates and add any ToolMessages from the update
        new_state = {**state, "messages": messages}
        if "messages" in response.update:
            new_state["messages"] = messages + response.update["messages"]
        new_state.update(response.update)  # merge other state updates
        return new_state
    else:
        # fallback if response is not a Command
        return {**state, "messages" : messages + [response]}

def Action_Items_Model_Should_Continue(state:AgentState):
    """check if the last message contains tool calls"""
    messages=state['messages']
    last_message=messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return 'Action_Items_Model_Tools'
    return END

Action_Items_Agent=(StateGraph(AgentState)
            .add_node("Action_Items_Model", Action_Items_Model)
            .add_node("Action_Items_Model_Tools", ToolNode(Action_Items_Model_Tools))
            .add_conditional_edges(
                'Action_Items_Model',

                Action_Items_Model_Should_Continue,
                {
                    'Action_Items_Model_Tools': 'Action_Items_Model_Tools',
                    END: END
                }
            )
            .add_edge(START, "Action_Items_Model")
            .add_edge("Action_Items_Model_Tools", "Action_Items_Model")
            ).compile()



Sentiment_Analysis_Model_Tools = [sentiment_analysis,update_state_Sentiment_Analysis_Agent]

def Sentiment_Analysis_Model(state:AgentState):
    messages=state['messages']
    messages=messages + [SystemMessage(content=Sentiment_Analysis_Model_Template)]
    Sentiment_Analysis_Model_With_Tools=model.bind_tools(Sentiment_Analysis_Model_Tools)
    response=safe_model_invoke(Sentiment_Analysis_Model_With_Tools, messages)
    
    if isinstance(response, Command):
        # For Command responses, merge the updates and add any ToolMessages from the update
        new_state = {**state, "messages": messages}
        if "messages" in response.update:
            new_state["messages"] = messages + response.update["messages"]
        new_state.update(response.update)  # merge other state updates
        return new_state
    else:
        # fallback if response is not a Command
        return {**state, "messages" : messages + [response]}

def Sentiment_Analysis_Model_Should_Continue(state:AgentState):
    """check if the last message contains tool calls"""
    messages=state['messages']
    last_message=messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return 'Sentiment_Analysis_Model_Tools'
    return END

Sentiment_Analysis_Agent=(StateGraph(AgentState)
            .add_node("Sentiment_Analysis_Model", Sentiment_Analysis_Model)
            .add_node("Sentiment_Analysis_Model_Tools", ToolNode(Sentiment_Analysis_Model_Tools))
            .add_conditional_edges(
                'Sentiment_Analysis_Model',
                Sentiment_Analysis_Model_Should_Continue,
                {
                    'Sentiment_Analysis_Model_Tools': 'Sentiment_Analysis_Model_Tools',
                    END: END
                }
            )
            .add_edge(START, "Sentiment_Analysis_Model")
            .add_edge("Sentiment_Analysis_Model_Tools", "Sentiment_Analysis_Model")
            ).compile()

Storage_Model_Tools = [update_state_Storage_Agent,insert_data_all,move_file_to_processed,make_embeddings_of_transcription]

def Storage_Agent_Model(state:AgentState):
    messages=state['messages']
    messages=messages + [SystemMessage(content=Storage_Model_Template)]
    Storage_Model_With_Tools=model.bind_tools(Storage_Model_Tools)
    response=safe_model_invoke(Storage_Model_With_Tools, messages)
    
    if isinstance(response, Command):
        # For Command responses, merge the updates and add any ToolMessages from the update
        new_state = {**state, "messages": messages}
        if "messages" in response.update:
            new_state["messages"] = messages + response.update["messages"]
        new_state.update(response.update)  # merge other state updates
        return new_state
    else:
        # fallback if response is not a Command
        return {**state, "messages" : messages + [response]}

def Storage_Agent_Should_Continue(state:AgentState):
    """check if the last message contains tool calls"""
    messages=state['messages']
    last_message=messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return 'Storage_Agent_Tools'
    return END

Storage_Agent=(StateGraph(AgentState)
            .add_node("Storage_Agent", Storage_Agent_Model)
            .add_node("Storage_Agent_Tools", ToolNode(Storage_Model_Tools))
            .add_conditional_edges(
                'Storage_Agent',
                Storage_Agent_Should_Continue,
                {
                    'Storage_Agent_Tools': 'Storage_Agent_Tools',
                    END: END
                }
            )
            .add_edge(START, "Storage_Agent")
            .add_edge("Storage_Agent_Tools", "Storage_Agent")
            ).compile()

workflow=StateGraph(AgentState)
workflow.add_node("Ingestion_Agent", Ingestion_Agent)
workflow.add_node("Speech_Agent", Speech_Agent)
workflow.add_node("Summarization_Agent", Summarization_Agent)
workflow.add_node("Topic_Classification_Agent", Topic_Classification_Agent)
workflow.add_node("Key_Points_Agent", Key_Points_Agent)
workflow.add_node("Action_Items_Agent", Action_Items_Agent)
workflow.add_node("Sentiment_Analysis_Agent", Sentiment_Analysis_Agent)
workflow.add_node("Storage_Agent", Storage_Agent)

workflow.add_edge(START, "Ingestion_Agent")
workflow.add_edge("Ingestion_Agent", "Speech_Agent")
workflow.add_edge('Speech_Agent', "Summarization_Agent")
workflow.add_edge("Summarization_Agent", "Key_Points_Agent")
workflow.add_edge("Summarization_Agent", "Action_Items_Agent")
workflow.add_edge("Speech_Agent", "Topic_Classification_Agent")
workflow.add_edge("Speech_Agent", "Sentiment_Analysis_Agent")
workflow.add_edge("Topic_Classification_Agent", "Storage_Agent")
workflow.add_edge("Key_Points_Agent", "Storage_Agent")
workflow.add_edge("Action_Items_Agent", "Storage_Agent")
workflow.add_edge("Sentiment_Analysis_Agent", "Storage_Agent")
workflow.add_edge("Storage_Agent", END)

graph=workflow.compile()