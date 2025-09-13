import logging
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
import os
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def truncate_messages_for_context(messages, max_tokens=5000):
    """Truncate messages to fit within context limits"""
    try:
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        total_chars = sum(len(str(msg.content)) for msg in messages)
        estimated_tokens = total_chars // 4
        
        if estimated_tokens <= max_tokens:
            return messages
        
        # If too long, keep only the most recent messages
        logger.warning(f"Messages too long ({estimated_tokens} estimated tokens), truncating...")
        
        # Keep system messages and recent messages
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        # Keep the most recent messages, ensuring tool call pairs stay together
        recent_messages = other_messages[-6:]  # Keep last 6 messages
        
        # Ensure tool call pairs are preserved
        validated_messages = []
        pending_tool_calls = set()
        
        for msg in recent_messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # This is an AIMessage with tool calls
                for tool_call in msg.tool_calls:
                    if 'id' in tool_call:
                        pending_tool_calls.add(tool_call['id'])
                validated_messages.append(msg)
            elif hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                # This is a ToolMessage
                if msg.tool_call_id in pending_tool_calls:
                    pending_tool_calls.remove(msg.tool_call_id)
                    validated_messages.append(msg)
                # Skip orphaned tool messages
            else:
                # Regular message
                validated_messages.append(msg)
        
        recent_messages = validated_messages
        
        truncated_messages = system_messages + recent_messages
        
        # Log what we're keeping
        logger.info(f"Truncated from {len(messages)} to {len(truncated_messages)} messages")
        
        return truncated_messages
        
    except Exception as e:
        logger.error(f"Error truncating messages: {e}")
        # Fallback: return last 5 messages
        return messages[-5:] if len(messages) > 5 else messages

def safe_model_invoke(model, messages, max_retries=3):
    """Safely invoke model with context management"""
    for attempt in range(max_retries):
        try:
            # Truncate messages if needed
            truncated_messages = truncate_messages_for_context(messages)
            
            # Invoke model
            response = model.invoke(truncated_messages)
            return response
            
        except Exception as e:
            if "context_length_exceeded" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Context length exceeded, retrying with shorter context (attempt {attempt + 1})")
                # Reduce context further
                messages = messages[-5:]  # Keep only last 5 messages
                continue
            else:
                logger.error(f"Model invocation failed: {e}")
                raise