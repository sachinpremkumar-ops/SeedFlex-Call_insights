import logging
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
import os
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def truncate_messages_for_context(messages, max_tokens=7000):
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
        
        # Keep the most recent messages
        recent_messages = other_messages[-10:]  # Keep last 10 messages
        
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