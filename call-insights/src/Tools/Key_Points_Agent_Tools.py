from typing import Optional, Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command


@tool
def extract_key_points(conversation: str, tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None):
    """
    Extract key points from the conversation.
    Args:
        conversation: The conversation text to extract key points from
    Returns:
        The extracted key points
    """
    # This is a placeholder - the actual extraction should be done by the LLM
    # The tool exists to prevent "tool not found" errors
    update_dict = {
        "messages": [
            ToolMessage(
                content="Key points extraction completed. Use update_state_Key_Points_Agent to update the state.",
                tool_call_id=tool_call_id or "extract_key_points",
            )
        ]
    }
    return Command(update=update_dict)


@tool
def update_state_Key_Points_Agent(
    processing_status: Optional[str] = None,
    processing_complete: Optional[bool] = None,
    key_points: Optional[str] = None,
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
    if key_points is not None:
        update_dict['key_points'] = key_points

    update_dict["messages"] = [
        ToolMessage(
            content="State updated successfully.",
            tool_call_id=tool_call_id or "key_points_state_updated",
        )
    ]
    # Return a Command to update the state
    return Command(update=update_dict)

