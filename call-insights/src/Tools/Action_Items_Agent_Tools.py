from typing import Optional, Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command


@tool
def extract_action_items(conversation: str, tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None):
    """
    Extract action items from the conversation.
    Args:
        conversation: The conversation text to extract action items from
    Returns:
        The extracted action items
    """
    # This is a placeholder - the actual extraction should be done by the LLM
    # The tool exists to prevent "tool not found" errors
    update_dict = {
        "messages": [
            ToolMessage(
                content="Action items extraction completed. Use update_state_Action_Items_Agent to update the state.",
                tool_call_id=tool_call_id or "extract_action_items",
            )
        ]
    }
    return Command(update=update_dict)


@tool
def update_state_Action_Items_Agent(
    processing_status: Optional[str] = None,
    processing_complete: Optional[bool] = None,
    action_items: Optional[str] = None,
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
    if action_items is not None:
        update_dict['action_items'] = action_items

    update_dict["messages"] = [
        ToolMessage(
            content="State updated successfully.",
            tool_call_id=tool_call_id or "action_items_state_updated",
        )
    ]
    # Return a Command to update the state
    return Command(update=update_dict)

