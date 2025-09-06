import json
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.genai.types import Content, Part, FunctionCall
import logging

from agent.agent_utils.llm_output_parser import parse_chat_template

def extract_tool_calls_after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Extracts tool calls from the LLM response and updates the callback context."""
    # breakpoint()
    logging.debug(f"Starting tool call extraction from LLM response: {llm_response}")
    content = llm_response.content
    logging.debug(f"Extracting tool calls from content: {content}")
    new_content = Content(parts=[], role=content.role)
    
    for part in content.parts:
        if part.text:
            text_content = part.text
            logging.debug(f"Processing text content: {text_content}")
            parsed_content = parse_chat_template(text_content)
            logging.debug(f"Parsed content: {parsed_content}")
            if parsed_content["has_tool_code"]:
                # If tool calls are present, add them to the callback context
                for tool_call_str in parsed_content.get("reformatted_output", {}).get("tool_calls", []):
                    tool_call = json.loads(tool_call_str)
                    logging.debug(f"Extracted tool call: {tool_call}")
                    new_content.parts.append(Part(text="\n\n"))
                    new_content.parts.append(
                        Part(
                            function_call=FunctionCall(args=tool_call["arguments"], name=tool_call["name"]),
                        )
                    )
            else:
                # If no tool calls, just keep the original text
                new_content.parts.append(Part(text=parsed_content["final_output"]))
        else:
            # If part has no text, just append it as is
            new_content.parts.append(part)
    logging.debug(f"New content after processing: {new_content}")
    llm_response.content = new_content
    
    logging.debug(f"Final LlmResponse after tool call extraction: {llm_response}")
    return None
