"""
Converter between OpenAI and ADK formats.
"""

import base64
import json
import logging
import uuid
from typing import Any, Dict, List

from google.genai import types

from ..models.openai_models import OpenAIMessage

logger = logging.getLogger(__name__)


class OpenAIConverter:
    """Converts between OpenAI and ADK message formats."""

    @staticmethod
    def messages_to_adk_content(messages: List[OpenAIMessage]) -> types.Content:
        """
        Convert OpenAI messages to ADK Content format.

        Only converts the last user message as that's what ADK's new_message expects.

        Args:
            messages: List of OpenAI messages

        Returns:
            ADK Content object

        Raises:
            ValueError: If no user message found
        """
        # Find the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_message = msg
                break

        if not last_user_message:
            raise ValueError("No user message found in request")

        # Convert content to parts
        parts = []
        content = last_user_message.content

        if isinstance(content, str):
            parts.append(types.Part.from_text(text=content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(types.Part.from_text(text=item.get("text", "")))
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "")
                        if url.startswith("data:"):
                            # Base64 image
                            header, data = url.split(",", 1)
                            mime_type = header.split(":")[1].split(";")[0]
                            parts.append(types.Part.from_bytes(
                                data=base64.b64decode(data),
                                mime_type=mime_type
                            ))
                        else:
                            # URL reference - pass as text for now
                            parts.append(types.Part.from_text(text=f"[Image: {url}]"))

        return types.Content(role="user", parts=parts)

    @staticmethod
    def events_to_openai_response(
        events: List[Any],
        model: str,
        created: int
    ) -> Dict[str, Any]:
        """
        Format ADK events as OpenAI chat completion response.

        Includes reasoning/thinking content in the response following the format
        used by providers like OpenRouter/LiteLLM:
        - message.reasoning: The full reasoning text
        - message.reasoning_details: Array of reasoning items with type/text

        Args:
            events: List of ADK events
            model: Model name
            created: Unix timestamp

        Returns:
            OpenAI format response dict
        """
        # Collect text content, separating thinking from visible content
        full_text = ""
        thinking_parts = []
        tool_calls = []

        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    # Check if this is thinking/reasoning content
                    is_thought = getattr(part, 'thought', False)

                    if hasattr(part, 'text') and part.text:
                        if is_thought:
                            # Collect thinking parts
                            thinking_parts.append(part.text)
                        else:
                            # Non-thinking text goes in content
                            full_text += part.text

                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(fc.args) if fc.args else "{}"
                            }
                        })

        # Build response message with reasoning fields
        message = {
            "role": "assistant",
            "content": full_text if full_text else None,
            "refusal": None,
            "annotations": None,
            "audio": None,
            "function_call": None,
            "tool_calls": tool_calls if tool_calls else None,
        }

        # Add reasoning content if present (like OpenRouter/LiteLLM format)
        thinking_text = "\n".join(thinking_parts)
        if thinking_text:
            message["reasoning"] = thinking_text
            message["reasoning_details"] = [
                {
                    "type": "reasoning.text",
                    "format": "text",
                    "index": idx,
                    "text": part
                }
                for idx, part in enumerate(thinking_parts)
            ]

        finish_reason = "tool_calls" if tool_calls else "stop"

        # Estimate token counts (rough approximation: ~4 chars per token)
        completion_tokens = len(full_text) // 4
        reasoning_tokens = len(thinking_text) // 4

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": 0,  # ADK doesn't provide this directly
                "completion_tokens": completion_tokens + reasoning_tokens,
                "total_tokens": completion_tokens + reasoning_tokens,
                "completion_tokens_details": {
                    "reasoning_tokens": reasoning_tokens,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": None,
                    "rejected_prediction_tokens": None
                }
            }
        }

    @staticmethod
    def event_to_openai_chunk(
        event: Any,
        model: str,
        created: int,
        chunk_id: str,
        is_first: bool = False,
        is_last: bool = False
    ) -> Dict[str, Any]:
        """
        Format a single ADK event as OpenAI streaming chunk.

        Includes reasoning/thinking content in the delta following the format
        used by providers like OpenRouter/LiteLLM.

        Args:
            event: ADK event
            model: Model name
            created: Unix timestamp
            chunk_id: Consistent ID for all chunks in this stream
            is_first: Whether this is the first chunk
            is_last: Whether this is the last chunk

        Returns:
            OpenAI streaming chunk dict
        """
        delta = {}

        if is_first:
            delta["role"] = "assistant"

        if event.content and event.content.parts:
            text_parts = []
            reasoning_parts = []

            for part in event.content.parts:
                is_thought = getattr(part, 'thought', False)

                if hasattr(part, 'text') and part.text:
                    if is_thought:
                        reasoning_parts.append(part.text)
                    else:
                        text_parts.append(part.text)

                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    delta["tool_calls"] = [{
                        "index": 0,
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(fc.args) if fc.args else "{}"
                        }
                    }]

            # Include visible content
            if text_parts:
                delta["content"] = "".join(text_parts)

            # Include reasoning content (like OpenRouter/LiteLLM format)
            if reasoning_parts:
                delta["reasoning"] = "\n".join(reasoning_parts)

        finish_reason = None
        if is_last:
            finish_reason = "tool_calls" if delta.get("tool_calls") else "stop"

        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason
            }]
        }

    @staticmethod
    def has_visible_content(event: Any) -> bool:
        """
        Check if an event has any visible content (text, reasoning, or function calls).

        Args:
            event: ADK event

        Returns:
            True if event has visible content, False otherwise
        """
        if not event.content or not event.content.parts:
            return False

        for part in event.content.parts:
            if (hasattr(part, 'text') and part.text) or \
               (hasattr(part, 'function_call') and part.function_call) or \
               (hasattr(part, 'function_response') and part.function_response):
                return True

        return False
