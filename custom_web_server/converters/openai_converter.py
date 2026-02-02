"""
Converter between OpenAI and ADK formats.
"""

import base64
import json
import uuid
from typing import Any, Dict, List

from google.genai import types

from ..models.openai_models import OpenAIMessage


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

        Args:
            events: List of ADK events
            model: Model name
            created: Unix timestamp

        Returns:
            OpenAI format response dict
        """
        # Collect all text content from events
        full_text = ""
        tool_calls = []

        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
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

        # Build response
        message = {
            "role": "assistant",
            "content": full_text if full_text else None
        }

        if tool_calls:
            message["tool_calls"] = tool_calls

        finish_reason = "tool_calls" if tool_calls else "stop"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": 0,  # ADK doesn't provide this directly
                "completion_tokens": len(full_text) // 4,  # Rough estimate
                "total_tokens": len(full_text) // 4
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
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    delta["content"] = part.text
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
                "finish_reason": finish_reason
            }]
        }
