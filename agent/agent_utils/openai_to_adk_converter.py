"""
OpenAI <-> Google ADK Agent Format Converter

This module provides bidirectional conversion between OpenAI API format
and Google ADK Agent Server format for the /run and /run_sse endpoints.

Author: Claude
Date: 2026-01-16
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import time
import hashlib
import json


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OpenAIMessage:
    """OpenAI message format"""
    role: str  # "system", "user", "assistant", "tool"
    content: Union[str, List[Dict[str, Any]]]  # text or multimodal content
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class OpenAIChatCompletionRequest:
    """OpenAI /v1/chat/completions request format"""
    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


@dataclass
class OpenAIChatCompletionResponse:
    """OpenAI /v1/chat/completions response format"""
    id: str
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    system_fingerprint: Optional[str] = None


@dataclass
class ADKRunRequest:
    """Google ADK /run endpoint request format"""
    app_name: str
    user_id: str
    session_id: str
    new_message: Dict[str, Any]  # Content object
    streaming: Optional[bool] = False
    state_delta: Optional[Dict[str, Any]] = None


@dataclass
class ADKContent:
    """ADK Content object structure"""
    parts: List[Dict[str, Any]]  # List of parts (text, functionCall, etc.)
    role: str  # "user" or "model"


@dataclass
class ADKEvent:
    """ADK Event object structure"""
    id: str
    timestamp: float
    author: str
    content: Optional[Dict[str, Any]] = None
    invocation_id: Optional[str] = None
    actions: Optional[Dict[str, Any]] = None
    partial: Optional[bool] = False


# ============================================================================
# CONVERTER CLASS
# ============================================================================

class OpenAIADKConverter:
    """
    Bidirectional converter between OpenAI and Google ADK Agent formats.
    
    Usage:
        converter = OpenAIADKConverter(
            app_name="my_agent",
            user_id_prefix="user_",
            session_id_prefix="session_"
        )
        
        # Convert OpenAI request to ADK format
        adk_request = converter.openai_to_adk_request(openai_request, user_id, session_id)
        
        # Convert ADK response to OpenAI format
        openai_response = converter.adk_to_openai_response(adk_events, model_name)
    """
    
    def __init__(
        self,
        app_name: str = "default_agent",
        user_id_prefix: str = "user_",
        session_id_prefix: str = "session_",
        default_model: str = "gemini-2.0-flash"
    ):
        """
        Initialize the converter.
        
        Args:
            app_name: Default ADK app name
            user_id_prefix: Prefix for auto-generated user IDs
            session_id_prefix: Prefix for auto-generated session IDs
            default_model: Default model name for responses
        """
        self.app_name = app_name
        self.user_id_prefix = user_id_prefix
        self.session_id_prefix = session_id_prefix
        self.default_model = default_model
    
    # ========================================================================
    # REQUEST CONVERSION: OpenAI -> ADK
    # ========================================================================
    
    def openai_to_adk_request(
        self,
        openai_request: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert OpenAI chat completion request to ADK /run request format.
        
        Args:
            openai_request: OpenAI /v1/chat/completions request
            user_id: Optional user ID (auto-generated if not provided)
            session_id: Optional session ID (auto-generated if not provided)
        
        Returns:
            ADK /run endpoint request dictionary
        
        Example:
            openai_req = {
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ],
                "temperature": 0.7,
                "stream": False
            }
            adk_req = converter.openai_to_adk_request(openai_req, "user_123", "session_456")
        """
        # Extract messages
        messages = openai_request.get("messages", [])
        
        # Auto-generate IDs if not provided
        if not user_id:
            user_id = self._generate_user_id(openai_request)
        if not session_id:
            session_id = self._generate_session_id(openai_request, user_id)
        
        # Convert messages to ADK Content format
        # ADK expects the last message as new_message
        # System messages and history are handled differently
        
        # Extract system message if present
        system_instruction = None
        conversation_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                system_instruction = self._extract_text_content(msg.get("content"))
            else:
                conversation_messages.append(msg)
        
        # Get the last user message as new_message
        if not conversation_messages:
            raise ValueError("No user messages found in request")
        
        last_message = conversation_messages[-1]
        new_message_content = self._convert_openai_message_to_adk_content(last_message)
        
        # Build ADK request
        adk_request = {
            "appName": self.app_name,  # TypeScript uses camelCase
            "userId": user_id,
            "sessionId": session_id,
            "newMessage": new_message_content,
            "streaming": openai_request.get("stream", False)
        }
        
        # Add state_delta if we need to pass system instructions or other state
        state_delta = {}
        if system_instruction:
            state_delta["system_instruction"] = system_instruction
        
        # Add generation config from OpenAI parameters
        generation_config = self._extract_generation_config(openai_request)
        if generation_config:
            state_delta["generation_config"] = generation_config
        
        if state_delta:
            adk_request["stateDelta"] = state_delta
        
        return adk_request
    
    def _convert_openai_message_to_adk_content(
        self,
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert a single OpenAI message to ADK Content format.
        
        Args:
            message: OpenAI message dictionary
        
        Returns:
            ADK Content object
        """
        role = message.get("role", "user")
        content = message.get("content")
        
        # Map OpenAI roles to ADK roles
        # OpenAI: system, user, assistant, tool
        # ADK: user, model
        adk_role = "model" if role == "assistant" else "user"
        
        # Convert content to parts
        parts = []
        
        if isinstance(content, str):
            # Simple text message
            parts.append({"text": content})
        
        elif isinstance(content, list):
            # Multimodal content
            for item in content:
                item_type = item.get("type")
                
                if item_type == "text":
                    parts.append({"text": item.get("text", "")})
                
                elif item_type == "image_url":
                    # Convert image URL to ADK format
                    image_url = item.get("image_url", {})
                    url = image_url.get("url", "")
                    
                    if url.startswith("data:"):
                        # Base64 encoded image
                        # Extract mime type and data
                        header, data = url.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]
                        
                        parts.append({
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": data
                            }
                        })
                    else:
                        # External URL - ADK needs to fetch and encode
                        # For now, pass as text with note
                        parts.append({
                            "text": f"[Image URL: {url}]"
                        })
        
        # Handle function/tool calls from assistant
        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tool_call in tool_calls:
                if tool_call.get("type") == "function":
                    function = tool_call.get("function", {})
                    parts.append({
                        "functionCall": {
                            "name": function.get("name"),
                            "args": json.loads(function.get("arguments", "{}"))
                        }
                    })
        
        # Handle tool responses
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            tool_content = message.get("content", "")
            
            parts.append({
                "functionResponse": {
                    "name": message.get("name", "unknown_tool"),
                    "response": {"result": tool_content}
                }
            })
        
        return {
            "role": adk_role,
            "parts": parts
        }
    
    def _extract_text_content(self, content: Any) -> str:
        """Extract plain text from OpenAI content."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            texts = []
            for item in content:
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return " ".join(texts)
        return ""
    
    def _extract_generation_config(
        self,
        openai_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract generation configuration from OpenAI request.
        
        Maps OpenAI parameters to ADK generation config.
        """
        config = {}
        
        if "temperature" in openai_request:
            config["temperature"] = openai_request["temperature"]
        
        if "max_tokens" in openai_request:
            config["maxOutputTokens"] = openai_request["max_tokens"]
        
        if "top_p" in openai_request:
            config["topP"] = openai_request["top_p"]
        
        if "stop" in openai_request:
            stop = openai_request["stop"]
            if isinstance(stop, str):
                config["stopSequences"] = [stop]
            elif isinstance(stop, list):
                config["stopSequences"] = stop
        
        return config
    
    def _generate_user_id(self, request: Dict[str, Any]) -> str:
        """Generate a user ID based on request content."""
        user = request.get("user")
        if user:
            return f"{self.user_id_prefix}{user}"
        
        # Hash the request to generate consistent ID
        request_str = json.dumps(request, sort_keys=True)
        hash_val = hashlib.md5(request_str.encode()).hexdigest()[:8]
        return f"{self.user_id_prefix}{hash_val}"
    
    def _generate_session_id(
        self,
        request: Dict[str, Any],
        user_id: str
    ) -> str:
        """Generate a session ID."""
        # Use timestamp + user_id for unique session
        timestamp = int(time.time())
        return f"{self.session_id_prefix}{user_id}_{timestamp}"
    
    # ========================================================================
    # RESPONSE CONVERSION: ADK -> OpenAI
    # ========================================================================
    
    def adk_to_openai_response(
        self,
        adk_events: Union[List[Dict[str, Any]], Dict[str, Any]],
        model: str,
        stream: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Convert ADK events to OpenAI response format.
        
        Args:
            adk_events: ADK event(s) from /run or /run_sse
            model: Model name to include in response
            stream: Whether this is a streaming response
        
        Returns:
            OpenAI chat completion response
        
        Example:
            # Non-streaming
            adk_events = [event1, event2, event3]
            openai_resp = converter.adk_to_openai_response(adk_events, "gpt-4")
            
            # Streaming
            for event in adk_events:
                chunk = converter.adk_to_openai_response(event, "gpt-4", stream=True)
        """
        if stream:
            # Streaming response - convert single event to chunk
            return self._adk_event_to_openai_chunk(adk_events, model)
        else:
            # Non-streaming - convert all events to complete response
            return self._adk_events_to_openai_complete(adk_events, model)
    
    def _adk_events_to_openai_complete(
        self,
        adk_events: List[Dict[str, Any]],
        model: str
    ) -> Dict[str, Any]:
        """
        Convert complete ADK event list to OpenAI response.
        
        Process:
        1. Find final response event (last event with model content)
        2. Extract text content
        3. Extract function calls if present
        4. Calculate token usage from all events
        """
        if not adk_events:
            raise ValueError("No events provided")
        
        # Find the final response event
        final_event = None
        for event in reversed(adk_events):
            content = event.get("content")
            if content and content.get("role") == "model":
                final_event = event
                break
        
        if not final_event:
            # No model response found, use last event
            final_event = adk_events[-1]
        
        # Extract content from final event
        content = final_event.get("content", {})
        parts = content.get("parts", [])
        
        # Build message content
        message_content = ""
        tool_calls = []
        
        for part in parts:
            # Extract text
            if "text" in part:
                message_content += part["text"]
            
            # Extract function calls
            if "functionCall" in part:
                func_call = part["functionCall"]
                tool_calls.append({
                    "id": func_call.get("id", f"call_{hashlib.md5(str(func_call).encode()).hexdigest()[:8]}"),
                    "type": "function",
                    "function": {
                        "name": func_call.get("name", ""),
                        "arguments": json.dumps(func_call.get("args", {}))
                    }
                })
        
        # Build choice object
        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": message_content if message_content else None
            },
            "finish_reason": self._determine_finish_reason(final_event, tool_calls)
        }
        
        if tool_calls:
            choice["message"]["tool_calls"] = tool_calls
        
        # Calculate usage (approximate)
        usage = self._calculate_usage_from_events(adk_events)
        
        # Generate response ID
        response_id = self._generate_response_id(final_event)
        
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": int(final_event.get("timestamp", time.time())),
            "model": model,
            "choices": [choice],
            "usage": usage
        }
    
    def _adk_event_to_openai_chunk(
        self,
        adk_event: Dict[str, Any],
        model: str
    ) -> Dict[str, Any]:
        """
        Convert a single ADK event to OpenAI streaming chunk.
        
        OpenAI streaming format:
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",  # Only in first chunk
                    "content": "text"      # Incremental text
                },
                "finish_reason": null  # null until last chunk
            }]
        }
        """
        content = adk_event.get("content", {})
        parts = content.get("parts", [])
        
        # Build delta object
        delta = {}
        
        # Add role only if this is the first chunk (has role field)
        role = content.get("role")
        if role == "model":
            delta["role"] = "assistant"
        
        # Extract text content for delta
        text_content = ""
        for part in parts:
            if "text" in part:
                text_content += part["text"]
        
        if text_content:
            delta["content"] = text_content
        
        # Handle function calls in streaming
        for part in parts:
            if "functionCall" in part:
                func_call = part["functionCall"]
                if "tool_calls" not in delta:
                    delta["tool_calls"] = []
                
                delta["tool_calls"].append({
                    "index": 0,
                    "id": func_call.get("id", f"call_{hashlib.md5(str(func_call).encode()).hexdigest()[:8]}"),
                    "type": "function",
                    "function": {
                        "name": func_call.get("name", ""),
                        "arguments": json.dumps(func_call.get("args", {}))
                    }
                })
        
        # Determine if this is final chunk
        is_final = not adk_event.get("partial", False)
        finish_reason = "stop" if is_final else None
        
        # Build streaming chunk
        chunk = {
            "id": self._generate_response_id(adk_event),
            "object": "chat.completion.chunk",
            "created": int(adk_event.get("timestamp", time.time())),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }
        
        return chunk
    
    def _determine_finish_reason(
        self,
        event: Dict[str, Any],
        tool_calls: List[Dict[str, Any]]
    ) -> str:
        """
        Determine OpenAI finish_reason from ADK event.
        
        Possible values:
        - "stop": Natural completion
        - "length": Max tokens reached
        - "tool_calls": Model called tools
        - "content_filter": Content filtered
        """
        if tool_calls:
            return "tool_calls"
        
        # Check actions for clues
        actions = event.get("actions", {})
        
        # Check if there's an error or special state
        # (ADK doesn't have direct equivalent, so default to "stop")
        
        return "stop"
    
    def _calculate_usage_from_events(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Calculate token usage from ADK events.
        
        Note: ADK doesn't always provide token counts in events.
        This is an approximation based on text content.
        """
        # Try to find usage metadata in events
        for event in events:
            # Some ADK events might include usage info
            if "usageMetadata" in event:
                metadata = event["usageMetadata"]
                return {
                    "prompt_tokens": metadata.get("promptTokenCount", 0),
                    "completion_tokens": metadata.get("candidatesTokenCount", 0),
                    "total_tokens": metadata.get("totalTokenCount", 0)
                }
        
        # Fallback: approximate token count
        total_text = ""
        for event in events:
            content = event.get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                if "text" in part:
                    total_text += part["text"]
        
        # Rough estimate: 1 token ≈ 4 characters
        estimated_tokens = len(total_text) // 4
        
        return {
            "prompt_tokens": 0,  # Can't determine from events alone
            "completion_tokens": estimated_tokens,
            "total_tokens": estimated_tokens
        }
    
    def _generate_response_id(self, event: Dict[str, Any]) -> str:
        """Generate OpenAI-style response ID."""
        event_id = event.get("id", "")
        return f"chatcmpl-{event_id}"
    
    # ========================================================================
    # STREAMING HELPERS
    # ========================================================================
    
    def format_sse_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Format OpenAI streaming chunk as SSE (Server-Sent Events).
        
        Args:
            chunk: OpenAI streaming chunk dictionary
        
        Returns:
            SSE formatted string
        
        Example:
            sse_data = converter.format_sse_chunk(chunk)
            # Returns: "data: {...}\n\n"
        """
        return f"data: {json.dumps(chunk)}\n\n"
    
    def format_sse_done(self) -> str:
        """Format SSE done message."""
        return "data: [DONE]\n\n"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Demonstrate converter usage with examples."""
    
    converter = OpenAIADKConverter(
        app_name="my_weather_agent",
        user_id_prefix="user_",
        session_id_prefix="sess_"
    )
    
    # ========================================================================
    # Example 1: Simple text request conversion
    # ========================================================================
    
    print("=" * 80)
    print("Example 1: OpenAI -> ADK Request Conversion")
    print("=" * 80)
    
    openai_request = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "temperature": 0.7,
        "stream": False
    }
    
    adk_request = converter.openai_to_adk_request(
        openai_request,
        user_id="user_123",
        session_id="session_456"
    )
    
    print("\nOpenAI Request:")
    print(json.dumps(openai_request, indent=2))
    
    print("\nADK Request:")
    print(json.dumps(adk_request, indent=2))
    
    # ========================================================================
    # Example 2: Response conversion
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("Example 2: ADK -> OpenAI Response Conversion")
    print("=" * 80)
    
    adk_events = [
        {
            "id": "event_1",
            "timestamp": 1737000000.0,
            "author": "weather_agent",
            "content": {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "Tokyo"}
                        }
                    }
                ]
            },
            "partial": False
        },
        {
            "id": "event_2",
            "timestamp": 1737000001.0,
            "author": "weather_agent",
            "content": {
                "role": "model",
                "parts": [
                    {"text": "The weather in Tokyo is sunny, 22°C."}
                ]
            },
            "partial": False
        }
    ]
    
    openai_response = converter.adk_to_openai_response(
        adk_events,
        model="gpt-4",
        stream=False
    )
    
    print("\nADK Events:")
    print(json.dumps(adk_events, indent=2))
    
    print("\nOpenAI Response:")
    print(json.dumps(openai_response, indent=2))
    
    # ========================================================================
    # Example 3: Streaming conversion
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("Example 3: Streaming Conversion")
    print("=" * 80)
    
    streaming_events = [
        {
            "id": "stream_1",
            "timestamp": 1737000000.0,
            "author": "agent",
            "content": {"role": "model", "parts": [{"text": "Hello"}]},
            "partial": True
        },
        {
            "id": "stream_2",
            "timestamp": 1737000000.5,
            "author": "agent",
            "content": {"role": "model", "parts": [{"text": ", how"}]},
            "partial": True
        },
        {
            "id": "stream_3",
            "timestamp": 1737000001.0,
            "author": "agent",
            "content": {"role": "model", "parts": [{"text": " can I help?"}]},
            "partial": False
        }
    ]
    
    print("\nStreaming chunks:")
    for event in streaming_events:
        chunk = converter.adk_to_openai_response(event, model="gpt-4", stream=True)
        sse_formatted = converter.format_sse_chunk(chunk)
        print(sse_formatted, end="")
    
    print(converter.format_sse_done())


if __name__ == "__main__":
    example_usage()