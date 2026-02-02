"""
OpenAI API compatible Pydantic models.
"""

import time
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class OpenAIMessage(BaseModel):
    """OpenAI chat message format."""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class OpenAIStreamOptions(BaseModel):
    """OpenAI stream_options format."""
    include_usage: Optional[bool] = False


class OpenAIChatCompletionRequest(BaseModel):
    """
    OpenAI /v1/chat/completions request format.

    Supports both OpenAI format and ADK format field names:
    - model / appName
    - user / userId
    - session_id / sessionId
    """
    # Accept both 'model' (OpenAI) and 'appName' (ADK)
    model: Optional[str] = Field(default=None)
    appName: Optional[str] = Field(default=None, exclude=True)
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stream_options: Optional[OpenAIStreamOptions] = None
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    # Accept both 'user' (OpenAI) and 'userId' (ADK)
    user: Optional[str] = Field(default=None)
    userId: Optional[str] = Field(default=None, exclude=True)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # Accept both 'session_id' and 'sessionId' (ADK)
    session_id: Optional[str] = Field(default=None)
    sessionId: Optional[str] = Field(default=None, exclude=True)

    @property
    def effective_model(self) -> str:
        """Get model name from either 'model' or 'appName'."""
        return self.model or self.appName or ""

    @property
    def effective_user(self) -> Optional[str]:
        """Get user from either 'user' or 'userId'."""
        return self.user or self.userId

    @property
    def effective_session_id(self) -> Optional[str]:
        """Get session_id from either 'session_id' or 'sessionId'."""
        return self.session_id or self.sessionId


class OpenAIModelInfo(BaseModel):
    """OpenAI model info format."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "google-adk"


class OpenAIModelsResponse(BaseModel):
    """OpenAI /v1/models response format."""
    object: str = "list"
    data: List[OpenAIModelInfo]
