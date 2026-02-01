"""
OpenAI-Compatible ADK Web Server

This module provides an extended AdkWebServer that includes OpenAI API
compatible endpoints (/v1/chat/completions, /v1/models) alongside the
standard ADK endpoints.

Author: Claude
Date: 2025-01-16
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from google.genai import types
from pydantic import BaseModel, Field
from starlette.types import Lifespan

from google.adk.cli.adk_web_server import AdkWebServer
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.utils.context_utils import Aclosing

from database_memory_service import DatabaseMemoryService
from database_session_service import DatabaseSessionService

# Authorization imports
from authorization import (
    AuthorizationPlugin,
    StaticPIP,
    Policy,
    PolicyRule,
)

logger = logging.getLogger("google_adk." + __name__)


# ============================================================================
# OpenAI API Request/Response Models
# ============================================================================

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
    """OpenAI /v1/chat/completions request format.

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


# ============================================================================
# OpenAI-Compatible ADK Web Server
# ============================================================================

class OpenAIAdkWebServer(AdkWebServer):
    """
    Extended AdkWebServer with OpenAI API compatible endpoints.

    This class adds the following endpoints:
    - POST /v1/chat/completions - OpenAI-compatible chat completion
    - GET /v1/models - List available models (ADK apps)

    Usage:
        server = OpenAIAdkWebServer(
            agent_loader=agent_loader,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
            credential_service=credential_service,
            eval_sets_manager=eval_sets_manager,
            eval_set_results_manager=eval_set_results_manager,
            agents_dir=agents_dir,
        )

        app = server.get_fast_api_app(allow_origins=["*"])
        # Now app has both ADK and OpenAI endpoints
    """

    def __init__(
        self,
        *,
        default_app_name: Optional[str] = None,
        default_user_id: str = "openai_user",
        policies: Optional[Dict[str, Policy]] = None,
        **kwargs
    ):
        """
        Initialize the OpenAI-compatible ADK web server.

        Args:
            default_app_name: Default ADK app to use when model name doesn't
                match any app. If None, first available app is used.
            default_user_id: Default user ID for OpenAI requests.
            policies: Optional dict of authorization policies for agents/tools.
                Keys should be "resource_type:resource_name" (e.g., "agent:admin_agent").
            **kwargs: Arguments passed to parent AdkWebServer.
        """
        super().__init__(**kwargs)
        self.default_app_name = default_app_name
        self.default_user_id = default_user_id
        self._session_cache: Dict[str, str] = {}  # Maps user+model to session_id

        # Setup authorization if policies are configured
        self.policies = policies or {}
        self._pip = StaticPIP(policies=self.policies) if self.policies else None
        self._auth_plugin = AuthorizationPlugin(pip=self._pip) if self._pip else None

    async def get_runner_async(self, app_name: str):
        """
        Get a runner for the specified app, with authorization plugin if configured.

        Overrides parent method to inject AuthorizationPlugin when policies are set.
        """
        runner = await super().get_runner_async(app_name)

        # Add authorization plugin if configured
        if self._auth_plugin:
            # Check if plugin is already added (avoid duplicates)
            has_auth_plugin = any(
                isinstance(p, AuthorizationPlugin) for p in runner._plugins
            )
            if not has_auth_plugin:
                runner._plugins.append(self._auth_plugin)
                logger.debug(f"Authorization plugin added to runner for app: {app_name}")

        return runner

    def _get_app_name_for_model(self, model: str) -> str:
        """
        Map OpenAI model name to ADK app name.

        The model parameter can be:
        1. An exact ADK app name
        2. A model identifier that we map to default app

        Args:
            model: Model name from OpenAI request

        Returns:
            ADK app name to use
        """
        available_apps = self.agent_loader.list_agents()

        # Direct match
        if model in available_apps:
            return model

        # Use default or first available
        if self.default_app_name and self.default_app_name in available_apps:
            return self.default_app_name

        if available_apps:
            return available_apps[0]

        raise ValueError("No ADK apps available")

    async def _get_or_create_session(
        self,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Get existing session or create a new one.

        Args:
            app_name: ADK app name
            user_id: User ID
            session_id: Optional specific session ID to use

        Returns:
            Session ID
        """
        if session_id:
            # Check if session exists
            session = await self.session_service.get_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id
            )
            if session:
                return session_id

        # Create new session
        session = await self.session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id or f"openai_{uuid.uuid4().hex[:12]}",
        )
        return session.id

    def _convert_messages_to_adk_content(
        self,
        messages: List[OpenAIMessage]
    ) -> types.Content:
        """
        Convert OpenAI messages to ADK Content format.

        Only converts the last user message as that's what ADK's new_message expects.
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
                            import base64
                            parts.append(types.Part.from_bytes(
                                data=base64.b64decode(data),
                                mime_type=mime_type
                            ))
                        else:
                            # URL reference - pass as text for now
                            parts.append(types.Part.from_text(text=f"[Image: {url}]"))

        return types.Content(role="user", parts=parts)

    def _format_openai_response(
        self,
        events: List[Any],
        model: str,
        created: int
    ) -> Dict[str, Any]:
        """
        Format ADK events as OpenAI chat completion response.
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

    def _format_openai_chunk(
        self,
        event: Any,
        model: str,
        created: int,
        chunk_id: str,
        is_first: bool = False,
        is_last: bool = False
    ) -> Dict[str, Any]:
        """
        Format a single ADK event as OpenAI streaming chunk.
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

    def get_fast_api_app(
        self,
        lifespan: Optional[Lifespan[FastAPI]] = None,
        allow_origins: Optional[List[str]] = None,
        web_assets_dir: Optional[str] = None,
        **kwargs
    ) -> FastAPI:
        """
        Create FastAPI app with both ADK and OpenAI endpoints.

        All arguments are passed to parent's get_fast_api_app().
        """
        # Get the base FastAPI app with ADK endpoints
        app = super().get_fast_api_app(
            lifespan=lifespan,
            allow_origins=allow_origins,
            web_assets_dir=web_assets_dir,
            **kwargs
        )

        # Add OpenAI-compatible endpoints
        self._add_openai_endpoints(app)

        return app

    def _add_openai_endpoints(self, app: FastAPI) -> None:
        """Add OpenAI-compatible endpoints to the FastAPI app."""

        @app.get("/v1/models", tags=["OpenAI Compatible"])
        async def list_models() -> OpenAIModelsResponse:
            """
            List available models (ADK apps).

            Maps ADK apps to OpenAI model format.
            """
            apps = self.agent_loader.list_agents()
            models = [
                OpenAIModelInfo(
                    id=app_name,
                    created=int(time.time()),
                    owned_by="google-adk"
                )
                for app_name in apps
            ]
            return OpenAIModelsResponse(data=models)

        @app.get("/v1/models/{model_id}", tags=["OpenAI Compatible"])
        async def get_model(model_id: str) -> OpenAIModelInfo:
            """Get info about a specific model (ADK app)."""
            apps = self.agent_loader.list_agents()
            if model_id not in apps:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            return OpenAIModelInfo(
                id=model_id,
                created=int(time.time()),
                owned_by="google-adk"
            )

        @app.post("/v1/chat/completions", tags=["OpenAI Compatible"], response_model=None)
        async def chat_completions(
            request: OpenAIChatCompletionRequest
        ):
            """
            OpenAI-compatible chat completion endpoint.

            Supports both streaming and non-streaming responses.
            The 'model' parameter maps to ADK app names.
            """
            created = int(time.time())

            # Log request details for debugging
            logger.info(
                "Chat completion request: model=%s (appName=%s), stream=%s, messages=%d",
                request.model,
                request.appName,
                request.stream,
                len(request.messages)
            )

            try:
                # Map model to app name (supports both 'model' and 'appName')
                app_name = self._get_app_name_for_model(request.effective_model)

                # Get user ID from request or use default
                user_id = request.effective_user or self.default_user_id

                # Get or create session (supports both 'session_id' and 'sessionId')
                session_id = await self._get_or_create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=request.effective_session_id,
                )

                # Convert messages to ADK format
                new_message = self._convert_messages_to_adk_content(request.messages)

                # Get runner
                runner = await self.get_runner_async(app_name)

                # Log the stream value for debugging
                logger.info(
                    "Stream decision: request.stream=%s (type=%s), will use streaming=%s",
                    request.stream,
                    type(request.stream).__name__,
                    bool(request.stream)
                )

                if request.stream:
                    # Streaming response
                    logger.info("Using STREAMING handler")
                    return await self._handle_streaming_completion(
                        runner=runner,
                        user_id=user_id,
                        session_id=session_id,
                        new_message=new_message,
                        model=request.effective_model,
                        created=created
                    )
                else:
                    # Non-streaming response
                    logger.info("Using NON-STREAMING handler")
                    return await self._handle_completion(
                        runner=runner,
                        user_id=user_id,
                        session_id=session_id,
                        new_message=new_message,
                        model=request.effective_model,
                        created=created
                    )

            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.exception("Error in chat completion: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

    async def _handle_completion(
        self,
        runner,
        user_id: str,
        session_id: str,
        new_message: types.Content,
        model: str,
        created: int
    ) -> Dict[str, Any]:
        """Handle non-streaming chat completion."""
        events = []
        async with Aclosing(
            runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=new_message,
            )
        ) as agen:
            async for event in agen:
                events.append(event)

        return self._format_openai_response(events, model, created)

    async def _handle_streaming_completion(
        self,
        runner,
        user_id: str,
        session_id: str,
        new_message: types.Content,
        model: str,
        created: int
    ) -> StreamingResponse:
        """Handle streaming chat completion."""
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        run_config = RunConfig(streaming_mode=StreamingMode.SSE)
        logger.info(
            "Starting streaming response: model=%s, chunk_id=%s, run_config.streaming_mode=%s",
            model, chunk_id, run_config.streaming_mode
        )
        print(f"[DEBUG] Streaming handler called with run_config={run_config}")

        async def generate():
            is_first = True
            last_event = None

            try:
                print(f"[DEBUG] Calling runner.run_async with streaming_mode={run_config.streaming_mode}")
                async with Aclosing(
                    runner.run_async(
                        user_id=user_id,
                        session_id=session_id,
                        new_message=new_message,
                        run_config=run_config,
                    )
                ) as agen:
                    async for event in agen:
                        # Only emit events with content
                        if event.content and event.content.parts:
                            has_content = any(
                                (hasattr(p, 'text') and p.text) or
                                (hasattr(p, 'function_call') and p.function_call)
                                for p in event.content.parts
                            )
                            if has_content:
                                chunk = self._format_openai_chunk(
                                    event=event,
                                    model=model,
                                    created=created,
                                    chunk_id=chunk_id,
                                    is_first=is_first,
                                    is_last=False
                                )
                                yield f"data: {json.dumps(chunk)}\n\n"
                                is_first = False
                                last_event = event

                # Send final chunk with finish_reason
                if last_event:
                    final_chunk = self._format_openai_chunk(
                        event=last_event,
                        model=model,
                        created=created,
                        chunk_id=chunk_id,
                        is_first=False,
                        is_last=True
                    )
                    # Clear delta content for final chunk
                    final_chunk["choices"][0]["delta"] = {}
                    yield f"data: {json.dumps(final_chunk)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception("Error in streaming: %s", e)
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "server_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )


# ============================================================================
# Convenience function to create the server
# ============================================================================

def create_openai_adk_server(
    agents_dir: str,
    default_app_name: Optional[str] = None,
    policies: Optional[Dict[str, Policy]] = None,
    **kwargs
) -> OpenAIAdkWebServer:
    """
    Convenience function to create an OpenAI-compatible ADK web server.

    Args:
        agents_dir: Directory containing ADK agents
        default_app_name: Default app to use for OpenAI requests
        policies: Optional dict of authorization policies for agents/tools.
            Keys should be "resource_type:resource_name" (e.g., "agent:admin_agent").
        **kwargs: Additional arguments for services

    Returns:
        Configured OpenAIAdkWebServer instance

    Example:
        from openai_adk_web_server import create_openai_adk_server
        from authorization import Policy, PolicyRule

        # Without authorization
        server = create_openai_adk_server(
            agents_dir="./agents",
            default_app_name="my_agent"
        )

        # With authorization policies
        policies = {
            "agent:admin_agent": Policy(
                resource_type="agent",
                resource_name="admin_agent",
                rules=[PolicyRule(type="rbac", required_roles=["admin"])]
            ),
        }
        server = create_openai_adk_server(
            agents_dir="./agents",
            policies=policies,
        )

        app = server.get_fast_api_app(allow_origins=["*"])

        # Run with uvicorn
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
    from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
    from google.adk.evaluation.local_eval_sets_manager import LocalEvalSetsManager
    from google.adk.evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager
    from google.adk.cli.utils.agent_loader import AgentLoader

    # Create services with defaults if not provided
    session_service = kwargs.get('session_service') or InMemorySessionService()
    artifact_service = kwargs.get('artifact_service') or InMemoryArtifactService()
    memory_service = kwargs.get('memory_service') or InMemoryMemoryService()
    credential_service = kwargs.get('credential_service') or InMemoryCredentialService()

    # Create agent loader
    agent_loader = kwargs.get('agent_loader') or AgentLoader(agents_dir)

    # Create eval managers
    eval_sets_manager = kwargs.get('eval_sets_manager') or LocalEvalSetsManager(agents_dir)
    eval_set_results_manager = kwargs.get('eval_set_results_manager') or LocalEvalSetResultsManager(agents_dir)

    return OpenAIAdkWebServer(
        agent_loader=agent_loader,
        session_service=session_service,
        memory_service=memory_service,
        artifact_service=artifact_service,
        credential_service=credential_service,
        eval_sets_manager=eval_sets_manager,
        eval_set_results_manager=eval_set_results_manager,
        agents_dir=agents_dir,
        default_app_name=default_app_name,
        policies=policies,
    )


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Example: Create and run the server
    server = create_openai_adk_server(
        agents_dir="./",
        default_app_name=None,
        memory_service=DatabaseMemoryService(
            database_url="sqlite:///memory.db"
        ),
        session_service=DatabaseSessionService(
            database_url="sqlite:///session.db"
        ),

    )

    app = server.get_fast_api_app(
        allow_origins=["*"]  # Allow all origins for development
    )

    print("Starting OpenAI-compatible ADK server...")
    print("OpenAI endpoints:")
    print("  GET  /v1/models")
    print("  POST /v1/chat/completions")
    print("\nADK endpoints also available at standard paths")

    uvicorn.run(app, host="0.0.0.0", port=8000)
