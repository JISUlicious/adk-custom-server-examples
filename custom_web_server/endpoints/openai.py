"""
OpenAI-compatible API endpoints.

Provides:
- GET /v1/models - List available models
- GET /v1/models/{model_id} - Get model info
- POST /v1/chat/completions - Chat completion (streaming/non-streaming)
"""

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from google.adk.agents.run_config import RunConfig, StreamingMode

from ..auth import get_auth_info
from ..converters import OpenAIConverter
from ..models import (
    OpenAIChatCompletionRequest,
    OpenAIModelInfo,
    OpenAIModelsResponse,
)

if TYPE_CHECKING:
    from ..server import CustomWebServer

logger = logging.getLogger(__name__)


def create_openai_router(server: "CustomWebServer") -> APIRouter:
    """
    Create OpenAI-compatible API router.

    Args:
        server: CustomWebServer instance

    Returns:
        Configured APIRouter
    """
    router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])

    @router.get("/models")
    async def list_models() -> OpenAIModelsResponse:
        """List available models (ADK apps)."""
        apps = server.list_apps()
        models = [
            OpenAIModelInfo(
                id=app_name,
                created=int(time.time()),
                owned_by="google-adk"
            )
            for app_name in apps
        ]
        return OpenAIModelsResponse(data=models)

    @router.get("/models/{model_id}")
    async def get_model(model_id: str) -> OpenAIModelInfo:
        """Get info about a specific model (ADK app)."""
        apps = server.list_apps()
        if model_id not in apps:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        return OpenAIModelInfo(
            id=model_id,
            created=int(time.time()),
            owned_by="google-adk"
        )

    @router.post("/chat/completions", response_model=None)
    async def chat_completions(
        request: OpenAIChatCompletionRequest,
        auth_info: Dict[str, Any] = Depends(get_auth_info),
    ):
        """
        OpenAI-compatible chat completion endpoint.

        Supports both streaming and non-streaming responses.
        The 'model' parameter maps to ADK app names.
        """
        created = int(time.time())

        logger.info(
            "Chat completion request: model=%s, stream=%s, messages=%d",
            request.effective_model,
            request.stream,
            len(request.messages)
        )

        try:
            # Get app name from model
            app_name = _get_app_name(server, request.effective_model)

            # Get user ID
            user_id = request.effective_user or server.config.default_user_id

            # Get or create session
            session_id = await _get_or_create_session(
                server=server,
                app_name=app_name,
                user_id=user_id,
                session_id=request.effective_session_id,
            )

            # Inject auth info into session state
            session = await server.session_service.get_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
            )
            if session:
                await server.session_service.update_session_state(
                    session=session,
                    state_delta={"_auth": auth_info},
                )

            # Convert messages to ADK format
            new_message = OpenAIConverter.messages_to_adk_content(request.messages)

            # Get runner
            runner = await server.get_runner(app_name)

            if request.stream:
                return await _handle_streaming(
                    runner=runner,
                    user_id=user_id,
                    session_id=session_id,
                    new_message=new_message,
                    model=request.effective_model,
                    created=created,
                )
            else:
                return await _handle_completion(
                    runner=runner,
                    user_id=user_id,
                    session_id=session_id,
                    new_message=new_message,
                    model=request.effective_model,
                    created=created,
                )

        except ValueError as e:
            logger.error("ValueError in chat completion: %s", e, exc_info=True)
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Error in chat completion: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    return router


def _get_app_name(server: "CustomWebServer", model: str) -> str:
    """Map model name to ADK app name."""
    apps = server.list_apps()

    # Direct match
    if model in apps:
        return model

    # Use default or first available
    if server.default_app_name and server.default_app_name in apps:
        return server.default_app_name

    if apps:
        return apps[0]

    raise ValueError("No ADK apps available")


async def _get_or_create_session(
    server: "CustomWebServer",
    app_name: str,
    user_id: str,
    session_id: str | None = None,
) -> str:
    """Get existing session or create a new one."""
    if session_id:
        session = await server.session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if session:
            return session_id

    # Create new session
    session = await server.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id or f"openai_{uuid.uuid4().hex[:12]}",
    )
    return session.id


async def _handle_completion(
    runner,
    user_id: str,
    session_id: str,
    new_message,
    model: str,
    created: int,
):
    """Handle non-streaming chat completion."""
    events = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
    ):
        events.append(event)

    return OpenAIConverter.events_to_openai_response(events, model, created)


async def _handle_streaming(
    runner,
    user_id: str,
    session_id: str,
    new_message,
    model: str,
    created: int,
):
    """Handle streaming chat completion."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    run_config = RunConfig(streaming_mode=StreamingMode.SSE)

    async def generate():
        is_first = True
        last_event = None

        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=new_message,
                run_config=run_config,
            ):
                # Skip non-partial (aggregated) events to avoid duplicate text
                # ADK's progressive SSE sends partial chunks + final aggregated event
                # OpenAI standard only sends incremental chunks
                if hasattr(event, 'partial') and event.partial is False:
                    # This is the final aggregated event - skip content but use for finish
                    last_event = event
                    continue

                # Emit partial events with visible content
                if OpenAIConverter.has_visible_content(event):
                    chunk = OpenAIConverter.event_to_openai_chunk(
                        event=event,
                        model=model,
                        created=created,
                        chunk_id=chunk_id,
                        is_first=is_first,
                        is_last=False
                    )
                    delta = chunk["choices"][0]["delta"]
                    if delta.get("content") or delta.get("reasoning") or delta.get("tool_calls"):
                        yield f"data: {json.dumps(chunk)}\n\n"
                        is_first = False
                        last_event = event

            # Send final chunk with finish_reason (empty delta, just signals completion)
            if last_event:
                final_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": "stop"
                    }]
                }
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
