"""
ADK API endpoints.

Provides:
- Session management
- Agent execution (run, run_sse, run_live)
- Artifact management

Matches the official Google ADK server API (adk_web_server.py).
"""

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from google.genai import types
from pydantic import BaseModel, Field

from ..auth import get_auth_info

if TYPE_CHECKING:
    from ..server import CustomWebServer

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models (matching official ADK server)
# =============================================================================

class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    session_id: Optional[str] = Field(
        default=None,
        description="The ID of the session to create. If not provided, a random ID will be generated."
    )
    state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The initial state of the session."
    )
    # Note: Official also supports 'events' field for initializing with events


class UpdateSessionRequest(BaseModel):
    """Request to update session state without running the agent."""
    state_delta: Dict[str, Any]


class RunAgentRequest(BaseModel):
    """Request to run an agent (matches official ADK RunAgentRequest)."""
    model_config = {"populate_by_name": True}

    app_name: str = Field(alias="appName")
    user_id: str = Field(alias="userId")
    session_id: str = Field(alias="sessionId")
    new_message: Dict[str, Any] = Field(alias="newMessage")  # Will be converted to types.Content
    streaming: bool = False
    state_delta: Optional[Dict[str, Any]] = Field(default=None, alias="stateDelta")
    invocation_id: Optional[str] = Field(default=None, alias="invocationId")  # For resume long running functions


class UpdateMemoryRequest(BaseModel):
    """Request to add a session to the memory service."""
    session_id: str


class SaveArtifactRequest(BaseModel):
    """Request to save an artifact."""
    filename: str
    content: str  # Base64 encoded
    mime_type: str = "application/octet-stream"


# =============================================================================
# Router Factory
# =============================================================================

def create_adk_router(server: "CustomWebServer") -> APIRouter:
    """
    Create ADK API router.

    Args:
        server: CustomWebServer instance for accessing services

    Returns:
        Configured APIRouter
    """
    router = APIRouter(tags=["ADK"])

    # -------------------------------------------------------------------------
    # App Management
    # -------------------------------------------------------------------------

    @router.get("/list-apps")
    async def list_apps(
        detailed: bool = Query(
            default=False, description="Return detailed app information"
        )
    ):
        """List available ADK apps (matches official ADK server)."""
        apps = server.list_apps()
        if detailed:
            # Return detailed app info if supported by runner factory
            return {"apps": [{"name": app} for app in apps]}
        return apps

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    @router.post(
        "/apps/{app_name}/users/{user_id}/sessions",
        response_model_exclude_none=True,
    )
    async def create_session(
        app_name: str,
        user_id: str,
        request: Optional[CreateSessionRequest] = None,
    ):
        """Create a new session (matches official ADK server)."""
        from google.adk.errors.already_exists_error import AlreadyExistsError

        state = request.state if request else None
        session_id = request.session_id if request else None

        try:
            session = await server.session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                state=state,
                session_id=session_id,
            )
            logger.info("New session created: %s", session.id)
            return session
        except AlreadyExistsError as e:
            raise HTTPException(
                status_code=409,
                detail=f"Session already exists: {session_id}"
            ) from e
        except Exception as e:
            logger.error("Internal server error during session creation: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
        response_model_exclude_none=True,
    )
    async def create_session_with_id(
        app_name: str,
        user_id: str,
        session_id: str,
        state: Optional[Dict[str, Any]] = None,
    ):
        """Create a new session with a specific session ID (deprecated in official ADK)."""
        from google.adk.errors.already_exists_error import AlreadyExistsError

        try:
            session = await server.session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                state=state,
                session_id=session_id,
            )
            logger.info("New session created: %s", session.id)
            return session
        except AlreadyExistsError as e:
            raise HTTPException(
                status_code=409,
                detail=f"Session already exists: {session_id}"
            ) from e
        except Exception as e:
            logger.error("Internal server error during session creation: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.get(
        "/apps/{app_name}/users/{user_id}/sessions",
        response_model_exclude_none=True,
    )
    async def list_sessions(app_name: str, user_id: str):
        """List sessions for a user (returns Session objects like official ADK)."""
        list_sessions_response = await server.session_service.list_sessions(
            app_name=app_name,
            user_id=user_id,
        )
        # Return sessions directly, filtering out eval sessions
        return [
            session
            for session in list_sessions_response.sessions
            if not session.id.startswith("eval_")  # Filter eval sessions like official ADK
        ]

    @router.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
        response_model_exclude_none=True,
    )
    async def get_session(app_name: str, user_id: str, session_id: str):
        """Get a specific session (returns Session object like official ADK)."""
        session = await server.session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        # Return the Session object directly - FastAPI handles serialization
        return session

    @router.patch("/apps/{app_name}/users/{user_id}/sessions/{session_id}")
    async def update_session_state(
        app_name: str,
        user_id: str,
        session_id: str,
        request: UpdateSessionRequest,
    ):
        """Update session state."""
        session = await server.session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        updated = await server.session_service.update_session_state(
            session=session,
            state_delta=request.state,
        )
        return {"state": updated.state}

    @router.delete("/apps/{app_name}/users/{user_id}/sessions/{session_id}")
    async def delete_session(app_name: str, user_id: str, session_id: str):
        """Delete a session."""
        await server.session_service.delete_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        return {"deleted": True}

    # -------------------------------------------------------------------------
    # Agent Execution
    # -------------------------------------------------------------------------

    @router.post("/run")
    async def run_agent(
        request: RunAgentRequest,
        auth_info: dict = Depends(get_auth_info),
    ):
        """
        Run an agent (non-streaming).

        Returns all events after completion.

        Authorization:
            The AuthorizationPlugin evaluates policies using RBAC and/or ABAC:

            Headers:
            - Authorization: Bearer <token> (for JWT-based auth)
            - X-User-Id: User identifier
            - X-User-Roles: Comma-separated roles for RBAC (e.g., "admin,analyst")
            - X-User-Attributes: JSON for ABAC (e.g., '{"department":"finance"}')

            RBAC Example (role-based):
                X-User-Id: alice
                X-User-Roles: admin,analyst

            ABAC Example (attribute-based):
                X-User-Id: bob
                X-User-Attributes: {"department":"finance","clearance":"high"}

            Combined:
                X-User-Id: carol
                X-User-Roles: manager
                X-User-Attributes: {"department":"finance"}
        """
        try:
            runner = await server.get_runner(request.app_name)

            # Ensure session exists and inject auth into session state
            session = await server.session_service.get_session(
                app_name=request.app_name,
                user_id=request.user_id,
                session_id=request.session_id,
            )

            # Merge auth info with request state_delta
            combined_state_delta = {"_auth": auth_info}
            if request.state_delta:
                combined_state_delta.update(request.state_delta)

            if not session:
                # Create session with auth info and state delta in initial state
                session = await server.session_service.create_session(
                    app_name=request.app_name,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    state=combined_state_delta,
                )
            else:
                # Update existing session with auth info and state delta
                # This persists to DB so the runner sees it when it fetches the session
                await server.session_service.update_session_state(
                    session=session,
                    state_delta=combined_state_delta,
                )

            # Convert message to ADK Content
            new_message = types.Content(
                role=request.new_message.get("role", "user"),
                parts=[
                    types.Part.from_text(text=p.get("text", ""))
                    for p in request.new_message.get("parts", [])
                    if p.get("text")
                ],
            )

            # Prepare runner.run_async kwargs
            run_kwargs = {
                "user_id": request.user_id,
                "session_id": request.session_id,
                "new_message": new_message,
            }
            # Add state_delta if provided
            if request.state_delta:
                run_kwargs["state_delta"] = request.state_delta

            events = []
            async for event in runner.run_async(**run_kwargs):
                events.append(_serialize_event(event))

            return {"events": events}

        except Exception as e:
            logger.exception("Error running agent: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/run_sse")
    async def run_agent_sse(
        request: RunAgentRequest,
        auth_info: dict = Depends(get_auth_info),
    ):
        """
        Run an agent with Server-Sent Events streaming.

        Returns events as SSE stream, ending with [DONE].

        Authorization:
            The AuthorizationPlugin evaluates policies using RBAC and/or ABAC:

            Headers:
            - Authorization: Bearer <token> (for JWT-based auth)
            - X-User-Id: User identifier
            - X-User-Roles: Comma-separated roles for RBAC (e.g., "admin,analyst")
            - X-User-Attributes: JSON for ABAC (e.g., '{"department":"finance"}')

            RBAC Example (role-based):
                X-User-Id: alice
                X-User-Roles: admin,analyst

            ABAC Example (attribute-based):
                X-User-Id: bob
                X-User-Attributes: {"department":"finance","clearance":"high"}

            Combined:
                X-User-Id: carol
                X-User-Roles: manager
                X-User-Attributes: {"department":"finance"}
        """
        try:
            runner = await server.get_runner(request.app_name)

            # Ensure session exists and inject auth into session state
            session = await server.session_service.get_session(
                app_name=request.app_name,
                user_id=request.user_id,
                session_id=request.session_id,
            )

            # Merge auth info with request state_delta
            combined_state_delta = {"_auth": auth_info}
            if request.state_delta:
                combined_state_delta.update(request.state_delta)

            if not session:
                # Create session with auth info and state delta in initial state
                session = await server.session_service.create_session(
                    app_name=request.app_name,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    state=combined_state_delta,
                )
            else:
                # Update existing session with auth info and state delta
                await server.session_service.update_session_state(
                    session=session,
                    state_delta=combined_state_delta,
                )

            # Convert message to ADK Content
            new_message = types.Content(
                role=request.new_message.get("role", "user"),
                parts=[
                    types.Part.from_text(text=p.get("text", ""))
                    for p in request.new_message.get("parts", [])
                    if p.get("text")
                ],
            )

            async def generate():
                try:
                    from google.adk.agents.run_config import RunConfig, StreamingMode

                    # Determine streaming mode based on request
                    stream_mode = StreamingMode.SSE if request.streaming else StreamingMode.NONE

                    # Prepare runner.run_async kwargs
                    run_kwargs = {
                        "user_id": request.user_id,
                        "session_id": request.session_id,
                        "new_message": new_message,
                        "run_config": RunConfig(streaming_mode=stream_mode),
                    }
                    # Add state_delta and invocation_id if provided
                    if request.state_delta:
                        run_kwargs["state_delta"] = request.state_delta
                    if request.invocation_id:
                        run_kwargs["invocation_id"] = request.invocation_id

                    async for event in runner.run_async(**run_kwargs):
                        # Handle artifact_delta splitting like official ADK server
                        # This ensures proper rendering in ADK Web UI
                        events_to_stream = [event]
                        if (
                            hasattr(event, "actions") and
                            hasattr(event.actions, "artifact_delta") and
                            event.actions.artifact_delta and
                            hasattr(event, "content") and
                            event.content and
                            hasattr(event.content, "parts") and
                            event.content.parts
                        ):
                            # Split into content event and artifact event
                            content_event = event.model_copy(deep=True)
                            content_event.actions.artifact_delta = {}
                            artifact_event = event.model_copy(deep=True)
                            artifact_event.content = None
                            events_to_stream = [content_event, artifact_event]

                        for event_to_stream in events_to_stream:
                            # Use model_dump_json for proper serialization
                            if hasattr(event_to_stream, "model_dump_json"):
                                sse_event = event_to_stream.model_dump_json(
                                    exclude_none=True,
                                    by_alias=True,
                                )
                            else:
                                sse_event = json.dumps(_serialize_event(event_to_stream))
                            logger.debug("Generated SSE event: %s", sse_event[:200])
                            yield f"data: {sse_event}\n\n"

                except Exception as e:
                    logger.exception("Error in SSE stream: %s", e)
                    yield f'data: {{"error": "{str(e)}"}}\n\n'

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
            )

        except Exception as e:
            logger.exception("Error starting SSE: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @router.websocket("/run_live")
    async def run_live(websocket: WebSocket):
        """
        WebSocket endpoint for live agent interaction.

        Expects JSON messages with:
        - app_name: str
        - user_id: str
        - session_id: str
        - message: dict (ADK Content format)
        - auth: dict (optional) - Authorization info for RBAC/ABAC

        Authorization:
            Unlike HTTP endpoints, WebSocket auth is passed in each message payload
            since headers are only available at connection time.

            The AuthorizationPlugin evaluates policies using RBAC and/or ABAC:

            auth object structure:
            {
                "is_authenticated": true,
                "user_id": "alice",
                "roles": ["admin", "analyst"],       // For RBAC
                "attributes": {"department": "finance"}  // For ABAC
            }

            RBAC Example (role-based):
            {
                "app_name": "my_app",
                "user_id": "alice",
                "session_id": "sess_123",
                "message": {"role": "user", "parts": [{"text": "Hello"}]},
                "auth": {
                    "is_authenticated": true,
                    "user_id": "alice",
                    "roles": ["admin", "analyst"]
                }
            }

            ABAC Example (attribute-based):
            {
                "app_name": "my_app",
                "user_id": "bob",
                "session_id": "sess_456",
                "message": {"role": "user", "parts": [{"text": "Hello"}]},
                "auth": {
                    "is_authenticated": true,
                    "user_id": "bob",
                    "attributes": {"department": "finance", "clearance": "high"}
                }
            }

            Combined RBAC + ABAC:
            {
                "app_name": "my_app",
                "user_id": "carol",
                "session_id": "sess_789",
                "message": {"role": "user", "parts": [{"text": "Hello"}]},
                "auth": {
                    "is_authenticated": true,
                    "user_id": "carol",
                    "roles": ["manager"],
                    "attributes": {"department": "finance"}
                }
            }
        """
        await websocket.accept()

        try:
            while True:
                # Receive message
                data = await websocket.receive_json()

                app_name = data.get("app_name")
                user_id = data.get("user_id")
                session_id = data.get("session_id")
                message_data = data.get("message", {})

                # Get auth info from message (for WebSocket, client sends auth in each message)
                auth_info = data.get("auth", {
                    "is_authenticated": False,
                    "user_id": user_id,
                    "roles": [],
                    "attributes": {},
                })

                if not all([app_name, user_id, session_id]):
                    await websocket.send_json({
                        "error": "Missing required fields: app_name, user_id, session_id"
                    })
                    continue

                try:
                    runner = await server.get_runner(app_name)

                    # Ensure session exists and inject auth
                    session = await server.session_service.get_session(
                        app_name=app_name,
                        user_id=user_id,
                        session_id=session_id,
                    )
                    if not session:
                        session = await server.session_service.create_session(
                            app_name=app_name,
                            user_id=user_id,
                            session_id=session_id,
                            state={"_auth": auth_info},
                        )
                    else:
                        await server.session_service.update_session_state(
                            session=session,
                            state_delta={"_auth": auth_info},
                        )

                    new_message = types.Content(
                        role=message_data.get("role", "user"),
                        parts=[
                            types.Part.from_text(text=p.get("text", ""))
                            for p in message_data.get("parts", [])
                            if p.get("text")
                        ],
                    )

                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=session_id,
                        new_message=new_message,
                    ):
                        await websocket.send_json(_serialize_event(event))

                    await websocket.send_json({"done": True})

                except Exception as e:
                    logger.exception("Error in live run: %s", e)
                    await websocket.send_json({"error": str(e)})

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.exception("WebSocket error: %s", e)

    # -------------------------------------------------------------------------
    # Memory
    # -------------------------------------------------------------------------

    @router.patch("/apps/{app_name}/users/{user_id}/memory")
    async def add_to_memory(
        app_name: str,
        user_id: str,
        request: UpdateMemoryRequest,
    ):
        """Add a session to memory."""
        if not server.memory_service:
            raise HTTPException(status_code=501, detail="Memory service not configured")

        session = await server.session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=request.session_id,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        await server.memory_service.add_session_to_memory(session)
        return {"added": True}

    # -------------------------------------------------------------------------
    # Artifacts
    # -------------------------------------------------------------------------

    @router.post("/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts")
    async def save_artifact(
        app_name: str,
        user_id: str,
        session_id: str,
        request: SaveArtifactRequest,
    ):
        """Save an artifact."""
        if not server.artifact_service:
            raise HTTPException(status_code=501, detail="Artifact service not configured")

        import base64
        content = base64.b64decode(request.content)
        artifact = types.Part.from_bytes(data=content, mime_type=request.mime_type)

        version = await server.artifact_service.save_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=request.filename,
            artifact=artifact,
        )
        return {"version": version, "filename": request.filename}

    @router.get("/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts")
    async def list_artifacts(app_name: str, user_id: str, session_id: str):
        """List artifacts in a session."""
        if not server.artifact_service:
            raise HTTPException(status_code=501, detail="Artifact service not configured")

        keys = await server.artifact_service.list_artifact_keys(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        return {"artifacts": keys}

    @router.get("/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{filename}")
    async def load_artifact(
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: Optional[int] = None,
    ):
        """Load an artifact."""
        if not server.artifact_service:
            raise HTTPException(status_code=501, detail="Artifact service not configured")

        artifact = await server.artifact_service.load_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
            version=version,
        )
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        import base64
        return {
            "filename": filename,
            "mime_type": artifact.inline_data.mime_type if artifact.inline_data else "application/octet-stream",
            "content": base64.b64encode(artifact.inline_data.data).decode() if artifact.inline_data else None,
        }

    @router.get("/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{filename}/versions")
    async def list_artifact_versions(
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
    ):
        """List versions of an artifact."""
        if not server.artifact_service:
            raise HTTPException(status_code=501, detail="Artifact service not configured")

        versions = await server.artifact_service.list_versions(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
        )
        return {"versions": versions}

    @router.delete("/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{filename}")
    async def delete_artifact(
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
    ):
        """Delete an artifact."""
        if not server.artifact_service:
            raise HTTPException(status_code=501, detail="Artifact service not configured")

        await server.artifact_service.delete_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
        )
        return {"deleted": True}

    return router


# =============================================================================
# Helpers
# =============================================================================

def _serialize_content(content: Any) -> Dict[str, Any]:
    """Serialize ADK Content to JSON-compatible dict."""
    result = {
        "role": getattr(content, "role", "user"),
        "parts": [],
    }

    if hasattr(content, "parts") and content.parts:
        for part in content.parts:
            part_data = {}
            if hasattr(part, "text") and part.text:
                part_data["text"] = part.text
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                part_data["functionCall"] = {
                    "name": fc.name,
                    "args": fc.args if hasattr(fc, "args") else {},
                }
            if hasattr(part, "function_response") and part.function_response:
                fr = part.function_response
                part_data["functionResponse"] = {
                    "name": fr.name,
                    "response": fr.response if hasattr(fr, "response") else {},
                }
            if part_data:
                result["parts"].append(part_data)

    return result


def _serialize_event(event: Any) -> Dict[str, Any]:
    """Serialize an ADK event to JSON-compatible dict."""
    result = {
        "id": getattr(event, "id", None),
        "author": getattr(event, "author", None),
    }

    if hasattr(event, "content") and event.content:
        result["content"] = _serialize_content(event.content)

    return result
