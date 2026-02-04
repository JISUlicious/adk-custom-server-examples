"""
ADK API endpoints.

Provides:
- Session management
- Agent execution (run, run_sse, run_live)
- Artifact management
- Evaluation management
"""

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from google.genai import types
from pydantic import BaseModel, Field

from ..auth import get_auth_info

if TYPE_CHECKING:
    from ..server import CustomWebServer

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    session_id: Optional[str] = None
    state: Optional[Dict[str, Any]] = None


class UpdateSessionStateRequest(BaseModel):
    """Request to update session state."""
    state: Dict[str, Any]


class RunRequest(BaseModel):
    """Request to run an agent."""
    app_name: str = Field(alias="appName")
    user_id: str = Field(alias="userId")
    session_id: str = Field(alias="sessionId")
    new_message: Dict[str, Any]
    streaming: bool = False

    model_config = {"populate_by_name": True}  # Accept both snake_case and camelCase


class AddSessionToMemoryRequest(BaseModel):
    """Request to add session to memory."""
    session_id: str


class CreateEvalSetRequest(BaseModel):
    """Request to create an eval set."""
    eval_set_id: str


class AddSessionToEvalRequest(BaseModel):
    """Request to add session to eval set."""
    session_id: str


class RunEvalRequest(BaseModel):
    """Request to run evaluation."""
    eval_metrics: Optional[List[str]] = None


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
    async def list_apps() -> List[str]:
        """List available ADK apps."""
        return server.list_apps()

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    @router.post("/apps/{app_name}/users/{user_id}/sessions")
    async def create_session(
        app_name: str,
        user_id: str,
        request: Optional[CreateSessionRequest] = None,
    ):
        """Create a new session.

        Optionally accepts a session_id in the request body to create
        a session with a specific ID. If not provided, a random ID
        will be generated.
        """
        state = request.state if request else None
        session_id = request.session_id if request else None

        try:
            session = await server.session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                state=state or {},
                session_id=session_id,
            )
        except Exception as e:
            # Handle AlreadyExistsError or similar
            if "already exists" in str(e).lower():
                raise HTTPException(
                    status_code=409,
                    detail=f"Session already exists: {session_id}"
                )
            raise HTTPException(status_code=500, detail=str(e))

        return {
            "id": session.id,
            "app_name": session.app_name,
            "user_id": session.user_id,
            "state": session.state,
        }

    @router.post("/apps/{app_name}/users/{user_id}/sessions/{session_id}")
    async def create_session_with_id(
        app_name: str,
        user_id: str,
        session_id: str,
        state: Optional[Dict[str, Any]] = None,
    ):
        """Create a new session with a specific session ID.

        This endpoint allows creating a session with an explicit session_id
        provided in the URL path.
        """
        try:
            session = await server.session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                state=state or {},
                session_id=session_id,
            )
        except Exception as e:
            # Handle AlreadyExistsError or similar
            if "already exists" in str(e).lower():
                raise HTTPException(
                    status_code=409,
                    detail=f"Session already exists: {session_id}"
                )
            raise HTTPException(status_code=500, detail=str(e))

        return {
            "id": session.id,
            "app_name": session.app_name,
            "user_id": session.user_id,
            "state": session.state,
        }

    @router.get("/apps/{app_name}/users/{user_id}/sessions")
    async def list_sessions(app_name: str, user_id: str):
        """List sessions for a user."""
        response = await server.session_service.list_sessions(
            app_name=app_name,
            user_id=user_id,
        )
        return {
            "sessions": [
                {
                    "id": s.id,
                    "app_name": s.app_name,
                    "user_id": s.user_id,
                }
                for s in response.sessions
            ]
        }

    @router.get("/apps/{app_name}/users/{user_id}/sessions/{session_id}")
    async def get_session(app_name: str, user_id: str, session_id: str):
        """Get a specific session."""
        session = await server.session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "id": session.id,
            "app_name": session.app_name,
            "user_id": session.user_id,
            "state": session.state,
            "events": [
                {
                    "id": e.id,
                    "author": e.author,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                }
                for e in session.events
            ] if session.events else [],
        }

    @router.patch("/apps/{app_name}/users/{user_id}/sessions/{session_id}")
    async def update_session_state(
        app_name: str,
        user_id: str,
        session_id: str,
        request: UpdateSessionStateRequest,
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
        request: RunRequest,
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
            if not session:
                # Create session with auth info in initial state
                session = await server.session_service.create_session(
                    app_name=request.app_name,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    state={"_auth": auth_info},
                )
            else:
                # Update existing session with auth info
                # This persists to DB so the runner sees it when it fetches the session
                await server.session_service.update_session_state(
                    session=session,
                    state_delta={"_auth": auth_info},
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

            events = []
            async for event in runner.run_async(
                user_id=request.user_id,
                session_id=request.session_id,
                new_message=new_message,
            ):
                events.append(_serialize_event(event))

            return {"events": events}

        except Exception as e:
            logger.exception("Error running agent: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/run_sse")
    async def run_agent_sse(
        request: RunRequest,
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
            if not session:
                # Create session with auth info in initial state
                session = await server.session_service.create_session(
                    app_name=request.app_name,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    state={"_auth": auth_info},
                )
            else:
                # Update existing session with auth info
                await server.session_service.update_session_state(
                    session=session,
                    state_delta={"_auth": auth_info},
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
                    async for event in runner.run_async(
                        user_id=request.user_id,
                        session_id=request.session_id,
                        new_message=new_message,
                    ):
                        data = json.dumps(_serialize_event(event))
                        yield f"data: {data}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.exception("Error in SSE stream: %s", e)
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
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
        request: AddSessionToMemoryRequest,
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

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    @router.post("/apps/{app_name}/eval-sets")
    async def create_eval_set(app_name: str, request: CreateEvalSetRequest):
        """Create an evaluation set."""
        if not server.eval_sets_manager:
            raise HTTPException(status_code=501, detail="Eval sets manager not configured")

        await server.eval_sets_manager.create_eval_set(
            app_name=app_name,
            eval_set_id=request.eval_set_id,
        )
        return {"eval_set_id": request.eval_set_id}

    @router.get("/apps/{app_name}/eval-sets")
    async def list_eval_sets(app_name: str):
        """List evaluation sets."""
        if not server.eval_sets_manager:
            raise HTTPException(status_code=501, detail="Eval sets manager not configured")

        eval_sets = await server.eval_sets_manager.list_eval_sets(app_name=app_name)
        return {"eval_sets": eval_sets}

    @router.get("/apps/{app_name}/eval-sets/{eval_set_id}")
    async def get_eval_set(app_name: str, eval_set_id: str):
        """Get an evaluation set."""
        if not server.eval_sets_manager:
            raise HTTPException(status_code=501, detail="Eval sets manager not configured")

        eval_set = await server.eval_sets_manager.get_eval_set(
            app_name=app_name,
            eval_set_id=eval_set_id,
        )
        if not eval_set:
            raise HTTPException(status_code=404, detail="Eval set not found")
        return eval_set

    @router.post("/apps/{app_name}/eval-sets/{eval_set_id}/add-session")
    async def add_session_to_eval(
        app_name: str,
        eval_set_id: str,
        request: AddSessionToEvalRequest,
    ):
        """Add a session to an evaluation set."""
        if not server.eval_sets_manager:
            raise HTTPException(status_code=501, detail="Eval sets manager not configured")

        await server.eval_sets_manager.add_session_to_eval_set(
            app_name=app_name,
            eval_set_id=eval_set_id,
            session_id=request.session_id,
        )
        return {"added": True}

    @router.post("/apps/{app_name}/eval-sets/{eval_set_id}/run")
    async def run_eval(
        app_name: str,
        eval_set_id: str,
        request: Optional[RunEvalRequest] = None,
    ):
        """Run an evaluation."""
        if not server.eval_sets_manager:
            raise HTTPException(status_code=501, detail="Eval sets manager not configured")

        # This would typically run the evaluation asynchronously
        # For now, return a placeholder
        return {
            "status": "started",
            "eval_set_id": eval_set_id,
            "message": "Evaluation started (implement actual eval logic)",
        }

    @router.get("/apps/{app_name}/eval-results")
    async def list_eval_results(app_name: str):
        """List evaluation results."""
        if not server.eval_set_results_manager:
            raise HTTPException(status_code=501, detail="Eval results manager not configured")

        results = await server.eval_set_results_manager.list_eval_set_results(
            app_name=app_name,
        )
        return {"results": results}

    return router


# =============================================================================
# Helpers
# =============================================================================

def _serialize_event(event: Any) -> Dict[str, Any]:
    """Serialize an ADK event to JSON-compatible dict."""
    result = {
        "id": getattr(event, "id", None),
        "author": getattr(event, "author", None),
    }

    if hasattr(event, "content") and event.content:
        content = event.content
        result["content"] = {
            "role": content.role,
            "parts": [],
        }
        if content.parts:
            for part in content.parts:
                part_data = {}
                if hasattr(part, "text") and part.text:
                    part_data["text"] = part.text
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    part_data["function_call"] = {
                        "name": fc.name,
                        "args": fc.args,
                    }
                if hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    part_data["function_response"] = {
                        "name": fr.name,
                        "response": fr.response,
                    }
                if part_data:
                    result["content"]["parts"].append(part_data)

    return result
