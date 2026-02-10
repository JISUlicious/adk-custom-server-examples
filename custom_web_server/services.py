"""
Service container for ADK services.

Groups all services for cleaner dependency injection.
"""

from dataclasses import dataclass
from typing import Optional

from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.artifacts.base_artifact_service import BaseArtifactService
from google.adk.auth.credential_service.base_credential_service import BaseCredentialService


@dataclass
class ServiceContainer:
    """
    Container for ADK services.

    Groups all services required by CustomWebServer for cleaner dependency injection.
    Only session_service is required; all others are optional.

    Attributes:
        session_service: Session storage service (required)
        memory_service: Memory service for RAG (optional)
        artifact_service: Artifact storage service (optional)
        credential_service: Credential management service (optional)

    Example:
        from google.adk.sessions.in_memory_session_service import InMemorySessionService
        from google.adk.memory.in_memory_memory_service import InMemoryMemoryService

        services = ServiceContainer(
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
    """

    session_service: BaseSessionService
    memory_service: Optional[BaseMemoryService] = None
    artifact_service: Optional[BaseArtifactService] = None
    credential_service: Optional[BaseCredentialService] = None
