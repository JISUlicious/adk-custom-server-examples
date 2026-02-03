"""
Custom Web Server for ADK.

A full-featured web server providing:
- ADK endpoints (sessions, run, artifacts, evaluation)
- OpenAI-compatible endpoints (/v1/chat/completions, /v1/models)
- ADK Web UI for development
- Pluggable runner factory pattern
- Authorization plugin support
"""

from .config import ServerConfig
from .server import CustomWebServer, create_server
from .services import ServiceContainer
from .factory import RunnerFactory, DefaultRunnerFactory, StaticRunnerFactory

__all__ = [
    # Core
    "CustomWebServer",
    "ServerConfig",
    # Services
    "ServiceContainer",
    # Factories
    "RunnerFactory",
    "DefaultRunnerFactory",
    "StaticRunnerFactory",
    # Convenience
    "create_server",
]
