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
from .auth import (
    JWTConfig,
    JWTValidator,
    get_auth_info,
    create_jwt_auth_dependency,
    create_role_checker,
    create_attribute_checker,
    generate_test_token,
)

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
    # Auth (JWT)
    "JWTConfig",
    "JWTValidator",
    "get_auth_info",
    "create_jwt_auth_dependency",
    "create_role_checker",
    "create_attribute_checker",
    "generate_test_token",
    # Convenience
    "create_server",
]
