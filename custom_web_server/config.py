"""
Server configuration for CustomWebServer.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ServerConfig:
    """
    Configuration for CustomWebServer.

    Attributes:
        host: Host to bind the server to
        port: Port to listen on
        allow_origins: CORS allowed origins
        log_level: Logging level
        reload: Enable auto-reload on file changes
        default_user_id: Default user ID for requests without user context
        web_ui_enabled: Enable ADK web UI
        url_prefix: URL prefix for reverse proxy scenarios
    """
    host: str = "0.0.0.0"
    port: int = 8000
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    log_level: str = "info"
    reload: bool = False
    default_user_id: str = "default_user"
    web_ui_enabled: bool = True
    url_prefix: str = ""
