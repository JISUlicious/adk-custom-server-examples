"""
Server configuration for CustomWebServer.
"""

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .auth import JWTConfig


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

        # JWT Authentication
        jwt_secret_key: Secret key for HS256 JWT validation
        jwt_public_key: PEM public key for RS256 JWT validation
        jwt_jwks_url: URL to fetch JWKS for key rotation
        jwt_algorithm: JWT signing algorithm (default: HS256)
        jwt_issuer: Expected JWT issuer (iss claim)
        jwt_audience: Expected JWT audience (aud claim)
        jwt_dev_mode: Allow insecure header-based auth (for development)
    """
    host: str = "0.0.0.0"
    port: int = 8000
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    log_level: str = "info"
    reload: bool = False
    default_user_id: str = "default_user"
    web_ui_enabled: bool = True
    url_prefix: str = ""

    # JWT Authentication
    jwt_secret_key: Optional[str] = None
    jwt_public_key: Optional[str] = None
    jwt_jwks_url: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_issuer: Optional[str] = None
    jwt_audience: Optional[str] = None
    jwt_dev_mode: bool = True  # Default True for backwards compatibility

    def get_jwt_config(self) -> Optional["JWTConfig"]:
        """
        Create JWTConfig from server config if JWT settings are provided.

        Returns:
            JWTConfig if any JWT settings are configured, None otherwise
        """
        from .auth import JWTConfig

        # If no JWT key is configured and not in dev mode, return None
        if not any([self.jwt_secret_key, self.jwt_public_key, self.jwt_jwks_url]):
            if not self.jwt_dev_mode:
                return None
            # Dev mode without keys - allow header-based auth
            return JWTConfig(dev_mode=True)

        return JWTConfig(
            secret_key=self.jwt_secret_key,
            public_key=self.jwt_public_key,
            jwks_url=self.jwt_jwks_url,
            algorithm=self.jwt_algorithm,
            issuer=self.jwt_issuer,
            audience=self.jwt_audience,
            dev_mode=self.jwt_dev_mode,
        )
