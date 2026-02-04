"""
JWT-based Authentication for ADK Web Server.

This module provides secure JWT authentication for the AuthorizationPlugin.
Roles and attributes are extracted from JWT claims (signed by the auth server),
NOT from client-provided headers.

Security Model:
    - Client sends: Authorization: Bearer <jwt_token>
    - Server validates: Signature, expiration, issuer, audience
    - Server extracts: user_id (sub), roles, attributes from token claims

The AuthorizationPlugin supports both:
- RBAC (Role-Based Access Control): Uses `roles` claim
- ABAC (Attribute-Based Access Control): Uses `attributes` claim

JWT Token Claims Expected:
    {
        "sub": "user_id",                           # Required: User identifier
        "roles": ["admin", "analyst"],              # Optional: For RBAC
        "attributes": {"department": "finance"},    # Optional: For ABAC
        "exp": 1234567890,                          # Required: Expiration
        "iss": "https://auth.example.com",          # Optional: Issuer
        "aud": "adk-api"                            # Optional: Audience
    }

Usage:
    from custom_web_server.auth import JWTConfig, create_jwt_auth_dependency

    # Configure JWT validation
    jwt_config = JWTConfig(
        secret_key="your-secret-key",  # Or use public_key for RS256
        algorithm="HS256",
        issuer="https://auth.example.com",
        audience="adk-api",
    )

    # Create the dependency
    get_auth_info = create_jwt_auth_dependency(jwt_config)

    # Use in endpoints
    @router.post("/run")
    async def run(request: RunRequest, auth_info: dict = Depends(get_auth_info)):
        # auth_info contains validated claims from JWT
        pass
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import Header, HTTPException, Request

logger = logging.getLogger(__name__)


# =============================================================================
# JWT Configuration
# =============================================================================


@dataclass
class JWTConfig:
    """
    Configuration for JWT validation.

    For symmetric algorithms (HS256, HS384, HS512):
        Use `secret_key` with the shared secret.

    For asymmetric algorithms (RS256, RS384, RS512, ES256, etc.):
        Use `public_key` with the PEM-encoded public key,
        or `jwks_url` to fetch keys from a JWKS endpoint.

    Attributes:
        secret_key: Shared secret for HMAC algorithms (HS256, etc.)
        public_key: PEM-encoded public key for RSA/EC algorithms
        jwks_url: URL to fetch JSON Web Key Set (for key rotation)
        algorithm: JWT algorithm (default: HS256)
        issuer: Expected token issuer (iss claim)
        audience: Expected token audience (aud claim)
        roles_claim: Claim name for roles (default: "roles")
        attributes_claim: Claim name for attributes (default: "attributes")
        leeway: Seconds of leeway for exp/nbf validation (default: 0)
        dev_mode: If True, also accept X-User-* headers (INSECURE, for dev only)
    """

    secret_key: Optional[str] = None
    public_key: Optional[str] = None
    jwks_url: Optional[str] = None
    algorithm: str = "HS256"
    issuer: Optional[str] = None
    audience: Optional[str] = None
    roles_claim: str = "roles"
    attributes_claim: str = "attributes"
    leeway: int = 0
    dev_mode: bool = False

    def __post_init__(self):
        if not any([self.secret_key, self.public_key, self.jwks_url]):
            if not self.dev_mode:
                raise ValueError(
                    "JWTConfig requires secret_key, public_key, or jwks_url "
                    "(or set dev_mode=True for development)"
                )


# =============================================================================
# JWT Validation
# =============================================================================


class JWTValidator:
    """
    Validates JWT tokens and extracts claims.

    This class handles:
    - Token signature validation
    - Expiration checking
    - Issuer/audience validation
    - Claim extraction for RBAC/ABAC
    """

    def __init__(self, config: JWTConfig):
        self.config = config
        self._jwks_client = None

    def _get_key(self) -> Any:
        """Get the key for JWT verification."""
        if self.config.secret_key:
            return self.config.secret_key
        if self.config.public_key:
            return self.config.public_key
        if self.config.jwks_url:
            # Lazy import and initialization
            try:
                import jwt
                if self._jwks_client is None:
                    self._jwks_client = jwt.PyJWKClient(self.config.jwks_url)
                return self._jwks_client
            except ImportError:
                raise ImportError("PyJWT is required: pip install PyJWT")
        raise ValueError("No key configured for JWT validation")

    def validate(self, token: str) -> Dict[str, Any]:
        """
        Validate a JWT token and return the payload.

        Args:
            token: The JWT token string (without "Bearer " prefix)

        Returns:
            Dict containing validated token claims

        Raises:
            HTTPException: If token is invalid, expired, or fails validation
        """
        try:
            import jwt
            from jwt.exceptions import (
                InvalidTokenError,
                ExpiredSignatureError,
                InvalidAudienceError,
                InvalidIssuerError,
            )
        except ImportError:
            raise ImportError("PyJWT is required: pip install PyJWT")

        try:
            # Build validation options
            options = {
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
                "require": ["exp", "sub"],
            }

            # Get the signing key
            key = self._get_key()

            # Handle JWKS (key rotation)
            if self.config.jwks_url and self._jwks_client:
                signing_key = self._jwks_client.get_signing_key_from_jwt(token)
                key = signing_key.key

            # Decode and validate
            payload = jwt.decode(
                token,
                key,
                algorithms=[self.config.algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer,
                leeway=self.config.leeway,
                options=options,
            )

            return payload

        except ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except InvalidAudienceError:
            raise HTTPException(status_code=401, detail="Invalid token audience")
        except InvalidIssuerError:
            raise HTTPException(status_code=401, detail="Invalid token issuer")
        except InvalidTokenError as e:
            logger.warning("JWT validation failed: %s", e)
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    def extract_auth_info(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract auth info from validated JWT payload.

        Args:
            payload: Validated JWT claims

        Returns:
            Auth info dict for AuthorizationPlugin
        """
        return {
            "is_authenticated": True,
            "user_id": payload.get("sub"),
            "roles": payload.get(self.config.roles_claim, []),
            "attributes": payload.get(self.config.attributes_claim, {}),
            "token_claims": payload,  # Include full claims for custom logic
        }


# =============================================================================
# FastAPI Dependencies
# =============================================================================

# Global validator instance (set by create_jwt_auth_dependency)
_jwt_validator: Optional[JWTValidator] = None
_jwt_config: Optional[JWTConfig] = None


def create_jwt_auth_dependency(config: JWTConfig):
    """
    Create a FastAPI dependency for JWT authentication.

    Args:
        config: JWT configuration

    Returns:
        FastAPI dependency function

    Example:
        jwt_config = JWTConfig(
            secret_key="your-secret",
            algorithm="HS256",
            issuer="https://auth.example.com",
        )
        get_auth_info = create_jwt_auth_dependency(jwt_config)

        @app.get("/protected")
        async def protected(auth: dict = Depends(get_auth_info)):
            return {"user": auth["user_id"]}
    """
    global _jwt_validator, _jwt_config
    _jwt_config = config
    _jwt_validator = JWTValidator(config)

    async def jwt_auth_dependency(
        request: Request,
        authorization: Optional[str] = Header(None),
        # Dev mode headers (only used if config.dev_mode=True)
        x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
        x_user_roles: Optional[str] = Header(None, alias="X-User-Roles"),
        x_user_attributes: Optional[str] = Header(None, alias="X-User-Attributes"),
    ) -> Dict[str, Any]:
        """
        Validate JWT and extract auth info.

        In production (dev_mode=False):
            - Only accepts Authorization: Bearer <token>
            - Roles/attributes come from token claims

        In development (dev_mode=True):
            - Also accepts X-User-* headers (INSECURE)
            - Useful for testing without an auth server
        """
        # Try JWT first
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
            payload = _jwt_validator.validate(token)
            return _jwt_validator.extract_auth_info(payload)

        # Dev mode: accept header-based auth (INSECURE)
        if config.dev_mode and x_user_id:
            logger.warning(
                "DEV MODE: Using header-based auth for user '%s'. "
                "Do not use in production!",
                x_user_id,
            )
            attributes = {}
            if x_user_attributes:
                try:
                    attributes = json.loads(x_user_attributes)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Invalid X-User-Attributes header")

            return {
                "is_authenticated": True,
                "user_id": x_user_id,
                "roles": x_user_roles.split(",") if x_user_roles else [],
                "attributes": attributes,
                "_dev_mode": True,  # Flag for debugging
            }

        # No valid auth
        return {
            "is_authenticated": False,
            "user_id": None,
            "roles": [],
            "attributes": {},
        }

    return jwt_auth_dependency


def get_auth_info(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    x_user_roles: Optional[str] = Header(None, alias="X-User-Roles"),
    x_user_attributes: Optional[str] = Header(None, alias="X-User-Attributes"),
) -> Dict[str, Any]:
    """
    Default auth dependency (uses global JWT config if set).

    This function is used when create_jwt_auth_dependency() has been called
    to set up the global JWT configuration.

    If no JWT config is set, falls back to dev mode behavior (header-based).
    """
    # Use configured JWT validator if available
    if _jwt_validator and _jwt_config:
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
            payload = _jwt_validator.validate(token)
            return _jwt_validator.extract_auth_info(payload)

        # Dev mode fallback
        if _jwt_config.dev_mode and x_user_id:
            logger.warning(
                "DEV MODE: Using header-based auth for user '%s'",
                x_user_id,
            )
            attributes = {}
            if x_user_attributes:
                try:
                    attributes = json.loads(x_user_attributes)
                except (json.JSONDecodeError, TypeError):
                    pass
            return {
                "is_authenticated": True,
                "user_id": x_user_id,
                "roles": x_user_roles.split(",") if x_user_roles else [],
                "attributes": attributes,
                "_dev_mode": True,
            }

        return {
            "is_authenticated": False,
            "user_id": None,
            "roles": [],
            "attributes": {},
        }

    # No JWT config - fall back to dev mode (for backwards compatibility)
    logger.warning(
        "No JWT configuration set. Using insecure header-based auth. "
        "Call create_jwt_auth_dependency() to configure JWT."
    )
    attributes = {}
    if x_user_attributes:
        try:
            attributes = json.loads(x_user_attributes)
        except (json.JSONDecodeError, TypeError):
            pass

    if x_user_id:
        return {
            "is_authenticated": True,
            "user_id": x_user_id,
            "roles": x_user_roles.split(",") if x_user_roles else [],
            "attributes": attributes,
            "_dev_mode": True,
        }

    return {
        "is_authenticated": False,
        "user_id": None,
        "roles": [],
        "attributes": {},
    }


# =============================================================================
# Helper Functions
# =============================================================================


def inject_auth_into_session(session, auth_info: Dict[str, Any]) -> None:
    """
    Inject auth info into session state.

    This sets session.state["_auth"] which the AuthorizationPlugin reads
    in its before_run_callback.

    Args:
        session: ADK Session object
        auth_info: Auth info dict from get_auth_info dependency
    """
    session.state["_auth"] = auth_info
    logger.debug(
        "Injected auth into session: user_id=%s, roles=%s",
        auth_info.get("user_id"),
        auth_info.get("roles"),
    )


async def require_auth(auth_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Dependency that requires authentication.

    Usage:
        @router.post("/admin/action")
        async def admin_action(auth: dict = Depends(require_auth)):
            ...  # Only authenticated users reach here
    """
    if auth_info is None or not auth_info.get("is_authenticated"):
        raise HTTPException(status_code=401, detail="Authentication required")
    return auth_info


def create_role_checker(required_roles: List[str], match: str = "any"):
    """
    Create a dependency that checks for required roles.

    Args:
        required_roles: List of roles to check
        match: "any" (user has at least one) or "all" (user has all)

    Returns:
        FastAPI dependency function

    Example:
        require_admin = create_role_checker(["admin"])
        require_manager_and_finance = create_role_checker(
            ["manager", "finance"], match="all"
        )

        @router.delete("/users/{user_id}")
        async def delete_user(auth: dict = Depends(require_admin)):
            ...
    """
    async def role_checker(auth_info: Dict[str, Any] = None) -> Dict[str, Any]:
        if auth_info is None or not auth_info.get("is_authenticated"):
            raise HTTPException(status_code=401, detail="Authentication required")

        user_roles = set(auth_info.get("roles", []))

        if match == "all":
            if not set(required_roles).issubset(user_roles):
                raise HTTPException(
                    status_code=403,
                    detail=f"Required roles (all): {required_roles}",
                )
        else:  # "any"
            if not user_roles.intersection(required_roles):
                raise HTTPException(
                    status_code=403,
                    detail=f"Required roles (any): {required_roles}",
                )

        return auth_info

    return role_checker


def create_attribute_checker(required_attributes: Dict[str, Any]):
    """
    Create a dependency that checks for required attributes (ABAC).

    Args:
        required_attributes: Dict of attribute key-value pairs to check

    Returns:
        FastAPI dependency function

    Example:
        require_finance_dept = create_attribute_checker({"department": "finance"})
        require_high_clearance = create_attribute_checker({
            "department": "finance",
            "clearance": "high",
        })

        @router.get("/financial-reports")
        async def get_reports(auth: dict = Depends(require_finance_dept)):
            ...
    """
    async def attribute_checker(auth_info: Dict[str, Any] = None) -> Dict[str, Any]:
        if auth_info is None or not auth_info.get("is_authenticated"):
            raise HTTPException(status_code=401, detail="Authentication required")

        user_attrs = auth_info.get("attributes", {})

        for key, value in required_attributes.items():
            if user_attrs.get(key) != value:
                raise HTTPException(
                    status_code=403,
                    detail=f"Required attribute: {key}={value}",
                )

        return auth_info

    return attribute_checker


# =============================================================================
# Token Generation (for testing/development)
# =============================================================================


def generate_test_token(
    config: JWTConfig,
    user_id: str,
    roles: Optional[List[str]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    expires_in: int = 3600,
) -> str:
    """
    Generate a test JWT token (for development/testing only).

    Args:
        config: JWT configuration (must have secret_key for signing)
        user_id: User identifier (becomes 'sub' claim)
        roles: List of roles for RBAC
        attributes: Dict of attributes for ABAC
        expires_in: Token expiration in seconds (default: 1 hour)

    Returns:
        Signed JWT token string

    Example:
        config = JWTConfig(secret_key="test-secret", dev_mode=True)
        token = generate_test_token(
            config,
            user_id="alice",
            roles=["admin", "analyst"],
            attributes={"department": "finance"},
        )
        # Use: Authorization: Bearer <token>
    """
    try:
        import jwt
        from datetime import datetime, timedelta, timezone
    except ImportError:
        raise ImportError("PyJWT is required: pip install PyJWT")

    if not config.secret_key:
        raise ValueError("secret_key required to generate tokens")

    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
        config.roles_claim: roles or [],
        config.attributes_claim: attributes or {},
    }

    if config.issuer:
        payload["iss"] = config.issuer
    if config.audience:
        payload["aud"] = config.audience

    return jwt.encode(payload, config.secret_key, algorithm=config.algorithm)
