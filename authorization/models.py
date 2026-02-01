"""
Simple authorization models for the Authorization Plugin.

Provides simplified dataclasses for RBAC/ABAC policy evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set


@dataclass
class UserContext:
    """
    User context for authorization decisions.

    Attributes:
        user_id: Unique user identifier
        roles: Set of user roles for RBAC
        attributes: Dictionary of user attributes for ABAC
        is_authenticated: Whether the user is authenticated
    """
    user_id: str
    roles: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    is_authenticated: bool = True

    @classmethod
    def anonymous(cls, user_id: str = "anonymous") -> "UserContext":
        """Create an anonymous (unauthenticated) user context."""
        return cls(
            user_id=user_id,
            roles=set(),
            attributes={},
            is_authenticated=False,
        )

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles

    def has_any_role(self, roles: List[str]) -> bool:
        """Check if user has any of the specified roles."""
        return bool(self.roles & set(roles))

    def has_all_roles(self, roles: List[str]) -> bool:
        """Check if user has all of the specified roles."""
        return set(roles).issubset(self.roles)

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get a user attribute value."""
        return self.attributes.get(key, default)


@dataclass
class ResourceContext:
    """
    Resource context for authorization decisions.

    Attributes:
        resource_type: Type of resource ("agent" or "tool")
        resource_name: Name/identifier of the resource
    """
    resource_type: str  # "agent" or "tool"
    resource_name: str

    @classmethod
    def for_agent(cls, name: str) -> "ResourceContext":
        """Create a ResourceContext for an agent."""
        return cls(resource_type="agent", resource_name=name)

    @classmethod
    def for_tool(cls, name: str) -> "ResourceContext":
        """Create a ResourceContext for a tool."""
        return cls(resource_type="tool", resource_name=name)


@dataclass
class PolicyRule:
    """
    A single policy rule for RBAC or ABAC.

    Attributes:
        type: Rule type ("rbac" or "abac")
        required_roles: List of required roles (for RBAC)
        match: How to match roles - "any" or "all" (for RBAC)
        conditions: Dictionary of attribute conditions (for ABAC)
    """
    type: str  # "rbac" or "abac"
    # For RBAC
    required_roles: Optional[List[str]] = None
    match: str = "any"  # "any" or "all"
    # For ABAC
    conditions: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.type == "rbac" and self.required_roles is None:
            self.required_roles = []
        if self.type == "abac" and self.conditions is None:
            self.conditions = {}


@dataclass
class Policy:
    """
    Policy definition for a resource.

    Attributes:
        resource_type: Type of resource ("agent" or "tool")
        resource_name: Name of the resource
        rules: List of PolicyRule objects
    """
    resource_type: str
    resource_name: str
    rules: List[PolicyRule] = field(default_factory=list)

    def get_key(self) -> str:
        """Get the policy lookup key."""
        return f"{self.resource_type}:{self.resource_name}"


@dataclass
class AuthzResult:
    """
    Result of an authorization decision.

    Attributes:
        allowed: Whether access is allowed
        reason: Human-readable reason for the decision
    """
    allowed: bool
    reason: str = ""

    @classmethod
    def allow(cls, reason: str = "Access granted") -> "AuthzResult":
        """Create an allow result."""
        return cls(allowed=True, reason=reason)

    @classmethod
    def deny(cls, reason: str = "Access denied") -> "AuthzResult":
        """Create a deny result."""
        return cls(allowed=False, reason=reason)

    def __bool__(self) -> bool:
        """Allow using result directly in conditions."""
        return self.allowed
