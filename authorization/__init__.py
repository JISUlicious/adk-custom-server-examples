"""
Simple Authorization Plugin for Google ADK.

Provides RBAC/ABAC authorization using the PIP/PDP pattern.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                   SimpleAuthorizationPlugin                  │
    ├─────────────────────────────────────────────────────────────┤
    │  before_run_callback                                        │
    │  ├── AuthN: Authenticate user (read from session._auth)     │
    │  └── PIP:   Load user context (roles, attributes)           │
    ├─────────────────────────────────────────────────────────────┤
    │  before_agent_callback                                      │
    │  ├── PEP: Intercept request                                 │
    │  ├── PDP: Evaluate RBAC/ABAC policies                       │
    │  └── Enforce: Allow or deny                                 │
    ├─────────────────────────────────────────────────────────────┤
    │  before_tool_callback                                       │
    │  └── Same as agent callback                                 │
    └─────────────────────────────────────────────────────────────┘

Quick Start:
    from authorization import (
        SimpleAuthorizationPlugin,
        StaticPIP,
        Policy,
        PolicyRule,
        UserContext,
    )

    # Define policies
    policies = {
        "agent:admin_agent": Policy(
            resource_type="agent",
            resource_name="admin_agent",
            rules=[PolicyRule(type="rbac", required_roles=["admin"])]
        ),
        "tool:delete_user": Policy(
            resource_type="tool",
            resource_name="delete_user",
            rules=[
                PolicyRule(type="rbac", required_roles=["admin"]),
                PolicyRule(type="abac", conditions={"department": "hr", "level__gte": 3}),
            ]
        ),
    }

    # Define users (optional - can also come from auth_info)
    users = {
        "alice": UserContext(user_id="alice", roles={"analyst"}),
        "bob": UserContext(user_id="bob", roles={"admin"}),
    }

    # Create PIP and plugin
    pip = StaticPIP(policies=policies, users=users)
    auth_plugin = SimpleAuthorizationPlugin(pip=pip)

    # Add to runner
    runner = Runner(app=my_app, plugins=[auth_plugin])
"""

# Simple Models
from .models import (
    UserContext,
    ResourceContext,
    Policy,
    PolicyRule,
    AuthzResult,
)

# Policy Information Point
from .pip import (
    PolicyInformationPoint,
    StaticPIP,
    DatabasePIP,
)

# Policy Decision Point
from .pdp import PolicyDecisionPoint

# Plugin (requires google.adk)
try:
    from .plugin import AuthorizationPlugin
except ImportError:
    AuthorizationPlugin = None  # type: ignore

__all__ = [
    # Models
    "UserContext",
    "ResourceContext",
    "Policy",
    "PolicyRule",
    "AuthzResult",
    # PIP
    "PolicyInformationPoint",
    "StaticPIP",
    "DatabasePIP",
    # PDP
    "PolicyDecisionPoint",
    # Plugin
    "AuthorizationPlugin",
]

__version__ = "0.1.0"
