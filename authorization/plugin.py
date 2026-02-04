"""
Simple Authorization Plugin for Google ADK.

A simplified authorization plugin that uses PIP (Policy Information Point)
and PDP (Policy Decision Point) for RBAC/ABAC authorization.
"""

import logging
from typing import Optional, Dict, Any

from google.genai import types
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from .pip import PolicyInformationPoint
from .pdp import PolicyDecisionPoint
from .models import UserContext, ResourceContext

logger = logging.getLogger(__name__)


class AuthorizationPlugin(BasePlugin):
    """
    Simple authorization plugin using PIP/PDP pattern.

    This plugin provides:
    - Authentication check via session._auth
    - User context loading via PIP
    - RBAC/ABAC policy evaluation via PDP

    Callbacks:
    - before_run_callback: Authenticate user, load user context via PIP
    - before_agent_callback: Check if user can access agent via PDP
    - before_tool_callback: Check if user can access tool via PDP

    Example:
        from authorization import AuthorizationPlugin, StaticPIP, Policy, PolicyRule

        # Define policies for agents and tools
        policies = {
            # Agent policy: only admins can access admin_agent
            "agent:admin_agent": Policy(
                resource_type="agent",
                resource_name="admin_agent",
                rules=[PolicyRule(type="rbac", required_roles=["admin"])]
            ),

            # Tool policy: only admins can use delete_user tool
            "tool:delete_user": Policy(
                resource_type="tool",
                resource_name="delete_user",
                rules=[PolicyRule(type="rbac", required_roles=["admin"])]
            ),

            # Tool policy with multiple roles (any role grants access)
            "tool:view_reports": Policy(
                resource_type="tool",
                resource_name="view_reports",
                rules=[PolicyRule(
                    type="rbac",
                    required_roles=["admin", "analyst", "manager"],
                    match="any"  # User needs ANY of these roles
                )]
            ),

            # Tool policy requiring ALL roles
            "tool:approve_budget": Policy(
                resource_type="tool",
                resource_name="approve_budget",
                rules=[PolicyRule(
                    type="rbac",
                    required_roles=["manager", "finance"],
                    match="all"  # User needs ALL of these roles
                )]
            ),

            # Tool policy with ABAC (attribute-based access control)
            "tool:access_confidential": Policy(
                resource_type="tool",
                resource_name="access_confidential",
                rules=[PolicyRule(
                    type="abac",
                    conditions={
                        "department": "legal",      # User must be in legal dept
                        "clearance_level": "high",  # User must have high clearance
                    }
                )]
            ),

            # Tool policy combining RBAC and ABAC
            "tool:transfer_funds": Policy(
                resource_type="tool",
                resource_name="transfer_funds",
                rules=[
                    # Rule 1: Admins can always transfer
                    PolicyRule(type="rbac", required_roles=["admin"]),
                    # Rule 2: Finance managers can transfer (if first rule fails)
                    PolicyRule(
                        type="rbac",
                        required_roles=["manager"],
                        conditions={"department": "finance"}
                    ),
                ]
            ),
        }

        # Define users with roles and attributes
        users = {
            "alice": UserContext(
                user_id="alice",
                roles={"analyst"},
                attributes={"department": "research"}
            ),
            "bob": UserContext(
                user_id="bob",
                roles={"admin"},
                attributes={"department": "it"}
            ),
            "carol": UserContext(
                user_id="carol",
                roles={"manager", "finance"},
                attributes={"department": "finance", "clearance_level": "high"}
            ),
        }

        # Create PIP and plugin
        pip = StaticPIP(policies=policies, users=users)
        auth_plugin = AuthorizationPlugin(pip=pip)

        # Add to runner
        runner = Runner(app=my_app, plugins=[auth_plugin])

        # With default_allow=False, tools without policies are denied
        auth_plugin_strict = AuthorizationPlugin(pip=pip, default_allow=False)
    """

    def __init__(
        self,
        pip: PolicyInformationPoint,
        pdp: Optional[PolicyDecisionPoint] = None,
        require_auth: bool = True,
        default_allow: bool = True,
        name: str = "authorization",
    ):
        """
        Initialize the simple authorization plugin.

        Args:
            pip: Policy Information Point for loading user context and policies
            pdp: Policy Decision Point for evaluating rules (defaults to PolicyDecisionPoint)
            require_auth: If True, authentication is required (default: True)
            default_allow: If True, allow access when no policy exists (default: True)
            name: Plugin name (default: "authorization")
        """
        super().__init__(name)
        self.pip = pip
        self.pdp = pdp or PolicyDecisionPoint()
        self.require_auth = require_auth
        self.default_allow = default_allow

    async def before_run_callback(
        self,
        *,
        invocation_context,
    ) -> Optional[types.Content]:
        """
        AuthN + PIP: Authenticate and load user context.

        This callback:
        1. Reads auth info from session._auth (set by middleware or caller)
        2. Checks if user is authenticated
        3. Loads full user context via PIP
        4. Stores user context in session for later callbacks
        """
        # Read auth info from session (set by middleware or caller)
        auth_info = invocation_context.session.state.get("_auth", {})
        logging.debug(f"AuthorizationPlugin: auth_info={auth_info}")
        if not auth_info.get("is_authenticated"):
            if self.require_auth:
                return self._error("Authentication required")

            # Anonymous user
            user_context = UserContext.anonymous()
            invocation_context.session.state["_user_context"] = user_context
            logger.debug("Anonymous user access")
            return None

        # Load full user context via PIP
        user_id = auth_info.get("user_id", invocation_context.user_id)
        user_context = await self.pip.get_user_context(
            user_id=user_id,
            auth_info=auth_info,
        )

        # Store for later callbacks
        invocation_context.session.state["_user_context"] = user_context
        logger.debug(f"Loaded user context for: {user_context.user_id}")

        return None

    async def before_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> Optional[types.Content]:
        """
        PEP + PDP: Check if user can access this agent.

        This callback:
        1. Gets user context from session
        2. Gets policy for agent from PIP
        3. Evaluates policy via PDP
        4. Returns error if access denied
        """
        invocation_context = callback_context._invocation_context
        user = self._get_user_context(invocation_context)
        logging.debug(f"Authorizing user '{user if user else 'unknown'}' for agent '{agent.name}'")
        if not user:
            return self._error("Not authenticated")

        resource = ResourceContext.for_agent(agent.name)
        policy = await self.pip.get_policy("agent", agent.name)

        if not policy:
            # No policy defined
            if self.default_allow:
                logger.debug(f"No policy for agent '{agent.name}', allowing by default")
                return None
            else:
                return self._error(f"Access denied: no policy for agent '{agent.name}'")

        result = self.pdp.evaluate(user, resource, policy)

        if not result.allowed:
            logger.info(f"Access denied to agent '{agent.name}': {result.reason}")
            return self._error(f"Access denied: {result.reason}")

        logger.debug(f"Access granted to agent '{agent.name}': {result.reason}")
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[Dict]:
        """
        PEP + PDP: Check if user can access this tool.

        This callback:
        1. Gets user context from session
        2. Gets policy for tool from PIP
        3. Evaluates policy via PDP
        4. Returns error dict if access denied
        """
        invocation_context = tool_context._invocation_context
        user = self._get_user_context(invocation_context)

        if not user:
            return {"error": "Not authenticated"}

        resource = ResourceContext.for_tool(tool.name)
        policy = await self.pip.get_policy("tool", tool.name)

        if not policy:
            # No policy defined
            if self.default_allow:
                logger.debug(f"No policy for tool '{tool.name}', allowing by default")
                return None
            else:
                return {"error": f"Access denied: no policy for tool '{tool.name}'"}

        result = self.pdp.evaluate(user, resource, policy)

        if not result.allowed:
            logger.info(f"Access denied to tool '{tool.name}': {result.reason}")
            return {"error": f"Access denied: {result.reason}"}

        logger.debug(f"Access granted to tool '{tool.name}': {result.reason}")
        return None

    def _get_user_context(
        self,
        invocation_context,
    ) -> Optional[UserContext]:
        """
        Get user context from session state.

        Falls back to session._auth if _user_context not set
        (for sub-agents that get a copy of invocation_context).
        """
        # Try direct access first
        user_context = invocation_context.session.state.get("_user_context")

        if user_context:
            return user_context

        # Fallback: reconstruct from _auth (for sub-agents)
        auth_info = invocation_context.session.state.get("_auth", {})

        if auth_info.get("is_authenticated"):
            return UserContext(
                user_id=auth_info.get("user_id", "unknown"),
                roles=set(auth_info.get("roles", [])),
                attributes=dict(auth_info.get("attributes", {})),
                is_authenticated=True,
            )

        if not self.require_auth:
            return UserContext.anonymous()

        return None

    def _error(self, message: str) -> types.Content:
        """Create an error response."""
        return types.Content(
            role="model",
            parts=[types.Part(text=message)],
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_user_context(self, invocation_context) -> Optional[UserContext]:
        """Get the current user context for an invocation."""
        return self._get_user_context(invocation_context)

    async def start(self) -> None:
        """Start the plugin (start PIP)."""
        await self.pip.start()
        logger.info("AuthorizationPlugin started")

    async def stop(self) -> None:
        """Stop the plugin (stop PIP)."""
        await self.pip.stop()
        logger.info("AuthorizationPlugin stopped")
