"""
Policy Information Point (PIP) - loads user context and policies.

The PIP is responsible for:
1. Getting user context (roles, attributes) from various sources
2. Getting policies for resources

Implementations:
- StaticPIP: In-memory policies and users (for testing or simple setups)
- DatabasePIP: Load from database (for production use)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

from .models import UserContext, Policy

logger = logging.getLogger(__name__)


class PolicyInformationPoint(ABC):
    """
    Abstract PIP - plug in any backend for user/policy information.

    Implement this interface to load user information and policies
    from your preferred data source (database, LDAP, config files, etc.).
    """

    @abstractmethod
    async def get_user_context(
        self,
        user_id: str,
        auth_info: Dict[str, Any],
    ) -> UserContext:
        """
        Get user's roles and attributes.

        Args:
            user_id: The user's identifier
            auth_info: Authentication info from session._auth
                       (may contain roles, attributes, etc.)

        Returns:
            SimpleUserContext with user's roles and attributes
        """
        pass

    @abstractmethod
    async def get_policy(
        self,
        resource_type: str,
        resource_name: str,
    ) -> Optional[Policy]:
        """
        Get policy for a resource.

        Args:
            resource_type: "agent" or "tool"
            resource_name: Name of the resource

        Returns:
            Policy if one exists, None otherwise
        """
        pass

    async def start(self) -> None:
        """Start the PIP (e.g., connect to database)."""
        pass

    async def stop(self) -> None:
        """Stop the PIP (e.g., close connections)."""
        pass


class StaticPIP(PolicyInformationPoint):
    """
    In-memory PIP for static policies and users.

    Use this for:
    - Testing
    - Simple setups with few policies
    - Policies defined in code

    Example:
        policies = {
            "agent:admin_agent": Policy(
                resource_type="agent",
                resource_name="admin_agent",
                rules=[PolicyRule(type="rbac", required_roles=["admin"])]
            ),
        }
        users = {
            "alice": SimpleUserContext(
                user_id="alice",
                roles={"analyst"},
                attributes={"department": "finance"}
            ),
        }
        pip = StaticPIP(policies=policies, users=users)
    """

    def __init__(
        self,
        policies: Optional[Dict[str, Policy]] = None,
        users: Optional[Dict[str, UserContext]] = None,
    ):
        """
        Initialize with static policies and users.

        Args:
            policies: Dict mapping "resource_type:resource_name" to Policy
            users: Dict mapping user_id to SimpleUserContext
        """
        self.policies = policies or {}
        self.users = users or {}

    async def get_user_context(
        self,
        user_id: str,
        auth_info: Dict[str, Any],
    ) -> UserContext:
        """
        Get user context from static users dict or auth_info.

        If user exists in users dict, returns that.
        Otherwise, creates context from auth_info.
        """
        # First check static users
        if user_id in self.users:
            return self.users[user_id]

        # Otherwise, create from auth_info
        roles = set(auth_info.get("roles", []))
        attributes = dict(auth_info.get("attributes", {}))

        return UserContext(
            user_id=user_id,
            roles=roles,
            attributes=attributes,
            is_authenticated=True,
        )

    async def get_policy(
        self,
        resource_type: str,
        resource_name: str,
    ) -> Optional[Policy]:
        """Get policy from static policies dict."""
        key = f"{resource_type}:{resource_name}"
        return self.policies.get(key)

    def add_policy(self, policy: Policy) -> None:
        """Add or update a policy."""
        key = policy.get_key()
        self.policies[key] = policy
        logger.debug(f"Added policy: {key}")

    def remove_policy(self, resource_type: str, resource_name: str) -> bool:
        """Remove a policy."""
        key = f"{resource_type}:{resource_name}"
        if key in self.policies:
            del self.policies[key]
            logger.debug(f"Removed policy: {key}")
            return True
        return False

    def add_user(self, user: UserContext) -> None:
        """Add or update a user."""
        self.users[user.user_id] = user
        logger.debug(f"Added user: {user.user_id}")

    def remove_user(self, user_id: str) -> bool:
        """Remove a user."""
        if user_id in self.users:
            del self.users[user_id]
            logger.debug(f"Removed user: {user_id}")
            return True
        return False


class DatabasePIP(PolicyInformationPoint):
    """
    Database-backed PIP for production use.

    Loads user information and policies from a database.
    Supports SQLite and PostgreSQL.

    Example:
        pip = DatabasePIP(database_url="sqlite:///auth.db")
        await pip.start()

        # Later
        await pip.stop()
    """

    def __init__(
        self,
        database_url: str,
        users_table: str = "auth_users",
        policies_table: str = "auth_policies",
        refresh_interval: int = 60,
    ):
        """
        Initialize with database connection info.

        Args:
            database_url: Database URL (sqlite:/// or postgresql://)
            users_table: Table name for user roles/attributes
            policies_table: Table name for policies
            refresh_interval: Seconds between cache refreshes (0 to disable)
        """
        self.database_url = database_url
        self.users_table = users_table
        self.policies_table = policies_table
        self.refresh_interval = refresh_interval
        self._connection = None
        self._db_type = "sqlite" if database_url.startswith("sqlite") else "postgresql"
        self._policy_cache: Dict[str, Policy] = {}
        self._user_cache: Dict[str, UserContext] = {}

    async def start(self) -> None:
        """Connect to database."""
        if self._db_type == "sqlite":
            import aiosqlite
            db_path = self.database_url.replace("sqlite:///", "").replace("sqlite://", "")
            self._connection = await aiosqlite.connect(db_path)
        else:
            import asyncpg
            self._connection = await asyncpg.connect(self.database_url)

        logger.info(f"DatabasePIP connected to {self._db_type}")

    async def stop(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
        logger.info("DatabasePIP disconnected")

    async def get_user_context(
        self,
        user_id: str,
        auth_info: Dict[str, Any],
    ) -> UserContext:
        """
        Get user context from database or auth_info.

        First checks auth_info for roles/attributes (from JWT, etc.).
        If not present, queries database.
        """
        # If auth_info has roles, use those (e.g., from JWT)
        if auth_info.get("roles"):
            return UserContext(
                user_id=user_id,
                roles=set(auth_info.get("roles", [])),
                attributes=dict(auth_info.get("attributes", {})),
                is_authenticated=True,
            )

        # Check cache
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        # Query database
        if not self._connection:
            logger.warning("DatabasePIP not connected, using auth_info only")
            return UserContext(
                user_id=user_id,
                roles=set(auth_info.get("roles", [])),
                attributes=dict(auth_info.get("attributes", {})),
                is_authenticated=True,
            )

        try:
            if self._db_type == "sqlite":
                async with self._connection.execute(
                    f"SELECT roles, attributes FROM {self.users_table} WHERE user_id = ?",
                    (user_id,)
                ) as cursor:
                    row = await cursor.fetchone()
            else:
                row = await self._connection.fetchrow(
                    f"SELECT roles, attributes FROM {self.users_table} WHERE user_id = $1",
                    user_id
                )

            if row:
                import json
                roles_data = row[0] if isinstance(row, tuple) else row["roles"]
                attrs_data = row[1] if isinstance(row, tuple) else row["attributes"]

                roles = set(json.loads(roles_data) if isinstance(roles_data, str) else roles_data)
                attributes = json.loads(attrs_data) if isinstance(attrs_data, str) else attrs_data

                user = UserContext(
                    user_id=user_id,
                    roles=roles,
                    attributes=attributes,
                    is_authenticated=True,
                )
                self._user_cache[user_id] = user
                return user

        except Exception as e:
            logger.error(f"Failed to load user from database: {e}")

        # Fallback to auth_info
        return UserContext(
            user_id=user_id,
            roles=set(auth_info.get("roles", [])),
            attributes=dict(auth_info.get("attributes", {})),
            is_authenticated=True,
        )

    async def get_policy(
        self,
        resource_type: str,
        resource_name: str,
    ) -> Optional[Policy]:
        """Get policy from database."""
        key = f"{resource_type}:{resource_name}"

        # Check cache
        if key in self._policy_cache:
            return self._policy_cache[key]

        if not self._connection:
            logger.warning("DatabasePIP not connected")
            return None

        try:
            if self._db_type == "sqlite":
                async with self._connection.execute(
                    f"SELECT rules FROM {self.policies_table} WHERE resource_type = ? AND resource_name = ?",
                    (resource_type, resource_name)
                ) as cursor:
                    row = await cursor.fetchone()
            else:
                row = await self._connection.fetchrow(
                    f"SELECT rules FROM {self.policies_table} WHERE resource_type = $1 AND resource_name = $2",
                    resource_type, resource_name
                )

            if row:
                import json
                from .models import PolicyRule

                rules_data = row[0] if isinstance(row, tuple) else row["rules"]
                rules_list = json.loads(rules_data) if isinstance(rules_data, str) else rules_data

                rules = [
                    PolicyRule(
                        type=r.get("type", "rbac"),
                        required_roles=r.get("required_roles"),
                        match=r.get("match", "any"),
                        conditions=r.get("conditions"),
                    )
                    for r in rules_list
                ]

                policy = Policy(
                    resource_type=resource_type,
                    resource_name=resource_name,
                    rules=rules,
                )
                self._policy_cache[key] = policy
                return policy

        except Exception as e:
            logger.error(f"Failed to load policy from database: {e}")

        return None

    def invalidate_cache(self) -> None:
        """Clear all cached data."""
        self._policy_cache.clear()
        self._user_cache.clear()
        logger.debug("DatabasePIP cache invalidated")

    def invalidate_user_cache(self, user_id: str) -> None:
        """Clear cached data for a specific user."""
        if user_id in self._user_cache:
            del self._user_cache[user_id]

    def invalidate_policy_cache(self, resource_type: str, resource_name: str) -> None:
        """Clear cached data for a specific policy."""
        key = f"{resource_type}:{resource_name}"
        if key in self._policy_cache:
            del self._policy_cache[key]
