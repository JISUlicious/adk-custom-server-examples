"""
Policy Decision Point (PDP) - evaluates RBAC and ABAC policies.

The PDP is responsible for making authorization decisions based on:
- User context (roles, attributes)
- Resource context (type, name)
- Policy rules (RBAC and ABAC)
"""

import logging
from typing import Optional

from .models import (
    UserContext,
    ResourceContext,
    Policy,
    PolicyRule,
    AuthzResult,
)

logger = logging.getLogger(__name__)


class PolicyDecisionPoint:
    """
    Evaluates RBAC and ABAC rules.

    The PDP takes a user context, resource context, and policy, then
    evaluates all rules to determine if access should be allowed.

    Rules are evaluated in order. First matching rule determines the outcome.

    Example:
        pdp = PolicyDecisionPoint()

        user = SimpleUserContext(user_id="bob", roles={"admin"})
        resource = SimpleResourceContext.for_tool("delete_user")
        policy = Policy(
            resource_type="tool",
            resource_name="delete_user",
            rules=[PolicyRule(type="rbac", required_roles=["admin"])]
        )

        result = pdp.evaluate(user, resource, policy)
        if result.allowed:
            print(f"Access granted: {result.reason}")
        else:
            print(f"Access denied: {result.reason}")
    """

    def evaluate(
        self,
        user: UserContext,
        resource: ResourceContext,
        policy: Policy,
    ) -> AuthzResult:
        """
        Evaluate policy rules against user and resource contexts.

        Args:
            user: The user context with roles and attributes
            resource: The resource being accessed
            policy: The policy containing rules to evaluate

        Returns:
            AuthzResult indicating if access is allowed and why
        """
        # Verify policy applies to this resource
        if policy.resource_type != resource.resource_type:
            return AuthzResult.deny(
                f"Policy resource type mismatch: policy={policy.resource_type}, resource={resource.resource_type}"
            )
        if policy.resource_name != resource.resource_name:
            return AuthzResult.deny(
                f"Policy resource name mismatch: policy={policy.resource_name}, resource={resource.resource_name}"
            )

        if not policy.rules:
            return AuthzResult.deny("No rules defined in policy")

        for rule in policy.rules:
            if rule.type == "rbac":
                result = self._evaluate_rbac(user, rule)
            elif rule.type == "abac":
                result = self._evaluate_abac(user, rule)
            else:
                logger.warning(f"Unknown rule type: {rule.type}")
                continue

            # First matching rule wins
            if result.allowed:
                return result

        return AuthzResult.deny("No matching rule")

    def _evaluate_rbac(
        self,
        user: UserContext,
        rule: PolicyRule,
    ) -> AuthzResult:
        """
        Check if user has required role(s).

        Args:
            user: The user context
            rule: The RBAC rule to evaluate

        Returns:
            AuthzResult indicating if the rule matched
        """
        required_roles = rule.required_roles or []

        if not required_roles:
            return AuthzResult.deny("No roles specified in rule")

        if rule.match == "all":
            # User must have ALL required roles
            if user.has_all_roles(required_roles):
                matched = list(set(required_roles) & user.roles)
                return AuthzResult.allow(f"Has all required roles: {matched}")
            missing = list(set(required_roles) - user.roles)
            return AuthzResult.deny(f"Missing roles: {missing}")
        else:
            # Default: user must have ANY of the required roles
            if user.has_any_role(required_roles):
                matched = list(set(required_roles) & user.roles)
                return AuthzResult.allow(f"Has role: {matched}")
            return AuthzResult.deny(f"Missing role (need one of: {required_roles})")

    def _evaluate_abac(
        self,
        user: UserContext,
        rule: PolicyRule,
    ) -> AuthzResult:
        """
        Check if user attributes match conditions.

        Supports operators:
        - Exact match: {"department": "hr"}
        - Greater than or equal: {"level__gte": 3}
        - Less than or equal: {"level__lte": 10}
        - Greater than: {"level__gt": 2}
        - Less than: {"level__lt": 5}
        - In list: {"status__in": ["active", "pending"]}
        - Contains: {"tags__contains": "admin"}

        Args:
            user: The user context
            rule: The ABAC rule to evaluate

        Returns:
            AuthzResult indicating if the rule matched
        """
        conditions = rule.conditions or {}

        if not conditions:
            return AuthzResult.deny("No conditions specified in ABAC rule")

        for key, expected in conditions.items():
            # Parse operator from key (e.g., "clearance__gte" -> "clearance", "gte")
            parts = key.split("__")
            attr_name = parts[0]
            operator = parts[1] if len(parts) > 1 else "eq"

            actual = user.get_attribute(attr_name)

            if not self._check_condition(actual, operator, expected):
                return AuthzResult.deny(
                    f"Condition failed: {attr_name} (actual={actual}, expected {operator} {expected})"
                )

        return AuthzResult.allow("All ABAC conditions satisfied")

    def _check_condition(
        self,
        actual,
        operator: str,
        expected,
    ) -> bool:
        """
        Check a single ABAC condition.

        Args:
            actual: The actual attribute value
            operator: The comparison operator
            expected: The expected value

        Returns:
            True if condition is satisfied
        """
        if actual is None:
            return False

        try:
            if operator == "eq":
                return actual == expected
            elif operator == "ne":
                return actual != expected
            elif operator == "gt":
                return actual > expected
            elif operator == "gte":
                return actual >= expected
            elif operator == "lt":
                return actual < expected
            elif operator == "lte":
                return actual <= expected
            elif operator == "in":
                return actual in expected
            elif operator == "contains":
                if isinstance(actual, (list, set, tuple)):
                    return expected in actual
                elif isinstance(actual, str):
                    return expected in actual
                return False
            elif operator == "startswith":
                return isinstance(actual, str) and actual.startswith(expected)
            elif operator == "endswith":
                return isinstance(actual, str) and actual.endswith(expected)
            else:
                logger.warning(f"Unknown ABAC operator: {operator}")
                return False
        except TypeError:
            # Type mismatch in comparison
            return False
