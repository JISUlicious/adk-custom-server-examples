"""
Runner factory implementations.

Provides pluggable runner creation strategies for CustomWebServer.
"""

import logging
from typing import Dict, List, Optional, Protocol, runtime_checkable

from google.adk.runners import Runner
from google.adk.apps import App
from google.adk.agents.base_agent import BaseAgent
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.cli.utils.agent_loader import AgentLoader

from .services import ServiceContainer


logger = logging.getLogger(__name__)


@runtime_checkable
class RunnerFactory(Protocol):
    """
    Protocol for runner factories.

    Implement this protocol to customize how runners are created.
    The factory pattern allows for:
    - Different services per agent
    - Custom plugin injection
    - Pre-configured runner pools
    - Lazy vs eager loading strategies
    """

    def list_apps(self) -> List[str]:
        """
        List available app names.

        Returns:
            List of app names that can be used with create_runner()
        """
        ...

    async def create_runner(self, app_name: str) -> Runner:
        """
        Create a runner for the given app.

        Args:
            app_name: Name of the app to create a runner for

        Returns:
            Configured Runner instance

        Raises:
            ValueError: If app_name is not available
        """
        ...


class DefaultRunnerFactory:
    """
    Default runner factory that loads agents from a directory.

    Uses shared services and plugins for all runners.
    Supports lazy loading - agents are loaded when first requested.

    Example:
        services = ServiceContainer(
            session_service=DatabaseSessionService("sqlite:///session.db"),
            memory_service=DatabaseMemoryService("sqlite:///memory.db"),
        )

        factory = DefaultRunnerFactory(
            agents_dir="./agents",
            services=services,
            plugins=[MyCustomPlugin(), AuthorizationPlugin(pip=my_pip)],
        )

        # List available apps
        apps = factory.list_apps()  # ["agent1", "agent2"]

        # Create a runner
        runner = await factory.create_runner("agent1")
    """

    def __init__(
        self,
        agents_dir: str,
        services: ServiceContainer,
        plugins: Optional[List[BasePlugin]] = None,
    ):
        """
        Initialize the DefaultRunnerFactory.

        Args:
            agents_dir: Directory containing ADK agent subdirectories
            services: ServiceContainer with required services
            plugins: List of plugins to add to all runners (optional)
        """
        self._agents_dir = agents_dir
        self._agent_loader = AgentLoader(agents_dir)
        self._services = services
        self._plugins = plugins or []

        logger.info(
            "DefaultRunnerFactory initialized: agents_dir=%s, plugins=%d",
            agents_dir,
            len(self._plugins),
        )

    def list_apps(self) -> List[str]:
        """List available ADK apps from the agents directory."""
        return self._agent_loader.list_agents()

    async def create_runner(self, app_name: str) -> Runner:
        """
        Create a runner for the specified app.

        Loads the agent from the agents directory and creates a runner
        with the shared services and plugins.

        Args:
            app_name: Name of the ADK app

        Returns:
            Configured Runner instance

        Raises:
            ValueError: If app_name is not found in agents directory
        """
        available_apps = self.list_apps()
        if app_name not in available_apps:
            raise ValueError(
                f"Unknown app: {app_name}. Available: {available_apps}"
            )

        # Load agent
        agent = self._agent_loader.load_agent(app_name)

        # Wrap in App if needed
        if isinstance(agent, BaseAgent):
            agentic_app = App(
                name=app_name,
                root_agent=agent,
                plugins=list(self._plugins),  # Copy plugins
            )
        else:
            agentic_app = agent

        # Create runner with shared services
        runner = Runner(
            app=agentic_app,
            session_service=self._services.session_service,
            memory_service=self._services.memory_service,
            artifact_service=self._services.artifact_service,
            credential_service=self._services.credential_service,
        )

        logger.info("Created runner for app: %s", app_name)
        return runner


class StaticRunnerFactory:
    """
    Factory that provides pre-configured runners.

    Use this when you need full control over runner configuration,
    such as different services per agent or complex setup requirements.

    Example:
        # Agent A: uses PostgreSQL
        runner_a = Runner(
            app=app_a,
            session_service=PostgresSessionService("postgres://..."),
            memory_service=PineconeMemoryService("..."),
        )

        # Agent B: uses in-memory (for testing)
        runner_b = Runner(
            app=app_b,
            session_service=InMemorySessionService(),
        )

        factory = StaticRunnerFactory({
            "agent_a": runner_a,
            "agent_b": runner_b,
        })

        # Use with CustomWebServer
        server = CustomWebServer(runner_factory=factory, services=services)
    """

    def __init__(self, runners: Dict[str, Runner]):
        """
        Initialize with pre-configured runners.

        Args:
            runners: Dict mapping app names to Runner instances
        """
        self._runners = runners

        logger.info(
            "StaticRunnerFactory initialized with %d runners: %s",
            len(runners),
            list(runners.keys()),
        )

    def list_apps(self) -> List[str]:
        """List available app names."""
        return list(self._runners.keys())

    async def create_runner(self, app_name: str) -> Runner:
        """
        Return the pre-configured runner for the given app.

        Args:
            app_name: Name of the app

        Returns:
            Pre-configured Runner instance

        Raises:
            ValueError: If app_name is not in the runners dict
        """
        if app_name not in self._runners:
            raise ValueError(
                f"Unknown app: {app_name}. Available: {list(self._runners.keys())}"
            )

        return self._runners[app_name]
