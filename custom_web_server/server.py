"""
CustomWebServer - Full-featured ADK web server.

Provides:
- ADK endpoints (sessions, run, artifacts, evaluation)
- OpenAI-compatible endpoints (/v1/chat/completions, /v1/models)
- ADK Web UI for development
- Pluggable runner factory for flexible agent configuration
- Easy service and plugin injection
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from google.adk.runners import Runner
from google.adk.plugins.base_plugin import BasePlugin

from .config import ServerConfig
from .services import ServiceContainer
from .factory import RunnerFactory, DefaultRunnerFactory
from .endpoints import (
    create_health_router,
    create_adk_router,
    create_openai_router,
)

logger = logging.getLogger(__name__)


class CustomWebServer:
    """
    Full-featured ADK web server with OpenAI compatibility.

    Features:
    - ADK session management
    - Agent execution (run, run_sse, run_live)
    - Artifact management
    - Evaluation management
    - OpenAI-compatible chat completions
    - ADK Web UI
    - Pluggable runner factory pattern

    Example (Simple - using DefaultRunnerFactory):
        from custom_web_server import CustomWebServer, ServerConfig, ServiceContainer
        from custom_web_server.factory import DefaultRunnerFactory
        from database_session_service import DatabaseSessionService

        services = ServiceContainer(
            session_service=DatabaseSessionService("sqlite:///session.db"),
        )

        factory = DefaultRunnerFactory(
            agents_dir="./agents",
            services=services,
            plugins=[MyPlugin()],
        )

        server = CustomWebServer(
            runner_factory=factory,
            services=services,
            config=ServerConfig(port=8000),
        )
        server.run()

    Example (Full Control - using StaticRunnerFactory):
        from custom_web_server import CustomWebServer, ServiceContainer
        from custom_web_server.factory import StaticRunnerFactory

        # Create runners with different configurations
        runner_a = Runner(app=app_a, session_service=postgres_service)
        runner_b = Runner(app=app_b, session_service=in_memory_service)

        factory = StaticRunnerFactory({"agent_a": runner_a, "agent_b": runner_b})
        services = ServiceContainer(session_service=postgres_service)

        server = CustomWebServer(runner_factory=factory, services=services)
        server.run()
    """

    def __init__(
        self,
        runner_factory: RunnerFactory,
        services: ServiceContainer,
        config: Optional[ServerConfig] = None,
        default_app_name: Optional[str] = None,
    ):
        """
        Initialize the CustomWebServer.

        Args:
            runner_factory: Factory for creating runners (DefaultRunnerFactory or StaticRunnerFactory)
            services: ServiceContainer with services for direct endpoint access
            config: Server configuration (optional, uses defaults)
            default_app_name: Default app when model name doesn't match (optional)
        """
        self._runner_factory = runner_factory
        self._runner_cache: Dict[str, Runner] = {}

        # Services for direct endpoint access (session CRUD, artifacts, etc.)
        self.session_service = services.session_service
        self.memory_service = services.memory_service
        self.artifact_service = services.artifact_service
        self.eval_sets_manager = services.eval_sets_manager
        self.eval_set_results_manager = services.eval_set_results_manager

        self.default_app_name = default_app_name
        self.config = config or ServerConfig()

        logger.info(
            "CustomWebServer initialized: apps=%s",
            self.list_apps(),
        )

    def list_apps(self) -> List[str]:
        """List available ADK apps."""
        return self._runner_factory.list_apps()

    async def get_runner(self, app_name: str) -> Runner:
        """
        Get a runner for the specified app.

        Creates and caches runners using the runner factory.

        Args:
            app_name: Name of the ADK app

        Returns:
            Configured Runner instance
        """
        # Return cached runner if available
        if app_name in self._runner_cache:
            return self._runner_cache[app_name]

        # Create runner via factory
        runner = await self._runner_factory.create_runner(app_name)

        # Cache and return
        self._runner_cache[app_name] = runner
        logger.debug("Cached runner for app: %s", app_name)

        return runner

    def create_app(self) -> FastAPI:
        """
        Create the FastAPI application.

        Returns:
            Configured FastAPI app with all endpoints
        """
        app = FastAPI(
            title="ADK Web Server",
            description="ADK Web Server with OpenAI-compatible API",
            version="1.0.0",
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add routers
        app.include_router(create_health_router())
        app.include_router(create_adk_router(self))
        app.include_router(create_openai_router(self))

        # Add ADK Web UI if enabled
        if self.config.web_ui_enabled:
            self._mount_web_ui(app)

        # Add dev UI config endpoint
        @app.get("/dev-ui/config", tags=["Dev UI"])
        async def get_dev_ui_config():
            """Get dev UI configuration."""
            return {
                "logo_text": "ADK",
                "url_prefix": self.config.url_prefix,
            }

        logger.info("FastAPI app created with %d routes", len(app.routes))

        return app

    def _mount_web_ui(self, app: FastAPI) -> None:
        """Mount ADK web UI static files."""
        try:
            # Try to find ADK web assets
            import google.adk.cli
            adk_cli_path = Path(google.adk.cli.__file__).parent
            web_assets_path = adk_cli_path / "browser" / "dist"

            if web_assets_path.exists():
                app.mount(
                    "/dev-ui",
                    StaticFiles(directory=str(web_assets_path), html=True),
                    name="dev-ui",
                )
                logger.info("ADK Web UI mounted at /dev-ui")
            else:
                logger.warning("ADK Web UI assets not found at %s", web_assets_path)

        except Exception as e:
            logger.warning("Failed to mount ADK Web UI: %s", e)

    def run(self) -> None:
        """
        Run the server with uvicorn.

        Uses configuration from self.config.
        """
        import uvicorn

        app = self.create_app()

        print(f"\nStarting ADK Web Server...")
        print(f"  Host: {self.config.host}")
        print(f"  Port: {self.config.port}")
        print(f"  Available apps: {self.list_apps()}")
        print(f"\nEndpoints:")
        print(f"  Health:    GET  /health")
        print(f"  ADK:       POST /run, /run_sse, WS /run_live")
        print(f"  OpenAI:    GET  /v1/models")
        print(f"             POST /v1/chat/completions")
        if self.config.web_ui_enabled:
            print(f"  Web UI:    GET  /dev-ui")
        print()

        uvicorn.run(
            app,
            host=self.config.host,
            port=self.config.port,
            log_level=self.config.log_level,
            reload=self.config.reload,
        )


# =============================================================================
# Convenience factory function
# =============================================================================

def create_server(
    agents_dir: str,
    session_db_url: Optional[str] = None,
    memory_db_url: Optional[str] = None,
    plugins: Optional[List[BasePlugin]] = None,
    **kwargs,
) -> CustomWebServer:
    """
    Convenience function to create a CustomWebServer with common configurations.

    Args:
        agents_dir: Directory containing ADK agents
        session_db_url: Database URL for sessions (e.g., "sqlite:///session.db")
        memory_db_url: Database URL for memory (e.g., "sqlite:///memory.db")
        plugins: List of plugins to add to all runners (optional)
        **kwargs: Additional arguments for CustomWebServer

    Returns:
        Configured CustomWebServer instance

    Example:
        from custom_web_server import create_server

        server = create_server(
            agents_dir="./agents",
            session_db_url="sqlite:///session.db",
            memory_db_url="sqlite:///memory.db",
            plugins=[MyPlugin()],
        )
        server.run()
    """
    # Import services
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
    from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
    from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService

    # Create session service
    if session_db_url:
        from database_session_service import DatabaseSessionService
        session_service = DatabaseSessionService(session_db_url)
    else:
        session_service = InMemorySessionService()

    # Create memory service
    if memory_db_url:
        from database_memory_service import DatabaseMemoryService
        memory_service = DatabaseMemoryService(memory_db_url)
    else:
        memory_service = InMemoryMemoryService()

    # Create other services
    artifact_service = kwargs.pop("artifact_service", None) or InMemoryArtifactService()
    credential_service = kwargs.pop("credential_service", None) or InMemoryCredentialService()

    # Create service container
    services = ServiceContainer(
        session_service=session_service,
        memory_service=memory_service,
        artifact_service=artifact_service,
        credential_service=credential_service,
    )

    # Create factory
    factory = DefaultRunnerFactory(
        agents_dir=agents_dir,
        services=services,
        plugins=plugins,
    )

    return CustomWebServer(
        runner_factory=factory,
        services=services,
        **kwargs,
    )
