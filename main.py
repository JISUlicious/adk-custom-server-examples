"""
Main entry point for the ADK Web Server.

Usage:
    python main.py [--host HOST] [--port PORT] [--agents-dir DIR]

Environment variables (can be set in .env file):
    HOST            - Server host (default: 0.0.0.0)
    PORT            - Server port (default: 8000)
    AGENTS_DIR      - Directory containing ADK agents (default: ./)
    SESSION_DB_URL  - Session database URL (default: sqlite:///session.db)
    MEMORY_DB_URL   - Memory database URL (default: sqlite:///memory.db)
    LOG_LEVEL       - Logging level (default: info)
    WEB_UI_ENABLED  - Enable ADK Web UI (default: true)
    RELOAD          - Enable auto-reload (default: false)

Database Support:
    - SQLite: Works out of the box (sync initialization)
    - PostgreSQL: Uses asyncpg with connection pooling (async initialization)
      The server automatically starts/stops connection pools via FastAPI lifespan.

Examples:
    # Run with defaults (SQLite)
    python main.py

    # With PostgreSQL
    export SESSION_DB_URL="postgresql://user:pass@localhost/db"
    export MEMORY_DB_URL="postgresql://user:pass@localhost/db"
    python main.py

    # Custom port (CLI overrides .env)
    python main.py --port 9000

    # Using .env file
    echo "PORT=9000" > .env
    python main.py
"""

import argparse
import logging
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from custom_web_server import (
    CustomWebServer,
    ServerConfig,
    ServiceContainer,
    DefaultRunnerFactory,
)
from database_session_service import DatabaseSessionService
from database_memory_service import DatabaseMemoryService

# Optional: Import authorization if you want to use policies
# from authorization import AuthorizationPlugin, StaticPIP, Policy, PolicyRule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_env(key: str, default: str = None) -> str:
    """Get environment variable with optional default."""
    return os.environ.get(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.environ.get(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    value = os.environ.get(key)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass
    return default


def parse_args():
    """Parse command line arguments (CLI args override env vars)."""
    parser = argparse.ArgumentParser(
        description="ADK Web Server with OpenAI-compatible API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (set in .env or shell):
  HOST            Server host
  PORT            Server port
  AGENTS_DIR      Directory containing ADK agents
  SESSION_DB_URL  Session database URL
  MEMORY_DB_URL   Memory database URL
  LOG_LEVEL       Logging level (debug, info, warning, error)
  WEB_UI_ENABLED  Enable ADK Web UI (true/false)
  RELOAD          Enable auto-reload (true/false)
        """,
    )
    parser.add_argument(
        "--host",
        default=get_env("HOST", "0.0.0.0"),
        help="Host to bind to (env: HOST, default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=get_env_int("PORT", 8000),
        help="Port to listen on (env: PORT, default: 8000)",
    )
    parser.add_argument(
        "--agents-dir",
        default=get_env("AGENTS_DIR", "./"),
        help="Directory containing ADK agents (env: AGENTS_DIR, default: ./)",
    )
    parser.add_argument(
        "--session-db",
        default=get_env("SESSION_DB_URL", "sqlite:///session.db"),
        help="Session database URL (env: SESSION_DB_URL, default: sqlite:///session.db)",
    )
    parser.add_argument(
        "--memory-db",
        default=get_env("MEMORY_DB_URL", "sqlite:///memory.db"),
        help="Memory database URL (env: MEMORY_DB_URL, default: sqlite:///memory.db)",
    )
    parser.add_argument(
        "--no-web-ui",
        action="store_true",
        default=not get_env_bool("WEB_UI_ENABLED", False),
        help="Disable ADK Web UI (env: WEB_UI_ENABLED=false)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=get_env_bool("RELOAD", False),
        help="Enable auto-reload on file changes (env: RELOAD=true)",
    )
    parser.add_argument(
        "--log-level",
        default=get_env("LOG_LEVEL", "info").lower(),
        choices=["debug", "info", "warning", "error"],
        help="Logging level (env: LOG_LEVEL, default: info)",
    )
    return parser.parse_args()


def create_plugins():
    """
    Create plugins for the runner factory.

    Customize this function to add plugins like authorization.
    Return an empty list to disable plugins.

    Example with authorization:
        from authorization import AuthorizationPlugin, StaticPIP, Policy, PolicyRule

        policies = {
            "agent:admin_agent": Policy(
                resource_type="agent",
                resource_name="admin_agent",
                rules=[PolicyRule(type="rbac", required_roles=["admin"])]
            ),
        }
        pip = StaticPIP(policies=policies)
        return [AuthorizationPlugin(pip=pip)]
    """
    return []  # No plugins by default


def main():
    """Main entry point."""
    args = parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    logger.info("Starting ADK Web Server...")
    logger.info("  Agents directory: %s", args.agents_dir)
    logger.info("  Session database: %s", args.session_db)
    logger.info("  Memory database: %s", args.memory_db)

    # Create services
    session_service = DatabaseSessionService(args.session_db)
    memory_service = DatabaseMemoryService(args.memory_db)

    # Create service container
    services = ServiceContainer(
        session_service=session_service,
        memory_service=memory_service,
    )

    # Create plugins (customize in create_plugins function)
    plugins = create_plugins()
    if plugins:
        logger.info("  Plugins: %d", len(plugins))

    # Create runner factory
    factory = DefaultRunnerFactory(
        agents_dir=args.agents_dir,
        services=services,
        plugins=plugins,
    )

    # Create server config
    config = ServerConfig(
        host=args.host,
        port=args.port,
        web_ui_enabled=not args.no_web_ui,
        reload=args.reload,
        log_level=args.log_level,
    )

    # Create and run server
    server = CustomWebServer(
        runner_factory=factory,
        services=services,
        config=config,
    )

    server.run()


if __name__ == "__main__":
    main()
