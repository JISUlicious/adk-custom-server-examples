"""
Database-backed Session Service for Google ADK.

Supports both SQLite and PostgreSQL (using asyncpg for PostgreSQL).
Similar to InMemorySessionService but persists sessions and events to a database.

Usage:
    # SQLite
    session_service = DatabaseSessionService("sqlite:///sessions.db")

    # PostgreSQL (requires calling start() before use)
    session_service = DatabaseSessionService("postgresql://user:pass@host/db")
    await session_service.start()  # Initialize connection pool

    # On shutdown
    await session_service.close()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Optional

from typing_extensions import override

from google.adk.events.event import Event
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)
from google.adk.sessions.session import Session
from google.adk.sessions.state import State

logger = logging.getLogger("google_adk." + __name__)


def _extract_state_delta(state: Optional[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Extracts app, user, and session state deltas from a state dictionary."""
    deltas = {"app": {}, "user": {}, "session": {}}
    if state:
        for key in state.keys():
            if key.startswith(State.APP_PREFIX):
                deltas["app"][key.removeprefix(State.APP_PREFIX)] = state[key]
            elif key.startswith(State.USER_PREFIX):
                deltas["user"][key.removeprefix(State.USER_PREFIX)] = state[key]
            elif not key.startswith(State.TEMP_PREFIX):
                deltas["session"][key] = state[key]
    return deltas


def _merge_state(
    app_state: dict[str, Any],
    user_state: dict[str, Any],
    session_state: dict[str, Any],
) -> dict[str, Any]:
    """Merge app, user, and session states into a single state dictionary."""
    merged_state = dict(session_state)
    for key in app_state.keys():
        merged_state[State.APP_PREFIX + key] = app_state[key]
    for key in user_state.keys():
        merged_state[State.USER_PREFIX + key] = user_state[key]
    return merged_state


class DatabaseSessionService(BaseSessionService):
    """A session service that uses a database for storage.

    Supports SQLite and PostgreSQL (using asyncpg).

    Args:
        database_url: Database connection URL. Examples:
            - "sqlite:///sessions.db" for SQLite file
            - "sqlite:///:memory:" for in-memory SQLite
            - "postgresql://user:password@localhost/dbname" for PostgreSQL
    """

    # SQLite schema
    SQLITE_SCHEMA = """
    -- Sessions table
    CREATE TABLE IF NOT EXISTS sessions (
        app_name TEXT NOT NULL,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        state TEXT DEFAULT '{}',
        create_time REAL NOT NULL,
        update_time REAL NOT NULL,
        PRIMARY KEY (app_name, user_id, session_id)
    );

    -- Events table
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        app_name TEXT NOT NULL,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        event_data TEXT NOT NULL,
        timestamp REAL NOT NULL,
        FOREIGN KEY (app_name, user_id, session_id)
            REFERENCES sessions(app_name, user_id, session_id) ON DELETE CASCADE
    );

    -- App state table
    CREATE TABLE IF NOT EXISTS app_state (
        app_name TEXT PRIMARY KEY,
        state TEXT DEFAULT '{}'
    );

    -- User state table
    CREATE TABLE IF NOT EXISTS user_state (
        app_name TEXT NOT NULL,
        user_id TEXT NOT NULL,
        state TEXT DEFAULT '{}',
        PRIMARY KEY (app_name, user_id)
    );

    -- Indexes for better query performance
    CREATE INDEX IF NOT EXISTS idx_events_session
        ON events(app_name, user_id, session_id);
    CREATE INDEX IF NOT EXISTS idx_events_timestamp
        ON events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_sessions_app_user
        ON sessions(app_name, user_id);
    """

    # PostgreSQL schema
    POSTGRES_SCHEMA = """
    -- Sessions table
    CREATE TABLE IF NOT EXISTS sessions (
        app_name TEXT NOT NULL,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        state JSONB DEFAULT '{}',
        create_time DOUBLE PRECISION NOT NULL,
        update_time DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (app_name, user_id, session_id)
    );

    -- Events table
    CREATE TABLE IF NOT EXISTS events (
        id SERIAL PRIMARY KEY,
        app_name TEXT NOT NULL,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        event_data JSONB NOT NULL,
        timestamp DOUBLE PRECISION NOT NULL,
        FOREIGN KEY (app_name, user_id, session_id)
            REFERENCES sessions(app_name, user_id, session_id) ON DELETE CASCADE
    );

    -- App state table
    CREATE TABLE IF NOT EXISTS app_state (
        app_name TEXT PRIMARY KEY,
        state JSONB DEFAULT '{}'
    );

    -- User state table
    CREATE TABLE IF NOT EXISTS user_state (
        app_name TEXT NOT NULL,
        user_id TEXT NOT NULL,
        state JSONB DEFAULT '{}',
        PRIMARY KEY (app_name, user_id)
    );

    -- Indexes for better query performance
    CREATE INDEX IF NOT EXISTS idx_events_session
        ON events(app_name, user_id, session_id);
    CREATE INDEX IF NOT EXISTS idx_events_timestamp
        ON events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_sessions_app_user
        ON sessions(app_name, user_id);
    """

    def __init__(self, database_url: str):
        """Initialize the database session service.

        Args:
            database_url: Database connection URL.
        """
        self.database_url = database_url
        self._is_postgres = database_url.startswith("postgresql://")
        self._is_memory = ":memory:" in database_url

        # For in-memory SQLite, keep a persistent connection
        self._persistent_conn: Optional[sqlite3.Connection] = None

        # For PostgreSQL, use shared pool from database_pool module
        self._pool = None  # asyncpg.Pool, set in start()
        self._started = False

        # Initialize SQLite immediately (sync), PostgreSQL requires start()
        if not self._is_postgres:
            self._init_sqlite()

    async def start(self) -> None:
        """Initialize PostgreSQL connection pool and schema.

        Must be called before using the service with PostgreSQL.
        For SQLite, this is a no-op as initialization happens in __init__.
        """
        if self._started:
            return

        if self._is_postgres:
            from database_pool import AsyncPgPool
            self._pool = await AsyncPgPool.get_pool(self.database_url)
            await self._init_postgres()

        self._started = True
        logger.info("DatabaseSessionService started: %s", self.database_url[:50])

    async def close(self) -> None:
        """Close database connections.

        For SQLite in-memory, closes the persistent connection.
        For PostgreSQL, the pool is managed by AsyncPgPool.close_all().
        """
        if self._persistent_conn:
            self._persistent_conn.close()
            self._persistent_conn = None

        self._started = False
        logger.info("DatabaseSessionService closed")

    def _init_sqlite(self):
        """Initialize SQLite schema."""
        with self._get_sqlite_cursor() as cursor:
            cursor.executescript(self.SQLITE_SCHEMA)

    async def _init_postgres(self):
        """Initialize PostgreSQL schema."""
        async with self._pool.acquire() as conn:
            await conn.execute(self.POSTGRES_SCHEMA)

    def _get_sqlite_path(self) -> str:
        """Extract SQLite file path from database URL."""
        if self.database_url.startswith("sqlite:///"):
            return self.database_url[10:]
        elif self.database_url.startswith("sqlite://"):
            return self.database_url[9:]
        return self.database_url

    @contextmanager
    def _get_sqlite_cursor(self):
        """Context manager for SQLite database cursor."""
        if self._is_memory:
            # For in-memory database, use persistent connection
            if self._persistent_conn is None:
                self._persistent_conn = sqlite3.connect(":memory:")
                self._persistent_conn.row_factory = sqlite3.Row
                self._persistent_conn.execute("PRAGMA foreign_keys = ON")
            cursor = self._persistent_conn.cursor()
            try:
                yield cursor
                self._persistent_conn.commit()
            except Exception:
                self._persistent_conn.rollback()
                raise
            finally:
                cursor.close()
        else:
            conn = sqlite3.connect(self._get_sqlite_path())
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                cursor = conn.cursor()
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()
                conn.close()

    def _serialize_json(self, data: Any) -> str:
        """Serialize data to JSON string."""
        return json.dumps(data, default=str)

    def _deserialize_json(self, data: Any) -> Any:
        """Deserialize JSON data."""
        if data is None:
            return {}
        if isinstance(data, str):
            return json.loads(data)
        return data  # Already a dict (PostgreSQL JSONB via asyncpg)

    # =========================================================================
    # SQLite helper methods (sync)
    # =========================================================================

    def _get_app_state_sqlite(self, cursor, app_name: str) -> dict[str, Any]:
        """Get app state from SQLite."""
        cursor.execute(
            "SELECT state FROM app_state WHERE app_name = ?",
            (app_name,)
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_json(row[0])
        return {}

    def _get_user_state_sqlite(self, cursor, app_name: str, user_id: str) -> dict[str, Any]:
        """Get user state from SQLite."""
        cursor.execute(
            "SELECT state FROM user_state WHERE app_name = ? AND user_id = ?",
            (app_name, user_id)
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_json(row[0])
        return {}

    def _upsert_app_state_sqlite(self, cursor, app_name: str, state: dict[str, Any]):
        """Upsert app state in SQLite."""
        state_json = self._serialize_json(state)
        cursor.execute("""
            INSERT INTO app_state (app_name, state) VALUES (?, ?)
            ON CONFLICT (app_name) DO UPDATE SET state = excluded.state
        """, (app_name, state_json))

    def _upsert_user_state_sqlite(self, cursor, app_name: str, user_id: str, state: dict[str, Any]):
        """Upsert user state in SQLite."""
        state_json = self._serialize_json(state)
        cursor.execute("""
            INSERT INTO user_state (app_name, user_id, state) VALUES (?, ?, ?)
            ON CONFLICT (app_name, user_id) DO UPDATE SET state = excluded.state
        """, (app_name, user_id, state_json))

    # =========================================================================
    # PostgreSQL helper methods (async)
    # =========================================================================

    async def _get_app_state_pg(self, conn, app_name: str) -> dict[str, Any]:
        """Get app state from PostgreSQL."""
        row = await conn.fetchrow(
            "SELECT state FROM app_state WHERE app_name = $1",
            app_name
        )
        if row:
            return self._deserialize_json(row["state"])
        return {}

    async def _get_user_state_pg(self, conn, app_name: str, user_id: str) -> dict[str, Any]:
        """Get user state from PostgreSQL."""
        row = await conn.fetchrow(
            "SELECT state FROM user_state WHERE app_name = $1 AND user_id = $2",
            app_name, user_id
        )
        if row:
            return self._deserialize_json(row["state"])
        return {}

    async def _upsert_app_state_pg(self, conn, app_name: str, state: dict[str, Any]):
        """Upsert app state in PostgreSQL."""
        state_json = self._serialize_json(state)
        await conn.execute("""
            INSERT INTO app_state (app_name, state) VALUES ($1, $2)
            ON CONFLICT (app_name) DO UPDATE SET state = EXCLUDED.state
        """, app_name, state_json)

    async def _upsert_user_state_pg(self, conn, app_name: str, user_id: str, state: dict[str, Any]):
        """Upsert user state in PostgreSQL."""
        state_json = self._serialize_json(state)
        await conn.execute("""
            INSERT INTO user_state (app_name, user_id, state) VALUES ($1, $2, $3)
            ON CONFLICT (app_name, user_id) DO UPDATE SET state = EXCLUDED.state
        """, app_name, user_id, state_json)

    # =========================================================================
    # Session CRUD - Create
    # =========================================================================

    @override
    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        """Creates a new session."""
        if self._is_postgres:
            return await self._create_session_pg(
                app_name=app_name,
                user_id=user_id,
                state=state,
                session_id=session_id,
            )
        else:
            return self._create_session_sqlite(
                app_name=app_name,
                user_id=user_id,
                state=state,
                session_id=session_id,
            )

    def _create_session_sqlite(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        """Create session in SQLite."""
        session_id = session_id.strip() if session_id and session_id.strip() else str(uuid.uuid4())
        current_time = time.time()

        state_deltas = _extract_state_delta(state)
        app_state_delta = state_deltas["app"]
        user_state_delta = state_deltas["user"]
        session_state = state_deltas["session"]

        with self._get_sqlite_cursor() as cursor:
            # Check if session already exists
            cursor.execute(
                "SELECT 1 FROM sessions WHERE app_name = ? AND user_id = ? AND session_id = ?",
                (app_name, user_id, session_id)
            )
            if cursor.fetchone():
                from google.adk.errors.already_exists_error import AlreadyExistsError
                raise AlreadyExistsError(f"Session with id {session_id} already exists.")

            # Get and update app state
            existing_app_state = self._get_app_state_sqlite(cursor, app_name)
            if app_state_delta:
                existing_app_state.update(app_state_delta)
                self._upsert_app_state_sqlite(cursor, app_name, existing_app_state)
            elif not existing_app_state:
                self._upsert_app_state_sqlite(cursor, app_name, {})

            # Get and update user state
            existing_user_state = self._get_user_state_sqlite(cursor, app_name, user_id)
            if user_state_delta:
                existing_user_state.update(user_state_delta)
                self._upsert_user_state_sqlite(cursor, app_name, user_id, existing_user_state)
            elif not existing_user_state:
                self._upsert_user_state_sqlite(cursor, app_name, user_id, {})

            # Create session
            state_json = self._serialize_json(session_state)
            cursor.execute("""
                INSERT INTO sessions (app_name, user_id, session_id, state, create_time, update_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (app_name, user_id, session_id, state_json, current_time, current_time))

            merged_state = _merge_state(existing_app_state, existing_user_state, session_state)

        return Session(
            id=session_id,
            app_name=app_name,
            user_id=user_id,
            state=merged_state,
            events=[],
            last_update_time=current_time,
        )

    async def _create_session_pg(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        """Create session in PostgreSQL."""
        session_id = session_id.strip() if session_id and session_id.strip() else str(uuid.uuid4())
        current_time = time.time()

        state_deltas = _extract_state_delta(state)
        app_state_delta = state_deltas["app"]
        user_state_delta = state_deltas["user"]
        session_state = state_deltas["session"]

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Check if session already exists
                row = await conn.fetchrow(
                    "SELECT 1 FROM sessions WHERE app_name = $1 AND user_id = $2 AND session_id = $3",
                    app_name, user_id, session_id
                )
                if row:
                    from google.adk.errors.already_exists_error import AlreadyExistsError
                    raise AlreadyExistsError(f"Session with id {session_id} already exists.")

                # Get and update app state
                existing_app_state = await self._get_app_state_pg(conn, app_name)
                if app_state_delta:
                    existing_app_state.update(app_state_delta)
                    await self._upsert_app_state_pg(conn, app_name, existing_app_state)
                elif not existing_app_state:
                    await self._upsert_app_state_pg(conn, app_name, {})

                # Get and update user state
                existing_user_state = await self._get_user_state_pg(conn, app_name, user_id)
                if user_state_delta:
                    existing_user_state.update(user_state_delta)
                    await self._upsert_user_state_pg(conn, app_name, user_id, existing_user_state)
                elif not existing_user_state:
                    await self._upsert_user_state_pg(conn, app_name, user_id, {})

                # Create session
                state_json = self._serialize_json(session_state)
                await conn.execute("""
                    INSERT INTO sessions (app_name, user_id, session_id, state, create_time, update_time)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, app_name, user_id, session_id, state_json, current_time, current_time)

                merged_state = _merge_state(existing_app_state, existing_user_state, session_state)

        return Session(
            id=session_id,
            app_name=app_name,
            user_id=user_id,
            state=merged_state,
            events=[],
            last_update_time=current_time,
        )

    # =========================================================================
    # Session CRUD - Read
    # =========================================================================

    @override
    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        """Gets a session."""
        if self._is_postgres:
            return await self._get_session_pg(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                config=config,
            )
        else:
            return self._get_session_sqlite(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                config=config,
            )

    def _get_session_sqlite(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        """Get session from SQLite."""
        with self._get_sqlite_cursor() as cursor:
            cursor.execute("""
                SELECT session_id, state, update_time
                FROM sessions
                WHERE app_name = ? AND user_id = ? AND session_id = ?
            """, (app_name, user_id, session_id))

            row = cursor.fetchone()
            if not row:
                return None

            session_state = self._deserialize_json(row[1])
            update_time = row[2]

            # Get events with optional filtering
            query = """
                SELECT event_id, event_data, timestamp
                FROM events
                WHERE app_name = ? AND user_id = ? AND session_id = ?
            """
            params = [app_name, user_id, session_id]

            if config and config.after_timestamp:
                query += " AND timestamp >= ?"
                params.append(config.after_timestamp)

            query += " ORDER BY timestamp DESC"

            if config and config.num_recent_events:
                query += " LIMIT ?"
                params.append(config.num_recent_events)

            cursor.execute(query, params)
            event_rows = cursor.fetchall()

            # Convert events (reverse to get chronological order)
            events = []
            for event_row in reversed(event_rows):
                event_data = self._deserialize_json(event_row[1])
                events.append(Event.model_validate(event_data))

            # Get app and user states
            app_state = self._get_app_state_sqlite(cursor, app_name)
            user_state = self._get_user_state_sqlite(cursor, app_name, user_id)
            merged_state = _merge_state(app_state, user_state, session_state)

        return Session(
            id=session_id,
            app_name=app_name,
            user_id=user_id,
            state=merged_state,
            events=events,
            last_update_time=update_time,
        )

    async def _get_session_pg(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        """Get session from PostgreSQL."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT session_id, state, update_time
                FROM sessions
                WHERE app_name = $1 AND user_id = $2 AND session_id = $3
            """, app_name, user_id, session_id)

            if not row:
                return None

            session_state = self._deserialize_json(row["state"])
            update_time = row["update_time"]

            # Build events query with optional filtering
            query = """
                SELECT event_id, event_data, timestamp
                FROM events
                WHERE app_name = $1 AND user_id = $2 AND session_id = $3
            """
            params = [app_name, user_id, session_id]
            param_idx = 4

            if config and config.after_timestamp:
                query += f" AND timestamp >= ${param_idx}"
                params.append(config.after_timestamp)
                param_idx += 1

            query += " ORDER BY timestamp DESC"

            if config and config.num_recent_events:
                query += f" LIMIT ${param_idx}"
                params.append(config.num_recent_events)

            event_rows = await conn.fetch(query, *params)

            # Convert events (reverse to get chronological order)
            events = []
            for event_row in reversed(event_rows):
                event_data = self._deserialize_json(event_row["event_data"])
                events.append(Event.model_validate(event_data))

            # Get app and user states
            app_state = await self._get_app_state_pg(conn, app_name)
            user_state = await self._get_user_state_pg(conn, app_name, user_id)
            merged_state = _merge_state(app_state, user_state, session_state)

        return Session(
            id=session_id,
            app_name=app_name,
            user_id=user_id,
            state=merged_state,
            events=events,
            last_update_time=update_time,
        )

    # =========================================================================
    # Session CRUD - List
    # =========================================================================

    @override
    async def list_sessions(
        self, *, app_name: str, user_id: Optional[str] = None
    ) -> ListSessionsResponse:
        """Lists all the sessions for a user."""
        if self._is_postgres:
            return await self._list_sessions_pg(app_name=app_name, user_id=user_id)
        else:
            return self._list_sessions_sqlite(app_name=app_name, user_id=user_id)

    def _list_sessions_sqlite(
        self, *, app_name: str, user_id: Optional[str] = None
    ) -> ListSessionsResponse:
        """List sessions from SQLite."""
        with self._get_sqlite_cursor() as cursor:
            if user_id is not None:
                cursor.execute("""
                    SELECT session_id, user_id, state, update_time
                    FROM sessions
                    WHERE app_name = ? AND user_id = ?
                """, (app_name, user_id))
            else:
                cursor.execute("""
                    SELECT session_id, user_id, state, update_time
                    FROM sessions
                    WHERE app_name = ?
                """, (app_name,))

            rows = cursor.fetchall()

            # Get app state
            app_state = self._get_app_state_sqlite(cursor, app_name)

            # Get user states
            user_states_map = {}
            if user_id is not None:
                user_states_map[user_id] = self._get_user_state_sqlite(cursor, app_name, user_id)
            else:
                cursor.execute(
                    "SELECT user_id, state FROM user_state WHERE app_name = ?",
                    (app_name,)
                )
                for user_row in cursor.fetchall():
                    user_states_map[user_row[0]] = self._deserialize_json(user_row[1])

            sessions = []
            for row in rows:
                sess_id = row[0]
                sess_user_id = row[1]
                session_state = self._deserialize_json(row[2])
                update_time = row[3]

                user_state = user_states_map.get(sess_user_id, {})
                merged_state = _merge_state(app_state, user_state, session_state)

                sessions.append(Session(
                    id=sess_id,
                    app_name=app_name,
                    user_id=sess_user_id,
                    state=merged_state,
                    events=[],
                    last_update_time=update_time,
                ))

        return ListSessionsResponse(sessions=sessions)

    async def _list_sessions_pg(
        self, *, app_name: str, user_id: Optional[str] = None
    ) -> ListSessionsResponse:
        """List sessions from PostgreSQL."""
        async with self._pool.acquire() as conn:
            if user_id is not None:
                rows = await conn.fetch("""
                    SELECT session_id, user_id, state, update_time
                    FROM sessions
                    WHERE app_name = $1 AND user_id = $2
                """, app_name, user_id)
            else:
                rows = await conn.fetch("""
                    SELECT session_id, user_id, state, update_time
                    FROM sessions
                    WHERE app_name = $1
                """, app_name)

            # Get app state
            app_state = await self._get_app_state_pg(conn, app_name)

            # Get user states
            user_states_map = {}
            if user_id is not None:
                user_states_map[user_id] = await self._get_user_state_pg(conn, app_name, user_id)
            else:
                user_rows = await conn.fetch(
                    "SELECT user_id, state FROM user_state WHERE app_name = $1",
                    app_name
                )
                for user_row in user_rows:
                    user_states_map[user_row["user_id"]] = self._deserialize_json(user_row["state"])

            sessions = []
            for row in rows:
                sess_id = row["session_id"]
                sess_user_id = row["user_id"]
                session_state = self._deserialize_json(row["state"])
                update_time = row["update_time"]

                user_state = user_states_map.get(sess_user_id, {})
                merged_state = _merge_state(app_state, user_state, session_state)

                sessions.append(Session(
                    id=sess_id,
                    app_name=app_name,
                    user_id=sess_user_id,
                    state=merged_state,
                    events=[],
                    last_update_time=update_time,
                ))

        return ListSessionsResponse(sessions=sessions)

    # =========================================================================
    # Session CRUD - Delete
    # =========================================================================

    @override
    async def delete_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        """Deletes a session."""
        if self._is_postgres:
            await self._delete_session_pg(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        else:
            self._delete_session_sqlite(
                app_name=app_name, user_id=user_id, session_id=session_id
            )

    def _delete_session_sqlite(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        """Delete session from SQLite."""
        with self._get_sqlite_cursor() as cursor:
            cursor.execute("""
                DELETE FROM sessions
                WHERE app_name = ? AND user_id = ? AND session_id = ?
            """, (app_name, user_id, session_id))

    async def _delete_session_pg(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        """Delete session from PostgreSQL."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                DELETE FROM sessions
                WHERE app_name = $1 AND user_id = $2 AND session_id = $3
            """, app_name, user_id, session_id)

    # =========================================================================
    # Event operations
    # =========================================================================

    @override
    async def append_event(self, session: Session, event: Event) -> Event:
        """Appends an event to a session."""
        if event.partial:
            return event

        # Trim temp state
        event = self._trim_temp_delta_state(event)

        if self._is_postgres:
            return await self._append_event_pg(session, event)
        else:
            return self._append_event_sqlite(session, event)

    def _append_event_sqlite(self, session: Session, event: Event) -> Event:
        """Append event in SQLite."""
        with self._get_sqlite_cursor() as cursor:
            # Verify session exists and check staleness
            cursor.execute("""
                SELECT update_time FROM sessions
                WHERE app_name = ? AND user_id = ? AND session_id = ?
            """, (session.app_name, session.user_id, session.id))

            row = cursor.fetchone()
            if not row:
                logger.warning(f"Session {session.id} not found, cannot append event")
                return event

            stored_update_time = row[0]
            if stored_update_time > session.last_update_time:
                raise ValueError(
                    f"Session {session.id} has been updated since last read. "
                    f"Stored: {stored_update_time}, Session: {session.last_update_time}"
                )

            # Handle state deltas
            if event.actions and event.actions.state_delta:
                state_deltas = _extract_state_delta(event.actions.state_delta)
                app_state_delta = state_deltas["app"]
                user_state_delta = state_deltas["user"]
                session_state_delta = state_deltas["session"]

                if app_state_delta:
                    app_state = self._get_app_state_sqlite(cursor, session.app_name)
                    app_state.update(app_state_delta)
                    self._upsert_app_state_sqlite(cursor, session.app_name, app_state)

                if user_state_delta:
                    user_state = self._get_user_state_sqlite(cursor, session.app_name, session.user_id)
                    user_state.update(user_state_delta)
                    self._upsert_user_state_sqlite(cursor, session.app_name, session.user_id, user_state)

                if session_state_delta:
                    cursor.execute("""
                        SELECT state FROM sessions
                        WHERE app_name = ? AND user_id = ? AND session_id = ?
                    """, (session.app_name, session.user_id, session.id))
                    sess_row = cursor.fetchone()
                    session_state = self._deserialize_json(sess_row[0])
                    session_state.update(session_state_delta)

                    state_json = self._serialize_json(session_state)
                    cursor.execute("""
                        UPDATE sessions SET state = ?
                        WHERE app_name = ? AND user_id = ? AND session_id = ?
                    """, (state_json, session.app_name, session.user_id, session.id))

            # Store event
            event_data = self._serialize_json(event.model_dump(mode="json"))
            cursor.execute("""
                INSERT INTO events (app_name, user_id, session_id, event_id, event_data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session.app_name, session.user_id, session.id, event.id, event_data, event.timestamp))

            # Update session timestamp
            cursor.execute("""
                UPDATE sessions SET update_time = ?
                WHERE app_name = ? AND user_id = ? AND session_id = ?
            """, (event.timestamp, session.app_name, session.user_id, session.id))

        # Update in-memory session
        session.events.append(event)
        session.last_update_time = event.timestamp

        return event

    async def _append_event_pg(self, session: Session, event: Event) -> Event:
        """Append event in PostgreSQL."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Verify session exists and check staleness
                row = await conn.fetchrow("""
                    SELECT update_time FROM sessions
                    WHERE app_name = $1 AND user_id = $2 AND session_id = $3
                """, session.app_name, session.user_id, session.id)

                if not row:
                    logger.warning(f"Session {session.id} not found, cannot append event")
                    return event

                stored_update_time = row["update_time"]
                if stored_update_time > session.last_update_time:
                    raise ValueError(
                        f"Session {session.id} has been updated since last read. "
                        f"Stored: {stored_update_time}, Session: {session.last_update_time}"
                    )

                # Handle state deltas
                if event.actions and event.actions.state_delta:
                    state_deltas = _extract_state_delta(event.actions.state_delta)
                    app_state_delta = state_deltas["app"]
                    user_state_delta = state_deltas["user"]
                    session_state_delta = state_deltas["session"]

                    if app_state_delta:
                        app_state = await self._get_app_state_pg(conn, session.app_name)
                        app_state.update(app_state_delta)
                        await self._upsert_app_state_pg(conn, session.app_name, app_state)

                    if user_state_delta:
                        user_state = await self._get_user_state_pg(conn, session.app_name, session.user_id)
                        user_state.update(user_state_delta)
                        await self._upsert_user_state_pg(conn, session.app_name, session.user_id, user_state)

                    if session_state_delta:
                        sess_row = await conn.fetchrow("""
                            SELECT state FROM sessions
                            WHERE app_name = $1 AND user_id = $2 AND session_id = $3
                        """, session.app_name, session.user_id, session.id)
                        session_state = self._deserialize_json(sess_row["state"])
                        session_state.update(session_state_delta)

                        state_json = self._serialize_json(session_state)
                        await conn.execute("""
                            UPDATE sessions SET state = $1
                            WHERE app_name = $2 AND user_id = $3 AND session_id = $4
                        """, state_json, session.app_name, session.user_id, session.id)

                # Store event
                event_data = self._serialize_json(event.model_dump(mode="json"))
                await conn.execute("""
                    INSERT INTO events (app_name, user_id, session_id, event_id, event_data, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, session.app_name, session.user_id, session.id, event.id, event_data, event.timestamp)

                # Update session timestamp
                await conn.execute("""
                    UPDATE sessions SET update_time = $1
                    WHERE app_name = $2 AND user_id = $3 AND session_id = $4
                """, event.timestamp, session.app_name, session.user_id, session.id)

        # Update in-memory session
        session.events.append(event)
        session.last_update_time = event.timestamp

        return event
