"""
Database-backed Memory Service for Google ADK

This module provides a persistent memory service that stores session events
in a database instead of in-memory storage. Supports SQLite (default) and
PostgreSQL (using asyncpg).

Features:
- Persistent storage across restarts
- Full-text search (FTS5 for SQLite, tsvector for PostgreSQL)
- Thread-safe operations
- asyncpg for PostgreSQL connections (async, high-performance)

Requirements:
- SQLite: Built-in (no additional dependencies)
- PostgreSQL: pip install asyncpg

Usage:
    # SQLite
    memory_service = DatabaseMemoryService("sqlite:///memory.db")

    # PostgreSQL (requires calling start() before use)
    memory_service = DatabaseMemoryService("postgresql://user:pass@host/db")
    await memory_service.start()  # Initialize connection pool

    # On shutdown
    await memory_service.close()
"""

from __future__ import annotations

import json
import re
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional
from contextlib import contextmanager

from typing_extensions import override

from google.genai import types
from google.adk.memory.base_memory_service import BaseMemoryService, SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.memory import _utils

if TYPE_CHECKING:
    from google.adk.sessions.session import Session

logger = logging.getLogger(__name__)


# ============================================================================
# Database Models (using raw SQL for minimal dependencies)
# ============================================================================

SQLITE_SCHEMA = """
-- Sessions table
CREATE TABLE IF NOT EXISTS memory_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(app_name, user_id, session_id)
);

-- Events table
CREATE TABLE IF NOT EXISTS memory_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_pk INTEGER NOT NULL,
    event_id TEXT,
    author TEXT,
    timestamp REAL,
    content_json TEXT NOT NULL,
    content_text TEXT,  -- Extracted text for FTS
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_pk) REFERENCES memory_sessions(id) ON DELETE CASCADE
);

-- Full-text search virtual table (SQLite FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS memory_events_fts USING fts5(
    content_text,
    content='memory_events',
    content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS memory_events_ai AFTER INSERT ON memory_events BEGIN
    INSERT INTO memory_events_fts(rowid, content_text) VALUES (new.id, new.content_text);
END;

CREATE TRIGGER IF NOT EXISTS memory_events_ad AFTER DELETE ON memory_events BEGIN
    INSERT INTO memory_events_fts(memory_events_fts, rowid, content_text) VALUES('delete', old.id, old.content_text);
END;

CREATE TRIGGER IF NOT EXISTS memory_events_au AFTER UPDATE ON memory_events BEGIN
    INSERT INTO memory_events_fts(memory_events_fts, rowid, content_text) VALUES('delete', old.id, old.content_text);
    INSERT INTO memory_events_fts(rowid, content_text) VALUES (new.id, new.content_text);
END;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_sessions_user ON memory_sessions(app_name, user_id);
CREATE INDEX IF NOT EXISTS idx_events_session ON memory_events(session_pk);
"""

POSTGRESQL_SCHEMA = """
-- Sessions table
CREATE TABLE IF NOT EXISTS memory_sessions (
    id SERIAL PRIMARY KEY,
    app_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(app_name, user_id, session_id)
);

-- Events table with tsvector for full-text search
CREATE TABLE IF NOT EXISTS memory_events (
    id SERIAL PRIMARY KEY,
    session_pk INTEGER NOT NULL REFERENCES memory_sessions(id) ON DELETE CASCADE,
    event_id TEXT,
    author TEXT,
    timestamp DOUBLE PRECISION,
    content_json JSONB NOT NULL,
    content_text TEXT,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', COALESCE(content_text, ''))) STORED,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_sessions_user ON memory_sessions(app_name, user_id);
CREATE INDEX IF NOT EXISTS idx_events_session ON memory_events(session_pk);
CREATE INDEX IF NOT EXISTS idx_events_tsv ON memory_events USING GIN(content_tsv);
"""


# ============================================================================
# Helper Functions
# ============================================================================

def extract_text_from_content(content: types.Content) -> str:
    """Extract all text from a Content object."""
    if not content or not content.parts:
        return ""

    texts = []
    for part in content.parts:
        if hasattr(part, 'text') and part.text:
            texts.append(part.text)

    return " ".join(texts)


def content_to_json(content: types.Content) -> str:
    """Serialize Content object to JSON."""
    if not content:
        return "{}"

    # Convert to dict format that can be restored
    data = {
        "role": content.role,
        "parts": []
    }

    if content.parts:
        for part in content.parts:
            part_dict = {}
            if hasattr(part, 'text') and part.text:
                part_dict["text"] = part.text
            if hasattr(part, 'function_call') and part.function_call:
                part_dict["functionCall"] = {
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args) if part.function_call.args else {}
                }
            if hasattr(part, 'function_response') and part.function_response:
                part_dict["functionResponse"] = {
                    "name": part.function_response.name,
                    "response": part.function_response.response
                }
            if part_dict:
                data["parts"].append(part_dict)

    return json.dumps(data)


def json_to_content(json_str: str) -> types.Content:
    """Deserialize JSON to Content object."""
    if isinstance(json_str, dict):
        data = json_str  # asyncpg returns JSONB as dict directly
    else:
        data = json.loads(json_str)

    parts = []
    for part_dict in data.get("parts", []):
        if "text" in part_dict:
            parts.append(types.Part.from_text(text=part_dict["text"]))
        elif "functionCall" in part_dict:
            fc = part_dict["functionCall"]
            parts.append(types.Part(
                function_call=types.FunctionCall(
                    name=fc.get("name", ""),
                    args=fc.get("args", {})
                )
            ))
        elif "functionResponse" in part_dict:
            fr = part_dict["functionResponse"]
            parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=fr.get("name", ""),
                    response=fr.get("response", {})
                )
            ))

    return types.Content(
        role=data.get("role", "user"),
        parts=parts
    )


# ============================================================================
# Database Memory Service
# ============================================================================

class DatabaseMemoryService(BaseMemoryService):
    """A database-backed memory service for production use.

    Supports SQLite (default) and PostgreSQL. Uses full-text search for
    efficient memory retrieval.

    Usage:
        # SQLite (default, creates file if not exists)
        memory_service = DatabaseMemoryService(
            database_url="sqlite:///memory.db"
        )

        # PostgreSQL (requires start() before use)
        memory_service = DatabaseMemoryService(
            database_url="postgresql://user:pass@localhost/dbname"
        )
        await memory_service.start()

        # In-memory SQLite (for testing)
        memory_service = DatabaseMemoryService(
            database_url="sqlite:///:memory:"
        )
    """

    def __init__(
        self,
        database_url: str = "sqlite:///adk_memory.db",
        max_results: int = 100,
    ):
        """
        Initialize the database memory service.

        Args:
            database_url: Database connection URL.
                - SQLite: "sqlite:///path/to/db.db" or "sqlite:///:memory:"
                - PostgreSQL: "postgresql://user:pass@host:port/dbname"
            max_results: Maximum number of results to return from search.
        """
        self.database_url = database_url
        self.max_results = max_results
        self._is_postgresql = database_url.startswith("postgresql")
        self._is_memory = ":memory:" in database_url
        self._persistent_connection = None  # For in-memory SQLite databases

        # For PostgreSQL, use shared pool from database_pool module
        self._pool = None  # asyncpg.Pool, set in start()
        self._started = False

        # Initialize SQLite immediately (sync), PostgreSQL requires start()
        if not self._is_postgresql:
            self._init_sqlite()

    async def start(self) -> None:
        """Initialize PostgreSQL connection pool and schema.

        Must be called before using the service with PostgreSQL.
        For SQLite, this is a no-op as initialization happens in __init__.
        """
        if self._started:
            return

        if self._is_postgresql:
            from database_pool import AsyncPgPool
            self._pool = await AsyncPgPool.get_pool(self.database_url)
            await self._init_postgres()

        self._started = True
        logger.info("DatabaseMemoryService started: %s", self.database_url[:50])

    async def close(self) -> None:
        """Close database connections.

        For SQLite in-memory, closes the persistent connection.
        For PostgreSQL, the pool is managed by AsyncPgPool.close_all().
        """
        if self._persistent_connection:
            self._persistent_connection.close()
            self._persistent_connection = None

        self._started = False
        logger.info("DatabaseMemoryService closed")

    def _get_connection(self):
        """Get SQLite database connection (sync)."""
        import sqlite3
        # For in-memory databases, reuse the same connection
        if self._is_memory:
            if self._persistent_connection is None:
                self._persistent_connection = sqlite3.connect(
                    ":memory:", check_same_thread=False
                )
                self._persistent_connection.row_factory = sqlite3.Row
            return self._persistent_connection

        # Extract path from sqlite:///path
        db_path = self.database_url.replace("sqlite:///", "")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _get_cursor(self):
        """Context manager for SQLite database cursor."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            # Don't close persistent connections (in-memory databases)
            if not self._is_memory and not self._is_postgresql:
                conn.close()

    def _init_sqlite(self):
        """Initialize SQLite database schema."""
        with self._get_cursor() as cursor:
            # SQLite needs to execute statements one by one
            for statement in SQLITE_SCHEMA.split(";"):
                statement = statement.strip()
                if statement:
                    try:
                        cursor.execute(statement)
                    except Exception as e:
                        # Ignore errors for "IF NOT EXISTS" statements
                        if "already exists" not in str(e).lower():
                            logger.debug(f"Schema statement skipped: {e}")

        logger.info(f"SQLite database initialized: {self.database_url}")

    async def _init_postgres(self):
        """Initialize PostgreSQL database schema."""
        async with self._pool.acquire() as conn:
            await conn.execute(POSTGRESQL_SCHEMA)

        logger.info(f"PostgreSQL database initialized: {self.database_url[:50]}...")

    @override
    async def add_session_to_memory(self, session: Session):
        """Add a session's events to memory storage.

        Args:
            session: The session to add to memory.
        """
        if not session.events:
            return

        if self._is_postgresql:
            await self._add_session_to_memory_pg(session)
        else:
            self._add_session_to_memory_sqlite(session)

        logger.debug(
            f"Added session to memory: app={session.app_name}, "
            f"user={session.user_id}, session={session.id}, "
            f"events={len(session.events)}"
        )

    def _add_session_to_memory_sqlite(self, session: Session):
        """Add session to SQLite database."""
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO memory_sessions (app_name, user_id, session_id, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (app_name, user_id, session_id)
                DO UPDATE SET updated_at = excluded.updated_at
            """, (session.app_name, session.user_id, session.id, datetime.now(timezone.utc).isoformat()))

            cursor.execute("""
                SELECT id FROM memory_sessions
                WHERE app_name = ? AND user_id = ? AND session_id = ?
            """, (session.app_name, session.user_id, session.id))
            session_pk = cursor.fetchone()[0]

            cursor.execute("DELETE FROM memory_events WHERE session_pk = ?", (session_pk,))

            for event in session.events:
                if not event.content or not event.content.parts:
                    continue

                content_json = content_to_json(event.content)
                content_text = extract_text_from_content(event.content)

                if not content_text:
                    continue

                cursor.execute("""
                    INSERT INTO memory_events
                    (session_pk, event_id, author, timestamp, content_json, content_text)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_pk,
                    event.id,
                    event.author,
                    event.timestamp,
                    content_json,
                    content_text
                ))

    async def _add_session_to_memory_pg(self, session: Session):
        """Add session to PostgreSQL database."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow("""
                    INSERT INTO memory_sessions (app_name, user_id, session_id, updated_at)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (app_name, user_id, session_id)
                    DO UPDATE SET updated_at = EXCLUDED.updated_at
                    RETURNING id
                """, session.app_name, session.user_id, session.id, datetime.now(timezone.utc))
                session_pk = row['id']

                await conn.execute("DELETE FROM memory_events WHERE session_pk = $1", session_pk)

                for event in session.events:
                    if not event.content or not event.content.parts:
                        continue

                    content_json = content_to_json(event.content)
                    content_text = extract_text_from_content(event.content)

                    if not content_text:
                        continue

                    await conn.execute("""
                        INSERT INTO memory_events
                        (session_pk, event_id, author, timestamp, content_json, content_text)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """, session_pk, event.id, event.author, event.timestamp, content_json, content_text)

    @override
    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Search memories using full-text search.

        Args:
            app_name: The application name.
            user_id: The user ID.
            query: The search query.

        Returns:
            SearchMemoryResponse with matching memories.
        """
        if not query.strip():
            return SearchMemoryResponse()

        if self._is_postgresql:
            response = await self._search_memory_pg(app_name, user_id, query)
        else:
            response = self._search_memory_sqlite(app_name, user_id, query)

        logger.debug(
            f"Memory search: app={app_name}, user={user_id}, "
            f"query='{query}', results={len(response.memories)}"
        )

        return response

    def _search_memory_sqlite(
        self, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        """Search memories in SQLite using FTS5."""
        response = SearchMemoryResponse()

        with self._get_cursor() as cursor:
            cursor.execute("""
                SELECT e.content_json, e.author, e.timestamp
                FROM memory_events e
                JOIN memory_sessions s ON e.session_pk = s.id
                JOIN memory_events_fts fts ON e.id = fts.rowid
                WHERE s.app_name = ? AND s.user_id = ?
                AND memory_events_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (app_name, user_id, query, self.max_results))

            rows = cursor.fetchall()

            for row in rows:
                try:
                    content = json_to_content(row['content_json'])
                    response.memories.append(
                        MemoryEntry(
                            content=content,
                            author=row['author'],
                            timestamp=_utils.format_timestamp(row['timestamp']) if row['timestamp'] else None,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error deserializing memory entry: {e}")
                    continue

        return response

    async def _search_memory_pg(
        self, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        """Search memories in PostgreSQL using tsvector."""
        response = SearchMemoryResponse()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT e.content_json, e.author, e.timestamp
                FROM memory_events e
                JOIN memory_sessions s ON e.session_pk = s.id
                WHERE s.app_name = $1 AND s.user_id = $2
                AND e.content_tsv @@ plainto_tsquery('english', $3)
                ORDER BY ts_rank(e.content_tsv, plainto_tsquery('english', $3)) DESC
                LIMIT $4
            """, app_name, user_id, query, self.max_results)

            for row in rows:
                try:
                    content = json_to_content(row['content_json'])
                    response.memories.append(
                        MemoryEntry(
                            content=content,
                            author=row['author'],
                            timestamp=_utils.format_timestamp(row['timestamp']) if row['timestamp'] else None,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error deserializing memory entry: {e}")
                    continue

        return response

    async def search_memory_keyword(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Search memories using simple keyword matching (fallback).

        This method provides compatibility with the InMemoryMemoryService
        behavior when FTS is not available or desired.

        Args:
            app_name: The application name.
            user_id: The user ID.
            query: The search query.

        Returns:
            SearchMemoryResponse with matching memories.
        """
        if not query.strip():
            return SearchMemoryResponse()

        # Extract keywords from query
        query_words = set(word.lower() for word in re.findall(r'[A-Za-z]+', query))

        if not query_words:
            return SearchMemoryResponse()

        if self._is_postgresql:
            return await self._search_memory_keyword_pg(app_name, user_id, query_words)
        else:
            return self._search_memory_keyword_sqlite(app_name, user_id, query_words)

    def _search_memory_keyword_sqlite(
        self, app_name: str, user_id: str, query_words: set
    ) -> SearchMemoryResponse:
        """Keyword search in SQLite."""
        response = SearchMemoryResponse()

        with self._get_cursor() as cursor:
            cursor.execute("""
                SELECT e.content_json, e.author, e.timestamp, e.content_text
                FROM memory_events e
                JOIN memory_sessions s ON e.session_pk = s.id
                WHERE s.app_name = ? AND s.user_id = ?
                ORDER BY e.timestamp DESC
                LIMIT ?
            """, (app_name, user_id, self.max_results * 10))

            for row in cursor.fetchall():
                if not row['content_text']:
                    continue

                event_words = set(word.lower() for word in re.findall(r'[A-Za-z]+', row['content_text']))

                if any(qw in event_words for qw in query_words):
                    try:
                        response.memories.append(
                            MemoryEntry(
                                content=json_to_content(row['content_json']),
                                author=row['author'],
                                timestamp=_utils.format_timestamp(row['timestamp']) if row['timestamp'] else None,
                            )
                        )
                        if len(response.memories) >= self.max_results:
                            break
                    except Exception as e:
                        logger.warning(f"Error deserializing memory entry: {e}")

        return response

    async def _search_memory_keyword_pg(
        self, app_name: str, user_id: str, query_words: set
    ) -> SearchMemoryResponse:
        """Keyword search in PostgreSQL."""
        response = SearchMemoryResponse()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT e.content_json, e.author, e.timestamp, e.content_text
                FROM memory_events e
                JOIN memory_sessions s ON e.session_pk = s.id
                WHERE s.app_name = $1 AND s.user_id = $2
                ORDER BY e.timestamp DESC
                LIMIT $3
            """, app_name, user_id, self.max_results * 10)

            for row in rows:
                if not row['content_text']:
                    continue

                event_words = set(word.lower() for word in re.findall(r'[A-Za-z]+', row['content_text']))

                if any(qw in event_words for qw in query_words):
                    try:
                        response.memories.append(
                            MemoryEntry(
                                content=json_to_content(row['content_json']),
                                author=row['author'],
                                timestamp=_utils.format_timestamp(row['timestamp']) if row['timestamp'] else None,
                            )
                        )
                        if len(response.memories) >= self.max_results:
                            break
                    except Exception as e:
                        logger.warning(f"Error deserializing memory entry: {e}")

        return response

    def clear_memory(self, app_name: Optional[str] = None, user_id: Optional[str] = None):
        """Clear memories from the database (sync, SQLite only).

        For PostgreSQL, use clear_memory_async().

        Args:
            app_name: If provided, only clear memories for this app.
            user_id: If provided, only clear memories for this user.
                     Requires app_name to also be provided.
        """
        if self._is_postgresql:
            raise RuntimeError("Use clear_memory_async() for PostgreSQL")

        self._clear_memory_sqlite(app_name, user_id)
        logger.info(f"Cleared memory: app={app_name}, user={user_id}")

    async def clear_memory_async(self, app_name: Optional[str] = None, user_id: Optional[str] = None):
        """Clear memories from the database (async).

        Args:
            app_name: If provided, only clear memories for this app.
            user_id: If provided, only clear memories for this user.
                     Requires app_name to also be provided.
        """
        if self._is_postgresql:
            await self._clear_memory_pg(app_name, user_id)
        else:
            self._clear_memory_sqlite(app_name, user_id)

        logger.info(f"Cleared memory: app={app_name}, user={user_id}")

    def _clear_memory_sqlite(self, app_name: Optional[str], user_id: Optional[str]):
        """Clear memories from SQLite."""
        with self._get_cursor() as cursor:
            if app_name and user_id:
                cursor.execute(
                    "DELETE FROM memory_sessions WHERE app_name = ? AND user_id = ?",
                    (app_name, user_id)
                )
            elif app_name:
                cursor.execute(
                    "DELETE FROM memory_sessions WHERE app_name = ?",
                    (app_name,)
                )
            else:
                cursor.execute("DELETE FROM memory_sessions")

    async def _clear_memory_pg(self, app_name: Optional[str], user_id: Optional[str]):
        """Clear memories from PostgreSQL."""
        async with self._pool.acquire() as conn:
            if app_name and user_id:
                await conn.execute(
                    "DELETE FROM memory_sessions WHERE app_name = $1 AND user_id = $2",
                    app_name, user_id
                )
            elif app_name:
                await conn.execute(
                    "DELETE FROM memory_sessions WHERE app_name = $1",
                    app_name
                )
            else:
                await conn.execute("DELETE FROM memory_sessions")

    def get_stats(self) -> dict:
        """Get statistics about stored memories (sync, SQLite only).

        For PostgreSQL, use get_stats_async().

        Returns:
            Dictionary with memory statistics.
        """
        if self._is_postgresql:
            raise RuntimeError("Use get_stats_async() for PostgreSQL")

        return self._get_stats_sqlite()

    async def get_stats_async(self) -> dict:
        """Get statistics about stored memories (async).

        Returns:
            Dictionary with memory statistics.
        """
        if self._is_postgresql:
            return await self._get_stats_pg()
        else:
            return self._get_stats_sqlite()

    def _get_stats_sqlite(self) -> dict:
        """Get stats from SQLite."""
        with self._get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM memory_sessions")
            session_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM memory_events")
            event_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT app_name) FROM memory_sessions")
            app_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM memory_sessions")
            user_count = cursor.fetchone()[0]

        return {
            "sessions": session_count,
            "events": event_count,
            "apps": app_count,
            "users": user_count,
            "database_url": self.database_url,
        }

    async def _get_stats_pg(self) -> dict:
        """Get stats from PostgreSQL."""
        async with self._pool.acquire() as conn:
            session_count = await conn.fetchval("SELECT COUNT(*) FROM memory_sessions")
            event_count = await conn.fetchval("SELECT COUNT(*) FROM memory_events")
            app_count = await conn.fetchval("SELECT COUNT(DISTINCT app_name) FROM memory_sessions")
            user_count = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM memory_sessions")

        return {
            "sessions": session_count,
            "events": event_count,
            "apps": app_count,
            "users": user_count,
            "database_url": self.database_url,
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def test_memory_service():
        # Create in-memory SQLite database for testing
        service = DatabaseMemoryService(database_url="sqlite:///:memory:")

        print("=== Database Memory Service Test ===")
        print(f"Stats: {service.get_stats()}")

        # Note: Full testing requires actual Session objects from ADK
        print("\nTo use with ADK:")
        print("  from database_memory_service import DatabaseMemoryService")
        print("  memory_service = DatabaseMemoryService('sqlite:///memory.db')")
        print("  server = OpenAIAdkWebServer(memory_service=memory_service, ...)")

    asyncio.run(test_memory_service())
