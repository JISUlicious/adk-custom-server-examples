"""
Shared asyncpg connection pool for PostgreSQL.

Provides a centralized connection pool manager that can be shared across
multiple services (session, memory, etc.) for efficient connection reuse.

Usage:
    from database_pool import AsyncPgPool

    # Get a shared pool (creates if not exists)
    pool = await AsyncPgPool.get_pool("postgresql://user:pass@host/db")

    # Use the pool
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM table WHERE id = $1", id)

    # On shutdown
    await AsyncPgPool.close_all()
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AsyncPgPool:
    """
    Shared asyncpg connection pool manager.

    Maintains a registry of pools by database URL, allowing multiple services
    to share the same pool for efficient connection reuse.

    This is a class with class methods, not meant to be instantiated directly.
    """

    _pools: dict[str, "asyncpg.Pool"] = {}

    @classmethod
    async def get_pool(
        cls,
        database_url: str,
        min_size: int = 2,
        max_size: int = 10,
    ) -> "asyncpg.Pool":
        """
        Get or create a connection pool for the given database URL.

        If a pool already exists for this URL, returns the existing pool.
        Otherwise, creates a new pool with the specified parameters.

        Args:
            database_url: PostgreSQL connection URL
            min_size: Minimum number of connections in the pool
            max_size: Maximum number of connections in the pool

        Returns:
            asyncpg.Pool instance
        """
        import asyncpg

        if database_url not in cls._pools:
            logger.info(
                "Creating asyncpg pool: min_size=%d, max_size=%d",
                min_size,
                max_size,
            )
            cls._pools[database_url] = await asyncpg.create_pool(
                database_url,
                min_size=min_size,
                max_size=max_size,
            )
            logger.info("asyncpg pool created successfully")

        return cls._pools[database_url]

    @classmethod
    async def close_pool(cls, database_url: str) -> None:
        """
        Close the pool for a specific database URL.

        Args:
            database_url: PostgreSQL connection URL
        """
        if database_url in cls._pools:
            pool = cls._pools.pop(database_url)
            await pool.close()
            logger.info("asyncpg pool closed for: %s", database_url[:50] + "...")

    @classmethod
    async def close_all(cls) -> None:
        """
        Close all connection pools.

        Call this on application shutdown to cleanly release all connections.
        """
        for url, pool in list(cls._pools.items()):
            await pool.close()
            logger.info("asyncpg pool closed")
        cls._pools.clear()

    @classmethod
    def has_pool(cls, database_url: str) -> bool:
        """
        Check if a pool exists for the given database URL.

        Args:
            database_url: PostgreSQL connection URL

        Returns:
            True if a pool exists, False otherwise
        """
        return database_url in cls._pools

    @classmethod
    def pool_count(cls) -> int:
        """
        Get the number of active pools.

        Returns:
            Number of pools currently managed
        """
        return len(cls._pools)
