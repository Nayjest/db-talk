"""
Minimalistic async database connector
"""

# pylint: disable=global-statement
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncContextManager

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_scoped_session,
    async_sessionmaker,
    create_async_engine,
)


class NotInitializedError(RuntimeError):
    """Raised when the database component is not initialized."""

    def __init__(self):
        super().__init__(
            "Database component is not initialized. Call init_db(db_url, **kwargs) first."
        )


@dataclass
class DbConfig:
    """
    Database component configuration.
    """

    db_url: str = field()
    """
    Async database URL.

    Documentation:
      https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html

    Examples:
      - SQLite (file-based): sqlite+aiosqlite:///./test.db
      - SQLite (in-memory): sqlite+aiosqlite:///:memory:
      - PostgreSQL: postgresql+asyncpg://user:password@localhost/dbname
      - MySQL: mysql+aiomysql://user:password@localhost/dbname
    """
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    """
    Additional keyword arguments for SQLAlchemy async engine creation.
    Documentation:
      https://docs.sqlalchemy.org/en/20/core/engines.html#sqlalchemy.create_engine
    """
    session_kwargs: dict[str, Any] = field(default_factory=dict)
    """
    Additional keyword arguments for SQLAlchemy async session creation.
    Documentation:
      https://docs.sqlalchemy.org/en/20/orm/session_api.html#sqlalchemy.orm.sessionmaker
    """

    def __post_init__(self):
        self.engine_kwargs.setdefault("pool_pre_ping", True)
        self.session_kwargs.setdefault("expire_on_commit", False)
        self.session_kwargs.setdefault("autoflush", True)


@dataclass
class DbComponent:
    """
    Async database component.
    """

    config: DbConfig = field()
    engine: AsyncEngine = field(init=False)
    session_factory: async_scoped_session[AsyncSession] = field(init=False)

    def __post_init__(self):
        self.engine = create_async_engine(
            self.config.db_url, **self.config.engine_kwargs
        )
        self.session_factory = async_scoped_session(
            async_sessionmaker(
                bind=self.engine, class_=AsyncSession, **self.config.session_kwargs
            ),
            scopefunc=asyncio.current_task,
        )

    def get_unmanaged_session(self) -> AsyncSession:
        """
        Returns async database session.
        Unmanaged; caller is responsible for closing it.
        """
        return self.session_factory()

    @asynccontextmanager
    async def session(self) -> AsyncContextManager[AsyncSession]:
        """Managed async session with automatic commit/rollback."""
        s = self.get_unmanaged_session()
        try:
            yield s
            await s.commit()
        except Exception:
            await s.rollback()
            raise
        finally:
            await s.close()

    async def health_check(self) -> bool:
        """Check if database connection is alive."""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as e:
            logging.error("Database health check failed: %s", e)
            return False

    async def dispose(self):
        """Dispose the engine and close all sessions."""
        await self.session_factory.close()
        await self.engine.dispose()


_db: DbComponent | None = None
_db_lock = threading.Lock()


def init_db(
    db_url: str,
    engine_kwargs: dict[str, Any] | None = None,
    session_kwargs: dict[str, Any] | None = None,
) -> DbComponent:
    """
    Initializes the async database component.
    """
    global _db
    with _db_lock:
        if _db is not None:
            raise RuntimeError("Database component is already initialized.")
        logging.info("Initializing async database connection...")
        config = DbConfig(
            db_url=db_url,
            engine_kwargs=engine_kwargs or {},
            session_kwargs=session_kwargs or {},
        )
        _db = DbComponent(config=config)
        logging.info("Async database connection initialized.")
        return _db


def db() -> DbComponent:
    """
    Database component facade.
    Returns the initialized database component.
    Raises NotInitializedError if not initialized.
    """
    if _db is None:
        raise NotInitializedError()
    return _db


def is_initialized() -> bool:
    """Check if database component is initialized."""
    return _db is not None


def db_session() -> AsyncContextManager[AsyncSession]:
    """
    Returns an async context manager for database sessions.
    Note: This is synchronous but returns an async context manager.
    """
    if _db is None:
        raise NotInitializedError()
    return _db.session()


async def dispose_db():
    """
    Disposes the async database component.
    """
    global _db
    async with _db_lock:
        if _db is not None:
            logging.info("Disposing async database connection...")
            await _db.dispose()
            _db = None
