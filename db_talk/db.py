"""
Minimalistic database connector
"""

# pylint: disable=global-statement
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ContextManager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, scoped_session, sessionmaker


class NotInitializedError(RuntimeError):
    """Raised when the database component is not initialized."""

    def __init__(self):
        super().__init__(
            "Database component is not initialized. Call init_db(db_url, **kwargs) first."
        )


@dataclass
class DbConfig:
    """
    Database component.
    """

    db_url: str = field()
    """
    Database URL.

    Documentation:
      https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls

    Examples:
      - SQLite (file-based): sqlite:///./test.db
      - SQLite (in-memory): sqlite:///:memory:
      - PostgreSQL: postgresql+psycopg2://user:password@localhost/dbname
      - MySQL: mysql+pymysql://user:password@localhost/dbname
    """
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    """
    Additional keyword arguments for SQLAlchemy engine creation.
    Documentation:
      https://docs.sqlalchemy.org/en/20/core/engines.html#sqlalchemy.create_engine
    """
    session_kwargs: dict[str, Any] = field(default_factory=dict)
    """
    Additional keyword arguments for SQLAlchemy session creation.
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
    Database component for LM Proxy.
    """

    config: DbConfig = field()
    engine: Engine = field(init=False)
    session_factory: scoped_session[Session] = field(init=False)

    def __post_init__(self):
        self.engine = create_engine(self.config.db_url, **self.config.engine_kwargs)
        self.session_factory = scoped_session(
            sessionmaker(bind=self.engine, **self.config.session_kwargs)
        )

    def get_unmanaged_session(self) -> Session:
        """
        Returns database session.
        Unmanaged; caller is responsible for closing it.
        """
        return self.session_factory()

    @contextmanager
    def session(self) -> ContextManager[Session]:
        """Managed session with automatic commit/rollback."""
        s = self.get_unmanaged_session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def health_check(self) -> bool:
        """Check if database connection is alive."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as e:
            logging.error("Database health check failed: %s", e)
            return False

    def dispose(self):
        """Dispose the engine and close all sessions."""
        self.session_factory.remove()
        self.engine.dispose()


_db: DbComponent | None = None
_db_lock = threading.Lock()


def init_db(
    db_url: str,
    engine_kwargs: dict[str, Any] | None = None,
    session_kwargs: dict[str, Any] | None = None,
) -> DbComponent:
    """
    Initializes the database component.
    """
    global _db
    with _db_lock:
        if _db is not None:
            raise RuntimeError("Database component is already initialized.")
        logging.info("Initializing database connection...")
        config = DbConfig(
            db_url=db_url,
            engine_kwargs=engine_kwargs or {},
            session_kwargs=session_kwargs or {},
        )
        _db = DbComponent(config=config)
        logging.info("Database connection initialized.")
        return _db


def db() -> DbComponent:
    """
    Database component facade.
    Returns the initialized database component.
    Raises NotInitializedError if not initialized.
    """
    with _db_lock:
        if _db is None:
            raise NotInitializedError()
        return _db


def is_initialized() -> bool:
    """Check if database component is initialized."""
    with _db_lock:
        return _db is not None


def db_session() -> ContextManager[Session]:
    return db().session()


def dispose_db():
    """
    Disposes the database component.
    """
    global _db
    with _db_lock:
        if _db is not None:
            logging.info("Disposing database connection...")
            _db.dispose()
            _db = None
