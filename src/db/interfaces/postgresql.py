import logging
from contextlib import contextmanager
from typing import Generator, Optional

from pydantic import Field
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from src.db.interfaces.base import BaseDatabase

logger = logging.getLogger(__name__)


class PostgreSQLSettings(BaseSettings):
    """PostgreSQL-specific config. Reads POSTGRES_* env vars."""

    database_url: str = Field(
        default="postgresql://rag_user:rag_password@localhost:5432/rag_db",
        description="PostgreSQL database URL",
    )
    echo_sql: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=20, description="Connection pool size")
    max_overflow: int = Field(default=0, description="Maximum pool overflow")

    class Config:
        env_prefix = "POSTGRES_"  # reads POSTGRES_DATABASE_URL, POSTGRES_ECHO_SQL etc.


# This is SQLAlchemy's base class for all ORM models
# Like JPA's @Entity base — all models inherit from this
Base = declarative_base()


class PostgreSQLDatabase(BaseDatabase):
    """Concrete PostgreSQL implementation of BaseDatabase."""

    def __init__(self, config: PostgreSQLSettings):
        self.config = config
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None

    def startup(self) -> None:
        """Initialize the database connection and create tables."""
        try:
            # Hide password from logs
            safe_url = self.config.database_url.split("@")[1] if "@" in self.config.database_url else "localhost"
            logger.info(f"Connecting to PostgreSQL at: {safe_url}")

            self.engine = create_engine(
                self.config.database_url,
                echo=self.config.echo_sql,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,  # test connection before using from pool
            )

            self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

            # Test the connection — like a smoke test
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.info("Database connection test successful")

            # Create tables that don't exist yet — idempotent, safe to run every startup
            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()
            Base.metadata.create_all(bind=self.engine)
            updated_tables = inspector.get_table_names()

            new_tables = set(updated_tables) - set(existing_tables)
            if new_tables:
                logger.info(f"Created new tables: {', '.join(new_tables)}")
            else:
                logger.info("All tables already exist")

            logger.info("PostgreSQL initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise

    def teardown(self) -> None:
        """Close all connections in the pool."""
        if self.engine:
            self.engine.dispose()
            logger.info("PostgreSQL connections closed")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a DB session as a context manager.
        Like Spring's @Transactional — auto rollback on error.

        Usage:
            with database.get_session() as session:
                session.query(Paper).all()
        """
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call startup() first.")

        session = self.session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
