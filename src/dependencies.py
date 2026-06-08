from typing import Annotated, Generator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from src.db.interfaces.base import BaseDatabase


def get_database(request: Request) -> BaseDatabase:
    """
    Pull the database off app.state.
    app.state.database is set in lifespan() in main.py.
    Like Spring's ApplicationContext.getBean(DataSource.class)
    """
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    """
    Open a session, yield to route, close after.
    The yield makes this a context manager dependency.
    FastAPI handles open/close automatically per request.
    """
    with database.get_session() as session:
        yield session


# ── Type aliases ──────────────────────────────────────────────────────────────
# These make route signatures clean and readable.
# Instead of: db: Annotated[Session, Depends(get_db_session)]
# You write:   db: SessionDep

DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
