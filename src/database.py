from contextlib import contextmanager

from src.db.factory import make_database

_database = None  # module-level singleton


def get_database():
    """Get or create the global database instance."""
    global _database
    if _database is None:
        _database = make_database()
    return _database


@contextmanager
def get_db_session():
    """Convenience context manager for scripts and notebooks."""
    database = get_database()
    with database.get_session() as session:
        yield session
