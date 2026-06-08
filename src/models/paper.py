import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID

from src.db.interfaces.postgresql import Base


class Paper(Base):
    """
    SQLAlchemy ORM model for academic papers.
    Like @Entity in JPA — maps directly to the 'papers' table.
    """

    __tablename__ = "papers"

    # Primary key — UUID generated in Python, not the DB
    # Like @GeneratedValue with a UUID strategy
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Unique identifier from arXiv — like a natural key
    arxiv_id = Column(String, unique=True, nullable=False, index=True)

    title = Column(String, nullable=False)

    # JSON column — stores a Python list ["Author 1", "Author 2"]
    # PostgreSQL has native JSON support — no join table needed
    authors = Column(JSON, nullable=False)

    abstract = Column(Text, nullable=False)  # Text = unlimited length String

    # Another JSON list ["cs.AI", "cs.LG"]
    categories = Column(JSON, nullable=False)

    published_date = Column(DateTime, nullable=False)

    pdf_url = Column(String, nullable=False)

    # Audit columns — auto-set, never touched by application code
    # Like @CreatedDate / @LastModifiedDate in Spring Data
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
