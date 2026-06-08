from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from src.models.paper import Paper
from src.schemas.paper import PaperCreate


class PaperRepository:
    """
    All database operations for Paper.
    Like Spring's @Repository / JpaRepository.
    Never contains business logic — only DB reads and writes.
    """

    def __init__(self, session: Session):
        self.session = session  # injected — never created here

    def create(self, paper: PaperCreate) -> Paper:
        """Insert a new paper. Like save() in JpaRepository."""
        db_paper = Paper(**paper.model_dump())  # schema → model
        self.session.add(db_paper)
        self.session.commit()
        self.session.refresh(db_paper)  # reload from DB to get generated fields
        return db_paper

    def get_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:
        """Find by natural key. Like findByArxivId() in Spring Data."""
        return self.session.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()

    def get_by_id(self, paper_id: UUID) -> Optional[Paper]:
        """Find by primary key. Like findById() in JpaRepository."""
        return self.session.query(Paper).filter(Paper.id == paper_id).first()

    def get_all(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        """
        Paginated list, newest first.
        Like findAll(Pageable pageable) in JpaRepository.
        """
        return self.session.query(Paper).order_by(Paper.published_date.desc()).limit(limit).offset(offset).all()

    def update(self, paper: Paper) -> Paper:
        """Persist changes to an existing paper."""
        self.session.add(paper)
        self.session.commit()
        self.session.refresh(paper)
        return paper

    def upsert(self, paper_create: PaperCreate) -> Paper:
        """
        Insert if not exists, update if exists.
        No equivalent in basic JpaRepository — you'd write
        a custom @Query or use merge() in JPA.
        Critical for Week 2 — arXiv pipeline runs daily,
        same paper might come in multiple times.
        """
        existing = self.get_by_arxiv_id(paper_create.arxiv_id)
        if existing:
            # update all fields on the existing DB object
            for key, value in paper_create.model_dump(exclude_unset=True).items():
                setattr(existing, key, value)
            return self.update(existing)
        else:
            return self.create(paper_create)
