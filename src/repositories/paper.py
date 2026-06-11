from typing import List, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.models.paper import Paper
from src.schemas.arxiv.paper import PaperCreate


class PaperRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, paper: PaperCreate) -> Paper:
        db_paper = Paper(**paper.model_dump())
        self.session.add(db_paper)
        self.session.commit()
        self.session.refresh(db_paper)
        return db_paper

    def get_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:
        stmt = select(Paper).where(Paper.arxiv_id == arxiv_id)
        return self.session.scalar(stmt)

    def get_by_id(self, paper_id: UUID) -> Optional[Paper]:
        stmt = select(Paper).where(Paper.id == paper_id)
        return self.session.scalar(stmt)

    def get_all(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        stmt = select(Paper).order_by(Paper.published_date.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def get_count(self) -> int:
        stmt = select(func.count(Paper.id))
        return self.session.scalar(stmt) or 0

    def get_processed_papers(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        """Papers with PDF content successfully parsed."""
        stmt = (
            select(Paper)
            .where(Paper.pdf_processed == True)
            .order_by(Paper.pdf_processing_date.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.scalars(stmt))

    def get_unprocessed_papers(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        """Papers stored without PDF content — need processing."""
        stmt = select(Paper).where(Paper.pdf_processed == False).order_by(Paper.published_date.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def get_processing_stats(self) -> dict:
        """
        Dashboard stats for monitoring the pipeline.
        Like a @Query aggregation in Spring Data.
        Called by the daily report task in Airflow.
        """
        total = self.get_count()
        processed_stmt = select(func.count(Paper.id)).where(Paper.pdf_processed == True)
        processed = self.session.scalar(processed_stmt) or 0
        text_stmt = select(func.count(Paper.id)).where(Paper.raw_text != None)
        with_text = self.session.scalar(text_stmt) or 0

        return {
            "total_papers": total,
            "processed_papers": processed,
            "papers_with_text": with_text,
            "processing_rate": (processed / total * 100) if total > 0 else 0,
            "text_extraction_rate": (with_text / processed * 100) if processed > 0 else 0,
        }

    def update(self, paper: Paper) -> Paper:
        self.session.add(paper)
        self.session.commit()
        self.session.refresh(paper)
        return paper

    def upsert(self, paper_create: PaperCreate) -> Paper:
        """
        Insert if new, update if exists.
        Critical for daily pipeline — same paper can appear multiple times.
        """
        existing = self.get_by_arxiv_id(paper_create.arxiv_id)
        if existing:
            for key, value in paper_create.model_dump(exclude_unset=True).items():
                setattr(existing, key, value)
            return self.update(existing)
        return self.create(paper_create)
