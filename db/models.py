from sqlalchemy.orm import DeclarativeBase
import uuid
from datetime import datetime
from sqlalchemy import Text, Integer, Float, BigInteger, ForeignKey, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
	pass

class Repo(Base):
	__tablename__="repos"

	id: Mapped[uuid.UUID]=mapped_column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4)
	github_repo_id: Mapped[int]=mapped_column(Integer,unique=True)
	full_name:Mapped[str]=mapped_column(Text)
	installation_id:Mapped[int]=mapped_column(Integer)
	created_at:Mapped[datetime]=mapped_column(server_default=func.now())

class IndexedFile(Base):
	__tablename__ = "indexed_files"
	__table_args__ = (UniqueConstraint("repo_id", "file_path"),)

	id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	repo_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("repos.id"))
	file_path: Mapped[str] = mapped_column(Text)
	content_hash: Mapped[str | None] = mapped_column(Text, nullable=True)  # NULL = failed
	status: Mapped[str] = mapped_column(Text, default="indexed")  # indexed|failed|failed_permanent|deleted
	retry_count: Mapped[int] = mapped_column(Integer, default=0)
	chunk_count: Mapped[int] = mapped_column(Integer, default=0)
	indexed_at: Mapped[datetime | None] = mapped_column(nullable=True)


class PRReview(Base):
	__tablename__ = "pr_reviews"

	id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	repo_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("repos.id"))
	pr_number: Mapped[int] = mapped_column(Integer)
	status: Mapped[str] = mapped_column(Text, default="pending")  # pending|posted|failed
	langfuse_trace_id: Mapped[str | None] = mapped_column(Text, nullable=True)
	latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
	raw_output: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
	created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ReviewFeedback(Base):
	__tablename__ = "review_feedback"

	id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	review_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pr_reviews.id"))
	action: Mapped[str] = mapped_column(Text)   # "+1" or "-1"
	value: Mapped[float] = mapped_column(Float)  # 1.0 or 0.0
	timestamp: Mapped[datetime] = mapped_column(server_default=func.now())


