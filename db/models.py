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
	id:Mapped[uuid.UUID]=mapped_column(
		UUID(as_uuid=True),
		primary_key=True,
		default=uuid.uuid4
	)
	github_repo_id:Mapped[int]=mapped_column(
		Integer,
		unique=True
	)
	full_name:Mapped[str]=mapped_column(
		Text
    )
	installations_id:Mapped[int]=mapped_column(
		Integer
    )
	created_at:Mapped[datetime]=mapped_column(
		server_default=func.now()
    )

class IndexedFile(Base):
	__tablename__ = "indexed_files"
	__table_args__ = (UniqueConstraint("repo_id", "file_path"),)

	id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	repo_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("repos.id"))
	file_path: Mapped[str] = mapped_column(Text)
	content_hash: Mapped[str | None] = mapped_column(Text, nullable=True)  # NULL = needs retry
	indexed_at: Mapped[datetime | None] = mapped_column(nullable=True)


class PRReview(Base):
	__tablename__ = "pr_reviews"

	id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	repo_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("repos.id"))
	pr_number: Mapped[int] = mapped_column(Integer)
	pr_title: Mapped[str | None] = mapped_column(Text, nullable=True)
	github_review_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
	status: Mapped[str] = mapped_column(Text, default="pending")  # pending | posted | failed
	raw_llm_output: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
	langfuse_trace_id: Mapped[str | None] = mapped_column(Text, nullable=True)
	created_at: Mapped[datetime] = mapped_column(server_default=func.now())

class ReviewFeedback(Base):
	__tablename__ = "review_feedback"

	id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	pr_review_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pr_reviews.id"))
	comment_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
	event: Mapped[str] = mapped_column(Text)  # dismissed | resolved | replied
	langfuse_score: Mapped[float | None] = mapped_column(Float, nullable=True)
	recorded_at: Mapped[datetime] = mapped_column(server_default=func.now())


