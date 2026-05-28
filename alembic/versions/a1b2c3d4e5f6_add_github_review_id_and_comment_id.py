"""add github_review_id and comment_id

Revision ID: a1b2c3d4e5f6
Revises: bf2430722724
Create Date: 2026-05-28 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'bf2430722724'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('pr_reviews', sa.Column('github_review_id', sa.BigInteger(), nullable=True))
    op.add_column('review_feedback', sa.Column('comment_id', sa.BigInteger(), nullable=True))


def downgrade() -> None:
    op.drop_column('review_feedback', 'comment_id')
    op.drop_column('pr_reviews', 'github_review_id')
