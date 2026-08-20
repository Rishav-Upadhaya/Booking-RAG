"""drop score from chunks

Revision ID: 3f7b2e5a1c9d
Revises: 8c2f1a9d4b7e
Create Date: 2026-08-21 02:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3f7b2e5a1c9d"
down_revision: str | Sequence[str] | None = "8c2f1a9d4b7e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop the never-persisted retrieval score column."""
    op.drop_column("chunks", "score")


def downgrade() -> None:
    """Restore the column."""
    op.add_column("chunks", sa.Column("score", sa.Float(), nullable=True))