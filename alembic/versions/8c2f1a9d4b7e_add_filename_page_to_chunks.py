"""add filename and page to chunks

Revision ID: 8c2f1a9d4b7e
Revises: 179a8d6e6f3e
Create Date: 2026-08-20 17:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8c2f1a9d4b7e"
down_revision: str | Sequence[str] | None = "179a8d6e6f3e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "chunks", sa.Column("filename", sa.String(length=255), nullable=True)
    )
    op.add_column("chunks", sa.Column("page", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("chunks", "page")
    op.drop_column("chunks", "filename")
