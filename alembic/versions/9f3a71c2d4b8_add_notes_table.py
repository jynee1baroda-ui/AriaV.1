"""add notes table

Revision ID: 9f3a71c2d4b8
Revises: d50e172ecb6b
Create Date: 2026-07-30 22:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9f3a71c2d4b8'
down_revision: Union[str, Sequence[str], None] = 'd50e172ecb6b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('notes',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('session_id', sa.String(length=64), nullable=True),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_notes_created_at', 'notes', ['created_at'], unique=False)
    op.create_index('ix_notes_session_id_id', 'notes', ['session_id', 'id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_notes_session_id_id', table_name='notes')
    op.drop_index('ix_notes_created_at', table_name='notes')
    op.drop_table('notes')
