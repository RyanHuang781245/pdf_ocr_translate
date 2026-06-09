"""add audit and system error logs

Revision ID: 0002_audit_system_logs
Revises: 0001_baseline_schema
Create Date: 2026-06-09 00:00:00
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

from app.services import job_store, state

# revision identifiers, used by Alembic.
revision = "0002_audit_system_logs"
down_revision = "0001_baseline_schema"
branch_labels = None
depends_on = None


def _configure_schema(bind) -> None:
    if bind.dialect.name == "mssql":
        job_store.configure_database_schema(state.DATABASE_SCHEMA)
        job_store.ensure_database_schema(bind)
    else:
        job_store.configure_database_schema("dbo")


def _create_table_if_missing(bind, table) -> None:
    inspector = inspect(bind)
    schema = job_store.inspection_schema(bind)
    existing_tables = {name.lower() for name in inspector.get_table_names(schema=schema)}
    if table.name.lower() not in existing_tables:
        table.create(bind=bind, checkfirst=True)


def _create_missing_indexes(bind, table) -> None:
    inspector = inspect(bind)
    schema = job_store.inspection_schema(bind)
    existing_indexes = {index["name"].lower() for index in inspector.get_indexes(table.name, schema=schema)}
    for index in table.indexes:
        if index.name and index.name.lower() not in existing_indexes:
            index.create(bind=bind, checkfirst=True)


def upgrade() -> None:
    bind = op.get_bind()
    _configure_schema(bind)

    for table in (
        job_store.AuditLogRecord.__table__,
        job_store.SystemErrorLogRecord.__table__,
    ):
        _create_table_if_missing(bind, table)
        _create_missing_indexes(bind, table)


def downgrade() -> None:
    bind = op.get_bind()
    _configure_schema(bind)

    for table in (
        job_store.SystemErrorLogRecord.__table__,
        job_store.AuditLogRecord.__table__,
    ):
        table.drop(bind=bind, checkfirst=True)
