from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text

from app.config import BaseConfig, CONFIG_BY_NAME
from app.services import auth_store, job_store  # noqa: F401
from app.services.state import normalize_database_url

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = job_store.Base.metadata
ALEMBIC_VERSION_TABLE = "alembic_version"
ALEMBIC_VERSION_TABLE_SCHEMA = "translation"


def _quote_identifier(identifier: str) -> str:
    return "[" + str(identifier).replace("]", "]]") + "]"


def _resolve_config_class():
    config_name = (
        os.environ.get("ALEMBIC_CONFIG_NAME")
        or os.environ.get("APP_ENV")
        or os.environ.get("FLASK_ENV")
        or "default"
    )
    return CONFIG_BY_NAME.get(str(config_name).lower(), BaseConfig)


def _resolve_database_url() -> str:
    config_cls = _resolve_config_class()
    return str(
        normalize_database_url(
            os.environ.get("ALEMBIC_DATABASE_URL") or getattr(config_cls, "DATABASE_URL", "")
        )
    )


def _database_url() -> str:
    database_url = _resolve_database_url()
    if not database_url:
        raise RuntimeError("Database URL is required for Alembic. Set ALEMBIC_DATABASE_URL or DATABASE_URL.")
    config.set_main_option("sqlalchemy.url", database_url)
    return database_url


def _version_table_schema(*, use_mssql: bool) -> str | None:
    return ALEMBIC_VERSION_TABLE_SCHEMA if use_mssql else None


def _ensure_mssql_version_table(connection) -> None:
    schema_literal = ALEMBIC_VERSION_TABLE_SCHEMA.replace("'", "''")
    schema_identifier = _quote_identifier(ALEMBIC_VERSION_TABLE_SCHEMA)
    table_literal = ALEMBIC_VERSION_TABLE.replace("'", "''")
    table_identifier = _quote_identifier(ALEMBIC_VERSION_TABLE)
    constraint_identifier = _quote_identifier("pk_translation_alembic_version")
    qualified_name = f"{schema_identifier}.{table_identifier}"

    connection.execute(
        text(
            f"IF SCHEMA_ID(N'{schema_literal}') IS NULL "
            f"EXEC(N'CREATE SCHEMA {schema_identifier}');"
        )
    )
    connection.execute(
        text(
            f"IF OBJECT_ID(N'{schema_literal}.{table_literal}', N'U') IS NULL "
            "BEGIN "
            f"CREATE TABLE {qualified_name} ("
            "version_num NVARCHAR(255) NOT NULL, "
            f"CONSTRAINT {constraint_identifier} PRIMARY KEY (version_num)"
            "); "
            "END"
        )
    )


def run_migrations_offline() -> None:
    database_url = _database_url()
    use_mssql = database_url.lower().startswith("mssql")
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        version_table=ALEMBIC_VERSION_TABLE,
        version_table_schema=_version_table_schema(use_mssql=use_mssql),
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    _database_url()
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        use_mssql = connection.dialect.name == "mssql"
        if use_mssql:
            with connection.begin():
                _ensure_mssql_version_table(connection)

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            version_table=ALEMBIC_VERSION_TABLE,
            version_table_schema=_version_table_schema(use_mssql=use_mssql),
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
