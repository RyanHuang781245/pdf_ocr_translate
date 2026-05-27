from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, UniqueConstraint, inspect, select, text
from sqlalchemy.orm import Mapped, mapped_column

from . import job_store

ROLE_ADMIN = "admin"
ROLE_EDITOR = "editor"


class UserRecord(job_store.Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    work_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    email: Mapped[str | None] = mapped_column(String(200), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=job_store.utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RoleRecord(job_store.Base):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)


class UserRoleRecord(job_store.Base):
    __tablename__ = "user_roles"

    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role_id: Mapped[int] = mapped_column(Integer, ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True)

    __table_args__ = (UniqueConstraint("user_id", name="uq_user_roles_user_id"),)


@dataclass(frozen=True)
class LocalUserSnapshot:
    user_id: int
    work_id: str
    display_name: str
    email: str | None
    role_names: tuple[str, ...]
    is_active: bool


class LocalAuthorizationError(RuntimeError):
    pass


_AUTH_TABLES = (
    UserRecord.__table__,
    RoleRecord.__table__,
    UserRoleRecord.__table__,
)


def _engine():
    engine = getattr(job_store, '_engine', None)
    if engine is None:
        raise RuntimeError('Database engine not initialized.')
    return engine


def ensure_auth_schema() -> None:
    engine = _engine()
    job_store.Base.metadata.create_all(bind=engine, tables=_AUTH_TABLES, checkfirst=True)

    inspector = inspect(engine)
    existing_tables = {name.lower() for name in inspector.get_table_names()}
    if 'users' not in existing_tables:
        return
    existing_columns = {col['name'].lower() for col in inspector.get_columns('users')}
    dialect_name = engine.dialect.name
    with engine.begin() as conn:
        if 'display_name' not in existing_columns:
            conn.execute(text('ALTER TABLE users ADD display_name NVARCHAR(200) NULL;'))
        if 'email' not in existing_columns:
            conn.execute(text('ALTER TABLE users ADD email NVARCHAR(200) NULL;'))
        if 'last_login_at' not in existing_columns:
            conn.execute(text('ALTER TABLE users ADD last_login_at DATETIME2 NULL;'))
        if 'is_active' not in existing_columns:
            if dialect_name == 'mssql':
                conn.execute(text("ALTER TABLE users ADD is_active BIT NOT NULL CONSTRAINT DF_users_is_active DEFAULT(1);"))
            else:
                conn.execute(text('ALTER TABLE users ADD is_active BOOLEAN NOT NULL DEFAULT 1;'))


def seed_roles() -> None:
    with job_store.session_scope() as session:
        existing = {row for row in session.scalars(select(RoleRecord.name)).all()}
        for role_name in (ROLE_ADMIN, ROLE_EDITOR):
            if role_name not in existing:
                session.add(RoleRecord(name=role_name))


def _normalize_work_id(value: object) -> str:
    return ' '.join(str(value or '').split()).strip()


def get_role(name: str) -> RoleRecord | None:
    cleaned = _normalize_work_id(name)
    if not cleaned:
        return None
    with job_store.session_scope() as session:
        return session.scalar(select(RoleRecord).where(RoleRecord.name == cleaned))


def get_user_by_work_id(work_id: str) -> UserRecord | None:
    cleaned = _normalize_work_id(work_id)
    if not cleaned:
        return None
    with job_store.session_scope() as session:
        return session.scalar(select(UserRecord).where(UserRecord.work_id == cleaned))


def user_exists(work_id: str) -> bool:
    return get_user_by_work_id(work_id) is not None


def is_user_active(work_id: str) -> bool | None:
    snapshot = get_local_user_snapshot(work_id)
    if snapshot is None:
        return None
    return snapshot.is_active


def _get_or_create_user(session, *, work_id: str, display_name: str | None, email: str | None, active: bool = True) -> UserRecord:
    user = session.scalar(select(UserRecord).where(UserRecord.work_id == work_id))
    if user is None:
        user = UserRecord(
            work_id=work_id,
            display_name=display_name or work_id,
            email=email,
            is_active=active,
            created_at=job_store.utcnow(),
        )
        session.add(user)
        session.flush()
        return user
    if display_name and user.display_name != display_name:
        user.display_name = display_name
    if email and user.email != email:
        user.email = email
    return user


def upsert_local_user(
    *,
    work_id: str,
    display_name: str | None = None,
    email: str | None = None,
    active: bool | None = None,
) -> LocalUserSnapshot:
    cleaned_work_id = _normalize_work_id(work_id)
    if not cleaned_work_id:
        raise ValueError("work_id is required.")

    with job_store.session_scope() as session:
        user = _get_or_create_user(
            session,
            work_id=cleaned_work_id,
            display_name=display_name,
            email=email,
            active=True if active is None else bool(active),
        )
        if active is not None:
            user.is_active = bool(active)
        role_names = _get_role_names_for_user_id(session, user.id)
        return LocalUserSnapshot(
            user_id=int(user.id),
            work_id=str(user.work_id),
            display_name=str(user.display_name or user.work_id),
            email=str(user.email).strip() if user.email else None,
            role_names=role_names,
            is_active=bool(user.is_active),
        )


def _get_role_names_for_user_id(session, user_id: int) -> tuple[str, ...]:
    rows = session.execute(
        select(RoleRecord.name)
        .join(UserRoleRecord, RoleRecord.id == UserRoleRecord.role_id)
        .where(UserRoleRecord.user_id == user_id)
    ).all()
    return tuple(str(name).strip() for (name,) in rows if str(name).strip())


def get_role_names_for_user_id(user_id: int) -> tuple[str, ...]:
    with job_store.session_scope() as session:
        return _get_role_names_for_user_id(session, user_id)


def get_effective_role_names(work_id: str) -> tuple[str, ...]:
    snapshot = get_local_user_snapshot(work_id)
    if snapshot is None or not snapshot.role_names:
        return (ROLE_EDITOR,)
    return snapshot.role_names


def upsert_user_role_for_work_id(*, work_id: str, role_name: str, display_name: str | None = None, email: str | None = None, active: bool = True) -> None:
    cleaned_work_id = _normalize_work_id(work_id)
    cleaned_role = _normalize_work_id(role_name)
    if not cleaned_work_id or not cleaned_role:
        raise ValueError('work_id and role_name are required.')

    with job_store.session_scope() as session:
        role = session.scalar(select(RoleRecord).where(RoleRecord.name == cleaned_role))
        if role is None:
            role = RoleRecord(name=cleaned_role)
            session.add(role)
            session.flush()
        user = _get_or_create_user(
            session,
            work_id=cleaned_work_id,
            display_name=display_name,
            email=email,
            active=active,
        )
        existing = session.scalar(select(UserRoleRecord).where(UserRoleRecord.user_id == user.id))
        if existing is None:
            session.add(UserRoleRecord(user_id=user.id, role_id=role.id))
        else:
            existing.role_id = role.id
        user.is_active = bool(active)


def assign_role(work_id: str, role_name: str) -> None:
    upsert_user_role_for_work_id(work_id=work_id, role_name=role_name)


def bootstrap_admins(config: Any) -> None:
    raw = str(config.get('BOOTSTRAP_ADMIN') or '').strip()
    work_ids = [_normalize_work_id(item) for item in raw.split(',') if _normalize_work_id(item)]
    if not work_ids:
        return
    for work_id in work_ids:
        upsert_user_role_for_work_id(work_id=work_id, role_name=ROLE_ADMIN, display_name=work_id, active=True)


def bootstrap_auth_store(config: Any) -> None:
    ensure_auth_schema()
    seed_roles()
    bootstrap_admins(config)


def get_local_user_snapshot(work_id: str) -> LocalUserSnapshot | None:
    cleaned = _normalize_work_id(work_id)
    if not cleaned:
        return None
    with job_store.session_scope() as session:
        user = session.scalar(select(UserRecord).where(UserRecord.work_id == cleaned))
        if user is None:
            return None
        role_names = _get_role_names_for_user_id(session, user.id)
        return LocalUserSnapshot(
            user_id=int(user.id),
            work_id=str(user.work_id),
            display_name=str(user.display_name or user.work_id),
            email=str(user.email).strip() if user.email else None,
            role_names=role_names,
            is_active=bool(user.is_active),
        )


def touch_last_login(work_id: str) -> None:
    cleaned = _normalize_work_id(work_id)
    if not cleaned:
        return
    with job_store.session_scope() as session:
        user = session.scalar(select(UserRecord).where(UserRecord.work_id == cleaned))
        if user is not None:
            user.last_login_at = datetime.now(timezone.utc)


def sync_authenticated_user(auth_user: Any, *, active: bool | None = None) -> LocalUserSnapshot:
    work_id = _normalize_work_id(getattr(auth_user, "work_id", ""))
    if not work_id:
        raise LocalAuthorizationError("登入資料缺少工號。")
    display_name = _normalize_work_id(getattr(auth_user, "display_name", "")) or work_id
    email = _normalize_work_id(getattr(auth_user, "email", "")) or None
    snapshot = upsert_local_user(
        work_id=work_id,
        display_name=display_name,
        email=email,
        active=active,
    )
    touch_last_login(work_id)
    return get_local_user_snapshot(work_id) or snapshot
