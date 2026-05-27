from __future__ import annotations

from typing import Any, Callable

from . import auth_store

AUTHZ_MODE_AD_ALL_USERS = "ad_all_users"
AUTHZ_MODE_LOCAL_ALLOWLIST = "local_allowlist"
AUTHZ_MODE_AD_GROUP_GATE = "ad_group_gate"
AUTHZ_MODE_HYBRID = "hybrid"

_SUPPORTED_AUTHZ_MODES = {
    AUTHZ_MODE_AD_ALL_USERS,
    AUTHZ_MODE_LOCAL_ALLOWLIST,
    AUTHZ_MODE_AD_GROUP_GATE,
    AUTHZ_MODE_HYBRID,
}


def get_authz_mode(config: Any) -> str:
    raw = " ".join(str(config.get("AUTHZ_MODE") or "").split()).strip().lower()
    if raw in _SUPPORTED_AUTHZ_MODES:
        return raw

    # Backward compatibility for phase-3 deployments that only set AUTH_REQUIRE_LOCAL_USER.
    if bool(config.get("AUTH_REQUIRE_LOCAL_USER", False)):
        return AUTHZ_MODE_LOCAL_ALLOWLIST
    return AUTHZ_MODE_AD_ALL_USERS


def authorize_login(
    config: Any,
    auth_user: Any,
    *,
    group_gate_checker: Callable[[Any], bool] | None = None,
) -> None:
    work_id = " ".join(str(getattr(auth_user, "work_id", "") or "").split()).strip()
    if not work_id:
        raise auth_store.LocalAuthorizationError("登入資料缺少工號。")

    mode = get_authz_mode(config)
    is_active = auth_store.is_user_active(work_id)
    if is_active is False:
        raise auth_store.LocalAuthorizationError("您的帳號已被停用。")

    if mode == AUTHZ_MODE_LOCAL_ALLOWLIST:
        if not auth_store.user_exists(work_id):
            raise auth_store.LocalAuthorizationError("您的帳號未獲得授權。")
        return

    if mode == AUTHZ_MODE_AD_GROUP_GATE:
        if group_gate_checker is None:
            raise auth_store.LocalAuthorizationError("尚未設定 AD 群組授權檢查。")
        if not group_gate_checker(auth_user):
            raise auth_store.LocalAuthorizationError("您的帳號未獲得授權。")
        return

    if mode in {AUTHZ_MODE_AD_ALL_USERS, AUTHZ_MODE_HYBRID}:
        return

    raise auth_store.LocalAuthorizationError(f"不支援的授權模式：{mode}")


def resolve_effective_roles(config: Any, auth_user: Any) -> tuple[str, ...]:
    mode = get_authz_mode(config)
    work_id = " ".join(str(getattr(auth_user, "work_id", "") or "").split()).strip()
    if not work_id:
        return (auth_store.ROLE_EDITOR,)

    if mode == AUTHZ_MODE_LOCAL_ALLOWLIST:
        snapshot = auth_store.get_local_user_snapshot(work_id)
        if snapshot is not None and snapshot.role_names:
            return snapshot.role_names
        return (auth_store.ROLE_EDITOR,)

    return auth_store.get_effective_role_names(work_id)
