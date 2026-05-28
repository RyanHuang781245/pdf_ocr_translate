# Authentication Settings

這份文件整理目前 `pdf_ocr_translate` 的登入與授權設定方式。

目前建議正式環境優先採用 `ad_all_users`。其他模式保留作為相容或後續擴充，不建議作為目前部署的主路徑。

程式入口可對照：
- [app/services/state.py](/home/NE025/pdf_ocr_translate/app/services/state.py:162)
- [app/config.py](/home/NE025/pdf_ocr_translate/app/config.py:50)
- [app/services/auth_policy.py](/home/NE025/pdf_ocr_translate/app/services/auth_policy.py:7)
- [app/services/auth_service.py](/home/NE025/pdf_ocr_translate/app/services/auth_service.py:162)

## 基本開關

先分成兩層：

1. 是否啟用登入系統
2. 是否使用真 LDAP 驗證

最小設定：

```env
AUTH_ENABLED=1
SECRET_KEY=your-secret
```

說明：
- `AUTH_ENABLED=1`：啟用登入保護
- `AUTH_ENABLED=0`：關閉登入保護，沿用匿名模式
- `SECRET_KEY`：Flask session 必填
- `SESSION_COOKIE_NAME`：同一 host 上若還有其他 Flask 站台，建議使用專屬 cookie 名稱，避免互相覆蓋

建議同 host 多站台時一併設定：

```env
SESSION_COOKIE_NAME=pdf_ocr_translate_session
```

## Owner 權限開關

這個設定用來控制 job / template 是否要依 `owner_work_id` 限制存取。

```env
OWNER_ACCESS_ENABLED=1
```

說明：
- `OWNER_ACCESS_ENABLED=1`：啟用 owner 權限封鎖
  - 一般使用者只能看到自己的 job / template
  - 一般使用者不能打開、下載、修改別人的 job
- `OWNER_ACCESS_ENABLED=0`：暫時關閉 owner 權限封鎖
  - 列表、下載、編輯、API 會先不做 owner 限制
  - 適合舊資料還沒補 `owner_work_id` 的過渡期

建議：
- 這是相容舊資料用的過渡開關
- 舊資料補完 `owner_work_id` 後，應改回 `OWNER_ACCESS_ENABLED=1`

## Stub 與 LDAP

### Stub 模式

用來驗證登入頁、session、導頁與未登入攔截。

```env
AUTH_ENABLED=1
AUTH_STUB_ENABLED=1
AUTHZ_MODE=ad_all_users
SECRET_KEY=dev-secret
```

行為：
- 不連 LDAP
- 不驗密碼
- 輸入工號或使用者名稱即可登入
- 適合本機測試，不適合正式環境

### LDAP 模式

正式模式請改成：

```env
AUTH_ENABLED=1
AUTH_STUB_ENABLED=0
SECRET_KEY=your-secret

LDAP_HOST=your-ldap-host
LDAP_BASE_DN=dc=example,dc=com
LDAP_BIND_DN=cn=svc_account,ou=service,dc=example,dc=com
LDAP_BIND_PASSWORD=your-bind-password
```

常用可選設定：

```env
LDAP_PORT=389
LDAP_USE_SSL=0
LDAP_USER_LOGIN_ATTR=sAMAccountName
LDAP_USER_OBJECT_FILTER=(&(objectClass=user)(!(objectClass=computer)))
LDAP_USER_DISPLAY_ATTR=displayName
LDAP_USER_EMAIL_ATTR=mail
LDAP_USER_SEARCH_SCOPE=SUBTREE
```

說明：
- `LDAP_HOST`：LDAP 或 AD 主機
- `LDAP_BASE_DN`：搜尋使用者的 base DN
- `LDAP_BIND_DN` / `LDAP_BIND_PASSWORD`：服務帳號，用來先搜尋使用者 DN
- `LDAP_USER_LOGIN_ATTR`：登入識別欄位，預設 `sAMAccountName`
- `LDAP_USER_DISPLAY_ATTR`：顯示名稱欄位
- `LDAP_USER_EMAIL_ATTR`：電子郵件欄位

## 授權模式

`AUTHZ_MODE` 決定「LDAP 驗證成功後，是否允許登入系統」。

目前支援：
- `ad_all_users`
- `local_allowlist`
- `ad_group_gate`
- `hybrid`

---

## 1. ad_all_users

用途：所有有效 AD 使用者都可以登入。

設定：

```env
AUTH_ENABLED=1
AUTH_STUB_ENABLED=0
AUTHZ_MODE=ad_all_users
SECRET_KEY=your-secret

LDAP_HOST=...
LDAP_BASE_DN=...
LDAP_BIND_DN=...
LDAP_BIND_PASSWORD=...
```

行為：
- LDAP 帳密正確即可登入
- 第一次登入會自動同步到本機 `users`
- 若本機沒有角色，預設為 `editor`
- 若本機 `users.is_active=0`，仍會被擋下
- `admin` 需要靠本機角色設定

適用情境：
- 內部 AD 使用者都可使用系統
- 只想另外控管少數管理員權限
- 目前建議的預設模式

---

## 2. local_allowlist

用途：只有本機授權名單中的使用者可以登入。

設定：

```env
AUTH_ENABLED=1
AUTH_STUB_ENABLED=0
AUTHZ_MODE=local_allowlist
SECRET_KEY=your-secret

LDAP_HOST=...
LDAP_BASE_DN=...
LDAP_BIND_DN=...
LDAP_BIND_PASSWORD=...
```

行為：
- LDAP 帳密正確後，還要存在本機 `users`
- 若不在授權名單，會顯示 `您的帳號未獲得授權。`
- 若在本機但沒有角色，預設為 `editor`
- 若本機 `users.is_active=0`，會顯示 `您的帳號已被停用。`

適用情境：
- 系統只開放特定人員
- 想用白名單精準控管登入資格

---

## 3. ad_group_gate

用途：LDAP 驗證成功後，再依 AD 群組決定是否可登入。

設定：

```env
AUTH_ENABLED=1
AUTH_STUB_ENABLED=0
AUTHZ_MODE=ad_group_gate
SECRET_KEY=your-secret

LDAP_HOST=...
LDAP_BASE_DN=...
LDAP_BIND_DN=...
LDAP_BIND_PASSWORD=...

LDAP_GROUP_GATE_ENABLED=1
ALLOWED_GROUP_DN=cn=your-group,ou=groups,dc=example,dc=com
```

目前狀態：
- 設定鍵已保留
- 群組授權檢查流程尚未正式實作完成
- 目前如果啟用這個模式，會回報 `尚未實作 AD 群組授權檢查。`

適用情境：
- 未來要做部門或 AD 群組控管時使用

---

## 4. hybrid

用途：AD 使用者可登入，但系統內功能權限仍主要靠本機角色控制。

設定：

```env
AUTH_ENABLED=1
AUTH_STUB_ENABLED=0
AUTHZ_MODE=hybrid
SECRET_KEY=your-secret

LDAP_HOST=...
LDAP_BASE_DN=...
LDAP_BIND_DN=...
LDAP_BIND_PASSWORD=...
```

目前行為：
- 登入層面接近 `ad_all_users`
- 差別主要在設計意圖，方便未來擴充更細的功能權限

適用情境：
- 想先開放 AD 登入
- 後續還會加上模組級或資源級權限限制

## BOOTSTRAP_ADMIN

用途：建立第一批本機管理員。

設定：

```env
BOOTSTRAP_ADMIN=u123456,u234567
```

行為：
- app 啟動時，會確保這些工號在本機具有 `admin` 角色
- 適合初始化第一批管理員
- 不適合當日常授權機制

建議：
- 第一批管理員建立完成後，將此設定縮到最小，或視需要移除

## 舊設定相容

目前仍保留：

```env
AUTH_REQUIRE_LOCAL_USER=0
```

用途：
- 只作為相容舊版設定的 fallback
- 若未設定 `AUTHZ_MODE`，則：
  - `AUTH_REQUIRE_LOCAL_USER=1` 視為 `local_allowlist`
  - `AUTH_REQUIRE_LOCAL_USER=0` 視為 `ad_all_users`

建議：
- 新設定請優先使用 `AUTHZ_MODE`
- 不要再把 `AUTH_REQUIRE_LOCAL_USER` 當主要控制開關
- 若未特別需求，請讓它維持 `0`

## 推薦配置

### 推薦 1：所有 AD 使用者都能登入

```env
AUTH_ENABLED=1
AUTH_STUB_ENABLED=0
AUTHZ_MODE=ad_all_users
SECRET_KEY=your-secret

LDAP_HOST=...
LDAP_BASE_DN=...
LDAP_BIND_DN=...
LDAP_BIND_PASSWORD=...

BOOTSTRAP_ADMIN=你的工號
```

### 推薦 2：只有白名單可登入

```env
AUTH_ENABLED=1
AUTH_STUB_ENABLED=0
AUTHZ_MODE=local_allowlist
SECRET_KEY=your-secret

LDAP_HOST=...
LDAP_BASE_DN=...
LDAP_BIND_DN=...
LDAP_BIND_PASSWORD=...
```
