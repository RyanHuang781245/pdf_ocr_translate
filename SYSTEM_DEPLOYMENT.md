# 系統部署流程

這份文件整理 `pdf_ocr_translate` 的首次環境建立、正式部署、服務檢查與部署後驗證流程。

## 1. 系統組成

目前系統主要由以下部分組成：

- 執行環境：Python 3.10 - 3.11
- Web App：Flask + Gunicorn
- Web service：`uo_regulations_translate.service`
- Worker service：`uo_regulations_translate_worker.service`
- Log 清理排程：
  - `uo_regulations_translate_log_cleanup.service`
  - `uo_regulations_translate_log_cleanup.timer`
- 模板備份排程：
  - `uo_regulations_translate_template_backup.service`
  - `uo_regulations_translate_template_backup.timer`
- Database：MSSQL
- Reverse proxy：Nginx
- Migration：Alembic

重要路徑：

```text
/home/NE025/pdf_ocr_translate
├── .env
├── deploy.sh
├── alembic.ini
├── migrations/
├── deploy/
│   ├── nginx-site.conf.template
│   └── systemd/
├── scripts/
│   ├── init_sqlserver_schema.sql
│   ├── install_systemd_units.sh
│   ├── install_nginx_site.sh
│   ├── backup_templates.sh
│   └── restore_templates.sh
├── out/
│   ├── pdf_overlay/
│   ├── pdf_rebuild/
│   ├── word_overlay/
│   ├── templates/
│   └── uploads/
├── backups/
│   └── templates/
└── logs/
```

## 2. 環境設定

正式環境設定放在 `.env`。各參數用途請先看：

- [ENVIRONMENT.md](/home/NE025/pdf_ocr_translate/ENVIRONMENT.md)
- [setting_setting.md](/home/NE025/pdf_ocr_translate/setting_setting.md)

必要設定至少包含：

```env
APP_ENV=production
AUTO_SCHEMA_MANAGEMENT=0

DATABASE_SCHEMA=translation
DATABASE_URL='mssql+pyodbc://user:password@host/database?driver=ODBC Driver 18 for SQL Server&TrustServerCertificate=yes'

AUTH_ENABLED=1
AUTH_STUB_ENABLED=0
AUTHZ_MODE=ad_all_users
SECRET_KEY=your-secret

OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-azure-openai-endpoint
```

正式環境建議：

- `AUTO_SCHEMA_MANAGEMENT=0`：schema 由 Alembic / SQL script 管理，不由 app 啟動時自動建立。
- `AUTH_STUB_ENABLED=0`：正式環境不可使用 stub 登入。
- `.env` 不要提交到 Git。

## 3. 首次建立 uv 環境

專案使用 `uv` 建立 Python 虛擬環境。`pyproject.toml` 目前要求：

```text
requires-python = ">=3.10,<3.12"
```

若系統尚未安裝 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

重新載入 shell 後確認：

```bash
command -v uv
uv --version
```

部署時 `deploy.sh` 會執行：

```bash
uv sync --frozen
```

這會依照 `pyproject.toml` 與 `uv.lock` 建立或同步 `.venv`。部署後可確認：

```bash
cd /home/NE025/pdf_ocr_translate
.venv/bin/python --version
.venv/bin/python -c "import flask, sqlalchemy, pyodbc; print('python env ok')"
```

## 4. 部署前檢查

進入專案目錄：

```bash
cd /home/NE025/pdf_ocr_translate
```

確認必要檔案：

```bash
test -f .env && echo ".env OK"
test -f deploy.sh && echo "deploy.sh OK"
test -f alembic.ini && echo "alembic.ini OK"
```

確認工具：

```bash
command -v uv
command -v systemctl
command -v nginx
```

確認 ODBC driver：

```bash
odbcinst -q -d | grep -F "ODBC Driver 18 for SQL Server"
```

確認 `.env` 有資料庫設定：

```bash
grep -n "DATABASE_URL\|DATABASE_SCHEMA" .env
```

確認 Alembic 目前狀態：

```bash
source .env
.venv/bin/alembic current
.venv/bin/alembic heads
```

目前 Alembic version table 應在：

```text
[translation].[alembic_version]
```

## 5. 首次資料庫 Schema 建立

正式環境第一次部署前，若資料庫還沒有 schema/table，可先執行 SQL script：

```bash
sqlcmd -S <server> -d <database> -U <user> -P '<password>' -i scripts/init_sqlserver_schema.sql
```

之後再由 Alembic 補 migration：

```bash
source .env
.venv/bin/alembic upgrade head
```

部署後可驗證：

```bash
.venv/bin/flask --app app.py schema-preflight
```

## 6. 正式部署流程

一般部署：

```bash
cd /home/NE025/pdf_ocr_translate
bash deploy.sh
```

`deploy.sh` 會做：

1. 進入 `APP_ROOT`，預設 `/home/NE025/pdf_ocr_translate`
2. 載入 `.env`
3. 設定 `APP_ENV=production`
4. 設定 Alembic 使用 `DATABASE_URL`
5. 視設定執行 `git pull`
6. 執行 `uv sync --frozen`
7. 停止 Web / Worker service
8. 安裝或更新 systemd units
9. 視設定安裝或更新 Nginx site
10. 執行 `alembic upgrade head`
11. 執行 `schema-preflight`
12. 執行 `seed-bootstrap`
13. 重啟 Web / Worker / timer
14. 顯示 service 狀態

常用部署參數：

```bash
RUN_GIT_PULL=1 DEPLOY_BRANCH=main bash deploy.sh
```

指定部署目錄：

```bash
APP_ROOT=/home/NE025/pdf_ocr_translate bash deploy.sh
```

指定 systemd 使用者：

```bash
APP_USER=NE025 bash deploy.sh
```

只部署程式、不安裝 systemd units：

```bash
INSTALL_SYSTEMD_UNITS=0 bash deploy.sh
```

只安裝/啟動本次服務，不設定開機自動啟動：

```bash
ENABLE_SYSTEMD_UNITS=0 bash deploy.sh
```

如果目前環境不是 systemd，但仍要跑 migration 與 preflight：

```bash
MANAGE_SYSTEMD_SERVICES=0 bash deploy.sh
```

調整 Web worker 數：

```bash
WEB_WORKERS=2 bash deploy.sh
```

調整 Nginx listen port：

```bash
NGINX_LISTEN_PORT=81 bash deploy.sh
```

## 7. systemd 服務

部署會安裝以下 units：

```text
uo_regulations_translate.service
uo_regulations_translate_worker.service
uo_regulations_translate_log_cleanup.service
uo_regulations_translate_log_cleanup.timer
uo_regulations_translate_template_backup.service
uo_regulations_translate_template_backup.timer
```

檢查服務：

```bash
sudo systemctl status uo_regulations_translate --no-pager
sudo systemctl status uo_regulations_translate_worker --no-pager
sudo systemctl status uo_regulations_translate_log_cleanup.timer --no-pager
sudo systemctl status uo_regulations_translate_template_backup.timer --no-pager
```

查看 log：

```bash
journalctl -u uo_regulations_translate --no-pager -n 100
journalctl -u uo_regulations_translate_worker --no-pager -n 100
journalctl -u uo_regulations_translate_log_cleanup.service --no-pager -n 100
journalctl -u uo_regulations_translate_template_backup.service --no-pager -n 100
```

手動重啟：

```bash
sudo systemctl restart uo_regulations_translate
sudo systemctl restart uo_regulations_translate_worker
```

手動執行 log 清理：

```bash
sudo systemctl start uo_regulations_translate_log_cleanup.service
```

手動執行模板備份：

```bash
sudo systemctl start uo_regulations_translate_template_backup.service
```

檢查 timer 排程：

```bash
systemctl list-timers | grep uo_regulations_translate
```

## 8. Nginx

部署預設 `ENABLE_NGINX=1`，會依照：

```text
deploy/nginx-site.conf.template
```

產生並安裝 site config。預設：

- site name：`uo_regulations_translate`
- listen port：`81`
- proxy target：`unix:/home/NE025/pdf_ocr_translate/uo_regulations_translate.sock`

手動 render：

```bash
bash scripts/install_nginx_site.sh \
  --output-file /tmp/uo_regulations_translate \
  --app-root /home/NE025/pdf_ocr_translate \
  --listen-port 81
```

手動安裝：

```bash
sudo bash scripts/install_nginx_site.sh \
  --install \
  --app-root /home/NE025/pdf_ocr_translate \
  --site-name uo_regulations_translate \
  --listen-port 81
```

檢查 Nginx：

```bash
sudo nginx -t
ls -l /etc/nginx/sites-enabled
sudo systemctl reload nginx
```

## 9. 部署後驗證

Schema 檢查：

```bash
cd /home/NE025/pdf_ocr_translate
.venv/bin/flask --app app.py schema-preflight
```

初始化預設資料：

```bash
.venv/bin/flask --app app.py seed-bootstrap
```

確認 Alembic：

```bash
.venv/bin/alembic current
.venv/bin/alembic heads
```

確認服務：

```bash
sudo systemctl status uo_regulations_translate --no-pager
sudo systemctl status uo_regulations_translate_worker --no-pager
```

確認本機 HTTP：

```bash
curl -i http://127.0.0.1:81/ | sed -n '1,20p'
```

若已啟用登入，未登入時進入工作頁應導到登入頁：

```bash
curl -i http://127.0.0.1:81/workspace/pdf-overlay | sed -n '1,30p'
```

確認 timer：

```bash
systemctl list-timers | grep uo_regulations_translate
```

確認 app log：

```bash
ls -lh logs
tail -n 100 logs/app-web.log
```

## 10. 模板備份與還原

模板備份排程由：

```text
uo_regulations_translate_template_backup.timer
```

觸發：

```text
uo_regulations_translate_template_backup.service
```

備份內容：

```text
export/document_templates.json
out/templates/jobs/
```

手動備份：

```bash
cd /home/NE025/pdf_ocr_translate
APP_ROOT=/home/NE025/pdf_ocr_translate \
ENV_FILE=/home/NE025/pdf_ocr_translate/.env \
bash scripts/backup_templates.sh
```

手動還原：

```bash
cd /home/NE025/pdf_ocr_translate
APP_ROOT=/home/NE025/pdf_ocr_translate \
ENV_FILE=/home/NE025/pdf_ocr_translate/.env \
bash scripts/restore_templates.sh /path/to/templates_backup.tar.gz --yes
```

完整流程請看：

- [TEMPLATE_BACKUP_RESTORE.md](/home/NE025/pdf_ocr_translate/TEMPLATE_BACKUP_RESTORE.md)

## 11. Log 與資料清理

Log 參數在 `.env`：

```env
APP_LOG_DIR=/home/NE025/pdf_ocr_translate/logs
APP_LOG_LEVEL=INFO
APP_LOG_TO_FILE=1
APP_LOG_STDOUT=1
APP_LOG_MAX_MB=10
APP_LOG_BACKUP_COUNT=10
```

清理排程：

```env
CLEANUP_ON_CALENDAR="*-*-* 03:30:00"
AUDIT_LOG_RETENTION_DAYS=180
SYSTEM_ERROR_LOG_RETENTION_DAYS=180
```

手動清理：

```bash
.venv/bin/flask --app app.py audit-cleanup
.venv/bin/flask --app app.py system-error-cleanup
```

先看 dry-run：

```bash
.venv/bin/flask --app app.py audit-cleanup --dry-run
.venv/bin/flask --app app.py system-error-cleanup --dry-run
```

## 12. 測試注意事項

測試不能使用正式 schema。測試預設要求獨立 schema，例如：

```bash
TEST_DATABASE_SCHEMA=translation_test .venv/bin/pytest tests/test_app.py
```

測試 safety guard 會拒絕使用：

```text
dbo
translation
```

避免 pytest 清到正式 `[translation].[document_templates]`。

## 13. 常見問題

### `.venv/bin/python` 不存在

代表 `uv sync --frozen` 沒成功。先看部署輸出中的 `uv sync` 錯誤：

```bash
command -v uv
uv --version
uv sync --frozen
```

### `DATABASE_URL is empty after loading .env`

確認 `.env` 內有資料庫設定：

```bash
grep -n "DATABASE_URL" .env
```

### Alembic current 還是舊版

確認使用同一個 DB 與 schema：

```bash
source .env
echo "$DATABASE_SCHEMA"
.venv/bin/alembic current
.venv/bin/alembic heads
```

目前 version table 應是：

```text
[translation].[alembic_version]
```

### Nginx 連不到服務

檢查 socket 與服務：

```bash
sudo systemctl status uo_regulations_translate --no-pager
ls -l /home/NE025/pdf_ocr_translate/uo_regulations_translate.sock
sudo nginx -t
journalctl -u nginx --no-pager -n 100
```

### 502 Bad Gateway

通常是 Gunicorn 沒啟動、socket 權限不對或 Nginx proxy path 不一致：

```bash
sudo systemctl restart uo_regulations_translate
sudo systemctl status uo_regulations_translate --no-pager
sudo tail -n 100 /var/log/nginx/error.log
```

### 模板還原後顯示等待 OCR 完成

使用最新版 `scripts/restore_templates.sh` 還原。新版流程會先還原 `out/templates/jobs`，再匯入 `document_templates`，並重建缺失的 `template_source` job record。

可看還原輸出是否有：

```text
rebuilt_source_jobs=...
```

### `template_backup` 有備份但沒有資料

確認目前 DB 真的有模板資料：

```bash
.venv/bin/flask --app app.py template-backup --output /tmp/document_templates.json
python -m json.tool /tmp/document_templates.json | head -40
```

若 `template_count=0`，表示目前 `document_templates` 表是空的。
