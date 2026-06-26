# 系統部署說明

說明翻譯系統於 Ubuntu 上的部署架構、部署流程、systemd 服務、Nginx 反向代理、部署驗證、維護作業與常見故障排除。系統環境需求、第三方軟體、`.env` 與路徑設定請參考 [ENVIRONMENT.md](./ENVIRONMENT.md)。

## 1. 部署架構概述

系統採用 Nginx、Gunicorn、Flask App、背景 Worker 與 MSSQL 分層部署。使用者透過瀏覽器連線至 Nginx，Nginx 提供靜態檔案並將動態請求透過 Unix Socket 轉發至 Gunicorn。Gunicorn 載入 `wsgi.py` 中的 Flask application object；背景 Worker 由 `worker.py` 常駐執行，負責 OCR、PDF 翻譯、文件重建與 Word 翻譯等非同步工作。

```text
使用者瀏覽器
    |
    | HTTP Request
    v
Nginx
    |-- /static/ -> static/
    |
    | reverse proxy
    v
Gunicorn Unix Socket
    |
    | WSGI
    v
Flask App
    |-- SQLAlchemy / pyodbc -> MSSQL
    |-- OpenAI / Azure OpenAI -> 翻譯模型
    |-- Triton / PP-Structure -> OCR 或版面服務
    |-- out/ -> 上傳、任務、輸出、模板與翻譯記憶
    |-- logs/ -> 應用程式 log

systemd
    |-- uo_regulations_translate.service -> Web
    |-- uo_regulations_translate_worker.service -> 背景任務
    |-- uo_regulations_translate_log_cleanup.timer -> log / audit 清理
    |-- uo_regulations_translate_template_backup.timer -> 模板備份
```

## 2. 專案目錄結構

專案主要目錄與檔案用途如下：

- `app.py`：Flask 開發入口，可供本機開發或簡易測試使用。
- `wsgi.py`：Gunicorn 使用的 WSGI 入口，匯入 `create_app()` 並建立 `app` 物件。
- `worker.py`：背景任務 Worker 入口，處理資料庫中的待執行工作。
- `app/`：Flask 應用程式主要程式碼，包含 blueprints、services、templates、models 與系統設定。
- `app/config.py`：應用程式設定檔，負責讀取環境變數並設定資料庫、認證、模型、路徑與 log。
- `app/services/state.py`：集中定義路徑、模型、認證、worker 併發與預設設定。
- `ocr_pipeline/`：OCR 合併、段落對齊與下層文件處理邏輯。
- `static/`：前端靜態資源目錄，由 Nginx 直接提供。
- `migrations/`：Alembic migration 檔案，用於管理資料庫 schema 版本。
- `scripts/init_sqlserver_schema.sql`：首次建立 MSSQL schema/table 的 SQL script。
- `pyproject.toml`：Python 專案設定與依賴套件宣告。
- `uv.lock`：鎖定 Python 套件版本，部署時用於建立可重現的虛擬環境。
- `.venv/`：Python 虛擬環境目錄，部署時由 `uv sync --frozen` 建立或更新。
- `.env`：部署環境變數檔案。變數內容請參考 [ENVIRONMENT.md](./ENVIRONMENT.md)。
- `deploy.sh`：正式部署腳本，負責同步虛擬環境、安裝 systemd unit、執行 migration、schema 檢查、預設資料初始化、重啟服務與選擇性安裝 Nginx 設定。
- `deploy/systemd/*.service.template`：systemd service 範本，部署時會產生實際 unit 檔。
- `deploy/systemd/*.timer.template`：systemd timer 範本，用於排程 log 清理與模板備份。
- `deploy/nginx-site.conf.template`：Nginx site 設定範本。
- `build/systemd/`：部署腳本產生的 systemd unit 檔輸出目錄。
- `build/nginx/`：部署腳本產生的 Nginx site 設定輸出目錄。
- `out/`：系統執行資料，包含上傳檔、任務資料、輸出檔、模板來源檔與翻譯記憶。
- `backups/templates/`：模板備份檔預設輸出目錄。
- `logs/`：應用程式 log 輸出目錄。

## 3. 後端服務部署流程

部署前請先完成 [ENVIRONMENT.md](./ENVIRONMENT.md) 所列作業系統、第三方軟體、`.env`、資料庫與路徑權限設定。

確認專案與 `.env`：

```bash
cd /home/NE025/pdf_ocr_translate
test -f .env && echo OK
```

同步 Python 虛擬環境：

```bash
uv sync --frozen
```

確認 Python 環境：

```bash
.venv/bin/python --version
.venv/bin/python -c "import flask, sqlalchemy, pyodbc; print('python env ok')"
```

確認 ODBC Driver：

```bash
odbcinst -q -d | grep -F "ODBC Driver 18 for SQL Server"
```

在啟動 systemd 服務前，可先以 Flask 或 Gunicorn 進行基本啟動測試：

```bash
.venv/bin/flask --app app.py --debug run --port 5001
```

或使用 Gunicorn 測試 WSGI 入口：

```bash
.venv/bin/gunicorn --workers 2 --worker-class gthread --threads 4 --timeout 300 --bind unix:uo_regulations_translate.sock wsgi:app
```

正式部署：

```bash
bash deploy.sh
```

若部署目錄不是預設路徑，應明確指定：

```bash
APP_ROOT=/opt/pdf_ocr_translate ENV_FILE=/opt/pdf_ocr_translate/.env bash deploy.sh
```

若要同步更新程式碼：

```bash
RUN_GIT_PULL=1 DEPLOY_BRANCH=main bash deploy.sh
```

若要調整 Web worker 數量或 Nginx listen port：

```bash
WEB_WORKERS=2 NGINX_LISTEN_PORT=81 bash deploy.sh
```

若目前環境不是 systemd，但仍要同步依賴、執行 migration 與 schema 檢查：

```bash
MANAGE_SYSTEMD_SERVICES=0 ENABLE_NGINX=0 bash deploy.sh
```

`deploy.sh` 會依序執行：

1. 進入 `APP_ROOT`，預設 `/home/NE025/pdf_ocr_translate`。
2. 載入 `.env`。
3. 設定 `APP_ENV=production` 與 Alembic 使用的資料庫連線。
4. 視 `RUN_GIT_PULL` 執行 `git pull origin <branch>`。
5. 執行 `uv sync --frozen`。
6. 停止 Web / Worker service。
7. 安裝或更新 systemd units。
8. 視 `ENABLE_NGINX` 安裝或更新 Nginx site。
9. 執行 `alembic upgrade head`。
10. 執行 `flask --app app.py schema-preflight`。
11. 執行 `flask --app app.py seed-bootstrap`。
12. 重啟 Web、Worker、log cleanup timer 與 template backup timer。
13. 顯示服務狀態。

## 4. Gunicorn 與 systemd 服務設定

本系統使用 systemd 管理 Web 服務、背景 Worker、排程清理與模板備份。systemd unit 範本位於：

```text
deploy/systemd/
```

部署會安裝下列 units：

| Unit 名稱 | 類型 | 用途 |
| --- | --- | --- |
| `uo_regulations_translate.service` | long-running service | 啟動 Gunicorn，提供 Flask Web 系統服務。 |
| `uo_regulations_translate_worker.service` | long-running service | 啟動 `worker.py`，處理 OCR、翻譯、文件重建與 Word 任務。 |
| `uo_regulations_translate_log_cleanup.service` | oneshot service | 執行 `system-error-cleanup` 與 `audit-cleanup`。 |
| `uo_regulations_translate_log_cleanup.timer` | timer | 依排程觸發 log / audit 清理。 |
| `uo_regulations_translate_template_backup.service` | oneshot service | 執行 `scripts/backup_templates.sh` 備份模板資料。 |
| `uo_regulations_translate_template_backup.timer` | timer | 依排程觸發模板備份。 |

### Web 服務

`uo_regulations_translate.service` 是主要 Web 服務，負責啟動 Gunicorn 並載入 Flask App。範本位於：

```text
deploy/systemd/uo_regulations_translate.service.template
```

核心設定範例：

```ini
[Service]
User={{APP_USER}}
Group=www-data
WorkingDirectory={{APP_ROOT}}
EnvironmentFile={{ENV_FILE}}
Environment="PATH={{APP_ROOT}}/.venv/bin:..."
ExecStart={{APP_ROOT}}/.venv/bin/gunicorn --workers {{WEB_WORKERS}} --worker-class gthread --threads 4 --timeout 300 --bind {{WEB_BIND}} --user {{APP_USER}} --group www-data -m 007 --error-logfile - wsgi:app
Restart=always
RestartSec=5
```

重要設定說明：

- `User` / `Group`：指定服務執行身分，需具備專案目錄、`out/`、`logs/` 與備份目錄讀寫權限。
- `WorkingDirectory`：指定 Gunicorn 啟動時的工作目錄。
- `EnvironmentFile`：指定 systemd 啟動服務時載入的 `.env`。
- `ExecStart`：定義 Gunicorn 啟動命令與 WSGI 入口 `wsgi:app`。
- `--bind unix:/home/NE025/pdf_ocr_translate/uo_regulations_translate.sock`：預設 Gunicorn 監聽 Unix Socket。
- `-m 007`：設定 socket 建立時的 umask，使同群組服務可存取 socket。
- `Restart=always`：服務異常結束時自動重啟。

### Worker 服務

`uo_regulations_translate_worker.service` 負責處理非同步工作：

```ini
[Service]
Type=simple
User={{APP_USER}}
Group=www-data
WorkingDirectory={{APP_ROOT}}
EnvironmentFile={{ENV_FILE}}
Environment="PYTHONUNBUFFERED=1"
ExecStart={{APP_ROOT}}/.venv/bin/python worker.py
Restart=always
RestartSec=5
```

Worker 併發與輪詢行為由 `.env` 控制，常見變數包含 `WORKER_POLL_SECONDS`、`WORKER_OCR_MAX_RUNNING`、`WORKER_PDF_TRANSLATE_MAX_RUNNING`、`WORKER_DOC_MAX_RUNNING` 與 `WORKER_WORD_MAX_RUNNING`。

### Log 清理服務與 timer

`uo_regulations_translate_log_cleanup.service` 為 `oneshot` 服務：

```ini
Type=oneshot
ExecStart={{APP_ROOT}}/.venv/bin/flask --app app.py system-error-cleanup
ExecStart={{APP_ROOT}}/.venv/bin/flask --app app.py audit-cleanup
StandardOutput=journal
StandardError=journal
```

`uo_regulations_translate_log_cleanup.timer` 由 `CLEANUP_ON_CALENDAR` 控制排程，預設由 `deploy.sh` 傳入 `*-*-* 03:30:00`。

### 模板備份服務與 timer

`uo_regulations_translate_template_backup.service` 為 `oneshot` 服務：

```ini
Type=oneshot
Environment="APP_ROOT={{APP_ROOT}}"
Environment="ENV_FILE={{ENV_FILE}}"
Environment="STARTUP_WARMUP_ENABLED=0"
ExecStart=/usr/bin/env bash {{APP_ROOT}}/scripts/backup_templates.sh
StandardOutput=journal
StandardError=journal
```

`uo_regulations_translate_template_backup.timer` 由 `TEMPLATE_BACKUP_ON_CALENDAR` 控制排程，預設由 `deploy.sh` 傳入 `*-*-* 02:30:00`。

## 5. Nginx 反向代理設定

Nginx site 範本位於：

```text
deploy/nginx-site.conf.template
```

預設設定：

| 項目 | 預設值 |
| --- | --- |
| Site name | `uo_regulations_translate` |
| Listen port | `81` |
| Static alias | `/home/NE025/pdf_ocr_translate/static/` |
| Proxy target | `unix:/home/NE025/pdf_ocr_translate/uo_regulations_translate.sock` |
| Upload limit | `client_max_body_size 200M` |
| Proxy timeout | 300 秒 |

部署腳本預設 `ENABLE_NGINX=1`，會呼叫 `scripts/install_nginx_site.sh` 產生並安裝 site config。

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

## 6. 系統啟動、重啟與狀態檢查

啟動或重啟服務：

```bash
sudo systemctl restart uo_regulations_translate
sudo systemctl restart uo_regulations_translate_worker
sudo systemctl restart uo_regulations_translate_log_cleanup.timer
sudo systemctl restart uo_regulations_translate_template_backup.timer
```

檢查服務狀態：

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

手動執行清理或備份：

```bash
sudo systemctl start uo_regulations_translate_log_cleanup.service
sudo systemctl start uo_regulations_translate_template_backup.service
```

檢查 timer 排程：

```bash
systemctl list-timers | grep uo_regulations_translate
```

## 7. 部署驗證

部署後建議依序確認下列項目。

```bash
cd /home/NE025/pdf_ocr_translate
.venv/bin/alembic current
.venv/bin/alembic heads
.venv/bin/flask --app app.py schema-preflight
.venv/bin/flask --app app.py seed-bootstrap
sudo systemctl status uo_regulations_translate --no-pager
sudo systemctl status uo_regulations_translate_worker --no-pager
curl -i http://127.0.0.1:81/ | sed -n '1,20p'
systemctl list-timers | grep uo_regulations_translate
ls -lh logs
tail -n 100 logs/app-web.log
```

若已啟用登入，未登入時進入工作頁應導到登入頁：

```bash
curl -i http://127.0.0.1:81/workspace/pdf-overlay | sed -n '1,30p'
```

## 8. 模板備份與還原

模板備份排程由 `uo_regulations_translate_template_backup.timer` 觸發 `uo_regulations_translate_template_backup.service`。備份內容包含：

```text
export/document_templates.json
out/templates/jobs/
```

模板備份採輪詢保留機制，預設最多保留最新 3 份 `.tar.gz` 備份 archive；超過數量時會刪除較舊的 archive 與對應 `.sha256`。可用 `TEMPLATE_BACKUP_MAX_FILES` 調整保留份數。

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

完整流程請參考 [TEMPLATE_BACKUP_RESTORE.md](./TEMPLATE_BACKUP_RESTORE.md)。

## 9. Log 與資料清理

Log 參數在 `.env`：

```env
APP_LOG_DIR=/home/NE025/pdf_ocr_translate/logs
APP_LOG_LEVEL=INFO
APP_LOG_TO_FILE=1
APP_LOG_STDOUT=1
APP_LOG_MAX_MB=10
APP_LOG_BACKUP_COUNT=10
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

## 10. 測試注意事項

測試不能使用正式 schema。測試預設要求獨立 schema，例如：

```bash
TEST_DATABASE_SCHEMA=translation_test .venv/bin/pytest tests/test_app.py
```

測試 safety guard 會拒絕使用 `dbo` 與 `translation`，避免 pytest 清到正式 `[translation].[document_templates]` 或其他正式資料表。

## 11. 維護與故障排除

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

正式環境目前 Alembic version table 應位於 `[translation].[alembic_version]`。

### 502 Bad Gateway

通常是 Gunicorn 沒啟動、socket 權限不對或 Nginx proxy path 不一致：

```bash
sudo systemctl restart uo_regulations_translate
sudo systemctl status uo_regulations_translate --no-pager
ls -l /home/NE025/pdf_ocr_translate/uo_regulations_translate.sock
sudo nginx -t
sudo tail -n 100 /var/log/nginx/error.log
```

### Permission denied

確認 systemd `User` / `Group` 對 `out/`、`logs/`、`backups/templates/` 與 socket 所在目錄有讀寫權限：

```bash
sudo chown -R NE025:www-data /home/NE025/pdf_ocr_translate/out
sudo chown -R NE025:www-data /home/NE025/pdf_ocr_translate/logs
sudo chown -R NE025:www-data /home/NE025/pdf_ocr_translate/backups
```

### 環境變數未載入

systemd 服務只讀 `EnvironmentFile` 指定的 `.env`，不會自動讀取互動 shell 的 `.bashrc`：

```bash
sudo systemctl cat uo_regulations_translate
sudo systemctl restart uo_regulations_translate
sudo systemctl restart uo_regulations_translate_worker
```

### 資料庫連線失敗

確認 `DATABASE_URL`、ODBC Driver 與 SQL Server 網路連線：

```bash
odbcinst -q -d | grep -F "ODBC Driver 18 for SQL Server"
source .env
.venv/bin/python -c "from app.services.state import DATABASE_URL; print(bool(DATABASE_URL))"
.venv/bin/flask --app app.py schema-preflight
```

### OCR 或版面服務失敗

確認 `.env` 中的 `TRITON_URL`、`PP_STRUCTURE_URL` 是否正確，且 Web / Worker 主機可連線：

```bash
grep -n "TRITON_URL\|PP_STRUCTURE_URL" .env
journalctl -u uo_regulations_translate_worker --no-pager -n 200
```

### OpenAI / Azure OpenAI 呼叫失敗

確認 `.env` 中的 API key、endpoint 與 deployment 名稱，並查看 Worker log：

```bash
grep -n "OPENAI_BASE_URL\|BATCH_TRANSLATE_DEPLOYMENT\|DOC_TRANSLATE_DEPLOYMENT\|WORD_TRANSLATE_DEPLOYMENT" .env
journalctl -u uo_regulations_translate_worker --no-pager -n 200
```

### 背景任務未執行

確認 Worker service 正常、資料庫可連線，並檢查 worker 併發上限：

```bash
sudo systemctl status uo_regulations_translate_worker --no-pager
journalctl -u uo_regulations_translate_worker --no-pager -n 200
grep -n "WORKER_.*MAX_RUNNING\|WORKER_POLL_SECONDS" .env
```

### 模板還原後顯示等待 OCR 完成

使用最新版 `scripts/restore_templates.sh` 還原。新版流程會先還原 `out/templates/jobs`，再匯入 `document_templates`，並重建缺失的 `template_source` job record。可看還原輸出是否有 `rebuilt_source_jobs=...`。

### `template_backup` 有備份但沒有資料

確認目前 DB 真的有模板資料：

```bash
.venv/bin/flask --app app.py template-backup --output /tmp/document_templates.json
python -m json.tool /tmp/document_templates.json | head -40
```

若 `template_count=0`，表示目前 `document_templates` 表是空的。
