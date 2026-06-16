# 模板備份與還原說明

本文說明翻譯系統的欄位翻譯模板備份、排程備份、備份檔驗證、還原流程與常見問題。本文只涵蓋模板資料備份，不包含整個 MSSQL database、所有 job、上傳檔、翻譯輸出檔或系統設定檔備份。

## 1. 備份範圍

模板備份由 `scripts/backup_templates.sh` 執行，備份內容包含：

| 路徑 | 來源 | 用途 |
| --- | --- | --- |
| `export/document_templates.json` | 由 Flask CLI 從 DB `document_templates` 匯出 | 模板 metadata、模板設定與 DB 內保存的模板內容。 |
| `out/templates/jobs/` | 本機檔案系統 | 模板來源 PDF、OCR、編輯檔案與模板來源 job 檔案。 |
| `out/templates/document_templates.json` | 本機檔案系統，若存在才納入 | 舊版或備用 JSON 檔。 |

備份不包含：

- `.env`、API key、DB 連線字串或 LDAP 密碼。
- 整個 MSSQL database。
- 一般使用者上傳檔、一般 OCR / 翻譯 job 輸出。
- `logs/`、`.venv/`、`PaddleX/` 或其他大型 runtime 目錄。

## 2. 重要路徑與參數

| 項目 | 預設值 | 說明 |
| --- | --- | --- |
| `APP_ROOT` | `/home/NE025/pdf_ocr_translate` | 專案根目錄。 |
| `ENV_FILE` | `$APP_ROOT/.env` | 備份與還原時載入的環境變數檔。 |
| `TEMPLATE_BACKUP_ROOT` | `$APP_ROOT/backups/templates` | 備份 archive 輸出目錄。 |
| `TEMPLATE_BACKUP_MAX_FILES` | `3` | 輪詢保留份數，最多保留最新 3 份 `.tar.gz` 備份 archive。 |
| `TEMPLATE_BACKUP_ON_CALENDAR` | `*-*-* 02:30:00` | systemd timer 排程，由 `deploy.sh` render 到 timer。 |
| `SKIP_PRE_RESTORE_BACKUP` | `0` | 還原前是否略過目前狀態備份。預設不略過。 |

備份檔命名格式：

```text
<hostname>_templates_<YYYY-MM-DD_HHMMSS>.tar.gz
<hostname>_templates_<YYYY-MM-DD_HHMMSS>.tar.gz.sha256
```

## 3. systemd 排程備份

部署後會安裝下列 unit：

| Unit | 用途 |
| --- | --- |
| `uo_regulations_translate_template_backup.service` | 執行 `scripts/backup_templates.sh`。 |
| `uo_regulations_translate_template_backup.timer` | 依 `TEMPLATE_BACKUP_ON_CALENDAR` 定期觸發備份。 |

檢查 timer：

```bash
systemctl status uo_regulations_translate_template_backup.timer --no-pager
systemctl list-timers | grep uo_regulations_translate_template_backup
```

查看最近備份執行紀錄：

```bash
journalctl -u uo_regulations_translate_template_backup.service --no-pager -n 100
```

立即觸發一次備份：

```bash
sudo systemctl start uo_regulations_translate_template_backup.service
sudo systemctl status uo_regulations_translate_template_backup.service --no-pager
```

## 4. 手動備份

手動執行備份：

```bash
cd /home/NE025/pdf_ocr_translate
APP_ROOT=/home/NE025/pdf_ocr_translate \
ENV_FILE=/home/NE025/pdf_ocr_translate/.env \
bash scripts/backup_templates.sh
```

指定備份輸出目錄與保留份數：

```bash
cd /home/NE025/pdf_ocr_translate
APP_ROOT=/home/NE025/pdf_ocr_translate \
ENV_FILE=/home/NE025/pdf_ocr_translate/.env \
TEMPLATE_BACKUP_ROOT=/home/NE025/pdf_ocr_translate/backups/templates \
TEMPLATE_BACKUP_MAX_FILES=3 \
bash scripts/backup_templates.sh
```

成功時輸出會包含：

```text
backup_file=/home/NE025/pdf_ocr_translate/backups/templates/<hostname>_templates_<timestamp>.tar.gz
```

## 5. 輪詢保留機制

模板備份使用輪詢保留機制。每次備份完成後，系統會在 `TEMPLATE_BACKUP_ROOT` 內依修改時間保留最新 `TEMPLATE_BACKUP_MAX_FILES` 份 `.tar.gz` archive，超過數量的舊 archive 會被刪除，對應的 `.sha256` 也會一併刪除。

預設：

```env
TEMPLATE_BACKUP_MAX_FILES=3
```

也就是最多保留最新 3 份備份檔。若要增加保留份數，可調整 `.env`：

```env
TEMPLATE_BACKUP_MAX_FILES=5
```

修改後重新部署或重啟 timer：

```bash
bash deploy.sh
systemctl cat uo_regulations_translate_template_backup.timer
```

## 6. 備份檔驗證

列出備份檔：

```bash
ls -lh /home/NE025/pdf_ocr_translate/backups/templates
```

驗證 checksum：

```bash
cd /home/NE025/pdf_ocr_translate/backups/templates
sha256sum -c <backup-file>.tar.gz.sha256
```

檢查壓縮包內容：

```bash
tar -tzf <backup-file>.tar.gz | sed -n '1,80p'
```

壓縮包至少應包含：

```text
export/document_templates.json
```

若系統已有模板來源 job，通常也會包含：

```text
out/templates/jobs/
```

檢查匯出的 DB JSON：

```bash
mkdir -p /tmp/template-backup-check
tar -xzf <backup-file>.tar.gz -C /tmp/template-backup-check export/document_templates.json
python -m json.tool /tmp/template-backup-check/export/document_templates.json | sed -n '1,80p'
```

## 7. 還原前注意事項

還原會覆蓋目前模板資料，請先確認：

1. 已選定正確的 `.tar.gz` 備份檔。
2. 若同目錄存在 `.sha256`，還原腳本會自動驗證 checksum。
3. 還原腳本預設會先建立目前狀態的模板備份，再執行還原。
4. 還原 DB 時會使用 `template-restore --replace`，也就是先清空目前 DB 內的 `document_templates`，再匯入備份內容。
5. 還原檔案時會移除目前 `out/templates/jobs`，再以備份包內的 `out/templates/jobs` 取代。

建議先暫停使用者對模板的新增、編輯與刪除操作，再執行還原。

## 8. 手動還原

標準還原流程：

```bash
cd /home/NE025/pdf_ocr_translate
APP_ROOT=/home/NE025/pdf_ocr_translate \
ENV_FILE=/home/NE025/pdf_ocr_translate/.env \
bash scripts/restore_templates.sh /path/to/<backup-file>.tar.gz --yes
```

若已經另外備份過目前狀態，可以略過還原前備份：

```bash
cd /home/NE025/pdf_ocr_translate
SKIP_PRE_RESTORE_BACKUP=1 \
APP_ROOT=/home/NE025/pdf_ocr_translate \
ENV_FILE=/home/NE025/pdf_ocr_translate/.env \
bash scripts/restore_templates.sh /path/to/<backup-file>.tar.gz --yes
```

成功時輸出會包含：

```text
restored_archive=/path/to/<backup-file>.tar.gz
```

## 9. 只還原 DB

若只需要還原 `document_templates` DB 內容，不還原 `out/templates/jobs/` 檔案，可先從 archive 取出 JSON，再使用 Flask CLI：

```bash
mkdir -p /tmp/template-db-restore
tar -xzf /path/to/<backup-file>.tar.gz -C /tmp/template-db-restore export/document_templates.json

cd /home/NE025/pdf_ocr_translate
APP_ROOT=/home/NE025/pdf_ocr_translate \
ENV_FILE=/home/NE025/pdf_ocr_translate/.env \
STARTUP_WARMUP_ENABLED=0 \
.venv/bin/flask --app app.py template-restore \
  --input /tmp/template-db-restore/export/document_templates.json \
  --replace
```

`--replace` 會先清空目前 DB 內的 `document_templates`，再匯入備份內容。

## 10. 還原後驗證

檢查還原腳本輸出與 service log：

```bash
journalctl -u uo_regulations_translate_template_backup.service --no-pager -n 100
```

確認模板 DB 可匯出：

```bash
cd /home/NE025/pdf_ocr_translate
.venv/bin/flask --app app.py template-backup --output /tmp/document_templates_after_restore.json
python -m json.tool /tmp/document_templates_after_restore.json | sed -n '1,80p'
```

確認模板來源檔案存在：

```bash
find /home/NE025/pdf_ocr_translate/out/templates/jobs -maxdepth 2 -type f | sed -n '1,80p'
```

確認 Web / Worker 狀態：

```bash
sudo systemctl status uo_regulations_translate --no-pager
sudo systemctl status uo_regulations_translate_worker --no-pager
```

若 Web 畫面仍顯示舊資料，可重新整理瀏覽器或重啟 Web service：

```bash
sudo systemctl restart uo_regulations_translate
```

## 11. 常見問題

### `Required path not found: .venv/bin/flask`

代表 Python 虛擬環境尚未建立或部署未完成。先同步環境：

```bash
cd /home/NE025/pdf_ocr_translate
uv sync --frozen
```

### `ENV_FILE not found`

確認 `ENV_FILE` 指向正確 `.env`：

```bash
APP_ROOT=/home/NE025/pdf_ocr_translate
ENV_FILE=$APP_ROOT/.env
test -f "$ENV_FILE" && echo OK
```

### checksum 驗證失敗

代表 `.tar.gz` 與 `.sha256` 不一致，可能是備份檔未完整複製、被修改或檔案損壞。請改用其他備份檔，不建議強行還原。

### `Restore archive missing export/document_templates.json`

代表該 `.tar.gz` 不是本系統模板備份檔，或備份檔結構不完整。先檢查內容：

```bash
tar -tzf /path/to/<backup-file>.tar.gz | sed -n '1,80p'
```

### 還原後模板顯示等待 OCR 完成

使用 `scripts/restore_templates.sh` 還原完整 archive，不要只匯入 DB JSON。完整還原會先還原 `out/templates/jobs`，再匯入 `document_templates`，程式會在匯入時重建缺失的 `template_source` job record。

### 備份檔超過 3 份

確認備份是透過最新版 `scripts/backup_templates.sh` 執行，且 `TEMPLATE_BACKUP_ROOT` 指向同一個備份目錄。輪詢只會清理同一層目錄中的 `*.tar.gz` 與對應 `.sha256`。

### 要暫時保留更多備份

在 `.env` 或手動備份命令中調整：

```env
TEMPLATE_BACKUP_MAX_FILES=5
```

或：

```bash
TEMPLATE_BACKUP_MAX_FILES=5 bash scripts/backup_templates.sh
```
