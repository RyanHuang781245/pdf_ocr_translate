# Template Backup And Restore

這份文件說明欄位翻譯模板的備份與還原流程。

備份內容包含：
- `export/document_templates.json`：從 DB `document_templates` 匯出的模板資料
- `out/templates/jobs/`：模板來源 PDF、OCR、編輯檔案等來源 job 檔案

## 手動備份

```bash
cd /home/NE025/pdf_ocr_translate
APP_ROOT=/home/NE025/pdf_ocr_translate \
ENV_FILE=/home/NE025/pdf_ocr_translate/.env \
bash scripts/backup_templates.sh
```

備份檔預設輸出到：

```text
backups/templates
```

## 驗證備份

```bash
cd /home/NE025/pdf_ocr_translate/backups/templates
sha256sum -c 最新檔名.tar.gz.sha256
tar -tzf 最新檔名.tar.gz | head -50
```

壓縮包至少應包含：

```text
export/document_templates.json
out/templates/jobs/
```

## 手動還原

還原會先自動建立目前狀態的模板備份，然後：

1. 將 DB `document_templates` 同步成備份 JSON 內容
2. 將 `out/templates/jobs` 同步成備份包內的檔案

```bash
cd /home/NE025/pdf_ocr_translate
APP_ROOT=/home/NE025/pdf_ocr_translate \
ENV_FILE=/home/NE025/pdf_ocr_translate/.env \
bash scripts/restore_templates.sh /path/to/templates_backup.tar.gz --yes
```

若已經另外備份過目前狀態，可以略過還原前備份：

```bash
SKIP_PRE_RESTORE_BACKUP=1 \
APP_ROOT=/home/NE025/pdf_ocr_translate \
ENV_FILE=/home/NE025/pdf_ocr_translate/.env \
bash scripts/restore_templates.sh /path/to/templates_backup.tar.gz --yes
```

## 只還原 DB

```bash
cd /home/NE025/pdf_ocr_translate
.venv/bin/flask --app app.py template-restore \
  --input /path/to/document_templates.json \
  --replace
```

`--replace` 會先清空目前 DB 內的 `document_templates`，再匯入備份內容。

## systemd 排程

部署後可檢查 timer：

```bash
systemctl status uo_regulations_translate_template_backup.timer --no-pager
systemctl list-timers | grep uo_regulations_translate_template_backup
```

立即觸發一次備份：

```bash
sudo systemctl start uo_regulations_translate_template_backup.service
sudo systemctl status uo_regulations_translate_template_backup.service --no-pager
```
