#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/home/NE025/pdf_ocr_translate}"
ENV_FILE="${ENV_FILE:-$APP_ROOT/.env}"
BACKUP_ROOT="${TEMPLATE_BACKUP_ROOT:-$APP_ROOT/backups/templates}"
RETENTION_DAYS="${TEMPLATE_BACKUP_RETENTION_DAYS:-30}"
HOSTNAME_SHORT="${HOSTNAME_SHORT:-$(hostname -s)}"
STAMP="$(date +%F_%H%M%S)"
ARCHIVE_NAME="${ARCHIVE_NAME:-${HOSTNAME_SHORT}_templates_${STAMP}.tar.gz}"
ARCHIVE_PATH="$BACKUP_ROOT/$ARCHIVE_NAME"
CHECKSUM_PATH="${ARCHIVE_PATH}.sha256"
WORK_DIR="$(mktemp -d)"
INCLUDE_FILE="$WORK_DIR/include.txt"
EXPORT_PATH="$WORK_DIR/document_templates.json"

cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1"
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Required path not found: $path" >&2
    exit 66
  fi
}

require_path "$APP_ROOT"
require_path "$ENV_FILE"
require_path "$APP_ROOT/.venv/bin/flask"

mkdir -p "$BACKUP_ROOT"

log "Exporting document_templates to JSON"
APP_ROOT="$APP_ROOT" ENV_FILE="$ENV_FILE" STARTUP_WARMUP_ENABLED=0 "$APP_ROOT/.venv/bin/flask" --app app.py template-backup --output "$EXPORT_PATH"

mkdir -p "$WORK_DIR/export"
cp "$EXPORT_PATH" "$WORK_DIR/export/document_templates.json"

: > "$INCLUDE_FILE"

if [[ -d "$APP_ROOT/out/templates/jobs" ]]; then
  printf 'out/templates/jobs\n' >> "$INCLUDE_FILE"
fi

if [[ -f "$APP_ROOT/out/templates/document_templates.json" ]]; then
  printf 'out/templates/document_templates.json\n' >> "$INCLUDE_FILE"
fi

log "Creating archive: $ARCHIVE_PATH"
tar -czf "$ARCHIVE_PATH" \
  -C "$WORK_DIR" export/document_templates.json \
  -C "$APP_ROOT" \
  -T "$INCLUDE_FILE" \
  --ignore-failed-read

sha256sum "$ARCHIVE_PATH" > "$CHECKSUM_PATH"

log "Archive created"
log "Checksum written: $CHECKSUM_PATH"

find "$BACKUP_ROOT" -type f \( -name "*.tar.gz" -o -name "*.tar.gz.sha256" \) -mtime +"$RETENTION_DAYS" -delete

log "Retention cleanup complete"
printf 'backup_file=%s\n' "$ARCHIVE_PATH"
