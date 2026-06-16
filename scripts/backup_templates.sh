#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/home/NE025/pdf_ocr_translate}"
ENV_FILE="${ENV_FILE:-$APP_ROOT/.env}"
BACKUP_ROOT="${TEMPLATE_BACKUP_ROOT:-$APP_ROOT/backups/templates}"
MAX_BACKUP_FILES="${TEMPLATE_BACKUP_MAX_FILES:-3}"
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

if ! [[ "$MAX_BACKUP_FILES" =~ ^[0-9]+$ ]] || [[ "$MAX_BACKUP_FILES" -lt 1 ]]; then
  echo "TEMPLATE_BACKUP_MAX_FILES must be a positive integer." >&2
  exit 64
fi

mapfile -t old_archives < <(
  find "$BACKUP_ROOT" -maxdepth 1 -type f -name "*.tar.gz" -printf '%T@ %p\n'     | sort -nr     | awk -v keep="$MAX_BACKUP_FILES" 'NR > keep {sub(/^[^ ]+ /, ""); print}'
)

for old_archive in "${old_archives[@]}"; do
  rm -f "$old_archive" "$old_archive.sha256"
done

log "Retention cleanup complete: kept latest $MAX_BACKUP_FILES archive(s)"
printf 'backup_file=%s\n' "$ARCHIVE_PATH"
