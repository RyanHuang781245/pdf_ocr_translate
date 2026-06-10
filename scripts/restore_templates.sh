#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/home/NE025/pdf_ocr_translate}"
ENV_FILE="${ENV_FILE:-$APP_ROOT/.env}"
BACKUP_ROOT="${TEMPLATE_BACKUP_ROOT:-$APP_ROOT/backups/templates}"
SKIP_PRE_RESTORE_BACKUP="${SKIP_PRE_RESTORE_BACKUP:-0}"
RESTORE_ARCHIVE="${RESTORE_ARCHIVE:-${1:-}}"
CONFIRM="${2:-}"
TMP_DIR="$(mktemp -d)"
CONTENTS_FILE="$TMP_DIR/contents.txt"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1"
}

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/restore_templates.sh PATH_TO_TEMPLATE_BACKUP.tar.gz --yes

Environment:
  APP_ROOT=/home/NE025/pdf_ocr_translate
  ENV_FILE=$APP_ROOT/.env
  TEMPLATE_BACKUP_ROOT=$APP_ROOT/backups/templates
  SKIP_PRE_RESTORE_BACKUP=1  Skip creating a current-state template backup before restore
EOF
}

if [[ -z "$RESTORE_ARCHIVE" || "$CONFIRM" != "--yes" ]]; then
  usage
  exit 64
fi

if [[ ! -d "$APP_ROOT" ]]; then
  echo "APP_ROOT not found: $APP_ROOT" >&2
  exit 66
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ENV_FILE not found: $ENV_FILE" >&2
  exit 66
fi

if [[ ! -f "$RESTORE_ARCHIVE" ]]; then
  echo "Restore archive not found: $RESTORE_ARCHIVE" >&2
  exit 66
fi

case "$RESTORE_ARCHIVE" in
  *.tar.gz) ;;
  *)
    echo "Restore archive must be a .tar.gz file: $RESTORE_ARCHIVE" >&2
    exit 64
    ;;
esac

CHECKSUM_PATH="${RESTORE_ARCHIVE}.sha256"
if [[ -f "$CHECKSUM_PATH" ]]; then
  log "Verifying checksum: $CHECKSUM_PATH"
  expected_hash="$(awk '{print $1; exit}' "$CHECKSUM_PATH")"
  actual_hash="$(sha256sum "$RESTORE_ARCHIVE" | awk '{print $1; exit}')"
  if [[ -z "$expected_hash" || "$expected_hash" != "$actual_hash" ]]; then
    echo "Checksum verification failed: $RESTORE_ARCHIVE" >&2
    exit 65
  fi
else
  log "Checksum file not found; skipping checksum verification"
fi

log "Validating archive structure"
tar -tzf "$RESTORE_ARCHIVE" > "$CONTENTS_FILE"
if grep -Eq '(^/|(^|/)\.\.(/|$))' "$CONTENTS_FILE"; then
  echo "Restore archive contains unsafe paths: $RESTORE_ARCHIVE" >&2
  exit 65
fi
if ! grep -qx 'export/document_templates.json' "$CONTENTS_FILE"; then
  echo "Restore archive missing export/document_templates.json: $RESTORE_ARCHIVE" >&2
  exit 65
fi

if [[ "$SKIP_PRE_RESTORE_BACKUP" != "1" ]]; then
  log "Creating current-state template backup before restore"
  APP_ROOT="$APP_ROOT" ENV_FILE="$ENV_FILE" TEMPLATE_BACKUP_ROOT="$BACKUP_ROOT" bash "$(dirname "${BASH_SOURCE[0]}")/backup_templates.sh"
fi

log "Extracting archive"
tar -xzf "$RESTORE_ARCHIVE" -C "$TMP_DIR"

log "Restoring template source files"
rm -rf -- "$APP_ROOT/out/templates/jobs"
mkdir -p "$APP_ROOT/out/templates"
if [[ -d "$TMP_DIR/out/templates/jobs" ]]; then
  cp -a "$TMP_DIR/out/templates/jobs" "$APP_ROOT/out/templates/jobs"
else
  mkdir -p "$APP_ROOT/out/templates/jobs"
fi

log "Restoring document_templates table"
APP_ROOT="$APP_ROOT" ENV_FILE="$ENV_FILE" STARTUP_WARMUP_ENABLED=0 "$APP_ROOT/.venv/bin/flask" --app app.py template-restore --input "$TMP_DIR/export/document_templates.json" --replace

log "Template restore complete"
printf 'restored_archive=%s\n' "$RESTORE_ARCHIVE"
