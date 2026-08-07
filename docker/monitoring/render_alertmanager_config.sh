#!/usr/bin/env bash
# Generates alertmanager.yml (gitignored -- contains the real Telegram bot
# token) from alertmanager.yml.template (committed, placeholders only) using
# TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID from the repo's .env.
#
# Run before `docker compose up`, and again any time .env's Telegram values
# change: bash render_alertmanager_config.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$ROOT/.env" ]]; then
  echo "missing $ROOT/.env -- cannot read TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a

: "${TELEGRAM_BOT_TOKEN:?TELEGRAM_BOT_TOKEN not set in .env}"
: "${TELEGRAM_CHAT_ID:?TELEGRAM_CHAT_ID not set in .env}"

sed \
  -e "s|__TELEGRAM_BOT_TOKEN__|${TELEGRAM_BOT_TOKEN}|" \
  -e "s|__TELEGRAM_CHAT_ID__|${TELEGRAM_CHAT_ID}|" \
  "$DIR/alertmanager/alertmanager.yml.template" > "$DIR/alertmanager/alertmanager.yml"

echo "wrote $DIR/alertmanager/alertmanager.yml"
