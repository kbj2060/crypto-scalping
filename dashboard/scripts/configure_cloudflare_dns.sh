#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$ROOT/.env"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

API_TOKEN="${CLOUDFLARE_API_TOKEN:-}"
ACCOUNT_ID="${CLOUDFLARE_ACCOUNT_ID:-d4b593dcda0eb529778fd97f4145f066}"
ZONE_NAME="${CLOUDFLARE_ZONE_NAME:-thesan.xyz}"
TUNNEL_ID="${CLOUDFLARE_TUNNEL_ID:-0b97ed5d-a917-4c6b-af14-4e905e0f68e5}"
TARGET="${TUNNEL_ID}.cfargotunnel.com"

if [[ -z "$API_TOKEN" ]]; then
  echo "CLOUDFLARE_API_TOKEN is required in $ENV_FILE." >&2
  exit 1
fi

api() {
  local method="$1"
  local url="$2"
  local data="${3:-}"
  if [[ -n "$data" ]]; then
    curl -fsS -X "$method" \
      -H "Authorization: Bearer $API_TOKEN" \
      -H "Content-Type: application/json" \
      --data "$data" \
      "$url"
  else
    curl -fsS -X "$method" \
      -H "Authorization: Bearer $API_TOKEN" \
      -H "Content-Type: application/json" \
      "$url"
  fi
}

json_get_string() {
  local key="$1"
  sed -nE "s/.*\"$key\":\"([^\"]+)\".*/\\1/p" | head -n 1
}

echo "Verifying Cloudflare token..."
verify="$(api GET "https://api.cloudflare.com/client/v4/user/tokens/verify" || true)"
if [[ "$verify" != *'"success":true'* ]]; then
  echo "Cloudflare API token is invalid or missing permissions." >&2
  echo "$verify" | sed -E 's/(cfat_)[A-Za-z0-9_\-]+/\1REDACTED/g' >&2
  exit 1
fi

echo "Finding zone: $ZONE_NAME"
zone_response="$(api GET "https://api.cloudflare.com/client/v4/zones?name=$ZONE_NAME")"
ZONE_ID="$(printf '%s' "$zone_response" | json_get_string id)"

if [[ -z "$ZONE_ID" ]]; then
  echo "Could not find Cloudflare zone '$ZONE_NAME' in account '$ACCOUNT_ID'." >&2
  exit 1
fi

echo "Zone ID found."
records_response="$(api GET "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records?name=$ZONE_NAME&per_page=100")"

mapfile -t DELETE_IDS < <(
  printf '%s' "$records_response" |
    tr '{' '\n' |
    sed -nE '/"type":"(A|AAAA|CNAME)"/ s/.*"id":"([^"]+)".*/\1/p'
)

for record_id in "${DELETE_IDS[@]}"; do
  [[ -n "$record_id" ]] || continue
  echo "Deleting existing apex A/AAAA/CNAME record: $record_id"
  api DELETE "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records/$record_id" >/dev/null
done

echo "Creating proxied tunnel CNAME: $ZONE_NAME -> $TARGET"
payload="$(printf '{"type":"CNAME","name":"%s","content":"%s","ttl":1,"proxied":true}' "$ZONE_NAME" "$TARGET")"
create_response="$(api POST "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records" "$payload")"

if [[ "$create_response" != *'"success":true'* ]]; then
  echo "Failed to create Cloudflare DNS record." >&2
  echo "$create_response" >&2
  exit 1
fi

echo "Cloudflare DNS configured."
echo "Expected record: CNAME @ $TARGET Proxied"
