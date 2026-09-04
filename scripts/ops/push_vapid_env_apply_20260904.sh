#!/usr/bin/env bash
# 대시보드 웹푸시 -- 서버 .env에 VAPID 키를 만들어 넣는다. 재시작은 하지 않는다.
# (auto 모드 분류기가 서버 .env 편집을 차단하므로, 사용자가 직접 실행하도록 저장한 스크립트.
#  같은 이유로 만들어진 a4_env_apply_20260904.sh와 같은 패턴이다.)
#
# 실행(dev에서):
#   bash scripts/ops/handoff.sh launch server push_vapid --sync scripts/ops/push_vapid_env_apply_20260904.sh \
#     -- bash /home/llewyn/crypto-scalping/scripts/ops/push_vapid_env_apply_20260904.sh
#   bash scripts/ops/handoff.sh logs server push_vapid
#
# 키를 이 스크립트에 박아두지 않고 **서버에서 생성**한다 -- 개인키가 저장소에도, 동기화되는
# 파일에도 남지 않는다. 그래서 서버 키와 로컬 dev 키는 서로 다르고, 그게 맞다(로컬에서 만든
# 구독은 서버에서 동작하지 않아야 한다).
#
# 추가하는 키:
#   VAPID_PUBLIC_KEY   브라우저에 넘겨주는 applicationServerKey (/api/push/config로 노출)
#   VAPID_PRIVATE_KEY  서버 밖으로 나가지 않는 서명키
#   VAPID_SUBJECT      RFC 8292가 요구하는 연락처(mailto:)
# 백업: .env.bak_pre_vapid_20260904
#
# ⚠️ 이미 키가 있으면 아무것도 하지 않는다. 키를 갈아끼우면 **기존 구독이 전부 무효**가 되어
#    모든 기기가 알림 버튼을 다시 눌러야 한다.
set -u
cd /home/llewyn/crypto-scalping

if grep -q '^VAPID_PRIVATE_KEY=' .env; then
  echo "VAPID 키가 이미 있습니다 -- 변경하지 않습니다(재생성하면 기존 구독이 전부 끊깁니다)."
  grep -c '^VAPID_' .env
  exit 0
fi

cp -p .env .env.bak_pre_vapid_20260904 || { echo "BACKUP FAILED"; exit 1; }
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"

[ -n "$(tail -c1 .env)" ] && echo >> .env
{
  echo "# Web Push (RFC 8292 VAPID) -- 대시보드 PWA 알림, 2026-09-04."
  echo "# scripts/ops/push_vapid_env_apply_20260904.sh가 이 서버에서 생성했다."
  "$PY" "$(pwd)/scripts/push_webpush_20260904.py" generate-keys
  echo "VAPID_SUBJECT=mailto:kbj2060@gmail.com"
} >> .env

echo "--- 추가된 키(값 제외) ---"
grep -o '^VAPID_[A-Z_]*' .env
echo "--- mode/owner/size ---"; stat -c '%a %U %s %n' .env .env.bak_pre_vapid_20260904
echo "--- 대시보드가 읽는 것과 같은 경로로 파싱 확인 ---"
"$PY" - <<'PY'
import os
from dotenv import load_dotenv
load_dotenv()
pub, priv = os.getenv("VAPID_PUBLIC_KEY", ""), os.getenv("VAPID_PRIVATE_KEY", "")
import sys; sys.path.insert(0, os.getcwd())
from scripts.push_webpush_20260904 import vapid_public_key_from_private, b64u_decode
assert pub and priv, "키가 .env에서 안 읽힙니다"
assert vapid_public_key_from_private(priv) == pub, "공개키/개인키 쌍이 맞지 않습니다"
assert len(b64u_decode(pub)) == 65, "공개키가 비압축 P-256 점(65바이트)이 아닙니다"
print("VAPID 키 쌍 검증 OK (공개키 길이 65, 개인키에서 공개키 재유도 일치)")
PY
echo
echo "다음: 대시보드 재시작(새 .env를 읽어야 /api/push/config가 enabled=true가 됩니다)"
echo "      그 다음 supervisor_push_notifier.sh 기동."
