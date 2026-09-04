#!/usr/bin/env bash
# Odyssey4 zig075 진입거부 · BTC 멀티슬롯 섀도우를 서버에서 내린다 (2026-09-04, 사용자 지시).
#
# 코딩 에이전트가 직접 못 하는 이유: sudoers NOPASSWD 화이트리스트가 `systemctl restart`만
# 허용하고 `stop`/`disable`은 의도적으로 제외한다(scripts/ops/systemd/deploy_watcher_sudoers의
# 주석 참고). 두 유닛 다 Restart=always라 프로세스만 죽이면 15초 뒤 되살아난다.
#
# 실행(서버에서 직접, 비밀번호 입력 필요):
#   bash scripts/ops/remove_odyssey4_multislot_shadows_20260904.sh
#
# 하는 일: stop -> disable -> 검증. 데이터와 유닛 파일은 건드리지 않는다.
# 되돌리기: sudo systemctl enable --now <유닛>
set -u
UNITS="btc-multislot-shadow.service eth-odyssey4-shadow.service"

echo "=== 실행 전 상태 ==="
for u in $UNITS; do
  printf "  %-30s active=%-8s enabled=%s\n" "$u" \
    "$(systemctl is-active "$u" 2>/dev/null)" "$(systemctl is-enabled "$u" 2>/dev/null)"
done

echo
echo "=== stop ==="
sudo systemctl stop $UNITS || { echo "!! stop 실패"; exit 1; }
echo "=== disable (재부팅해도 안 올라오게) ==="
sudo systemctl disable $UNITS || { echo "!! disable 실패"; exit 1; }

echo
echo "=== 실행 후 상태 (inactive + disabled 여야 정상) ==="
fail=0
for u in $UNITS; do
  a=$(systemctl is-active "$u" 2>/dev/null); e=$(systemctl is-enabled "$u" 2>/dev/null)
  printf "  %-30s active=%-8s enabled=%s\n" "$u" "$a" "$e"
  [ "$a" = "active" ] && fail=1
  [ "$e" = "enabled" ] && fail=1
done

echo
echo "=== 프로세스 잔존 확인 (아무것도 안 나와야 정상) ==="
ps -eo pid,etime,cmd | grep -E "[r]un_btc_multislot_shadow_loop|[o]dyssey4_zig075_entry_veto_shadow_cleanroom" || echo "  없음"

echo
if [ "$fail" = 0 ]; then
  echo "완료. 데이터(data/live/eth_odyssey4_shadow/, data/ensemble/omega4_6_1_btc_multislot_*)와"
  echo "유닛 파일은 그대로 뒀습니다 -- 다시 돌리려면 sudo systemctl enable --now <유닛>."
  echo
  echo "남은 정리(선택): /etc/sudoers.d/crypto-scalping-deploy-watcher 의 두 유닛 restart 항목."
  echo "  ⚠️ sudoers는 직접 편집하지 말고 반드시 visudo -c 로 검증 후 install 할 것"
  echo "     (문법 오류 하나로 sudo 전체가 막힙니다). 남겨둬도 무해합니다."
else
  echo "!! 검증 실패 -- 위 상태를 확인해주세요."; exit 1
fi
