#!/usr/bin/env bash
# trade_tape.duckdb 를 collector(Pi) -> server 로 복제. **Pi 에서 매시 cron 으로 돈다.**
#
# 왜 필요한가: 대시보드가 trade_tape.duckdb 를 로컬에서 읽는다(시간대별 분위 기준선,
# 1시간마다 -- dashboard/server.py:550 주석). 수집을 Pi 로 옮기면서 이 복제가 없으면
# 서버측 수집기를 내리는 순간 기준선 패널이 죽는다.
#
# 🔴수집기가 쓰는 중인 duckdb 를 그대로 rsync 하면 찢어진다. COPY FROM DATABASE 로
#   **일관된 스냅샷**을 먼저 뜬 뒤 그것만 보낸다.
#
# 🔴기본 목적지가 trade_tape.duckdb 가 **아니다**. 병행 테스트 동안은 서버 수집기가
#   그 파일을 소유하므로 덮으면 안 된다. 전환할 때 TT_DEST 를 바꾼다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
PY="${PYTHON_BIN:-$HOME/miniforge3/envs/quant_ai/bin/python}"
# 2026-09-26 다코인: 코인별 테이프(trade_tape_btc.duckdb, okx_trade_tape_xrp.duckdb …)도 같은 스크립트로
#   보낸다 -- TT_SRC 만 바꿔 cron 줄을 하나씩 더 둔다. 목적지 기본값은 원본 이름에 .from_pi 를 붙인
#   것이라 서버에 같은 이름의 수집기가 있어도 덮지 않는다.
SRC="${TT_SRC:-data/live/trade_tape.duckdb}"
DEST="${TT_DEST:-${SRC%.duckdb}.from_pi.duckdb}"   # 전환 시 원본과 같은 이름으로
# 2026-09-26 테이프 외 DB 도 보낸다(HL 멀티코인 포지션 -- 대시보드 SOL·XRP 고래 청산가). 스냅샷 검증 표만 바꾼다.
VERIFY_TABLE="${TT_VERIFY_TABLE:-trade_tape_1s}"

[[ -f "$SRC" ]] || { echo "[$(date -Iseconds)] 원본 없음: $SRC"; exit 0; }

# 일관 스냅샷. 락이 걸려 있으면 재시도하고, 끝내 못 열면 이번 주기를 건너뛴다
# (다음 시각에 다시 시도한다 -- 한 번 실패로 복제를 영구히 멈추지 않는다).
"$PY" - "$SRC" "$DEST" "$VERIFY_TABLE" <<'PYEOF'
import duckdb, pathlib, sys, time
src, dest, verify_table = sys.argv[1], sys.argv[2], sys.argv[3]
tmp = dest + ".tmp"
for f in (tmp, tmp + ".wal"):
    pathlib.Path(f).unlink(missing_ok=True)
con = None
for _ in range(60):
    try:
        con = duckdb.connect(src, read_only=True); break
    except Exception:
        time.sleep(1.0)
if con is None:
    print("락 해제 실패 -- 이번 주기 건너뜀"); sys.exit(3)
try:
    # duckdb 의 ATTACH 는 파라미터 바인딩을 안 받는다 -- 경로를 직접 넣되 따옴표를 이스케이프한다.
    # 🔴(READ_WRITE) 를 빼면 부모 연결의 read_only 를 물려받아 "database does not exist" 로 죽는다.
    con.execute("ATTACH '%s' AS snap (READ_WRITE)" % tmp.replace("'", "''"))
    # 🔴원본 카탈로그 이름은 "main" 이 아니라 **파일명**이다(trade_tape). 고정하지 않고 조회한다.
    srcdb = con.execute("SELECT current_database()").fetchone()[0]
    con.execute('COPY FROM DATABASE "%s" TO snap' % srcdb.replace('"', '""'))
    con.execute("DETACH snap")
finally:
    con.close()
# 열리는지 확인하고 나서야 제자리에 놓는다 -- 깨진 스냅샷을 보내지 않는다.
v = duckdb.connect(tmp, read_only=True)
n = v.execute('SELECT count(*) FROM "%s"' % verify_table.replace('"', '""')).fetchone()[0]
v.close()
pathlib.Path(tmp).replace(dest)
pathlib.Path(tmp + ".wal").unlink(missing_ok=True)
print(f"스냅샷 {n:,}행 -> {dest}")
PYEOF
rc=$?
[[ $rc -eq 0 ]] || { echo "[$(date -Iseconds)] 스냅샷 실패(rc=$rc) -- 전송 안 함"; exit $rc; }

# handoff.sh 가 이 저장소의 유일한 ssh/rsync 통로다. 새로 rsync 옵션을 쓰지 않는다.
bash scripts/ops/handoff.sh push server "$DEST" 2>&1 | tail -3
echo "[$(date -Iseconds)] 복제 완료: $DEST"
