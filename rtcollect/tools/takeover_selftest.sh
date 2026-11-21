#!/usr/bin/env bash
# takeover.sh 자체 시험(개발기 · 2026-11-21): 가짜 저장소 + 가짜 crontab(CRONTAB_CMD) + 가짜 프로세스(bash 고리)로 go·back 을 실제로 돌린다.
#   진짜 crontab·진짜 프로세스는 건드리지 않는다(가짜 저장소 폴더가 명령줄에 든 프로세스만 띄우고 치운다). 거래소 접속 없음.
#   bash rtcollect/tools/takeover_selftest.sh
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R="$(mktemp -d)"; export R
cleanup() { disown -a 2>/dev/null; for p in /proc/[0-9]*; do tr '\0' ' ' 2>/dev/null < "$p/cmdline" | grep -qF "$R/" && kill -9 "${p#/proc/}" 2>/dev/null; done; rm -rf "$R"; }
trap cleanup EXIT
FAIL=0; ok() { if eval "$2"; then echo "  ✓ $1"; else echo "  ✗ $1"; FAIL=1; fi; }
mkdir -p "$R"/{scripts/ops,rtcollect/tools,data/hot/rtcollect,data/live,logs/supervisor,tmp/rtbin}
cp "$SRC"/scripts/ops/{_supervise.sh,supervisor_trade_tape.sh,supervisor_rtcollect.sh} "$R/scripts/ops/"
cp "$SRC"/rtcollect/tools/{takeover.sh,shadow_tape.py,shadow_duck.py} "$R/rtcollect/tools/"
# 가짜 파이썬 수집기: 감독(부모)이 먼저 죽었는지 SIGTERM 때 적는다 · 쓰는 파일 mtime 을 민다
cat > "$R/scripts/live_trade_tape_collector_20260916.py" <<'X'
P0=$PPID
trap 'if kill -0 $P0 2>/dev/null; then echo "py_term $TAPE_MARKET parent_alive" >> $R/order.log; else echo "py_term $TAPE_MARKET parent_gone" >> $R/order.log; fi; [ -s $R/slow ] && for i in $(seq $(( $(cat $R/slow) * 2 ))); do touch "$f"; sleep 0.5; done; echo "py_exit $TAPE_MARKET" >> $R/order.log; exit 0' TERM   # slow 파일에 적힌 초만큼 SIGTERM 뒤 더 쓰다가 끝남(마지막 묶음 흉내)
echo "py_start $TAPE_SYMBOL $TAPE_MARKET" >> $R/order.log
f=$R/data/hot/binance_tape.sqlite; [ "$TAPE_MARKET" = futures ] || f=$R/data/hot/binance_spot_tape.sqlite
while :; do touch "$f"; sleep 0.3 & wait $!; done
X
# 가짜 rtcollect(이름이 rtcollect 라 pgrep -x 에 잡힌다)
cat > "$R/tmp/rtbin/rtcollect" <<'X'
#!/bin/bash
trap 'echo "rs_term takeover=${RTCOLLECT_TAKEOVER:-0}" >> $R/order.log; exit 0' TERM
echo "rs_start takeover=${RTCOLLECT_TAKEOVER:-0} db=${TAPE_DB_PATH:-shadow} rest=${RTCOLLECT_REST:-0} sym=$TAPE_SYMBOL" >> $R/order.log
f="${TAPE_DB_PATH:-data/hot/rtcollect/binance_tape.sqlite}"
sleep 2
while :; do touch "$f"; sleep 0.3 & wait $!; done
X
chmod +x "$R/tmp/rtbin/rtcollect"
# 같은 행의 두 SQLite(그림자 대조 통과) · 시험 실패용 사본(값 하나 다름)
python3 - <<'PY'
import os, sqlite3
R = os.environ["R"]
for f, bad in (("data/hot/binance_tape.sqlite", 0), ("data/hot/rtcollect/binance_tape.sqlite", 0), ("bad.sqlite", 1)):
    c = sqlite3.connect(f"{R}/{f}")
    c.execute("CREATE TABLE trade_tape_1s(symbol TEXT, ts_sec INTEGER, price_bin INTEGER, buy_qty REAL, PRIMARY KEY(symbol, ts_sec, price_bin)) WITHOUT ROWID")
    rows = [(sym, 1_763_683_200 + s, 1, 9.0 if (bad and s == 50) else 1.0) for s in range(600) for sym in ("ethusdt", "btcusdt")]
    if "rtcollect" in f:   # 그림자: 재기동 사이(T+300~T+340) 비었다
        rows = [r for r in rows if not 300 <= r[1] - 1_763_683_200 <= 340]
    c.executemany("INSERT INTO trade_tape_1s VALUES (?,?,?,?)", rows)
    c.commit()
PY
echo '{"collectors": {"bn_tape_all": {"runner": "live_trade_tape_collector_20260916.py", "match": {"TAPE_SYMBOL": "ethusdt,btcusdt"}, "env": {"TAPE_SYMBOL": "ethusdt,btcusdt"}, "supervisor": "supervisor_trade_tape.sh"},
 "okx_tape_all": {"runner": "live_okx_trade_tape_collector_20260923.py", "match": {}, "env": {}, "supervisor": "x"}}}' > "$R/data/live/multicoin_collectors.json"
# 가짜 crontab
printf '#!/bin/bash\nif [ "$1" = -l ]; then cat "%s/crontab.txt"; else cat > "%s/crontab.txt"; fi\n' "$R" "$R" > "$R/fakecron"; chmod +x "$R/fakecron"
cat > "$R/crontab.txt" <<X
# 서버 crontab 흉내
@reboot sleep 30 && cd $R && bash dashboard/supervise_server.sh >> logs/x.log 2>&1 &
@reboot sleep 60 && cd $R && TAPE_SYMBOL=ethusdt,btcusdt PYTHON_BIN=bash bash scripts/ops/supervisor_trade_tape.sh >> logs/supervisor/tape_reboot.log 2>&1 &
@reboot sleep 61 && cd $R && TAPE_SYMBOL=ethusdt TAPE_MARKET=spot PYTHON_BIN=bash bash scripts/ops/supervisor_trade_tape.sh >> logs/supervisor/tape_reboot.log 2>&1 &
@reboot sleep 105 && cd $R && RTCOLLECT_BIN=$R/tmp/rtbin/rtcollect TAPE_SYMBOL=ethusdt,btcusdt TAPE_MARKET=futures bash scripts/ops/supervisor_rtcollect.sh binance_tape >> logs/supervisor/rtcollect_reboot.log 2>&1 &
5 11 * * * cd $R && python scripts/seal_to_lake.py >> logs/seal.log 2>&1
X
cat >> "$R/crontab.txt" <<X
@reboot sleep 50 && cd $R && nice -n 10 bash scripts/ops/supervisor_eth_iceberg.sh >> logs/supervisor/eth_iceberg_reboot.log 2>&1 &
@reboot sleep 55 && cd $R && nice -n 10 bash scripts/ops/supervisor_eth_iceberg.sh XRPUSDT >> logs/supervisor/xrp_iceberg_reboot.log 2>&1 &
X
ORIG="$(cat "$R/crontab.txt")"
T0=1763683200
{ echo "[$(date -d @$((T0 - 100)) -Iseconds)] SUPERVISOR starting rtcollect(binance_tape_ethusdt_btcusdt) (pid will follow)"
  echo "[$(date -d @$((T0 + 320)) -Iseconds)] SUPERVISOR starting rtcollect(binance_tape_ethusdt_btcusdt) (pid will follow)"; } > "$R/logs/supervisor/rtcollect_binance_tape_ethusdt_btcusdt_20251121.log"
echo "== 0. shadow_start.sh cron 이 아이스버그 종목을 감독 첫 인자에서 읽는다(서버 줄 그대로)"
mkdir -p "$R/bin"; ln -s "$R/fakecron" "$R/bin/crontab"
sed "s#^ROOT=.*#ROOT=$R#" "$SRC/rtcollect/tools/shadow_start.sh" > "$R/ss.sh"; touch "$R/tmp/rtbin/rtduck"; chmod +x "$R/tmp/rtbin/rtduck"
ICE="$(PATH="$R/bin:$PATH" bash "$R/ss.sh" cron | grep 'supervisor_rtcollect.sh iceberg' | grep -oE 'ICEBERG_SYMBOL=[A-Z]+' | tr '\n' ' ')"
ok "아이스버그 줄 = ETHUSDT 하나 · XRPUSDT 하나 ($ICE)" '[ "$ICE" = "ICEBERG_SYMBOL=ETHUSDT ICEBERG_SYMBOL=XRPUSDT " ]'
cd "$R" || exit 1
# 띄우기: 파이썬 선물(넘길 것) · 파이썬 현물(남아야 함) · 그림자 Rust 선물
env TAPE_SYMBOL=ethusdt,btcusdt PYTHON_BIN=bash setsid nohup bash scripts/ops/supervisor_trade_tape.sh > logs/py_fut.log 2>&1 < /dev/null &
sleep 1   # 감독끼리 같은 순간에 뜨면 서로의 중복 검사에 걸린다(원본 감독 스크립트의 성질)
env TAPE_SYMBOL=ethusdt TAPE_MARKET=spot PYTHON_BIN=bash setsid nohup bash scripts/ops/supervisor_trade_tape.sh > logs/py_spot.log 2>&1 < /dev/null &
sleep 1
env RTCOLLECT_BIN="$R/tmp/rtbin/rtcollect" TAPE_SYMBOL=ethusdt,btcusdt setsid nohup bash scripts/ops/supervisor_rtcollect.sh binance_tape > /dev/null 2>&1 < /dev/null &
sleep 3
envof() { tr '\0' '\n' 2>/dev/null < "/proc/$1/environ" | grep "^$2=" | cut -d= -f2-; }
pyp() { for p in /proc/[0-9]*; do tr '\0' '\n' 2>/dev/null < "$p/cmdline" | grep -qx "$R/scripts/live_trade_tape_collector_20260916.py" && ! tr '\0' '\n' 2>/dev/null < "$p/cmdline" | grep -q '_supervise.sh$' && [ "$(envof "${p#/proc/}" TAPE_MARKET)" = "$1" ] && echo "${p#/proc/}"; done; }
rsp() { for p in $(pgrep -x rtcollect); do tr '\0' ' ' < /proc/$p/cmdline | grep -qF "$R/" && [ "$(envof $p RTCOLLECT_TAKEOVER)" = "$1" ] && echo $p; done; }
export CRONTAB_CMD="$R/fakecron" CHECK_PY=python3 WATCH_S=6 TERM_WAIT_S=5
FUT0="$(pyp futures)"; SPOT0="$(pyp spot)"; SH0="$(rsp '')"
echo "시작: 파이썬 선물 $FUT0 · 현물 $SPOT0 · 그림자 $SH0"
ok "시작 상태(파이썬 둘·그림자 하나)" '[ -n "$FUT0" ] && [ -n "$SPOT0" ] && [ -n "$SH0" ]' || cat logs/py_fut.log logs/py_spot.log

echo "== 0b. check: 그림자 재기동 구간은 빼고 · 그림자가 안 덮는 종목은 불통과"
OUT="$(bash rtcollect/tools/takeover.sh check binance_tape)"; RC=$?; echo "$OUT"
ok "재기동 구간을 빼면 통과(파이썬에만 0)" '[ "$RC" = 0 ] && echo "$OUT" | grep -q "그림자 꺼진 구간" && echo "$OUT" | grep -q "파이썬에만 0"'
ok "목록: 파이썬 두 종목 전부 그림자가 뜬다" 'echo "$OUT" | grep -q "목록: 파이썬 TAPE_SYMBOL=ETHUSDT,BTCUSDT 전부"'
mkdir -p "$R/hold"; mv "$R"/logs/supervisor/rtcollect_binance_tape_* "$R/hold/"   # 감독 로그를 전부 치운다(오늘 것 포함)
OUT="$(bash rtcollect/tools/takeover.sh check binance_tape)"; RC=$?
ok "재기동 시각을 모르면(로그 없음) 그 빈 구간이 «파이썬에만» → 불통과" '[ "$RC" != 0 ] && echo "$OUT" | grep -q "파이썬에만 [1-9]"'
mv "$R"/hold/* "$R/logs/supervisor/"
OUT="$(bash rtcollect/tools/takeover.sh check binance_spot)"; RC=$?
ok "현물: 그림자 Rust 가 안 뜨면 목록 불통과" '[ "$RC" != 0 ] && echo "$OUT" | grep -q "그림자 Rust 가 안 뜨는 종목 ETHUSDT"'
echo "== 1. check 가 불일치를 잡으면 go 는 아무것도 안 바꾼다"
cp data/hot/rtcollect/binance_tape.sqlite good.sqlite; cp bad.sqlite data/hot/rtcollect/binance_tape.sqlite
bash rtcollect/tools/takeover.sh go binance_tape ETHUSDT; RC=$?
ok "go 종료코드 ≠ 0" '[ "$RC" != 0 ]'
ok "crontab 그대로" '[ "$(cat crontab.txt)" = "$ORIG" ]'
ok "파이썬·그림자 그대로" '[ "$(pyp futures)" = "$FUT0" ] && [ "$(rsp "")" = "$SH0" ]'
ok "백업 파일도 안 만듦" '[ -z "$(ls logs/supervisor/crontab_backup_* 2>/dev/null)" ]'
cp good.sqlite data/hot/rtcollect/binance_tape.sqlite

echo "== 2. go"
: > order.log
bash rtcollect/tools/takeover.sh go binance_tape ETHUSDT; RC=$?
ok "go 종료코드 0" '[ "$RC" = 0 ]'
NEWC="$(cat crontab.txt)"
ok "crontab 줄 수 +1 (8 → 9)" '[ "$(wc -l < crontab.txt)" = 9 ]'
ok "주석 2줄(파이썬 선물 + 그림자) · 앞 표지" '[ "$(grep -c "^#takeover:binance_tape_ETHUSDT_BTCUSDT @reboot" crontab.txt)" = 2 ]'
ok "현물·대시보드·seal 줄은 그대로" 'grep -q "^@reboot sleep 61 .*TAPE_MARKET=spot" crontab.txt && grep -q "^@reboot sleep 30 .*supervise_server" crontab.txt && grep -q "^5 11 \* \* \*" crontab.txt && [ "$(grep -c "^@reboot.*supervisor_eth_iceberg" crontab.txt)" = 2 ]'
ok "더한 Rust 줄 = TAKEOVER·진짜 경로·REST·끝 표지" 'tail -1 crontab.txt | grep -q "RTCOLLECT_TAKEOVER=1 .*TAPE_DB_PATH=data/hot/binance_tape.sqlite RTCOLLECT_REST=1 .*supervisor_rtcollect.sh binance_tape .* #takeover:binance_tape_ETHUSDT_BTCUSDT$"'
ok "백업 = 원래 crontab" '[ "$(cat logs/supervisor/crontab_backup_*_takeover_binance_tape_ETHUSDT_BTCUSDT.txt)" = "$ORIG" ]'
ok "파이썬 선물 꺼짐 · 감독이 먼저(parent_gone)" '[ -z "$(pyp futures)" ] && grep -qx "py_term futures parent_gone" order.log'
ok "파이썬 현물은 살아 있음" '[ "$(pyp spot)" = "$SPOT0" ]'
ok "순서: 파이썬 끔 → 그림자 끔 → 진짜 경로 시작" '[ "$(grep -oE "^(py_term|rs_term|rs_start)[^ ]* ?[a-z=0-9]*" order.log | head -3 | tr "\n" "|")" = "py_term futures|rs_term takeover=0|rs_start takeover=1|" ]'
ok "그림자 Rust 없음 · 진짜 경로 Rust 하나" '[ -z "$(rsp "")" ] && [ "$(rsp 1 | wc -l)" = 1 ]'
ok "Rust 가 받은 값(진짜 경로·REST·종목)" 'grep -qx "rs_start takeover=1 db=data/hot/binance_tape.sqlite rest=1 sym=ethusdt,btcusdt" order.log'
ok "매니페스트: 선물 항목만 taken_over" 'python3 -c "import json; m=json.load(open(\"data/live/multicoin_collectors.json\")); assert list(m[\"taken_over\"])==[\"bn_tape_all\"] and list(m[\"collectors\"])==[\"okx_tape_all\"]"'
ok "두 번째 go 는 거부(이미 넘김)" '! bash rtcollect/tools/takeover.sh go binance_tape > /dev/null 2>&1 && [ "$(cat crontab.txt)" = "$NEWC" ]'

echo "== 3. back"
: > order.log
bash rtcollect/tools/takeover.sh back binance_tape; RC=$?
ok "back 종료코드 0" '[ "$RC" = 0 ]'
ok "crontab = 원래 것" '[ "$(cat crontab.txt)" = "$ORIG" ]'
ok "진짜 경로 Rust 꺼짐(rs_term takeover=1)" '[ -z "$(rsp 1)" ] && grep -qx "rs_term takeover=1" order.log'
sleep 2
ok "파이썬 선물이 다시 떴다(새 pid)" 'p="$(pyp futures)"; [ -n "$p" ] && [ "$p" != "$FUT0" ]'
ok "매니페스트 되돌림" 'python3 -c "import json; m=json.load(open(\"data/live/multicoin_collectors.json\")); assert sorted(m[\"collectors\"])==[\"bn_tape_all\",\"okx_tape_all\"] and not m.get(\"taken_over\")"'
ok "파이썬 현물은 끝까지 그대로" '[ "$(pyp spot)" = "$SPOT0" ]'
echo "== 4. 느리게 죽는 파이썬: 기다림을 넘기면 스스로 되돌린다(crontab·매니페스트·감독 다시 · 쓰는 프로세스 0 으로 안 끝남)"
env RTCOLLECT_BIN="$R/tmp/rtbin/rtcollect" TAPE_SYMBOL=ethusdt,btcusdt setsid nohup bash scripts/ops/supervisor_rtcollect.sh binance_tape > /dev/null 2>&1 < /dev/null &
sleep 3; echo 20 > "$R/slow"; : > order.log
FUT1="$(pyp futures)"
G0=$SECONDS; OUT="$(TERM_WAIT_S=12 bash rtcollect/tools/takeover.sh go binance_tape 2>&1)"; RC=$?; GW=$((SECONDS - G0)); echo "$OUT" | grep -vE '^  (trade_tape|목록|그림자 감독)'
ok "종료코드 8(되돌림)" '[ "$RC" = 8 ]'
ok "«12초 뒤에도 산다»는 실제로 12초 넘게 기다린 뒤에만(go 전체 ${GW}초)" '[ "$GW" -ge 12 ] && echo "$OUT" | grep -q "SIGTERM 1[2-4]초 뒤에도 산다(기다림 한도 12초"'
ok "기다리는 동안 «살아 있음 · mtime 전진» 을 찍음" 'echo "$OUT" | grep -q "기다림 10초: .*살아 있음 · 파일 mtime 전진"'
ok "crontab = 원래 것" '[ "$(cat crontab.txt)" = "$ORIG" ]'
ok "매니페스트 되돌림" 'python3 -c "import json; m=json.load(open(\"data/live/multicoin_collectors.json\")); assert sorted(m[\"collectors\"])==[\"bn_tape_all\",\"okx_tape_all\"] and not m.get(\"taken_over\")"'
ok "Rust 진짜 경로는 안 띄움(수집기가 혼자 쓰는 동안 단일 writer)" '[ -z "$(rsp 1)" ] && ! grep -q "rs_start takeover=1" order.log'
ok "끝에 «지금 쓰는 프로세스» 줄 · 파이썬 1(혼자 사는 수집기)" 'echo "$OUT" | grep -q "지금 쓰는 프로세스(binance_tape_ETHUSDT_BTCUSDT): 파이썬 1"'
rm -f "$R/slow"
for _ in $(seq 30); do p="$(pyp futures)"; [ -n "$p" ] && [ "$p" != "$FUT1" ] && break; sleep 1; done
ok "수집기가 끝난 뒤 지킴이가 감독을 다시 띄움(새 파이썬 pid)" 'p="$(pyp futures)"; [ -n "$p" ] && [ "$p" != "$FUT1" ] && grep -q "py_exit futures" order.log'
ok "그림자는 그대로(④ 전에 멈춤)" '[ -n "$(rsp "")" ]'
echo "== 4b. 40초 걸려 죽는 파이썬 + 기본 기다림(120초) → 넘김 성공 · 실제 기다린 초를 찍음"
sleep 3; echo 40 > "$R/slow"; : > order.log
G0=$SECONDS; OUT="$(TERM_WAIT_S=120 bash rtcollect/tools/takeover.sh go binance_tape 2>&1)"; RC=$?; GW=$((SECONDS - G0))
echo "$OUT" | grep -E "끔:|기다림|go binance"
ok "go 종료코드 0 · 40초쯤 기다림(go 전체 ${GW}초)" '[ "$RC" = 0 ] && [ "$GW" -ge 40 ] && echo "$OUT" | grep -qE "수집기 pid [0-9]+ \(SIGTERM 뒤 (39|4[0-9])초\)"'
ok "기다리는 동안 10초마다 «살아 있음»(30초 줄 있음)" 'echo "$OUT" | grep -q "기다림 30초: .*살아 있음"'
rm -f "$R/slow"
bash rtcollect/tools/takeover.sh back binance_tape > /dev/null 2>&1
ok "back 뒤 crontab 원래대로" '[ "$(cat crontab.txt)" = "$ORIG" ]'
sleep 3
echo "== 5. 바이낸스 가중치 머리 읽기(binance_ctx go 의 넘기기 전·뒤 단계) -- 127.0.0.1 가짜 서버"
cat > "$R/wsrv.py" <<'X'
import http.server, sys
seq = iter([100, 250, 180, 90, 90, 90])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("X-MBX-USED-WEIGHT-1M", str(next(seq))); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"{}")
    def log_message(self, *a): pass
s = http.server.HTTPServer(("127.0.0.1", 0), H); print(s.server_address[1], flush=True); s.serve_forever()
X
python3 "$R/wsrv.py" > "$R/wport" & sleep 1
WP="$(cat "$R/wport")"
W="$(WEIGHT_URL="http://127.0.0.1:$WP/fapi/v1/time" WEIGHT_N=3 WEIGHT_GAP=0 bash rtcollect/tools/takeover.sh weight binance_ctx 2>/dev/null | tr '\n' '|')"
ok "가중치 세 번 = 100 250 180 · 중앙 180 ($W)" '[ "$W" = "100 250 180|중앙 180|" ]'
J() { bash rtcollect/tools/takeover.sh weight-judge binance_ctx "" "$1" "$2" > /dev/null 2>&1; }
ok "11-22 실측(뒤 다섯 중 하나만 튐 1382) → 통과" 'J "1095 1067 1078 1059 1032" "1382 1020 1140 1098 1100"'
ok "다섯 번 다 +300 → 되돌림" '! J "1095 1067 1078 1059 1032" "1395 1367 1378 1359 1332"'
ok "두 번 연속 +200(중앙은 그대로) → 되돌림" '! J "1095 1067 1078 1059 1032" "1100 1300 1300 1090 1080"'
echo "$(( $(date +%s) + 600 ))" > data/live/binance_ban_until
WEIGHT_URL="http://127.0.0.1:$WP/fapi/v1/time" WEIGHT_N=1 bash rtcollect/tools/takeover.sh weight binance_ctx > /dev/null 2>&1; RC=$?
ok "밴 파일이 살아 있으면 부르지 않고 실패" '[ "$RC" != 0 ]'
rm -f data/live/binance_ban_until
echo "== 6. check bookdepth·ob5m = 재생 대조 결과(2시간 안이면 다시 안 돌림)"
printf '  bars: 비교 38행 · 불일치 0\n요약: 견준 38 · 설명 못 함 0\n' > logs/supervisor/replay_book_bookdepth_ETHUSDT.txt
OUT="$(bash rtcollect/tools/takeover.sh check bookdepth 2>&1)"; RC=$?
ok "재생 결과 불일치 0 → 통과 · 다시 안 돌림" '[ "$RC" = 0 ] && echo "$OUT" | grep -q "2시간 안 결과를 다시 쓴다" && echo "$OUT" | grep -q "견준 38 · 설명 못 함 0 · 통과"'
printf '요약: 견준 38 · 설명 못 함 2\n' > logs/supervisor/replay_book_ob5m_BTCUSDT.txt
bash rtcollect/tools/takeover.sh check ob5m BTCUSDT > /dev/null 2>&1; RC=$?
ok "재생 결과 불일치 있으면 불통과" '[ "$RC" != 0 ]'
echo "== 7. raster_archive: 넘긴 래스터 표지가 있어야 check 통과"
bash rtcollect/tools/takeover.sh check raster_archive > /dev/null 2>&1; RC=$?
ok "넘긴 래스터 없으면 불통과" '[ "$RC" != 0 ]'
cp crontab.txt crontab.keep; echo "@reboot sleep 100 && cd $R && x #takeover:raster_ETHUSDT" >> crontab.txt
OUT="$(bash rtcollect/tools/takeover.sh check raster_archive 2>&1)"; RC=$?
ok "넘긴 래스터 ETHUSDT 가 보이면 통과" '[ "$RC" = 0 ] && echo "$OUT" | grep -q "넘긴 래스터 종목 ETHUSDT"'
mv crontab.keep crontab.txt
[ "$FAIL" = 0 ] && echo "takeover selftest OK" || { echo "takeover selftest 실패"; exit 1; }
