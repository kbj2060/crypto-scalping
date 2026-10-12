"""아침 운영 감사관 (2026-10-12, 사용자 «7번 진행»): 사실 수집(읽기 전용) → Claude 판단(도구 없음) → 텔레그램.

왜: 이 저장소의 조용한 고장(aggTrade 경로 3주·손절 4일·푸시 0건·텔레그램 토큰 1년·뉴스 판정 12시간)은 전부 «값이 그럴듯해서»
임계 감시를 통과했다. 감시기(ops_watchdog)는 항목별 임계만 본다 -- 이 감사관은 하루 한 번 여러 원천을 한 장에 모아 «사람처럼» 읽는다.
안전: 수집은 전부 읽기(바이낸스는 GET /fapi/v1/openAlgoOrders 하나뿐, 주문·취소 호출 없음). Claude 는 `--tools ""` 로 도구가 없어
서버를 건드릴 수 없다 -- 받은 사실 JSON 을 읽고 문장만 쓴다. Claude 가 실패하면 결정적 요약을 대신 보낸다.

cron(서버 = KST): 30 8 * * * cd ~/crypto-scalping && nice -n 10 <python> scripts/ops/morning_audit_20261012.py >> logs/morning_audit.log 2>&1
  --dry-run  보내지 않고 출력만 · --no-llm  판단 없이 사실 요약만 · --selftest
"""
from __future__ import annotations

import asyncio
import html
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]   # live_manual_peg_execute 가 `scripts.x` 꼴로 import 한다
import ops_watchdog as W  # noqa: E402  (.env 로드 · send_telegram · KST)

OUT = ROOT / "data" / "live" / "ops_watchdog" / "morning_audit"
DASH = "http://127.0.0.1:8787"
CLAUDE = Path(os.getenv("AUDIT_CLAUDE_BIN", str(Path.home() / ".local" / "bin" / "claude")))
# crontab 에 있어야 하는 줄(부분 문자열). 지우거나 옮기면 여기도 고친다.
EXPECTED_CRON = {
    "전진 기록기(3자산)": "live_eth_direction_forward_recorder_20261010.py --record",
    "max pain 장부 pre": "live_eth_maxpain_oos_ledger_20261007.py pre",
    "max pain 장부 settle": "live_eth_maxpain_oos_ledger_20261007.py settle",
    "lake 봉인": "seal_to_lake.py",
    "대체데이터(공지·펀딩 스프레드)": "run_f4c_altdata_collector",
    "청산지도 채점": "live_liqmap_scorer_20261008.py",
    "배포 워처": "deploy_watcher.sh",
}
# 하루 한 번 이상 써야 하는 산출물(경로, 최대 나이 시간)
EXPECTED_FILES = {
    "전진 기록기 장부": ("data/live/eth_direction_forward.jsonl", 26),
    "max pain 장부": ("data/live/maxpain_oos_ledger.jsonl", 26),
    "청산지도 채점": ("data/live/liqmap_score.jsonl", 3),
}


def safe(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 -- 한 원천 실패가 감사 전체를 막으면 안 된다
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}


def get_json(path: str, timeout: float = 15) -> dict:
    with urllib.request.urlopen(DASH + path, timeout=timeout) as r:
        return json.loads(r.read())


# ── 순수 판정 함수(자체점검 대상) ─────────────────────────────────────────────────────────────
def position_flags(p: dict, equity: float, algos: list[dict]) -> dict:
    """포지션 한 줄 → 노출·청산거리·스톱 유무·스톱 시 손실(순자산 %)."""
    side = 1 if str(p.get("side")).upper() == "LONG" else -1
    entry, mark, qty = float(p["entry_price"]), float(p["mark_price"]), abs(float(p["qty"]))
    liq = float(p.get("liquidation_price") or 0)
    mine = [a for a in algos if str(a.get("positionSide", "")).upper() == str(p.get("side")).upper()
            and str(a.get("orderType", "")).upper().startswith("STOP")]
    stop = float(mine[0]["triggerPrice"]) if mine else None
    out = {"symbol": p["symbol"], "side": p["side"], "notional": round(qty * mark, 2),
           "notional_x_equity": round(qty * mark / equity, 2) if equity > 0 else None,
           "liq_distance_pct": round(abs(mark - liq) / mark * 100, 2) if liq > 0 else None,
           "stop": stop, "stop_count": len(mine)}
    if stop is not None:
        loss = max(0.0, (entry - stop) * side) * qty
        out["loss_at_stop_pct_equity"] = round(loss / equity * 100, 1) if equity > 0 else None
        out["stop_beyond_liq"] = bool(liq > 0 and (stop - liq) * side < 0)          # 스톱이 청산가보다 바깥 = 스톱이 안 걸린다
    return out


def apt_events(text: str, now: datetime, hours: float = 36) -> list[dict]:
    """/var/log/apt/history.log → 최근 hours 안의 설치·업그레이드(엔비디아 패키지 표시). 10-11 드라이버 자동 갱신이 뉴스 판정을 12시간 멈췄다."""
    out, start = [], None
    for line in text.splitlines():
        if line.startswith("Start-Date:"):
            start = datetime.strptime(line.split(":", 1)[1].strip(), "%Y-%m-%d  %H:%M:%S")
        elif start and (line.startswith("Upgrade:") or line.startswith("Install:")) and now - start <= timedelta(hours=hours):
            pk = re.findall(r"([a-z0-9][\w.+-]*):", line.split(":", 1)[1])
            out.append({"at": start.isoformat(sep=" "), "kind": line.split(":")[0], "n": len(pk),
                        "nvidia": sorted({x for x in pk if "nvidia" in x})[:6]})
    return out


def verdict_flags(items: list[dict], today: datetime) -> dict:
    d0 = today.date()
    due = [i for i in items if not i.get("done") and 0 <= (datetime.fromisoformat(i["date"]).date() - d0).days <= 7]
    over = [i for i in items if not i.get("done") and (datetime.fromisoformat(i["date"]).date() - d0).days < 0]
    f = lambda xs: [{"id": i["id"], "date": i["date"], "title": i["title"]} for i in xs]  # noqa: E731
    return {"due_7d": f(due), "overdue_not_done": f(over)}


# ── 수집 ──────────────────────────────────────────────────────────────────────────────────
def facts_watchdog() -> dict:
    snap = json.loads((W.OUT / "health_snapshot.json").read_text())
    bad = [{"component": c["component"], "status": c["status"], "summary": c["summary"]} for c in snap["checks"] if c["status"] != "OK"]
    since = (W.now_kst() - timedelta(hours=24)).isoformat()
    with sqlite3.connect(f"file:{W.OUT / 'incidents.sqlite'}?mode=ro", uri=True) as con:
        inc = con.execute("SELECT component, status, notification_kind, count(*) FROM incidents WHERE observed_at_kst >= ? "
                          "GROUP BY 1, 2, 3 ORDER BY 4 DESC LIMIT 20", [since]).fetchall()
    return {"snapshot_at": snap.get("updated_at_kst"), "checks": len(snap["checks"]), "not_ok": bad,
            "incidents_24h": [{"component": a, "status": b, "kind": k, "n": n} for a, b, k, n in inc]}


def facts_deploy() -> dict:
    g = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()  # noqa: E731
    log = ROOT / "logs" / "deploy_watcher_cron.log"
    tail = log.read_text(errors="replace").splitlines()[-3:] if log.exists() else []
    tracked_dirty = [x for x in g("status", "--porcelain", "--untracked-files=no").splitlines() if x]
    return {"head": g("rev-parse", "--short=8", "HEAD"), "origin_main": g("rev-parse", "--short=8", "origin/main"),
            "watcher_tail": tail, "tracked_dirty": len(tracked_dirty), "tracked_dirty_sample": tracked_dirty[:5]}


def facts_disk() -> dict:
    out = {}
    for name, p in (("root", "/"), ("D_drive", "/mnt/d")):
        if Path(p).exists():
            u = shutil.disk_usage(p)
            out[name] = {"free_gb": round(u.free / 2**30, 1), "used_pct": round(u.used / u.total * 100, 1)}
    return out


def facts_apt() -> list[dict]:
    p = Path("/var/log/apt/history.log")
    return apt_events(p.read_text(errors="replace"), datetime.now()) if p.exists() else []


async def _algos(symbols: list[str]) -> dict:
    import aiohttp
    import live_manual_peg_execute_20260912 as M
    key, sec = os.getenv("BINANCE_API_KEY", ""), os.getenv("BINANCE_SECRET_KEY", "")
    if not (key and sec):
        return {"error": "API 키 없음"}
    out = {}
    async with aiohttp.ClientSession() as s:
        off = await M._clock_offset(s)
        for sym in symbols:                                  # 🔴읽기 전용 GET 하나 -- 주문·취소 경로를 부르지 않는다
            r = await M.signed(s, "GET", "/fapi/v1/openAlgoOrders", {"symbol": sym}, key, sec, off)
            out[sym] = r if isinstance(r, list) else {"error": str(r)[:200]}
    return out


def facts_account() -> dict:
    a = get_json("/api/binance-account")
    b = a.get("balance") or 0
    equity = float(b.get("margin") or 0) if isinstance(b, dict) else float(b)   # margin = 지갑 + 미실현 = 순자산
    pos = [p for p in a.get("positions") or [] if abs(float(p.get("qty") or 0)) > 0]
    algos = asyncio.run(_algos(sorted({p["symbol"] for p in pos}))) if pos else {}
    rows = [position_flags(p, equity, algos.get(p["symbol"]) if isinstance(algos.get(p["symbol"]), list) else []) for p in pos]
    return {"equity": round(equity, 2), "positions": rows, "algo_errors": {k: v for k, v in algos.items() if isinstance(v, dict)}}


def facts_cron() -> dict:
    tab = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=10).stdout
    live = "\n".join(x for x in tab.splitlines() if x.strip() and not x.lstrip().startswith("#"))
    return {"missing": [k for k, v in EXPECTED_CRON.items() if v not in live]}


def facts_files() -> dict:
    out = {}
    for name, (rel, max_h) in EXPECTED_FILES.items():
        p = ROOT / rel
        age = (time.time() - p.stat().st_mtime) / 3600 if p.exists() else None
        out[name] = {"age_h": None if age is None else round(age, 1), "max_h": max_h, "stale": age is None or age > max_h}
    return out


def facts_paper() -> dict:
    d = get_json("/api/paper-arms")
    return {"available": d.get("available"), "age_s": d.get("age_s"), "stale": d.get("stale")}


def gather() -> dict:
    now = W.now_kst()
    cal = json.loads((ROOT / "dashboard" / "verdict_calendar.json").read_text())
    return {"generated_at_kst": now.isoformat(timespec="seconds"),
            "watchdog": safe(facts_watchdog), "deploy": safe(facts_deploy), "disk": safe(facts_disk),
            "apt_36h": safe(facts_apt), "account": safe(facts_account), "cron": safe(facts_cron),
            "outputs": safe(facts_files), "paper_engine": safe(facts_paper),
            "verdicts": safe(lambda: verdict_flags(cal["items"], now.replace(tzinfo=None)))}


# ── 판단·전송 ──────────────────────────────────────────────────────────────────────────────
PROMPT = """너는 개인 크립토 트레이딩 서버의 아침 운영 감사관이다. 아래 JSON 은 방금 스크립트가 읽기 전용으로 모은 사실이다.
규칙: 한국어. 12줄 이하. JSON 에 없는 것은 추측하지 마라. 매매 방향·시장 전망은 말하지 마라.
형식: 첫 줄 = 한 줄 결론(이상 없음/확인 필요/즉시 조치). 그다음 «🔴 즉시» «🟠 확인» «🟢 정상(한 줄 요약)» 순서로, 해당 없는 칸은 생략.
특히 볼 것: 감시기 not_ok·24h 사고 반복 · origin_main≠head · tracked_dirty · 디스크(D_drive free_gb<30 위험) · apt 의 nvidia 갱신 ·
포지션: stop_count=0(스톱 없음) · stop_beyond_liq · loss_at_stop_pct_equity>20 · notional_x_equity>6 · cron missing · outputs stale · paper stale · 판정 7일 안/지난 판정.
각 항목에 «무엇을 보면 되는지»를 짧게 붙여라.

사실(JSON):
"""


def judge(facts: dict) -> str | None:
    if not CLAUDE.exists():
        return None
    r = subprocess.run([str(CLAUDE), "-p", PROMPT + json.dumps(facts, ensure_ascii=False, default=str), "--tools", ""],
                       capture_output=True, text=True, timeout=240, cwd="/tmp")
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def fallback(f: dict) -> str:
    """Claude 없이 결정적 요약 -- 판단이 실패해도 사실은 간다."""
    L = []
    w = f["watchdog"] if "error" not in f["watchdog"] else {"not_ok": [{"component": "watchdog_read", "status": "ERR", "summary": f["watchdog"]["error"]}]}
    L += [f"감시기 이상 {len(w.get('not_ok', []))}건: " + ", ".join(f"{x['component']}={x['status']}" for x in w.get("not_ok", [])[:6])]
    for p in (f["account"].get("positions", []) if isinstance(f["account"], dict) else []):
        L.append(f"포지션 {p['symbol']} {p['side']} 명목 {p['notional']}(순자산 {p['notional_x_equity']}배) 스톱 {p['stop']} "
                 f"스톱 시 손실 {p.get('loss_at_stop_pct_equity')}% · 청산까지 {p['liq_distance_pct']}%")
    if isinstance(f["cron"], dict) and f["cron"].get("missing"):
        L.append("cron 빠짐: " + ", ".join(f["cron"]["missing"]))
    st = [k for k, v in (f["outputs"] if isinstance(f["outputs"], dict) else {}).items() if isinstance(v, dict) and v.get("stale")]
    if st:
        L.append("산출물 멈춤: " + ", ".join(st))
    v = f["verdicts"] if isinstance(f["verdicts"], dict) else {}
    if v.get("due_7d"):
        L.append("7일 안 판정: " + ", ".join(f"{x['id']}({x['date']})" for x in v["due_7d"]))
    return "\n".join(L)


def main() -> None:
    a = sys.argv[1:]
    if "--selftest" in a:
        return selftest()
    facts = gather()
    text = None if "--no-llm" in a else safe(lambda: judge(facts))
    body = text if isinstance(text, str) else fallback(facts)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = W.now_kst().strftime("%Y%m%d")
    (OUT / f"{stamp}.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1, default=str))
    (OUT / f"{stamp}.md").write_text(body + "\n")
    msg = f"🌅 <b>아침 운영 감사 {W.now_kst():%m-%d}</b>{'' if isinstance(text, str) else ' (Claude 없이 요약)'}\n" + html.escape(body)
    print(msg)
    if "--dry-run" not in a:
        print("telegram:", W.send_telegram(msg[:4000]))


def selftest() -> None:
    algos = [{"positionSide": "LONG", "orderType": "STOP_MARKET", "triggerPrice": "2485.55"}]
    p = {"symbol": "ETHUSDC", "side": "LONG", "qty": 5.358, "entry_price": 2552.47, "mark_price": 2533.0, "liquidation_price": 2261.27}
    f = position_flags(p, 1450.0, algos)
    assert f["stop_count"] == 1 and f["stop"] == 2485.55 and not f["stop_beyond_liq"], f
    assert abs(f["loss_at_stop_pct_equity"] - (2552.47 - 2485.55) * 5.358 / 1450 * 100) < 0.1, f
    assert f["notional_x_equity"] == round(5.358 * 2533 / 1450, 2)
    g = position_flags(p, 1450.0, [])                                            # 스톱 없음
    assert g["stop_count"] == 0 and "loss_at_stop_pct_equity" not in g
    s = position_flags({**p, "side": "SHORT", "liquidation_price": 2800}, 1450.0,
                       [{"positionSide": "SHORT", "orderType": "STOP_MARKET", "triggerPrice": "2900"}])
    assert s["stop_beyond_liq"] is True, s                                       # 숏 스톱 2900 > 청산 2800 = 스톱 전에 청산
    log = ("Start-Date: 2026-10-11  06:17:01\nUpgrade: libnvidia-compute-580:amd64 (580.173.02, 580.178.04), curl:amd64 (1, 2)\n"
           "Start-Date: 2026-09-01  06:00:00\nUpgrade: old:amd64 (1, 2)\n")
    ev = apt_events(log, datetime(2026, 10, 12, 8, 30))
    assert len(ev) == 1 and ev[0]["nvidia"] == ["libnvidia-compute-580"] and ev[0]["n"] == 2, ev
    items = [{"id": "a", "date": "2026-10-15", "title": "x"}, {"id": "b", "date": "2026-10-01", "title": "y"},
             {"id": "c", "date": "2026-10-01", "title": "z", "done": "ok"}, {"id": "d", "date": "2026-12-01", "title": "w"}]
    v = verdict_flags(items, datetime(2026, 10, 12))
    assert [x["id"] for x in v["due_7d"]] == ["a"] and [x["id"] for x in v["overdue_not_done"]] == ["b"], v
    txt = fallback({"watchdog": {"not_ok": []}, "account": {"positions": [f]}, "cron": {"missing": ["x"]},
                    "outputs": {"y": {"stale": True}}, "verdicts": {"due_7d": [{"id": "a", "date": "2026-10-15"}]}})
    assert "cron 빠짐: x" in txt and "산출물 멈춤: y" in txt and "a(2026-10-15)" in txt, txt
    print("selftest ok")


if __name__ == "__main__":
    main()
