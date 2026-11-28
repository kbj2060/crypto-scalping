#!/usr/bin/env python3
"""지지·저항 «가까워짐» 텔레그램 워커 (2026-11-04 사용자 «닿는 게 아니라 가까워지면 알림»).

2026-11-28 대시보드 밖으로 옮김(rtserve 전환 준비 -- 보내는 쪽은 늘 하나): 대시보드 안에 있던 sr_break_loop(뚫림 텔레그램 ·
/api/sr-watch 보기)와 chip_score_loop(청산 급증 배지 채점 jsonl)이 이제 이 워커에서 돈다. 값·문구·간격(10초)은 원본 그대로.
  레벨 = 응답 서버 /api/liquidation-map(원본 load_liquidation_map 그대로) · 1초 가격 = /api/supply-1s seconds(원본 footprint_state["sec"] 그대로)
  · 급증·미드 = /api/market-context burst·mid(원본 micro_state liq60·mid 그대로). 응답 서버 = 8787, rtserve 전환 표식(data/live/rtserve_live.on)이면 8790.
  보기는 data/live/sr_watch.json 에 쓴다 -- 파이썬 /api/sr-watch 와 나중의 rtserve 는 그 파일을 읽기만.
  ponytail: supply-1s 는 진행 중인 초를 빼고 가격을 소수 3자리로 준다 -- 원본보다 최대 1초 늦고 XRP(4자리)는 0.0005 안 오차. 필요하면 partial 도 읽는다.
«가까워짐»(아래 decide)은 그 보기(화면의 지지1~3·저항1~3 = srLevelsLive 와 같은 목록 + 1초 마지막 체결가)를 그대로 쓴다.

판정(코인마다 기준거리 THR_BP): 레벨↔현재가 거리 ≤ 기준 이고 «무장» 상태면 한 번 알리고 무장을 푼다.
무장 = 거리가 기준의 REARM_X 배 이상으로 멀어졌을 때(다시 다가오면 또 알림) · 같은 레벨은 COOLDOWN_S 안에 두 번 안 보낸다.
처음 본 레벨이 이미 기준 안이면 알리지 않는다(재시작·한 시간마다 바뀌는 청산맵 때문에 알림이 쏟아지지 않게) -- «멀리 있다가 다가온» 것만.
⚠️ 알림은 매매 트리거가 아니다 -- 문구는 서술형.

기준 = 그 코인 1시간 고저폭 중앙값(11-04 5분봉 200개 실측 ETH 58 · BTC 46 · SOL 75 · XRP 69 · SNDK 134 · SOXL 272bp)의 약 1/3.
ponytail: 상태는 메모리(재시작 직후 첫 폴링은 기준만 잡는다) · 기준은 상수표 -- 변동성 국면이 바뀌면 THR_BP 를 고친다.

실행: python scripts/live_sr_proximity_alert_20261104.py   (supervisor: scripts/ops/supervisor_sr_proximity.sh · crontab @reboot)
자체 시험: python -m pytest -q test/test_sr_proximity_20261104.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SR_WATCH_PATH = REPO_ROOT / "data/live/sr_watch.json"
POLL_S = 10
STALE_S = 180          # 1초 마지막 체결이 이보다 오래됐으면 그 코인은 건너뛴다(멈춘 가격으로 «가까움» 을 말하지 않는다)
REARM_X = 2.0
COOLDOWN_S = 30 * 60
FORGET_S = 2 * 3600    # 이 시간 안 보인 레벨 상태는 버린다
THR_BP = {"eth": 20, "btc": 15, "sol": 25, "xrp": 25, "sndk": 45, "soxl": 90,
          "skhynix": 45, "mu": 35}   # 2026-11-13 같은 규칙: vision 1시간봉 10-12~11-11 평일 고저폭 중앙 SKHYNIX 131 · MU 103bp 의 ~1/3 (같은 계산 SNDK 135 = 11-04 실측 134 재현)
DEFAULT_THR_BP = 25


def decide(state: dict, view: dict, now: float) -> list[dict]:
    """한 폴링의 판정(순수 함수 -- state 를 갱신한다). view = /api/sr-watch 의 assets. 반환 = 보낼 알림 [{asset, name, level, px, dist_bp, thr_bp, dp, key}]."""
    out = []
    for asset, v in view.items():
        px, t = float(v.get("px") or 0), float(v.get("t") or 0)
        if px <= 0 or now - t > STALE_S:
            continue
        thr = THR_BP.get(asset, DEFAULT_THR_BP)
        for kind, name, levels in (("sup", "지지", v.get("sup") or []), ("res", "저항", v.get("res") or [])):
            for i, lv in enumerate(levels):
                lv = float(lv)
                dist = abs(lv / px - 1) * 1e4
                k = (asset, kind, round(lv, 8))
                s = state.get(k)
                if s is None:   # 처음 본 레벨: 이미 가까우면 무장 안 함(알림 폭주 방지)
                    s = state[k] = {"armed": dist > thr, "last": 0.0, "seen": now}
                s["seen"] = now
                if dist >= REARM_X * thr:
                    s["armed"] = True
                if dist <= thr and s["armed"] and now - s["last"] >= COOLDOWN_S:
                    s.update(armed=False, last=now)
                    out.append({"asset": asset, "name": f"{name}{i + 1}", "level": lv, "px": px, "dist_bp": dist, "thr_bp": thr,
                                "dp": int(v.get("dp") or 2), "key": k})
    for k in [k for k, s in state.items() if now - s["seen"] > FORGET_S]:
        del state[k]
    return out


def message(asset: str, alerts: list[dict]) -> str:
    dp, px = alerts[0]["dp"], alerts[0]["px"]
    lines = [f"{'🔻' if a['name'].startswith('지지') else '🔺'} <b>{asset.upper()} {a['name']} 가까워짐</b> {a['level']:,.{dp}f} "
             f"({(a['level'] / px - 1) * 100:+.2f}% · 기준 {a['thr_bp'] / 100:.2f}%)" for a in alerts]
    return "\n".join(lines + [f"지금 {px:,.{dp}f}"])


def get(path: str) -> dict:
    """응답 서버(파이썬 8787 · rtserve 전환 표식이 있으면 8790 -- 매번 다시 본다)의 JSON."""
    port = 8790 if (REPO_ROOT / "data/live/rtserve_live.on").exists() else 8787
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=8) as r:
        return json.loads(r.read())


def sr_tick(srv, assets: list[str], st: dict, watch: dict, get=get, send=None) -> None:
    """원본 server.py sr_break_loop 의 한 바퀴(2026-10-25 «지지와 저항라인들을 뚫었을 때마다 텔레그램»). 줄마다 원본 그대로 --
    한 레벨은 지도 시각(밀도 이력 마지막 열의 정시)이 바뀔 때까지 한 번만 · 재시작 직후 첫 바퀴는 기준만 잡는다."""
    for asset in assets:
        try:
            m = get(f"/api/liquidation-map?asset={asset}")
            hh = m.get("heatmap_history") or []
            if not m.get("warmed_up") or not hh:
                continue
            ku = str(hh[-1].get("ts_utc"))[:13]
            s = st.get(asset)
            sec = get(f"/api/supply-1s?asset={asset}&since={s['sec'] if s else 0}").get("seconds") or []
            pts = [(int(r[0]), float(r[7])) for r in sec if r[7] and r[0] > (s["sec"] if s else 0)]   # r = [초, …, 가격] = 원본 (초, c[6])
            if not pts:
                continue
            if s is None or s["ku"] != ku:
                st[asset] = s = {"ku": ku, "broken": set(), "sec": (s or {}).get("sec", 0), "px": (s or {}).get("px")}
            prev = s["px"]
            s.update(sec=pts[-1][0], px=pts[-1][1])
            if prev is None:
                continue
            lo, hi = min(p for _, p in pts), max(p for _, p in pts)
            sup, res = [float(lv["price"]) for lv in m.get("support_levels") or []], [float(lv["price"]) for lv in m.get("resistance_levels") or []]
            hits = srv.sr_breaks(sup, res, prev, lo, hi, s["broken"])
            dp = srv.FLOW_SPECS[asset].price_dp
            watch[asset] = {"px": s["px"], "t": s["sec"], "dp": dp, "ku": ku, **srv.sr_watch_view(sup, res, s["px"], s["broken"])}
            if not hits:
                continue
            px = s["px"]
            lines = [f"{'🔻' if name.startswith('지지') else '🔺'} <b>{asset.upper()} {name} 뚫림</b> {lv:,.{dp}f}"
                     + (f" · 다음 {name[:2]} {nxt:,.{dp}f} ({(nxt / px - 1) * 100:+.2f}%)" if nxt else "")
                     for name, lv, nxt in hits]
            msg = "\n".join(lines + [f"지금 {px:,.{dp}f}"])
            sent = send(msg)
            print(f"sr break {asset}: {[h[0] for h in hits]} sent={sent}", flush=True)
        except Exception as exc:  # noqa: BLE001 -- 알림 하나로 워커가 죽으면 안 된다
            print(f"sr break {asset}: {type(exc).__name__}: {exc}", flush=True)
    tmp = SR_WATCH_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(watch, ensure_ascii=False), encoding="utf-8")
    tmp.replace(SR_WATCH_PATH)


def chip_tick(srv, cs: dict, now: float, get=get) -> None:
    """원본 server.py chip_score_loop 의 한 바퀴(2026-10-05 머리 칩 «실전 성적» -- 청산 급증 배지가 켜진 순간 → 15분 뒤 되돌림 bp).
    ponytail: 대기 중(15분 되돌림)은 메모리에만 -- 재기동하면 그 몇 건은 버린다(원본과 같다)."""
    burst_on, pend_burst, rows = cs.setdefault("burst_on", {}), cs.setdefault("pend", []), []
    try:
        mc = get("/api/market-context?asset=eth")
        if not mc.get("available") or mc.get("stale"):
            return                  # 응답 서버가 맥락을 못 만든 바퀴는 건너뛴다(묵은 값으로 켜짐을 판정하지 않는다)
        b = mc.get("burst") or {}   # 원본 micro_state 가 비면 burst 가 None -- 그때 원본은 liq·mid 둘 다 빈 값
        liq, mid = ({"long": b.get("long_usd_60s"), "short": b.get("short_usd_60s")}, mc.get("mid")) if b else ({}, None)
        for i, side in enumerate(("long", "short")):
            on = float(liq.get(side) or 0) > srv.LIQ_BURST_60S_USD[i]     # 배지와 같은 판정(mcLiqBurstSide)
            if on and not burst_on.get(side) and mid:
                pend_burst.append({"side": side, "t": now, "mid0": float(mid), "usd": float(liq.get(side) or 0)})
            burst_on[side] = on
        for b in [b for b in pend_burst if now >= b["t"] + 900]:
            pend_burst.remove(b)
            if mid:
                sgn = 1.0 if b["side"] == "long" else -1.0              # 롱 청산(가격 하락) 뒤 되돌림 = 위로
                rows.append({"k": "burst", "a": "eth", "t": int(b["t"]), "side": b["side"], "usd": round(b["usd"]),
                             "rev_bp": round((float(mid) / b["mid0"] - 1) * 1e4 * sgn, 2)})
        if rows:
            with open(srv.chip_score_path(now), "a", encoding="utf-8") as fh:
                fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    except Exception as exc:  # noqa: BLE001 -- 기록기 하나로 워커가 죽으면 안 된다
        print(f"chip score: {type(exc).__name__}: {exc}", flush=True)


def run() -> None:
    from scripts.ops_watchdog import send_telegram   # .env(TELEGRAM_BOT_TOKEN·CHAT_ID)를 읽는다
    from dashboard import server as srv              # 원본 판정 함수·상수(sr_breaks·sr_watch_view·LIQ_BURST_60S_USD·chip_score_path)
    assets = [a for a in srv.FLOW_SPECS if a == "eth" or a in srv.DASHBOARD_ASSETS]   # 원본 flows 와 같은 코인
    state: dict = {}
    st: dict = {}
    watch: dict = {}
    cs: dict = {}
    print(f"[sr_proximity] start assets={assets} thr_bp={THR_BP}", flush=True)
    while True:
        sr_tick(srv, assets, st, watch, send=send_telegram)
        chip_tick(srv, cs, time.time())
        view = watch
        if view:
            now = time.time()
            alerts = decide(state, view, now)
            for asset in dict.fromkeys(a["asset"] for a in alerts):
                mine = [a for a in alerts if a["asset"] == asset]
                sent = send_telegram(message(asset, mine))
                print(f"[sr_proximity] {asset} {[a['name'] for a in mine]} sent={sent}", flush=True)
                if sent is False:   # 보내기 실패 = 다시 무장해 다음 폴링에 재시도(설정이 없어 None 이면 재시도해도 소용없다 -- 그대로 둔다)
                    for a in mine:
                        state[a["key"]].update(armed=True, last=0.0)
        time.sleep(POLL_S)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
