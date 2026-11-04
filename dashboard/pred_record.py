"""⑤ 예측 줄 아래 «적중 기록»(15분·1시간·4h 방향 + 횡보·추세) 과 «4h 상대 강도» (2026-11-03).

기록: 서버 대시보드가 예측을 낼 때마다(화면이 열려 있든 아니든, 5분 배경 루프) 그 순간 화면이 보여 준 값과 신호를
data/live/pred_record/{COIN}.jsonl 에 한 줄씩 쌓는다. 채점은 읽을 때 한다 -- 신호(±1)였던 결정의 지평 뒤 종가가 기록에 있으면 적중/실패.
  k="s": {t, px, a(15분 분위), b(1시간 분위)} · k="4": {t, px, d(4h 예측 bp), s(신호)}  (t = 결정 시각 ms = 봉 시작 + 5분, px = 그 봉 종가)
신호 정의는 화면과 같다: 분위 ≥ 80 롱 · ≤ 20 숏(15분·1시간), 4h 는 예측이 아티팩트 문턱(hi·lo) 밖.
상대 강도: 9코인 종가를 4h 격자(UTC 0·4·8…)마다 xsec.jsonl 에 쌓고, 다음 4h 와의 순위상관(전진 IC)을 낸다. 연구 사전등록: docs/prereg/dashboard_candidates_forward_20261103.md.
    python dashboard/pred_record.py --selftest
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parents[1] / "data" / "live" / "pred_record"
BACKFILL = Path(__file__).resolve().parents[1] / "data" / "live" / "pred_record_backfill"   # 실시간 기록 시작(11-03 15시 UTC) 전 구간을 같은 모델로 다시 계산한 재구성 기록 -- 읽기 전용, 실시간 기록이 같은 결정(k, t)에 우선
BAND_SCORED = Path(__file__).resolve().parents[1] / "data" / "live" / "regime_band_forward_scored.jsonl"
X_COINS = ["ETH", "BTC", "SOL", "XRP", "ADA", "AVAX", "BNB", "DOGE", "LINK"]
BAR = 300_000
H = {"15m": 3, "1h": 12, "4h": 48}   # 지평(봉)
TAIL_BYTES = 8_000_000
DAYS = 14
RECENT = 24                           # 화면 막대 = 마지막으로 채점된 결정 이만큼(최신이 오른쪽)
GAP_MS = 30 * 60_000                  # 지평 뒤 종가가 이만큼 지나도 없으면 «대기»가 아니라 기록 구멍(재시작·수집 공백)
STALE_MS = 20 * 60_000                # 상대 강도 스냅숏이 이보다 묵으면 화면에 안 낸다(5분 루프가 멈췄다는 뜻)
_seen: dict = {}                      # (파일, k) -> 이미 쓴 t 집합 (프로세스 안)
_xs_latest: dict = {}                 # 마지막 상대 강도 스냅숏 {t, rel_bp}


def _tail(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - TAIL_BYTES))
        lines = f.read().decode("utf-8", "replace").splitlines()
    out = []
    for ln in lines[1:] if size > TAIL_BYTES else lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue                                  # 깨진 줄은 건너뛴다
    return out


def _rows(name: str, d: Path, bf: Path | None) -> list[dict]:
    """실시간 기록 + (있으면) 재구성 기록. 같은 (k, t) 는 실시간이 이긴다."""
    rows = _tail(d / f"{name}.jsonl")
    if bf is not None and (bf / f"{name}.jsonl").exists():
        have = {(r.get("k"), r.get("t")) for r in rows}
        rows += [r for r in _tail(bf / f"{name}.jsonl") if (r.get("k"), r.get("t")) not in have]
    return rows


def _append(path: Path, kind: str, rows: list[dict]) -> int:
    key = (str(path), kind)
    if key not in _seen:   # setdefault(key, <set>) 는 기본값을 매번 만들어 8MB 꼬리를 매번 읽는다
        _seen[key] = {r["t"] for r in _tail(path) if r.get("k") == kind}
    seen = _seen[key]
    new = [r for r in rows if r["t"] not in seen]
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in new))
        seen.update(r["t"] for r in new)
    return len(new)


def _px(closes: pd.Series, t_ms: int):
    v = closes.get(pd.Timestamp(t_ms - BAR, unit="ms"))
    return None if v is None or pd.isna(v) else float(v)


def _settled(payload: dict) -> list:
    """hist 에서 «피쳐가 온전한» 봉만. 최신 봉에 빈 피쳐(nan_feats: 봉 마감 직후 OI·롱숏비 REST 가 아직 안 올라옴)가 있으면 그 봉은 뺀다 --
    기록은 첫 값이 고정(dedupe)이라 NaN 이 섞인 예측이 신호로 박힐 수 있다. 뺀 봉은 다음 실행(5분 뒤)의 hist 에 다시 들어온다."""
    hist = payload.get("hist") or []
    return [h for h in hist if h[0] != payload.get("decided_at_ms")] if payload.get("nan_feats") else hist


def record_short(coin: str, payload: dict, closes: pd.Series, d: Path = DIR) -> int:
    """/api/dir-short 응답의 hist [[결정 ms, 15분 분위, 1시간 분위], …] 를 쌓는다."""
    if not (payload and payload.get("ok")):
        return 0
    rows = [{"k": "s", "t": int(t), "px": _px(closes, int(t)), "a": a, "b": b} for t, a, b in _settled(payload)]
    return _append(d / f"{coin.upper()}.jsonl", "s", [r for r in rows if r["px"] is not None])


def record_h4(coin: str, payload: dict, closes: pd.Series, d: Path = DIR) -> int:
    """/api/dir4h 응답의 hist [[결정 ms, 예측 bp], …] + 문턱 lo·hi(같은 bp 단위)."""
    if not (payload and payload.get("ok")):
        return 0
    lo, hi = payload["lo"], payload["hi"]
    rows = [{"k": "4", "t": int(t), "px": _px(closes, int(t)), "d": v, "s": 1 if v >= hi else -1 if v <= lo else 0} for t, v in _settled(payload)]
    return _append(d / f"{coin.upper()}.jsonl", "4", [r for r in rows if r["px"] is not None])


def _score(rows: list[dict], now_ms: int) -> dict:
    """신호였던 결정 → 지평 뒤 종가가 기록에 있으면 채점. 종가가 없는 결정(기록 구멍)·아직 안 지난 결정은 pending."""
    px = {r["t"]: r["px"] for r in rows}
    out = {}
    for name, ks, sig in (("15m", "s", lambda r: _s80(r.get("a"))), ("1h", "s", lambda r: _s80(r.get("b"))), ("4h", "4", lambda r: r.get("s"))):
        n = hit = pend = 0
        day: dict = {}
        seq: list = []   # 채점된 결정 [결정 ms, 적중 0/1] -- 화면은 하루 막대 대신 가장 마지막 검증 가능한 봉부터 보여 준다
        for r in sorted(rows, key=lambda r: r["t"]):
            if r.get("k") != ks:
                continue
            s = sig(r)
            if not s:
                continue
            end = px.get(r["t"] + H[name] * BAR)
            if end is None:
                if now_ms - (r["t"] + H[name] * BAR) < GAP_MS:   # 지평이 막 지났거나 아직 안 지남 = 대기. 한참 지났는데도 종가가 없으면 기록 구멍 -- 세지 않는다
                    pend += 1
                continue
            ret = end / r["px"] - 1
            if ret == 0:
                continue
            d = pd.Timestamp(r["t"], unit="ms").strftime("%m-%d")
            c = day.setdefault(d, [0, 0])
            c[0] += 1
            c[1] += int(ret * s > 0)
            seq.append([r["t"], int(ret * s > 0)])
            n += 1
            hit += int(ret * s > 0)
        if n or pend:
            out[name] = {"n": n, "hit": hit, "pending": pend, "daily": [[d, a, b] for d, (a, b) in sorted(day.items())][-DAYS:], "recent": seq[-RECENT:]}
    return out


def _s80(p):
    return None if p is None else 1 if p >= 80 else -1 if p <= 20 else 0


def _band(coin: str, scored: Path = BAND_SCORED) -> dict | None:
    rows = [r for r in _tail(scored) if str(r.get("coin")).upper() == coin.upper()]
    if not rows:
        return None
    n, ok = len(rows), sum(int(r["correct"]) for r in rows)
    ty = sum(int(r["y"]) for r in rows) / n
    day: dict = {}
    for r in rows:
        c = day.setdefault(str(r["ts"])[5:10], [0, 0])
        c[0] += 1
        c[1] += int(r["correct"])
    return {"n": n, "hit": ok, "base": max(ty, 1 - ty), "daily": [[d, a, b] for d, (a, b) in sorted(day.items())][-DAYS:],
            "recent": [[str(r["ts"]), int(r["correct"])] for r in sorted(rows, key=lambda r: str(r["ts"]))][-RECENT:]}


def xsec_record(px_by_coin: dict, d: Path = DIR) -> int:
    """px_by_coin = {코인: 종가 Series(봉 시작 ts)} -- 4h 격자 결정 시각(봉 시작 hour%4==3·55분)마다 9코인 종가를 쌓고 최신 스냅숏을 갱신."""
    frame = pd.DataFrame({c: s for c, s in px_by_coin.items() if len(s)}).dropna()
    if frame.shape[1] < len(X_COINS) or len(frame) < 49:
        return 0
    if frame.index[-1] - frame.index[-49] != pd.Timedelta(minutes=240):   # 결측 봉이 있으면 «49번째 앞»이 4h 앞이 아니다 -- 스냅숏을 갱신하지 않는다
        return 0
    last = frame.index[-1]
    rel = (frame.iloc[-1] / frame.iloc[-49] - 1)
    _xs_latest.update(t=int(pd.Timestamp(last).value // 10**6) + BAR, rel_bp={c: round(float(v) * 1e4, 1) for c, v in (rel - rel.mean()).items()})
    grid = frame[(frame.index.hour % 4 == 3) & (frame.index.minute == 55)]
    rows = [{"k": "x", "t": int(pd.Timestamp(ts).value // 10**6) + BAR, "px": {c: float(v) for c, v in r.items()}} for ts, r in grid.iterrows()]
    return _append(d / "xsec.jsonl", "x", rows)


def _xsec(d: Path = DIR, now_ms: int | None = None, bf: Path | None = None) -> dict:
    """전진 IC: 격자 행 t 마다 (직전 4h 상대 수익) 와 (다음 4h 상대 수익) 의 스피어만 순위상관 평균. now_ms 가 있고 스냅숏이 20분 넘게 묵었으면 latest 는 비운다."""
    rows = {r["t"]: r["px"] for r in _rows("xsec", d, bf) if r.get("k") == "x"}
    step, ics = 4 * 3600 * 1000, []
    for t, now in sorted(rows.items()):
        a, b = rows.get(t - step), rows.get(t + step)
        if not (a and b):
            continue
        cs = [c for c in X_COINS if c in now and c in a and c in b]
        past = pd.Series({c: now[c] / a[c] - 1 for c in cs})
        nxt = pd.Series({c: b[c] / now[c] - 1 for c in cs})
        ics.append(float(past.rank().corr(nxt.rank())))
    ics = [x for x in ics if x == x]   # 가격이 안 변한 코인뿐이면 NaN -- JSON 에 NaN 이 나가면 브라우저가 응답 전체를 못 읽는다
    fresh = _xs_latest and (now_ms is None or now_ms - _xs_latest["t"] <= STALE_MS)
    return {"n": len(ics), "ic": float(sum(ics) / len(ics)) if ics else None, "latest": dict(_xs_latest) if fresh else None}


def summary(now_ms: int, d: Path = DIR, band: Path = BAND_SCORED, bf: Path | None = BACKFILL) -> dict:
    coins = {}
    names = {p.stem for g in (d, bf) if g is not None and g.exists() for p in g.glob("*.jsonl")} - {"xsec"}
    for name in sorted(names):
        rows = [r for r in _rows(name, d, bf) if r["t"] >= now_ms - (DAYS + 1) * 86_400_000]
        coins[name.lower()] = _score(rows, now_ms)
    for c in set(coins) | {"eth", "btc", "sol", "xrp"}:
        b = _band(c, band)
        if b:
            coins.setdefault(c, {})["band"] = b
    xs = _xsec(d, now_ms, bf)
    return {"ok": bool(coins) or bool(xs["latest"]), "asof_ms": now_ms, "coins": coins, "xsec": xs}


def _selftest() -> None:
    d = Path(tempfile.mkdtemp())
    t0 = 1_793_000_000_000 // BAR * BAR
    idx = pd.date_range(pd.Timestamp(t0 - 20 * BAR, unit="ms"), periods=200, freq="5min")
    closes = pd.Series(range(100, 300), index=idx, dtype=float)                  # 단조 상승 -> 롱 신호만 적중
    hist = [[t0 + i * BAR, 95.0 if i % 2 == 0 else 50.0, 5.0] for i in range(60)]  # 15분 롱(짝수 i) · 1시간 숏
    assert record_short("ETH", {"ok": True, "hist": hist}, closes, d) == 60
    assert record_short("ETH", {"ok": True, "hist": hist}, closes, d) == 0         # 같은 결정은 두 번 안 쓴다
    assert record_h4("ETH", {"ok": True, "lo": -1, "hi": 1, "hist": [[t0 + i * BAR, 2.0] for i in range(60)]}, closes, d) == 60
    s = summary(t0 + 65 * BAR, d, d / "none", None)["coins"]["eth"]
    assert (s["15m"]["n"], s["15m"]["hit"], s["15m"]["pending"]) == (29, 29, 1), s["15m"]      # 짝수 i 30개 중 i=58 만 지평 뒤 종가 없음
    assert (s["1h"]["n"], s["1h"]["hit"], s["1h"]["pending"]) == (48, 0, 12), s["1h"]         # 상승장 숏 신호는 전부 실패
    assert (s["4h"]["n"], s["4h"]["hit"], s["4h"]["pending"]) == (12, 12, 48), s["4h"]
    assert [x[0] for x in s["15m"]["recent"]] == sorted(x[0] for x in s["15m"]["recent"]) and len(s["15m"]["recent"]) == RECENT and all(x[1] for x in s["15m"]["recent"]), s["15m"]["recent"]   # 최신이 끝 · 채점된 결정만
    g = summary(t0 + 500 * BAR, d, d / "none", None)["coins"]["eth"]                        # 한참 뒤에도 종가가 없는 결정 = 구멍 -> 대기로 세지 않는다
    assert all(v["pending"] == 0 for v in g.values()), g
    # 상대 강도: 직전 4h 에 오른 코인이 다음 4h 에 지는 반전 시장 -> IC < 0
    step = 4 * 3600 * 1000
    base = 1_793_000_000_000 // step * step
    # 앞 4코인 -> 상승 뒤 하락(반전), 뒤 5코인 -> 하락 뒤 상승
    rows = [{"k": "x", "t": base + i * step, "px": {c: 100 + (5 if (k < 4) == (i % 2 == 1) else -5) for k, c in enumerate(X_COINS)}} for i in range(5)]
    (d / "xsec.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    x = _xsec(d)
    assert x["n"] == 3 and x["ic"] < -0.9, x
    # 2026-11-03 서브 에이전트 검증 반영
    p2 = Path(tempfile.mkdtemp())
    ca = pd.Series(100.0, index=idx)
    nanp = {"ok": True, "decided_at_ms": t0 + 2 * BAR, "nan_feats": ["X_oi"], "hist": [[t0 + i * BAR, 95.0, 95.0] for i in range(3)]}
    assert record_short("ETH", nanp, ca, p2) == 2                                    # 빈 피쳐가 있는 최신 봉(i=2)은 뺀다
    assert record_short("ETH", dict(nanp, nan_feats=[]), ca, p2) == 1               # 다음 실행에서 온전해지면 그 봉이 들어온다
    gap = {c: pd.Series(100.0, index=idx).drop(idx[-10]) for c in X_COINS}                          # 결측 봉 1개
    _xs_latest.clear()
    assert xsec_record(gap, p2) == 0 and not _xs_latest                              # 결측이 있으면 스냅숏을 안 만든다
    _xs_latest.update(t=1_000_000, rel_bp={"ETH": 1.0})
    assert _xsec(p2, 1_000_000 + STALE_MS + 1)["latest"] is None and _xsec(p2, 1_000_000 + 1000)["latest"]   # 20분 넘게 묵으면 비운다
    flat = {"k": "x", "t": 0, "px": {c: 100.0 for c in X_COINS}}
    (p2 / "xsec.jsonl").write_text("".join(json.dumps(dict(flat, t=i * 4 * 3600 * 1000)) + "\n" for i in range(3)))
    assert _xsec(p2)["ic"] is None                                                   # 전부 같은 가격 -> NaN 대신 None
    # 재구성 기록: 같은 (k, t) 는 실시간이 이기고, 겹치지 않는 결정은 합쳐 채점
    live, bfd = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    (live / "ETH.jsonl").write_text(json.dumps({"k": "s", "t": t0, "px": 100.0, "a": 95.0, "b": 50.0}) + "\n" + json.dumps({"k": "s", "t": t0 + 3 * BAR, "px": 101.0, "a": 50.0, "b": 50.0}) + "\n")
    (bfd / "ETH.jsonl").write_text(json.dumps({"k": "s", "t": t0, "px": 1.0, "a": 5.0, "b": 50.0, "bf": 1}) + "\n"           # 같은 결정 -> 무시돼야 함(실시간 a=95 롱)
                                   + json.dumps({"k": "s", "t": t0 - 10 * BAR, "px": 100.0, "a": 5.0, "b": 50.0, "bf": 1}) + "\n"   # 재구성 숏 -> px 가 없는 t-7봉이라 대기가 아니라 구멍
                                   + json.dumps({"k": "s", "t": t0 - 3 * BAR, "px": 99.0, "a": 95.0, "b": 50.0, "bf": 1}) + "\n")  # 재구성 롱: 종료가 = t0 의 px 100 -> 적중
    m = summary(t0 + 100 * BAR, live, live / "none", bfd)["coins"]["eth"]["15m"]
    assert (m["n"], m["hit"], m["pending"]) == (2, 2, 0), m   # 실시간 t0(롱·종료가 101 적중) + 재구성 t0-3봉(롱·종료가 100 적중). 재구성 t0 는 무시, t0-10봉 숏은 종료가가 없어 구멍
    print("selftest OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
