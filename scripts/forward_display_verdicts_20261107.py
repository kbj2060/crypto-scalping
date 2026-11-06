#!/usr/bin/env python3
"""배포 표시 칸 전진 판정 (사전등록 docs/prereg/display_forward_verdicts_20261107.md).

서버에서 읽기 전용으로 실행한다. 판정일 전에는 거절하고, --status 는 건수만 낸다(적중 계산 안 함).
    python scripts/forward_display_verdicts_20261107.py --status
    python scripts/forward_display_verdicts_20261107.py run          # 판정일부터. 칸마다 결과 파일 1회만
    python scripts/forward_display_verdicts_20261107.py --selftest
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
OUT = ROOT / "tmp" / "display_forward_verdicts_20261107"
BAR, DAY = 300_000, 86_400_000
COINS = ["ETH", "BTC", "SOL", "XRP"]
X_COINS = ["ETH", "BTC", "SOL", "XRP", "ADA", "AVAX", "BNB", "DOGE", "LINK"]   # pred_record.X_COINS 와 같게
H = {"15m": ("a", 3), "1h": ("b", 12)}
START = pd.Timestamp("2026-11-08")
SHORT_END, SHORT_VERDICT = pd.Timestamp("2027-01-07"), pd.Timestamp("2027-01-08")
X_END, X_VERDICT = pd.Timestamp("2027-02-06"), pd.Timestamp("2027-02-07")
BOOT, SEED = 2000, 20261107
MIN_DAYS, MIN_ROWS = 0.75, 0.80
BAND_MD5 = "f62774c6"


def ms(ts):
    return int(ts.value // 10**6)


def read(path):
    if not Path(path).exists():
        return []
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def boot_days(days, stat):
    """일 블록 부트스트랩. days = 날짜별 배열 목록, stat(날짜 묶음 목록) -> 값."""
    rng = np.random.default_rng(SEED)
    n = len(days)
    return np.array([stat([days[i] for i in rng.integers(0, n, n)]) for _ in range(BOOT)])


def one_sided_p(b, side):
    return float(np.mean(b <= 0) if side == "lower" else np.mean(b >= 0))


def direction_rows(rows, key, h, t0, t1, kind="s"):
    """(날짜, 신호, 수익) 표. 신호 0 인 결정도 기준선에 필요해 남긴다."""
    px = {r["t"]: r["px"] for r in rows}   # 끝 가격은 화면 채점(pred_record._score)처럼 기록 종류를 가리지 않는다
    out = []
    for r in rows:
        if r.get("k") != kind or not (t0 <= r["t"] < t1):
            continue
        end = px.get(r["t"] + h * BAR)
        if end is None or end == r["px"]:
            continue
        if kind == "s":
            p = r.get(key)
            if p is None:
                continue
            s = 1 if p >= 80 else -1 if p <= 20 else 0
        else:
            s = r.get("s") or 0
        out.append((r["t"] // DAY, s, np.sign(end / r["px"] - 1)))
    return np.array(out, dtype=float).reshape(-1, 3)


def excess(a):
    """평균 초과 적중. a = (날짜, 신호, 부호). 롱·숏 신호가 다 있어야 값이 난다."""
    up, lo, hi = a[:, 2] > 0, a[:, 1] < 0, a[:, 1] > 0
    if not lo.any() or not hi.any():
        return np.nan
    return 0.5 * ((up[hi].mean() - up.mean()) + ((~up)[lo].mean() - (~up).mean()))


def same_day_excess(a):
    """보조: 기준선을 같은 날 오름 비율로. 신호 건수 가중 평균."""
    tot = w = 0.0
    for d in np.unique(a[:, 0]):
        g = a[a[:, 0] == d]
        up = g[:, 2] > 0
        n = (g[:, 1] != 0).sum()
        if n:
            tot += ((g[:, 1] != 0) * (np.where(g[:, 1] > 0, up, ~up) - np.where(g[:, 1] > 0, up.mean(), 1 - up.mean()))).sum()
            w += n
    return tot / w if w else np.nan


def score_direction(rows, key, h, t0, t1, kind="s"):
    a = direction_rows(rows, key, h, t0, t1, kind)
    days_total = (t1 - t0) // DAY
    expect = (t1 - t0) // BAR
    n_rows = sum(1 for r in rows if r.get("k") == kind and t0 <= r["t"] < t1)
    sig = a[a[:, 1] != 0] if len(a) else a
    res = {"rows": f"{n_rows}/{expect}", "scored_days": int(len(np.unique(sig[:, 0]))) if len(sig) else 0, "days_total": int(days_total)}
    if n_rows / expect < MIN_ROWS or res["scored_days"] < MIN_DAYS * days_total or np.isnan(excess(a)):
        return res | {"verdict": "보류(표본)"}
    up = a[:, 2] > 0
    hi, lo = a[:, 1] > 0, a[:, 1] < 0
    days = [a[a[:, 0] == d] for d in np.unique(a[:, 0])]
    b = boot_days(days, lambda ds: excess(np.vstack(ds)))
    b = b[~np.isnan(b)]
    low = float(np.quantile(b, 0.10))
    half = len(days) // 2
    return res | {"long_hit": f"{int(up[hi].sum())}/{int(hi.sum())}", "short_hit": f"{int((~up)[lo].sum())}/{int(lo.sum())}", "base_up": float(up.mean()),
                  "excess": float(excess(a)), "lower90": low, "p": one_sided_p(b, "lower"), "same_day_excess": float(same_day_excess(a)),
                  "halves": [float(excess(np.vstack(days[:half]))), float(excess(np.vstack(days[half:])))], "verdict": "통과" if low > 0 else "불통과"}


def score_band(rows, t0, t1):
    d = pd.DataFrame(rows)
    if d.empty:
        return {"verdict": "보류(표본)", "n": 0}
    d["t"] = pd.to_datetime(d.ts)
    d = d[(d.t >= pd.Timestamp(t0, unit="ms")) & (d.t < pd.Timestamp(t1, unit="ms"))]
    days_total, expect = (t1 - t0) // DAY, (t1 - t0) // 3_600_000 * 4
    res = {"rows": f"{len(d)}/{expect}", "days_total": int(days_total), "scored_days": int(d.t.dt.date.nunique())}
    if len(d) / expect < MIN_ROWS or res["scored_days"] < MIN_DAYS * days_total:
        return res | {"verdict": "보류(표본)"}
    acc = float(d.correct.mean())

    def gain(g):
        return g.correct.mean() - max(g.y.mean(), 1 - g.y.mean())

    days = [g[["correct", "y"]].to_numpy(float) for _, g in d.groupby(d.t.dt.date)]
    b = boot_days(days, lambda ds: (lambda m: m[:, 0].mean() - max(m[:, 1].mean(), 1 - m[:, 1].mean()))(np.vstack(ds)))
    low = float(np.quantile(b, 0.10))
    c6, c8 = d[d.confidence >= 0.6], d[d.confidence >= 0.8]
    a6 = float(c6.correct.mean()) if len(c6) else float("nan")
    ok = acc >= 0.648 and low > 0 and a6 >= 0.80
    return res | {"acc": f"{int(d.correct.sum())}/{len(d)}", "acc_rate": acc, "gain_vs_majority": float(gain(d)), "gain_lower90": low,
                  "conf60": f"{int(c6.correct.sum())}/{len(c6)}", "conf60_rate": a6, "conf80": f"{int(c8.correct.sum())}/{len(c8)}",
                  "by_coin": {c: float(g.correct.mean()) for c, g in d.groupby("coin")}, "verdict": "통과" if ok else "불통과"}


def score_xrev(rows, t0, t1):
    px = {r["t"]: r["px"] for r in rows if r.get("k") == "x"}
    step = 4 * 3_600_000
    ic = []
    for t, now in sorted(px.items()):
        a, b = px.get(t - step), px.get(t + step)
        if not (t0 <= t < t1 and a and b):
            continue
        cs = [c for c in X_COINS if c in now and c in a and c in b]
        past = pd.Series({c: now[c] / a[c] - 1 for c in cs})
        nxt = pd.Series({c: b[c] / now[c] - 1 for c in cs})
        v = past.rank().corr(nxt.rank())
        if v == v:
            ic.append((t // DAY, float(v)))
    days_total, expect = (t1 - t0) // DAY, (t1 - t0) // step
    a = np.array(ic, dtype=float).reshape(-1, 2)
    res = {"points": f"{len(a)}/{expect}", "days_total": int(days_total), "scored_days": int(len(np.unique(a[:, 0]))) if len(a) else 0}
    if len(a) / expect < MIN_ROWS or res["scored_days"] < MIN_DAYS * days_total:
        return res | {"verdict": "보류(표본)"}
    days = [a[a[:, 0] == d, 1] for d in np.unique(a[:, 0])]
    b = boot_days(days, lambda ds: np.concatenate(ds).mean())
    up = float(np.quantile(b, 0.90))
    return res | {"ic": float(a[:, 1].mean()), "upper90": up, "p": one_sided_p(b, "upper"), "verdict": "통과" if up < 0 else "불통과"}


def top_long_aux(rows, key, h, t0, t1):
    """보조(판정 아님): 분위 ≥ 90 롱만 -- 평균 수익 bp 와 전체 결정 평균 대비 초과. 사전 등록 dir_top_decile_long_20261107."""
    px = {r["t"]: r["px"] for r in rows}
    all_r, top = [], []
    for r in rows:
        if r.get("k") != "s" or not (t0 <= r["t"] < t1) or r.get(key) is None:
            continue
        end = px.get(r["t"] + h * BAR)
        if end is None:
            continue
        v = (end / r["px"] - 1) * 1e4
        all_r.append(v)
        if r[key] >= 90:
            top.append(v)
    return {"n_top": len(top), "n_all": len(all_r), "top_mean_bp": float(np.mean(top)) if top else None, "all_mean_bp": float(np.mean(all_r)) if all_r else None}


def top_long_4h_aux(rows, t0, t1, warm=288 * 14):
    """보조(판정 아님): 4h 예측이 과거 기록의 90분위 이상일 때 48봉 평균 수익 bp. 11-07 깨뜨리기에서 후보 탈락, 관찰만."""
    px = {r["t"]: r["px"] for r in rows}
    f = sorted((r for r in rows if r.get("k") == "4" and r.get("d") is not None), key=lambda r: r["t"])
    hist, top, all_r = [], [], []
    for r in f:
        end = px.get(r["t"] + 48 * BAR)
        if t0 <= r["t"] < t1 and end is not None and len(hist) >= warm:
            v = (end / r["px"] - 1) * 1e4
            all_r.append(v)
            if r["d"] >= np.quantile(hist, .9):
                top.append(v)
        hist.append(r["d"])
    return {"n_top": len(top), "n_all": len(all_r), "top_mean_bp": float(np.mean(top)) if top else None, "all_mean_bp": float(np.mean(all_r)) if all_r else None}


def holm(ps):
    order = sorted(ps, key=ps.get)
    out, cur = {}, 0.0
    for i, k in enumerate(order):
        cur = max(cur, min(1.0, ps[k] * (len(order) - i)))
        out[k] = cur
    return out


def status(today):
    t0 = ms(START)
    print(f"오늘 {today.date()} · 구간 시작 {START.date()} · 판정일 방향·띠 {SHORT_VERDICT.date()} · 코인 간 {X_VERDICT.date()} (건수만, 적중 계산 안 함)")
    for c in COINS:
        rows = read(LIVE / "pred_record" / f"{c}.jsonl")
        print(c, {k: sum(1 for r in rows if r.get("k") == k and r["t"] >= t0) for k in ("s", "4")})
    print("xsec", sum(1 for r in read(LIVE / "pred_record" / "xsec.jsonl") if r["t"] >= t0))
    print("band", sum(1 for r in read(LIVE / "regime_band_forward_scored.jsonl") if pd.Timestamp(r["ts"]) >= START))


def write_once(name, rep):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{name}_verdict.json"
    if f.exists():
        raise SystemExit(f"이미 판정함: {f}")
    f.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f, json.dumps(rep, ensure_ascii=False, indent=1))


def run(today):
    if today < SHORT_VERDICT:
        raise SystemExit(f"판정일 전이라 거절: {SHORT_VERDICT.date()} 부터")
    t0, t1 = ms(START), ms(SHORT_END)
    rep = {"prereg": "docs/prereg/display_forward_verdicts_20261107.md", "window": [str(START), str(SHORT_END)], "direction": {}, "aux_dir4h": {}, "aux_top_long": {}}
    for c in COINS:
        rows = read(LIVE / "pred_record" / f"{c}.jsonl")
        for name, (key, h) in H.items():
            rep["direction"][f"{c}_{name}"] = score_direction(rows, key, h, t0, t1)
            rep["aux_top_long"][f"{c}_{name}"] = top_long_aux(rows, key, h, t0, t1)
        if c != "BTC":
            rep["aux_dir4h"][c] = score_direction(rows, None, 48, t0, t1, kind="4")
            rep["aux_top_long"][f"{c}_4h"] = top_long_4h_aux(rows, t0, t1)
    ps = {k: v["p"] for k, v in rep["direction"].items() if "p" in v}
    rep["holm_p_aux"] = holm(ps) if ps else {}
    model = LIVE / "regime_band_h48_model.joblib"
    md5 = hashlib.md5(model.read_bytes()).hexdigest() if model.exists() else None
    rep["band"] = score_band(read(LIVE / "regime_band_forward_scored.jsonl"), t0, t1) | {"model_md5": md5, "model_md5_ok": bool(md5 and md5.startswith(BAND_MD5))}
    write_once("direction_band", rep)
    if today >= X_VERDICT:
        write_once("xrev", {"prereg": rep["prereg"], "window": [str(START), str(X_END)], "xrev": score_xrev(read(LIVE / "pred_record" / "xsec.jsonl"), t0, ms(X_END))})
    else:
        print(f"코인 간 반전은 {X_VERDICT.date()} 부터")


def selftest():
    t0, days = ms(pd.Timestamp("2026-01-01")), 40
    t1 = t0 + days * DAY
    rng = np.random.default_rng(1)
    n = (t1 - t0) // BAR + 12
    ret = rng.normal(0, 1e-3, n)
    px = 100 * np.exp(np.cumsum(ret))

    def recs(skill):   # skill=True: 신호가 3봉 뒤 방향을 맞힘 / False: 무작위
        out = []
        for i in range(n):
            fut = np.sign(px[min(i + 3, n - 1)] - px[i])
            s = rng.choice([1, -1, 0], p=[.2, .2, .6])
            s = (fut if s else 0) if skill else s
            a = 90.0 if s > 0 else 10.0 if s < 0 else 50.0
            out.append({"k": "s", "t": t0 + i * BAR, "px": float(px[i]), "a": a, "b": 50.0})
        return out

    good = score_direction(recs(True), "a", 3, t0, t1)
    bad = score_direction(recs(False), "a", 3, t0, t1)
    assert good["verdict"] == "통과" and good["excess"] > 0.4, good          # 양성 대조
    assert bad["verdict"] == "불통과" or abs(bad["excess"]) < 0.02, bad       # 무작위는 0 근처
    assert abs(bad["excess"]) < 0.02, bad
    thin = [r for r in recs(True) if r["t"] < t0 + 10 * DAY]
    assert score_direction(thin, "a", 3, t0, t1)["verdict"] == "보류(표본)"    # 표본 조건
    # 항상 오르는 시장: 신호가 전부 롱이어도 기준선(오름 비율 1)을 못 넘는다
    rising = [{"k": "s", "t": t0 + i * BAR, "px": 100.0 + i, "a": 90.0 if i % 2 else 10.0, "b": 50.0} for i in range(n)]
    x = direction_rows(rising, "a", 3, t0, t1)
    assert excess(x) <= 0, excess(x)
    # 띠: 정확도·다수결·확신 조건
    hours = pd.date_range("2026-01-01", periods=days * 24, freq="h")
    def band(acc, conf=.7):
        return [{"ts": str(t), "coin": c, "y": int(i % 2), "correct": int(rng.random() < acc), "confidence": conf} for i, t in enumerate(hours) for c in COINS]
    assert score_band(band(.85), t0, t1)["verdict"] == "통과"
    mid = [{"ts": str(t), "coin": c, "y": int(i % 2), "correct": 1 if i % 10 < 3 else int(i % 10 < 6), "confidence": .7 if i % 10 < 3 else .3} for i, t in enumerate(hours) for c in COINS]
    assert score_band(mid, t0, t1)["acc_rate"] == .6 and score_band(mid, t0, t1)["verdict"] == "불통과"   # 전체 60% -- 확신 칸은 100% 라도 64.8% 미달
    assert score_band(band(.70, conf=.5), t0, t1)["verdict"] == "불통과"       # 확신 0.6↑ 표본 없음(NaN) → 불통과
    assert score_band(band(.85)[:500], t0, t1)["verdict"] == "보류(표본)"
    # 코인 간: 완전 반전 → 통과, 완전 지속 → 불통과
    step = 4 * 3_600_000
    def xs(sign):
        rows, lv = [], {c: 100.0 for c in X_COINS}
        for i in range((t1 - t0) // step + 2):
            d = {c: rng.normal() * .01 for c in X_COINS} if i == 0 else {c: sign * prev[c] for c in X_COINS}
            lv = {c: lv[c] * (1 + d[c]) for c in X_COINS}
            prev = d
            rows.append({"k": "x", "t": t0 - step + i * step, "px": dict(lv)})
        return rows
    assert score_xrev(xs(-1), t0, t1)["verdict"] == "통과"
    assert score_xrev(xs(1), t0, t1)["verdict"] == "불통과"
    # 판정일 전 거절
    try:
        run(pd.Timestamp("2027-01-07"))
    except SystemExit as e:
        assert "거절" in str(e)
    else:
        raise AssertionError("판정일 전 거절 실패")
    assert holm({"a": .01, "b": .02, "c": .5}) == {"a": .03, "b": .04, "c": .5}
    rows4 = [{"k": "4", "t": t0 + i * BAR, "px": 100.0 + i, "d": float(i % 10)} for i in range(200)]
    a4 = top_long_4h_aux(rows4, t0, t1, warm=20)
    assert a4["n_top"] > 0 and a4["n_all"] == 200 - 20 - 48 and abs(a4["all_mean_bp"] - np.mean([(148 + i) / (100 + i) - 1 for i in range(20, 152)]) * 1e4) < 1e-6, a4
    aux = top_long_aux([{"k": "s", "t": t0 + i * BAR, "px": 100.0 + i, "a": 95.0 if i == 0 else 50.0} for i in range(10)], "a", 3, t0, t1)
    assert aux["n_top"] == 1 and abs(aux["top_mean_bp"] - 300) < 1e-9 and aux["n_all"] == 7, aux
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    now = pd.Timestamp.now("UTC").tz_localize(None)
    if a.selftest:
        selftest()
    elif a.status:
        status(now)
    elif a.cmd == "run":
        run(now)
    else:
        sys.exit(__doc__)
