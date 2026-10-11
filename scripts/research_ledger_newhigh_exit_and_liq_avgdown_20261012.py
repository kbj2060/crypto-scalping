"""A 신고가 익절 · B 청산 섞인 역행 중 물타기 (2026-10-12, 사용자 «둘 다 진행해줘»). 결과 보기 전 고정.

A «더 뻗을 확률» 모델(가격만, 실험 C 기준선과 같은 11피쳐 · HGB) — 4h(48봉) 신고/저가 사건, 목표 = 2h 안 +0.5ATR 더 못 뻗음.
  학습 2022~2024 사건(12봉 중복 제거). 문턱 = 학습 예측의 q75 · q90.
  A1 (2025-01~2026-10 전체 사건) 확률 묶음(낮음 <q75 · 중간 · 높음 ≥q90)별 사건 방향 수익 1h/2h/4h(bp)
      — «높음»에서 보유 수익이 낮을수록 그 자리 익절이 이득. 높음−낮음 차, 일 블록 CI.
  A2 실원장 재생: 보유 중 같은 방향 신고가(롱=신고가·숏=신저가) 5분봉 마감에 확률 ≥ 문턱이고 평가 이익이면 전량 익절.
      팔: 실제 · +익절(q75) · +익절(q90) · +무조건 신고가 익절(대조) — 각각 모델 거리 거래소 손절 있음/없음.
B 원장 물타기(첫 주문 제외 추가 주문, 첫 체결 시각·평균가) 직전 W(15·30분) «역행 + OI 감소(계약 수)» vs «역행 + OI 증가».
  OI = metrics 5분 oi_usd ÷ 그 시점 가격, 규약 «스탬프+5분 = 스냅샷»(10-11 실측 청산으로 확인) → 시각 t 에 아는 최신 스탬프만.
  결과(추가 이후 240분): 추가가 대비 최대 추가 역행 % · +0.5% 회복 여부 · 240분 뒤 수익(포지션 방향).
  비교 = 역행 크기 3분위 층별 차(OI↓ − OI↑), 일 블록 부트스트랩. 09-20 이후는 실측 청산(그 방향 강제청산 $)으로도 같은 비교.
경계: 사건·라벨은 확인 봉 종가 이후. 원장 기간 43일·대재앙 1건 — 결론 약함.

실행: python scripts/research_ledger_newhigh_exit_and_liq_avgdown_20261012.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_ledger_risk_system_replay_20261012 as R  # noqa: E402  sim · risk_model · bars5 · G · W · summarize

ROOT = R.ROOT
OUT = ROOT / "tmp/ledger_newhigh_liq_20261012"
LIQ = R.MAIN / "tmp/liqmap_validation_20261008"
RNG = np.random.default_rng(20261012)
BASE = ["side", "brk", "r1h", "r4h", "r24h", "d24", "atrp", "atr_r", "vr", "hour", "clpos"]


# ── A: 더 뻗을 확률 ─────────────────────────────────────────────────────────
def ext_frame(o: pd.DataFrame) -> pd.DataFrame:
    """5분봉 o(h,l,c,v; 인덱스 = 봉 시작 ms) → 신고/저가 사건 행(봉 i 종가까지 정보) + 라벨."""
    H, L, C, V = (o[k].to_numpy(float) for k in "hlcv")
    pc = np.r_[C[0], C[:-1]]
    A = pd.Series(np.maximum(H - L, np.maximum(abs(H - pc), abs(L - pc)))).rolling(12).mean().to_numpy()
    S = pd.Series
    hh, ll = S(H).shift(1).rolling(48).max().to_numpy(), S(L).shift(1).rolling(48).min().to_numpy()
    rhi, rlo = S(H).shift(12).rolling(276).max().to_numpy(), S(L).shift(12).rolling(276).min().to_numpy()
    vr = V / S(V).rolling(288).mean().shift(1).to_numpy()
    atr_r = A / S(A).rolling(288).mean().shift(1).to_numpy()
    lag = lambda n: np.r_[np.full(n, np.nan), C[:-n]]
    n = len(C); rows = []
    hour = (o.index.to_numpy() // 3_600_000) % 24
    for sgn in (1, -1):
        i = np.flatnonzero((H > hh) if sgn > 0 else (L < ll))
        i = i[(i > 600) & (i < n - 49)]
        f = pd.DataFrame({
            "i": i, "b": o.index.to_numpy()[i], "side": sgn,
            "brk": ((H[i] - hh[i]) if sgn > 0 else (ll[i] - L[i])) / A[i],
            "r1h": sgn * np.log(C[i] / lag(12)[i]) / (A[i] / C[i]), "r4h": sgn * np.log(C[i] / lag(48)[i]) / (A[i] / C[i]),
            "r24h": sgn * np.log(C[i] / lag(288)[i]) / (A[i] / C[i]),
            "d24": ((H[i] - rhi[i]) if sgn > 0 else (rlo[i] - L[i])) / A[i],
            "atrp": A[i] / C[i], "atr_r": atr_r[i], "vr": vr[i], "hour": hour[i],
            "clpos": sgn * (C[i] - (H[i] + L[i]) / 2) / np.maximum(H[i] - L[i], 1e-9)})
        fmax = np.array([H[j + 1:j + 25].max() for j in i]); fmin = np.array([L[j + 1:j + 25].min() for j in i])
        f["no_ext"] = ~((fmax >= H[i] + 0.5 * A[i]) if sgn > 0 else (fmin <= L[i] - 0.5 * A[i]))
        for h in (12, 24, 48):
            f[f"r{h}"] = sgn * (C[i + h] / C[i] - 1) * 1e4
        rows.append(f)
    return pd.concat(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=BASE).sort_values("b").reset_index(drop=True)


def dedup_side(df: pd.DataFrame, gap: int = 12) -> pd.DataFrame:
    keep = []
    for _, g in df.groupby("side"):
        last = -10**9
        for idx, i in zip(g.index, g.i):
            if i - last >= gap:
                keep.append(idx); last = i
    return df.loc[sorted(keep)]


def boot_diff(v, g, day, B=1000):
    """평균(g==1) − 평균(g==0), 일 블록 포아송 부트스트랩 95%."""
    days, di = np.unique(day, return_inverse=True)

    def est(w):
        ww = w[di]
        a = (ww * v * (g == 1)).sum() / max((ww * (g == 1)).sum(), 1e-9)
        b = (ww * v * (g == 0)).sum() / max((ww * (g == 0)).sum(), 1e-9)
        return a - b
    pt = est(np.ones(len(days)))
    bs = [est(RNG.poisson(1.0, len(days)).astype(float)) for _ in range(B)]
    return round(float(pt), 3), [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def part_a(o, k, trips, safe, bidx) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    ev = ext_frame(o)
    ts = pd.to_datetime(ev.b, unit="ms")
    tr = dedup_side(ev[ts < "2025-01-01"])
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=100, random_state=0)
    m.fit(tr[BASE], tr.no_ext)
    ev["p"] = m.predict_proba(ev[BASE])[:, 1]
    q75, q90 = np.quantile(m.predict_proba(tr[BASE])[:, 1], [0.75, 0.9])
    res = {"학습 사건": len(tr), "문턱 q75·q90": [round(float(q75), 3), round(float(q90), 3)]}
    te = dedup_side(ev[ts >= "2025-01-01"]).copy()
    te["day"] = te.b // 86_400_000
    te["grp"] = np.where(te.p >= q90, "높음", np.where(te.p >= q75, "중간", "낮음"))
    res["A1 2025~ 사건"] = len(te)
    res["A1 AUC(더 못 뻗음)"] = round(float(roc_auc_score(te.no_ext, te.p)), 3)
    res["A1 묶음별"] = {gname: {"n": int(len(g)), "더 못 뻗음": round(float(g.no_ext.mean()), 3),
                             **{f"보유 {h // 12}h bp": round(float(g[f"r{h}"].mean()), 2) for h in (12, 24, 48)}}
                      for gname, g in te.groupby("grp")}
    hl = te[te.grp != "중간"]
    res["A1 높음−낮음 보유 수익 bp(CI)"] = {f"{h // 12}h": boot_diff(hl[f"r{h}"].to_numpy(), (hl.grp == "높음").to_numpy().astype(int),
                                                           hl.day.to_numpy()) for h in (12, 24, 48)}
    exits = {}
    for name, th in (("q75", q75), ("q90", q90), ("무조건", -1.0)):
        sel = ev[(ev.b >= k.t[0]) & (ev.p >= th)]
        exits[name] = {s: set(k.i(b + 4 * 60_000) for b in sel[sel.side == s].b) for s in (1, -1)}
    E_start = min(trips, key=lambda t: t["t0"])["E0"]
    order = sorted(range(len(trips)), key=lambda i: trips[i]["t0"])
    arms = {}
    for stop in (False, True):
        for name in ("없음", "q75", "q90", "무조건"):
            rows, done = [], []
            for i in order:
                trp = trips[i]
                real = [(a, b) for t1, a, b in done if t1 <= trp["t0"]]
                E = max(trp["E0"] + sum(b - a for a, b in real), 1.0)
                d = R.d_at(safe, bidx, trp["t0"], trp["s"]) if stop else None
                r = R.sim(trp, k, E, d=d, scale=E / trp["E0"], exits=None if name == "없음" else exits[name])
                done.append((trp["t1"], trp["pnl"], r["pnl"])); rows.append({"big": trp["big"], **r})
            s = R.summarize(pd.DataFrame(rows), E_start)
            arms[f"{'모델손절 + ' if stop else ''}익절 {name}"] = {kk: s[kk] for kk in ("총손익 $", "10-06 큰 롱 $", "그 1건 뺀 손익 $",
                                                                                   "MDD %(순차)", "승률", "청산 방식")}
    res["A2 실원장"] = arms
    return res


# ── B: 청산 섞인 역행 중 물타기 ─────────────────────────────────────────────
def oi_contracts(k) -> pd.Series:
    """계약 수(ETH) 시계열, 인덱스 = 아는 시각(스탬프 + 5분)."""
    m = pd.read_parquet(LIQ / "metrics_5m.parquet")
    snap = m.ts.astype("int64").to_numpy() // 10**6 + 300_000
    idx = np.array([k.i(x) - 1 for x in snap])
    ok = (idx >= 0) & (idx < len(k.c))
    return pd.Series(m.oi_usd.to_numpy()[ok] / k.c[idx[ok]], index=snap[ok]).sort_index()


def part_b(k, trips) -> dict:
    oi = oi_contracts(k)
    oit, oiv = oi.index.to_numpy(), oi.to_numpy()
    at = lambda t: oiv[np.searchsorted(oit, t, side="right") - 1] if np.searchsorted(oit, t, side="right") > 0 else np.nan
    liq = pd.read_parquet(LIQ / "eth_liq_all_20261008.parquet")
    liq = liq[liq.venue == "binance"]; liq_t0 = int(liq.ts_ms.min())
    rows = []
    for tr in trips:
        s = tr["s"]
        for o in range(1, len(tr["ordq"])):
            t, p = int(tr["ordt"][o]), float(tr["ordp"][o])
            mi = k.i(t)
            if mi + 240 >= len(k.c) or mi < 60:
                continue
            r = {"day": t // 86_400_000, "big": tr["big"], "t": t, "s": s}
            for W in (15, 30):
                r[f"adv{W}"] = -s * (p / k.c[mi - W] - 1) * 100                   # + = 포지션 반대로 밀림(%)
                r[f"doi{W}"] = (at(t) / at(t - W * 60_000) - 1) * 100
                if t - W * 60_000 >= liq_t0:
                    w = liq[(liq.ts_ms < t) & (liq.ts_ms >= t - W * 60_000) & (liq.side == ("long" if s > 0 else "short"))]
                    r[f"liq{W}"] = float((w.qty * w.price).sum())
            seg_l, seg_h = k.l[mi + 1:mi + 241], k.h[mi + 1:mi + 241]
            r["mae"] = ((p - seg_l.min()) / p if s > 0 else (seg_h.max() - p) / p) * 100
            r["recover"] = float(((seg_h >= p * 1.005) if s > 0 else (seg_l <= p * 0.995)).any())
            r["ret240"] = s * (k.c[mi + 240] / p - 1) * 100
            rows.append(r)
    D = pd.DataFrame(rows)
    D.to_parquet(OUT / "adds.parquet")
    res = {"추가 주문 수": len(D),
           "10-06 큰 롱 추가": D[D.big][["adv15", "doi15", "adv30", "doi30", "mae", "recover"]].round(3).to_dict("records")}
    for W in (15, 30):
        x = D[(D[f"adv{W}"] > 0) & D[f"doi{W}"].notna()].copy()
        x["g"] = (x[f"doi{W}"] < 0).astype(int)
        x["q"] = pd.qcut(x[f"adv{W}"], 3, labels=False, duplicates="drop")
        out = {"n 역행 중 추가": len(x), "OI↓ 비율": round(float(x.g.mean()), 3)}
        for y in ("mae", "recover", "ret240"):
            out[y] = {"OI↓": round(float(x[x.g == 1][y].mean()), 3), "OI↑": round(float(x[x.g == 0][y].mean()), 3),
                      "역행 3분위별 차(OI↓−OI↑)·CI": [boot_diff(g[y].to_numpy(), g.g.to_numpy(), g.day.to_numpy())
                                                for _, g in x.groupby("q") if g.g.nunique() == 2]}
        res[f"B OI {W}분"] = out
        if f"liq{W}" in D:
            z = D[(D[f"adv{W}"] > 0) & D[f"liq{W}"].notna()].copy()
            if len(z) > 20:
                z["g"] = (z[f"liq{W}"] > z[f"liq{W}"].median()).astype(int)
                res[f"B 실측 청산 {W}분(09-20~)"] = {"n": len(z), **{y: {"청산 많음": round(float(z[z.g == 1][y].mean()), 3),
                                                                     "적음": round(float(z[z.g == 0][y].mean()), 3),
                                                                     "차·CI": boot_diff(z[y].to_numpy(), z.g.to_numpy(), z.day.to_numpy())}
                                                                for y in ("mae", "recover", "ret240")}}
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d1 = R.G.load_1m(); k = R.W.K(d1); trips = R.W.load_trips(k)
    o = R.bars5(d1)
    safe, bidx, _ = R.risk_model(o)
    rep = {"B": part_b(k, trips)}
    print(json.dumps(rep["B"], ensure_ascii=False, indent=1, default=float), flush=True)
    rep["A"] = part_a(o, k, trips, safe, bidx)
    print(json.dumps(rep["A"], ensure_ascii=False, indent=1, default=float), flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    return 0


def selftest():
    v = np.r_[np.ones(20), np.zeros(20)]; g = np.r_[np.ones(20), np.zeros(20)].astype(int)
    pt, ci = boot_diff(v, g, np.arange(40) % 8, B=50)
    assert pt == 1.0 and ci[0] == 1.0, (pt, ci)
    df = pd.DataFrame({"side": [1, 1, 1, -1], "i": [0, 5, 13, 6]})
    assert dedup_side(df).index.tolist() == [0, 2, 3]
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    sys.exit(selftest() if ap.parse_args().selftest else main())
