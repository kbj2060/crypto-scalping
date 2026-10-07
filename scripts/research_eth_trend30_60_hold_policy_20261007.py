"""ETH 30분·60분 추세(방향) 모델 + «추세 쪽은 길게·반대는 짧게» 보유 규칙 검정 (2026-10-07).

사용자: «핑퐁장에서 롱/숏을 잡는데 추세에 맞는 포지션은 오래, 반대는 짧게 들고 갈 것 — 30분이나 1시간 추세 예측 모델».
결과 보기 전 고정:
  피쳐 = 라이브 dashboard/flow_read.card30_features 와 같은 5분봉 26개(파리티 대조) + 일봉 추세(T5·1/3/7일 수익률).
  라벨 = 5분봉 τ 종가 → τ+6(30분)·τ+12(60분) 종가 로그수익 부호. 피쳐는 τ 종가까지 · 라벨은 τ+1 봉부터.
  분할 TRAIN 2022-04~2024-06(끝 12봉 엠바고) · VAL 2024-07~12 · TEST 2025-01~ (주 판정; 2026 따로 보고).
  모델 = HGB(기본값 + max_iter 300, early stop 끔, 시드 1개 — 승격 주장 아님).
  보유 규칙 검정: 진입(시각, 방향 s)마다 «정렬»= s·(p−0.5)>0 이면 Hl 분, 아니면 Hs 분 보유 후 종가 청산.
    (Hs, Hl) = (10, 30) ← 30분 모델 · (10, 60) ← 60분 모델. 비교 = 항상 Hs · 항상 Hl · 같은 정렬 비율 무작위(위약).
    판정 = 모델 규칙 − 무작위 기대값 = mean((a−ā)·s·(r_Hl − r_Hs)) 의 일 블록 부트스트랩 CI. 청산 횟수가 같아 비용 상쇄.
    진입 집합: ①사용자 실원장 왕복 첫 체결(2026-07-20~10-04) ②핑퐁 상태(ER6h 하위) 페이드 후보(2025~, pingpong_extract).
    비교선 정렬: 일봉 T5 부호 · 5분봉 SMA144 위/아래.
실행: python scripts/research_eth_trend30_60_hold_policy_20261007.py [--selftest]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path("/home/kbj20/crypto-scalping/data")
PREV = Path("/home/kbj20/crypto-scalping/.claude/worktrees/realtime-ledger-analysis-bdd373")
K1D = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping--claude-worktrees-realtime-ledger-analysis-bdd373/"
           "d97de56b-4d05-489d-94d9-511517645e4d/scratchpad/k1m")
OUT = ROOT / "tmp/trend30_60_hold_20261007"
ms = lambda s: int(pd.Timestamp(s, tz="UTC").value // 10**6)  # noqa: E731
TR_END, VA_END = ms("2024-07-01"), ms("2025-01-01")
HOLDS = {30: (10, 30), 60: (10, 60)}
SEED, N_BOOT = 20261007, 2000


def load_1m() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted((DATA / "binance_vision/klines1m").glob("ETHUSDT-1m-*.parquet"))]
    parts += [pd.read_parquet(p) for p in sorted(K1D.glob("ETHUSDT-*.parquet"))]
    k = pd.concat(parts)[["t", "h", "l", "c", "v", "tb"]].drop_duplicates("t").sort_values("t")
    k["t"] = k.t.astype("int64")
    full = pd.DataFrame({"t": np.arange(k.t.iloc[0], k.t.iloc[-1] + 1, 60_000)}).merge(k, on="t", how="left")
    full["c"] = full.c.ffill(); full["h"] = full.h.fillna(full.c); full["l"] = full.l.fillna(full.c)
    full[["v", "tb"]] = full[["v", "tb"]].fillna(0.0)
    return full.reset_index(drop=True)


def to_5m(k: pd.DataFrame) -> pd.DataFrame:
    g = k.assign(b=k.t // 300_000 * 300_000).groupby("b")
    b = pd.DataFrame({"h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "v": g.v.sum(), "tb": g.tb.sum(), "n": g.t.size()})
    b = b[b.n == 5].drop(columns="n")
    return b.reindex(np.arange(b.index[0], b.index[-1] + 1, 300_000)).rename_axis("t").reset_index()  # 빈 봉 NaN 유지


def features(b: pd.DataFrame) -> pd.DataFrame:
    """card30_features 의 벡터판(연구 build() 의 K 블록과 같은 정의) + 일봉 추세."""
    S = pd.Series
    h, l, c, v = (b[x].to_numpy(float) for x in ("h", "l", "c", "v"))
    F: dict[str, np.ndarray] = {}
    lr = np.log(c)
    for k in (1, 3, 6, 12, 24, 48, 144, 288):
        F[f"ret{k}"] = (lr - S(lr).shift(k).values) * 1e4
    for N in (6, 24, 144, 288):
        H, L = S(h).rolling(N).max().values, S(l).rolling(N).min().values
        F[f"pos{N}"] = (c - L) / np.maximum(H - L, 1e-9)
    atr = S(h - l).rolling(144).mean().values
    for N in (48, 144, 288):
        F[f"sma{N}"] = (c - S(c).rolling(N).mean().values) / atr
    imb = 2 * b.tb.to_numpy(float) / np.maximum(v, 1e-9) - 1
    for k in (1, 6, 12, 48):
        F[f"imb{k}"] = S(imb * v).rolling(k).sum().values / np.maximum(S(v).rolling(k).sum().values, 1e-9)
    rg = (S(h).rolling(6).max().values - S(l).rolling(6).min().values) / c * 1e4
    F["rg30"] = rg
    F["rgq"] = S(rg).rolling(288).rank(pct=True).values
    F["volz"] = (v - S(v).rolling(288).mean().values) / S(v).rolling(288).std().values
    pc = S(c).shift(1).values
    F["last_body"] = (c - pc) / np.maximum(h - l, 1e-9)
    F["wick"] = ((h - np.maximum(c, pc)) - (np.minimum(c, pc) - l)) / np.maximum(h - l, 1e-9)
    ts = pd.to_datetime(b.t, unit="ms")
    F["hour"], F["dow"] = ts.dt.hour.to_numpy(float), ts.dt.dayofweek.to_numpy(float)
    for d in (1, 3, 7):
        F[f"ret{d}d"] = (lr - S(lr).shift(288 * d).values) * 1e4
    F["t5"] = np.mean([np.sign(lr - S(lr).shift(288 * d).values) for d in (7, 14, 28, 56, 90)], axis=0)
    return pd.DataFrame(F)


def parity(b: pd.DataFrame, X: pd.DataFrame, n: int = 300) -> float:
    """라이브 card30_features 와 벡터판 최대 오차(5분 간격 끊긴 창은 라이브가 None 이라 건너뜀)."""
    from dashboard.flow_read import card30_features
    rng = np.random.default_rng(SEED)
    worst, done = 0.0, 0
    for i in rng.integers(400, len(b), n * 3):
        w = b.iloc[i - 292:i + 1]
        if w.isna().any().any():
            continue
        bars = [dict(time=int(r.t // 1000), high=r.h, low=r.l, close=r.c, volume=r.v, taker=r.tb) for r in w.itertuples()]
        live = card30_features(bars)
        worst = max(worst, max(abs(live[k] - X.at[i, k]) / max(1.0, abs(live[k])) for k in live))
        done += 1
        if done == n:
            break
    return worst


def boot_ci(x: np.ndarray, day: np.ndarray, rng) -> list[float]:
    u, inv = np.unique(day, return_inverse=True)
    s, cnt = np.bincount(inv, x), np.bincount(inv)
    idx = rng.integers(0, len(u), (N_BOOT, len(u)))
    m = s[idx].sum(1) / cnt[idx].sum(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def hold_gain(s, a, r_s, r_l):
    """정렬 a(0/1)·방향 s 일 때 모델 규칙 손익 · 항상짧게 · 항상길게 · 무작위 대비 이득 원소."""
    g = s * (r_l - r_s)
    pol = s * np.where(a == 1, r_l, r_s)
    return pol, s * r_s, s * r_l, (a - a.mean()) * g


def eval_entries(name, t_dec, side, k1, b, P, rng) -> dict:
    """t_dec(ms) 결정 시각 · side ±1. 피쳐 = t_dec 이전에 닫힌 마지막 5분봉 · 진입가 = t_dec 가 든 1분봉 종가."""
    bi = np.searchsorted(b.t.to_numpy(), t_dec // 300_000 * 300_000 - 300_000)
    ei = np.searchsorted(k1.t.to_numpy(), t_dec // 60_000 * 60_000)
    c1 = k1.c.to_numpy()
    ok = (bi < len(b)) & (ei + 61 < len(c1))
    bi, ei, side, t_dec = bi[ok], ei[ok], side[ok], t_dec[ok]
    day = t_dec // 86_400_000
    out = dict(n=int(ok.sum()), days=int(len(np.unique(day))))
    align = {f"model{h}": (P[h][bi] - 0.5) * side > 0 for h in HOLDS}
    align["t5"] = b.t5_sign.to_numpy()[bi] * side > 0
    align["sma144"] = np.sign(b.sma144.to_numpy()[bi]) * side > 0
    for h, (hs, hl) in HOLDS.items():
        r_s = np.log(c1[ei + hs] / c1[ei]) * 1e4; r_l = np.log(c1[ei + hl] / c1[ei]) * 1e4
        for an in (f"model{h}", "t5", "sma144"):
            a = align[an].astype(float)
            pol, short, long_, gain = hold_gain(side, a, r_s, r_l)
            out[f"H{hs}/{hl}·{an}"] = dict(
                aligned=float(a.mean()), policy_bp=float(pol.mean()), always_short_bp=float(short.mean()),
                always_long_bp=float(long_.mean()), gain_vs_random_bp=float(gain.mean()),
                gain_ci=boot_ci(gain, day, rng),
                aligned_long_bp=float((side * r_l)[a == 1].mean()) if a.sum() else None,
                counter_long_bp=float((side * r_l)[a == 0].mean()) if (a == 0).sum() else None)
    print(f"\n[{name}] n={out['n']} 일수={out['days']}")
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"  {k:22s} 정렬 {v['aligned']:.2f} · 규칙 {v['policy_bp']:+.2f} · 항상짧게 {v['always_short_bp']:+.2f} · "
                  f"항상길게 {v['always_long_bp']:+.2f} · 무작위 대비 {v['gain_vs_random_bp']:+.2f} "
                  f"[{v['gain_ci'][0]:+.2f},{v['gain_ci'][1]:+.2f}] · 길게 보유 시 정렬 {v['aligned_long_bp']:+.1f} / 반대 {v['counter_long_bp']:+.1f}")
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    k1 = load_1m(); b = to_5m(k1); X = features(b)
    res: dict = {"parity_max_rel_err": parity(b, X)}
    print("라이브 card30_features 파리티 최대 상대오차:", res["parity_max_rel_err"], flush=True)
    assert res["parity_max_rel_err"] < 1e-6
    b["t5_sign"] = np.sign(X.t5.to_numpy()); b["sma144"] = X.sma144.to_numpy()
    t = b.t.to_numpy(); c = b.c.to_numpy()
    P: dict[int, np.ndarray] = {}
    for h in HOLDS:
        n = h // 5
        fwd = np.log(np.roll(c, -n) / c) * 1e4; fwd[-n:] = np.nan
        y = (fwd > 0).astype(int)
        valid = np.isfinite(fwd) & (fwd != 0) & np.isfinite(X.ret288.to_numpy())
        tr = valid & (t >= ms("2022-04-01")) & (t < TR_END - 12 * 300_000)
        va = valid & (t >= TR_END) & (t < VA_END); te = valid & (t >= VA_END)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=False, random_state=SEED)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X)[:, 1]; P[h] = p
        r = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "base_up_test": float(y[te].mean())}
        for nm, msk in (("VAL", va), ("TEST", te), ("TEST_2026", te & (t >= ms("2026-01-01")))):
            day = t[msk] // 86_400_000
            pm, ym, fm = p[msk], y[msk], fwd[msk]
            conf = np.abs(pm - 0.5) >= np.quantile(np.abs(p[va] - 0.5), 0.8)    # VAL 상위 20% 확신 문턱
            sgn = np.sign(pm - 0.5)
            r[nm] = dict(auc=float(roc_auc_score(ym, pm)), acc=float(((pm > 0.5) == ym).mean()),
                         acc_conf20=float(((pm > 0.5) == ym)[conf].mean()), share_conf=float(conf.mean()),
                         bp_follow=float((sgn * fm).mean()), bp_follow_ci=boot_ci(sgn * fm, day, rng),
                         bp_follow_conf=float((sgn * fm)[conf].mean()), bp_follow_conf_ci=boot_ci((sgn * fm)[conf], day[conf], rng),
                         p_sd=float(pm.std()))
            print(f"[{h}분 {nm}] AUC {r[nm]['auc']:.4f} · 적중 {r[nm]['acc']:.3f} · 확신 상위20% 적중 {r[nm]['acc_conf20']:.3f} "
                  f"(비중 {r[nm]['share_conf']:.2f}) · 따라가기 {r[nm]['bp_follow']:+.2f}bp {r[nm]['bp_follow_ci']} · "
                  f"확신만 {r[nm]['bp_follow_conf']:+.2f} {r[nm]['bp_follow_conf_ci']}", flush=True)
        res[f"model{h}"] = r

    T = pd.read_parquet(PREV / "tmp/ledger_micro_20261007/trips.parquet")
    res["ledger"] = eval_entries("실원장 왕복 첫 체결", T.t0.to_numpy("int64"), T.d.to_numpy(int), k1, b, P, rng)
    C = pd.read_parquet(PREV / "tmp/pingpong_extract_20261007/candidates.parquet")
    C = C[C.pp & (C.t >= VA_END)]
    res["pingpong_fade"] = eval_entries("핑퐁 페이드 후보 2025~", C.t.to_numpy("int64") + 60_000, -C.d.to_numpy(int), k1, b, P, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def selftest() -> None:
    s = np.array([1, -1, 1, -1]); a = np.array([1, 1, 0, 0.0])
    r_s = np.array([1.0, 1.0, 1.0, 1.0]); r_l = np.array([5.0, -5.0, -5.0, 5.0])
    pol, short, long_, gain = hold_gain(s, a, r_s, r_l)
    assert pol.tolist() == [5, 5, 1, -1] and short.tolist() == [1, -1, 1, -1]
    assert np.isclose(gain.mean(), pol.mean() - (a.mean() * long_.mean() + (1 - a.mean()) * short.mean()))
    b = pd.DataFrame({"t": np.arange(0, 300_000 * 400, 300_000, dtype="int64")})
    t_dec = np.array([300_000 * 10 + 1, 300_000 * 10])     # 10번째 봉 시작 직후 → 9번째(닫힘) · 정확히 경계 → 9번째
    assert np.searchsorted(b.t.to_numpy(), t_dec // 300_000 * 300_000 - 300_000).tolist() == [9, 9]
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
