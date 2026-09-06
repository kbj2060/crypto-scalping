#!/usr/bin/env python3
"""증거신호 **앵커 재정의** — 단일 첫발동 대신 '어떤 조합이 중첩됐는가'로 (2026-09-07).

사용자: *"증거신호의 첫 발동은 정확도가 너무 낮아. 중첩으로 쌓이고 어떤 조합인지가 굉장히
중요해. 바닥 앵커와 천장 앵커에서의 최고의 조합이나 중첩을 찾아봐."*

## 왜 지금 이게 정당한가
`docs/experiments/eth_regime_outcome_label_20260906.md` §5가 방향축을 닫으면서 **부활 조건을
스스로 명시**했다: *"되살릴 조건은 모집단 자체가 바뀌어 레짐 빈도가 0.5에서 유의하게 벗어나는
것이고, 그건 발동 조건을 바꾼다는 뜻이다."* 조합 앵커가 정확히 그 변경이다.

## 이미 닫힌 것과 무엇이 다른가
`eth_confluence_vote_lift_revalidation_and_fade_guard_20260906.md`는 **동시발동 수(N=votes)** 를
검정해 페이드 진입 36셀 0통과로 닫았다. 여기서 새로 묻는 것은 **조합의 정체(어느 신호들인가)** 다.
N은 같아도 조합이 다르면 다른가? 그 문서가 남긴 유일한 생존 조각(천장 측면 ATR초과 리프트가
N에 대해 증가)이 특정 조합에서 오는 것인지도 같이 본다.

## 앵커 정의 (인과적, 뒤만 본다)
부분집합 S, 창 W에 대해 봉 t가 앵커  <=>
  (1) 모든 s in S 가 [t-W+1, t] 안에서 raw 발동한 적이 있다
  (2) 적어도 하나가 **정확히 t에** 발동한다  (t = 조합 완성 봉 -> 미래참조 불가)
  (3) 같은 (side, S, W) 앵커가 직전 GAP=12봉 안에 없다  (기존 first_fire 규약과 동일, 뒤만 봄)
raw 발동을 쓴다 — `_active`/`_fill`(지속창)은 신호마다 길이가 달라(taker 24봉·str_z 12봉)
'동시성'을 오염시킨다(위 재검증 문서 §1: 천장 표 총합의 21.1%가 장기유지 2종에서).

## 라벨 (전부 출구·비용 없음 — sim_exit 결함 무관)
  L1 앵커 정확도  P(K=12봉 안에 같은 종류 지그재그 피벗)   <- 사용자가 말한 "정확도"
  L2 레짐 편향    P(R1 > 0.5),  R1 = MFE_cont/(MFE_cont+MFE_fade),  H = 12 / 48
                  (09-06 순수 레짐 라벨 원문. 전 지평 0.5 동전이었던 그 통계량)
두 라벨 모두 **ATR 십분위 매칭 기준선**을 뺀 초과분으로 판정한다. 09-06 재검증이 보인 대로
raw lift 상승의 대부분은 "겹칠수록 변동성↑"이다. 기준선은 봉 i의 십분위 평균 m_d(i)를
앵커별로 붙여 excess_i = y_i - m_d(i) 로 만든다 (평균이 정확히 관측-매칭귀무).

## 사전 판정 (실행 전 고정)
  n >= 30 이고 서로 다른 날 >= 10  (VAL·OOS 각각)   <- 09-06 소표본 클러스터 착시 방지
  L1 통과: excess 일군집 CI 하한 > 0 이 VAL·OOS 둘 다
  L2 통과: excess 일군집 CI 가 0 제외 + VAL·OOS 부호 동일
  TRAIN 은 보고만 (선택에 쓰지 않는다 — 전수 격자라 선택 자유도가 없다)
  ⭐다중도 잣대 = **시간이동 플라시보**: 신호별로 독립 원형이동(±3~30일)해 조합 정체만 파괴하고
    각 신호의 발동률·자기상관은 보존. R회 반복해 통과 개수 분포를 얻는다.
    실제 통과 수가 그 분포 안이면 "조합 정체는 중요하지 않다"가 답이다.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CACHE = ROOT / "tmp/eth_anchor_combo_20260907/signals.parquet"
ZIGZAG_DIR = ROOT / "tmp/zigzag_action_labels_extended_20260809"
OUT_DIR = ROOT / "tmp/eth_anchor_combo_20260907"

SIGNALS = ["taker_delta_z_climax", "short_term_return_z", "liquidity_sweep", "orthogonal_combo",
           "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]
ABBR = {"taker_delta_z_climax": "taker", "short_term_return_z": "strz", "liquidity_sweep": "sweep",
        "orthogonal_combo": "orth", "smt_divergence": "smt", "fib_extension_exhaustion": "fib",
        "demarker_extreme": "dem", "kalman_deviation_meanrev": "kal"}

GAP_BARS = 12
K_PIVOT = 12
HORIZONS = (12, 48)
# 전방향 로컬극값 셀 (W봉 극값, t..t+D 안 도달). D=3 이 사용자 제안. W=48 을 헤드라인으로 둔다
# (기저율 0.20 -- 현행 지그재그 라벨 0.124 와 난이도 동급). 나머지는 민감도.
EXT_CELLS = ((12, 3), (24, 3), (48, 3), (96, 3), (48, 1), (48, 6))
WINDOWS_W = (1, 3, 12)
MAX_SUBSET = 3
MIN_N, MIN_DAYS = 30, 10
BOOT = 2000
PLACEBO_R = 20
RNG_SEED = 20260907

TRAIN_START = pd.Timestamp("2024-01-01")
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-17 15:00:00")


# ---------------------------------------------------------------- 데이터

def load_zigzag_pivots() -> pd.DataFrame:
    """analyze_..._confluence_..._20260814.load_zigzag_pivots() + 2024 (재검증 스크립트와 동일)."""
    frames = []
    for year in (2024, 2025, 2026):
        frames.append(pd.read_csv(ZIGZAG_DIR / f"zigzag_action_labels_{year}.csv",
                                  parse_dates=["timestamp"], usecols=["timestamp", "low", "high", "zigzag_action"]))
    zz = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    zz = zz.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    run_id = (zz["zigzag_action"] != zz["zigzag_action"].shift()).cumsum()
    piv = []
    for _, run in zz.groupby(run_id):
        a = int(run["zigzag_action"].iloc[0])
        if a == 2:
            piv.append({"timestamp": run.loc[run["low"].idxmin(), "timestamp"], "pivot_type": "bottom"})
        elif a == 1:
            piv.append({"timestamp": run.loc[run["high"].idxmax(), "timestamp"], "pivot_type": "top"})
    return pd.DataFrame(piv).sort_values("timestamp").reset_index(drop=True)


def rolling_extreme_fwd(x: np.ndarray, H: int, how: str) -> np.ndarray:
    """t+1..t+H 의 max/min. 끝 H봉은 NaN."""
    n = len(x)
    out = np.full(n, np.nan)
    if n <= H:
        return out
    s = pd.Series(x[::-1])
    r = s.rolling(H, min_periods=H).max() if how == "max" else s.rolling(H, min_periods=H).min()
    v = r.to_numpy()[::-1]
    out[:n - H] = v[1:n - H + 1]
    return out


def build_panel() -> dict:
    sig = pd.read_parquet(CACHE)
    ts = pd.to_datetime(sig["timestamp"].to_numpy())
    n = len(sig)
    high, low, close = (sig[c].to_numpy(float) for c in ("high", "low", "close"))

    piv = load_zigzag_pivots()
    pos_of = {t: i for i, t in enumerate(ts)}
    pivot_hit = {}
    for s in ("bottom", "top"):
        pp = np.sort(np.array([pos_of[t] for t in piv.loc[piv.pivot_type == s, "timestamp"] if t in pos_of]))
        idx = np.searchsorted(pp, np.arange(n), side="left")
        dist = np.full(n, np.inf)
        ok = idx < len(pp)
        dist[ok] = pp[idx[ok]] - np.arange(n)[ok]
        pivot_hit[s] = (dist <= K_PIVOT).astype(float)   # 다음 피벗까지 <= K (event_study 정의)

    # R1 경로우세도 (09-06 원문): MFE_* = 발동 봉 종가 대비 그 방향 최대 유리이탈
    r1 = {}
    for H in HORIZONS:
        up = (rolling_extreme_fwd(high, H, "max") - close) / close
        dn = (close - rolling_extreme_fwd(low, H, "min")) / close
        denom = up + dn
        # bottom 앵커: 페이드=롱 -> MFE_fade=up, MFE_cont=dn      top: 반대
        r1[("bottom", H)] = np.where(denom > 0, dn / np.maximum(denom, 1e-12), 0.5)
        r1[("top", H)] = np.where(denom > 0, up / np.maximum(denom, 1e-12), 0.5)
        for s in ("bottom", "top"):
            r1[(s, H)] = np.where(np.isfinite(denom), (r1[(s, H)] > 0.5).astype(float), np.nan)

    # L3 전방향 로컬극값 (사용자 제안 2026-09-07): "봉 b의 저가/고가가 [b, b+W] 구간의 극값"이
    # 되는 봉이 t..t+D 안에 있는가. ⭐중심창(+-W)을 쓰지 않는다 — sweep/str_z 계열은 발동 조건
    # 자체가 '직전 구간 저점 돌파'라 뒤쪽 절반이 기계적으로 충족돼 트리거 쪽으로 기운다
    # (2026-09-01 local_extreme 93% -> 23% 붕괴가 정확히 이 메커니즘).
    ext = {}
    for W, D in EXT_CELLS:
        for side, x, how in (("bottom", low, "min"), ("top", high, "max")):
            r = pd.Series(x[::-1]).rolling(W + 1, min_periods=W + 1)
            f = (r.max() if how == "max" else r.min()).to_numpy()[::-1]
            e = np.full(n, np.nan)
            e[:n - W] = ((x[:n - W] >= f[:n - W] - 1e-12) if how == "max"
                         else (x[:n - W] <= f[:n - W] + 1e-12)).astype(float)
            a = pd.Series(e[::-1]).rolling(D + 1, min_periods=1).max().to_numpy()[::-1]
            v = np.full(n, np.nan)
            v[:n - D] = a[D:]                      # any(e[t..t+D])
            ext[(side, W, D)] = v

    atr = sig["atr_pct"].to_numpy(float)
    dec = pd.qcut(pd.Series(atr).rank(method="first"), 10, labels=False, duplicates="drop").to_numpy(float)

    fire = {s: np.stack([sig[f"{s}_{name}"].fillna(False).to_numpy(bool) for name in SIGNALS], axis=1)
            for s in ("bottom", "top")}
    day = pd.Series(ts).dt.floor("D").to_numpy()
    return {"ts": ts, "n": n, "pivot_hit": pivot_hit, "r1": r1, "ext": ext, "dec": dec,
            "fire": fire, "day": day, "atr": atr}


# ---------------------------------------------------------------- 앵커

def within_windows(fire_side: np.ndarray) -> dict[int, np.ndarray]:
    """W봉 안에 발동한 적이 있는가 (뒤만 본다). 조합마다 다시 계산하지 않으려고 미리 만든다."""
    out = {1: fire_side}
    for W in WINDOWS_W:
        if W == 1:
            continue
        out[W] = np.stack([pd.Series(fire_side[:, j]).rolling(W, min_periods=1).max().to_numpy() > 0
                           for j in range(fire_side.shape[1])], axis=1)
    return out


def cofire_anchors(fire_side: np.ndarray, within_W: np.ndarray, subset: tuple[int, ...]) -> np.ndarray:
    """조합 완성 봉 인덱스. 뒤만 본다."""
    cols = list(subset)
    hit = within_W[:, cols].all(axis=1) & fire_side[:, cols].any(axis=1)
    idx = np.flatnonzero(hit)
    if len(idx) == 0:
        return idx
    keep, last = [], -10 ** 9
    for i in idx:
        if i - last > GAP_BARS:
            keep.append(i)
        last = i
    return np.array(keep, dtype=int)


def anyk_anchors(fire_side: np.ndarray, within_W: np.ndarray, k: int) -> np.ndarray:
    """정체 무관 '아무 k종이 W봉 안에 겹침' 앵커 — 조합 정체의 직접 대조군.
    (동시발동 수 축 = 09-06 votes 재검증이 검정한 그 축의 raw·창 명시 버전)"""
    hit = (within_W.sum(axis=1) >= k) & fire_side.any(axis=1)
    idx = np.flatnonzero(hit)
    if len(idx) == 0:
        return idx
    keep, last = [], -10 ** 9
    for i in idx:
        if i - last > GAP_BARS:
            keep.append(i)
        last = i
    return np.array(keep, dtype=int)


def first_fire_union(fire_side: np.ndarray) -> np.ndarray:
    """현행 규약 기준선: 신호별 GAP12 첫발동의 합집합(같은 봉 중복 제거)."""
    out = set()
    for j in range(fire_side.shape[1]):
        last = -10 ** 9
        for i in np.flatnonzero(fire_side[:, j]):
            if i - last > GAP_BARS:
                out.add(int(i))
            last = i
    return np.array(sorted(out), dtype=int)


# ---------------------------------------------------------------- 통계

def day_ci(x: np.ndarray, days: np.ndarray, rng: np.random.Generator, B: int = BOOT) -> tuple[float, float]:
    """일군집 부트스트랩 CI95 (날짜를 통째로 복원추출). 일별 (합, 개수)로 벡터화."""
    _, inv = np.unique(days, return_inverse=True)
    g = int(inv.max()) + 1
    if g < 3:
        return (float("nan"), float("nan"))
    s = np.bincount(inv, weights=x, minlength=g)
    c = np.bincount(inv, minlength=g).astype(float)
    pick = rng.integers(0, g, size=(B, g))
    means = s[pick].sum(axis=1) / c[pick].sum(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def diff_day_ci(x1: np.ndarray, d1: np.ndarray, x2: np.ndarray, d2: np.ndarray,
                rng: np.random.Generator, B: int = BOOT) -> tuple[float, float]:
    """평균1 - 평균2 의 일군집 CI95. 두 집합이 **같은 날짜 표집**을 공유하도록 리샘플한다
    (앵커 집합이 달라 짝비교는 불가하지만, 날짜 클러스터는 공통이다)."""
    days = np.unique(np.concatenate([d1, d2]))
    g = len(days)
    if g < 3:
        return (float("nan"), float("nan"))
    pos = {d: i for i, d in enumerate(days)}
    i1 = np.array([pos[d] for d in d1]); i2 = np.array([pos[d] for d in d2])
    s1 = np.bincount(i1, weights=x1, minlength=g); c1 = np.bincount(i1, minlength=g).astype(float)
    s2 = np.bincount(i2, weights=x2, minlength=g); c2 = np.bincount(i2, minlength=g).astype(float)
    pick = rng.integers(0, g, size=(B, g))
    n1, n2 = c1[pick].sum(axis=1), c2[pick].sum(axis=1)
    ok = (n1 > 0) & (n2 > 0)
    d = s1[pick].sum(axis=1)[ok] / n1[ok] - s2[pick].sum(axis=1)[ok] / n2[ok]
    if len(d) < B // 2:
        return (float("nan"), float("nan"))
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def decile_baseline(y: np.ndarray, dec: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """창 안 전체 봉에서 십분위별 평균 -> 길이 10 배열 (NaN 은 전체평균)."""
    out = np.full(10, np.nan)
    yy, dd = y[pool], dec[pool]
    ok = np.isfinite(yy) & np.isfinite(dd)
    yy, dd = yy[ok], dd[ok]
    for d in range(10):
        m = dd == d
        if m.sum() >= 20:
            out[d] = yy[m].mean()
    g = yy.mean() if len(yy) else np.nan
    return np.where(np.isfinite(out), out, g)


def evaluate(anchors: np.ndarray, y: np.ndarray, base: np.ndarray, dec: np.ndarray,
             day: np.ndarray, rng: np.random.Generator, ci: bool = True) -> dict | None:
    if len(anchors) == 0:
        return None
    yy, dd, dy = y[anchors], dec[anchors], day[anchors]
    ok = np.isfinite(yy) & np.isfinite(dd)
    yy, dd, dy = yy[ok], dd[ok].astype(int), dy[ok]
    if len(yy) < MIN_N:
        return None
    ndays = len(np.unique(dy))
    exc = yy - base[dd]
    lo, hi = day_ci(exc, dy, rng) if ci else (float("nan"), float("nan"))
    return {"n": int(len(yy)), "days": int(ndays), "obs": float(yy.mean()),
            "null": float(base[dd].mean()), "excess": float(exc.mean()), "ci_lo": lo, "ci_hi": hi}


# ---------------------------------------------------------------- 스크린

def subsets() -> list[tuple[int, ...]]:
    out = []
    for k in range(1, MAX_SUBSET + 1):
        out.extend(itertools.combinations(range(len(SIGNALS)), k))
    return out


def screen(P: dict, fire: dict, rng: np.random.Generator, verbose: bool = True) -> pd.DataFrame:
    ts, dec, day = P["ts"], P["dec"], P["day"]
    wins = {"TRAIN": (TRAIN_START, VAL_START - pd.Timedelta(seconds=1)),
            "VAL": (VAL_START, VAL_END), "OOS": (OOS_START, OOS_END)}
    pools, bases = {}, {}
    for w, (a, b) in wins.items():
        pool = np.flatnonzero((ts >= a) & (ts <= b))
        pools[w] = pool
        for side in ("bottom", "top"):
            bases[(w, side, "L1")] = decile_baseline(P["pivot_hit"][side], dec, pool)
            for H in HORIZONS:
                bases[(w, side, f"L2H{H}")] = decile_baseline(P["r1"][(side, H)], dec, pool)
            for W_, D_ in EXT_CELLS:
                bases[(w, side, f"L3W{W_}D{D_}")] = decile_baseline(P["ext"][(side, W_, D_)], dec, pool)

    combos = subsets()
    rows = []
    for side in ("bottom", "top"):
        f = fire[side]
        ww = within_windows(f)
        cand = [("first_fire_union", None, first_fire_union(f))]
        for W in WINDOWS_W:
            for k in range(2, 5):
                cand.append((f"any{k}", W, anyk_anchors(f, ww[W], k)))
        for S in combos:
            for W in WINDOWS_W:
                if len(S) == 1 and W != 1:
                    continue                       # 단일 신호는 창이 무의미
                cand.append(("+".join(ABBR[SIGNALS[i]] for i in S), W, cofire_anchors(f, ww[W], S)))
        for name, W, anc in cand:
            kk = 0 if W is None else (int(name[3:]) if name.startswith("any") else len(name.split("+")))
            rec = {"side": side, "combo": name, "k": kk, "W": W,
                   "kind": "base" if W is None else ("anyk" if name.startswith("any") else "combo"),
                   "anchors": anc}
            good = True
            for w in wins:
                lo_b, hi_b = pools[w][0], pools[w][-1]
                a = anc[(anc >= lo_b) & (anc <= hi_b)]
                for metric, yv in ([("L1", P["pivot_hit"][side])] +
                                   [(f"L2H{H}", P["r1"][(side, H)]) for H in HORIZONS] +
                                   [(f"L3W{W}D{D}", P["ext"][(side, W, D)]) for W, D in EXT_CELLS]):
                    r = evaluate(a, yv, bases[(w, side, metric)], dec, day, rng, ci=(w != "TRAIN"))
                    if r is None:
                        good = False
                        continue
                    for k2, v in r.items():
                        rec[f"{w}_{metric}_{k2}"] = v
            rec["ok"] = good
            rows.append(rec)
        if verbose:
            print(f"  {side}: {len(cand)}개 후보 평가 완료", flush=True)
    return pd.DataFrame(rows)


def verdict(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    enough = np.ones(len(d), bool)
    for w in ("VAL", "OOS"):
        enough &= (d.get(f"{w}_L1_n", pd.Series(0, index=d.index)).fillna(0) >= MIN_N)
        enough &= (d.get(f"{w}_L1_days", pd.Series(0, index=d.index)).fillna(0) >= MIN_DAYS)
    d["enough"] = enough
    d["pass_L1"] = enough & (d.get("VAL_L1_ci_lo", np.nan) > 0) & (d.get("OOS_L1_ci_lo", np.nan) > 0)
    for H in HORIZONS:
        v_lo, v_hi = d.get(f"VAL_L2H{H}_ci_lo", np.nan), d.get(f"VAL_L2H{H}_ci_hi", np.nan)
        o_lo, o_hi = d.get(f"OOS_L2H{H}_ci_lo", np.nan), d.get(f"OOS_L2H{H}_ci_hi", np.nan)
        pos = (v_lo > 0) & (o_lo > 0)
        neg = (v_hi < 0) & (o_hi < 0)
        d[f"pass_L2H{H}"] = enough & (pos | neg)
    for W_, D_ in EXT_CELLS:
        d[f"pass_L3W{W_}D{D_}"] = enough & (d.get(f"VAL_L3W{W_}D{D_}_ci_lo", np.nan) > 0) & (d.get(f"OOS_L3W{W_}D{D_}_ci_lo", np.nan) > 0)
    d["pass_any"] = d["pass_L1"] | np.logical_or.reduce([d[f"pass_L2H{H}"] for H in HORIZONS])
    return d


def placebo_fire(fire: dict, rng: np.random.Generator) -> dict:
    """신호별 독립 원형이동(±3~30일 = 864~8640봉). 조합 정체만 파괴, 발동률·자기상관 보존."""
    out = {}
    for side, f in fire.items():
        g = np.empty_like(f)
        for j in range(f.shape[1]):
            sh = int(rng.integers(864, 8641)) * (1 if rng.random() < 0.5 else -1)
            g[:, j] = np.roll(f[:, j], sh)
        out[side] = g
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    print("[1/3] 패널 구성 ...", flush=True)
    P = build_panel()
    print(f"      {P['n']:,}봉  {P['ts'][0]} ~ {P['ts'][-1]}", flush=True)

    print("[2/3] 실제 조합 스크린 ...", flush=True)
    real = verdict(screen(P, P["fire"], rng))
    real.to_parquet(OUT_DIR / "screen_real.parquet")
    n_pass = {c: int(real[c].sum()) for c in ("pass_L1", "pass_L2H12", "pass_L2H48", "pass_any")}
    n_eval = int(real["enough"].sum())
    print(f"      평가가능 {n_eval} / 전체 {len(real)} · 통과 {n_pass}", flush=True)

    print(f"[3/3] 시간이동 플라시보 R={PLACEBO_R} ...", flush=True)
    pl = []
    for r in range(PLACEBO_R):
        pr = verdict(screen(P, placebo_fire(P["fire"], rng), rng, verbose=False))
        pl.append({"rep": r, "enough": int(pr["enough"].sum()),
                   **{c: int(pr[c].sum()) for c in ("pass_L1", "pass_L2H12", "pass_L2H48", "pass_any")}})
        print(f"      rep {r+1:2d}/{PLACEBO_R}  {pl[-1]}", flush=True)
    pldf = pd.DataFrame(pl)
    pldf.to_csv(OUT_DIR / "placebo_pass_counts.csv", index=False)

    summary = {"n_candidates": len(real), "n_evaluable": n_eval, "real_pass": n_pass,
               "placebo": {c: {"mean": float(pldf[c].mean()), "p95": float(np.percentile(pldf[c], 95)),
                               "max": int(pldf[c].max())} for c in ("pass_L1", "pass_L2H12", "pass_L2H48", "pass_any")},
               "placebo_evaluable_mean": float(pldf["enough"].mean())}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
