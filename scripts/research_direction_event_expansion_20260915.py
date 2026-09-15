#!/usr/bin/env python3
"""**사건 트리거 방향 축 확장** — 규칙을 더 찾고 수익성을 증명한다. (2026-09-15)

§5.30 이 사건 트리거로 방향 축을 열었다(ETH 23셀/10군). 이 파일은 그 위에서 두 가지를 한다.

🔴**먼저 복원이다.** 23셀을 만든 파이프라인은 **git 어디에도 없었다**(이전 세션 scratchpad 에만
존재). 문서가 숫자를 인용하는데 재현 코드가 없으면 그 숫자는 검증 불가다. `--stage screen` 이
그 인과 스크린을 자립 구현으로 복원하고 `data/research/eth_past_failures_events_20260915/
eth_final_clean.csv` 와 대조한다.

그 다음 §5.30 G(미검정)와 B-4(정직한 경계 넷)를 하나씩 닫는다:
  `--stage cross`  크로스자산 트리거 — BTC 사건이 ETH 방향을 예측하는가 (이 저장소 초행)
  `--stage conj`   조합(AND) 트리거 — 규칙군 2개 교집합
  `--stage fdr`    BH-FDR 다중검정 보정 (경계 ①)
  `--stage oictl`  OI 계열 상호 통제 — doi·oiflow·oi_z 15셀이 같은 사건인가 (경계 ②)
  `--stage port`   노출 상한 + 가중 포트폴리오 = **진짜 숫자** (§5.30 B-5 ③ 경고 ②③)
  `--stage size`   사건별 크기 모델 (G절 1순위 — 방향모델은 §5.30 C 에서 기각됐다)

🔴규약(§5.30 E 의 착시 여섯에서 온 것 전부, 결과 보기 전 고정):
  · 분위 임계는 **인과**(확장창·당일 제외·최소 56일). 전수 임계는 어디에도 안 쓴다.
  · 배경은 **그 사건이 일어난 날**의 배경, **같은 측면**(롱 규칙엔 무작위 롱).
  · 점추정은 전수, 표본은 비겹침(H 봉 간격), CI 는 **날짜블록 부트스트랩**.
  · 통과 = 초과>0 & 날짜블록 CI 0배제 & 2024·2025·2026 각각 양수.
  · 사후 부호 선택 금지 — 두 측면을 다 세고 표에 방향을 반드시 싣는다(B-3 감사 2번).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/research/eth_event_expansion_20260915"
COST_BP = 10.0          # 호메로스 표준. §5.30 B-5 가 이 비용에서 +13.94bp 를 냈다.
YEARS = (2024, 2025, 2026)
HS = {"4h": 48, "12h": 144, "1d": 288}
QS = (0.01, 0.025, 0.05, 0.10)
MIN_HIST_DAYS = 56      # 인과 분위의 최소 관측
SEED = 20260915

KL = {
    "ETH": ["data/eth_5m_1year.csv", "data/eth_5m_2026_gap.csv"],
    "BTC": ["data/btc_5m_1year.csv", "data/btc_5m_2026_gap.csv"],
    "SOL": ["data/sol_5m_2024_2026.csv"],
    "BNB": ["data/bnb_5m_2024_2026.csv"], "XRP": ["data/xrp_5m_2024_2026.csv"],
    "DOGE": ["data/doge_5m_2024_2026.csv"], "ADA": ["data/ada_5m_2024_2026.csv"],
    "AVAX": ["data/avax_5m_2024_2026.csv"], "LINK": ["data/link_5m_2024_2026.csv"],
}
MCSV = {"ETH": "data/TOTAL_ETHUSDT_metrics_2024_2026.csv",
        "BTC": "data/TOTAL_BTCUSDT_metrics_2024_2026.csv",
        "SOL": "data/TOTAL_SOLUSDT_metrics_2024_2026.csv"}
AGGFEAT = ROOT / "data/binance_vision/aggfeat"
MPARQ = ROOT / "data/binance_vision/metrics"


# ── 패널 ─────────────────────────────────────────────────────────────────────
def _chg(x, k):
    o = np.full(len(x), np.nan); o[k:] = x[k:] - x[:-k]; return o


def _z(x, w=2016):
    s = pd.Series(x)
    return ((s - s.rolling(w, min_periods=w // 2).mean())
            / s.rolling(w, min_periods=w // 2).std().replace(0, np.nan)).to_numpy()


def load_metrics(asset: str) -> pd.DataFrame | None:
    if asset in MCSV:
        m = pd.read_csv(ROOT / MCSV[asset])
        m["timestamp"] = pd.to_datetime(m["create_time"])
        return m.drop(columns=["create_time", "symbol"]).drop_duplicates("timestamp")
    fs = sorted(MPARQ.glob(f"{asset}USDT_*.parquet"))
    if not fs:
        return None
    m = pd.concat([pd.read_parquet(f) for f in fs])
    m = m[~m.index.duplicated()].reset_index().rename(columns={"index": "timestamp"})
    return m


def panel(asset: str, ticks: bool = True) -> pd.DataFrame:
    """봉 + 메트릭(OI/LSR) + (있으면) 틱 주문흐름 → 사건 트리거 후보 피쳐 패널."""
    ks = []
    for f in KL[asset]:
        d = pd.read_csv(ROOT / f)
        if "timestamp" not in d.columns:
            d["timestamp"] = pd.to_datetime(d["ts"])
        ks.append(d[["timestamp", "close", "high", "low", "volume", "trades", "taker_buy_base"]])
    df = pd.concat(ks)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    m = load_metrics(asset)
    if m is not None:
        m["timestamp"] = pd.to_datetime(m["timestamp"])
        df = df.merge(m, on="timestamp", how="left")
    else:
        for c in ("sum_open_interest", "count_long_short_ratio",
                  "sum_toptrader_long_short_ratio", "sum_taker_long_short_vol_ratio"):
            df[c] = np.nan
    df = df[df["timestamp"] >= "2024-01-01"].reset_index(drop=True)

    c = df["close"].to_numpy(float); lc = np.log(c)
    hi = df["high"].to_numpy(float); lw = df["low"].to_numpy(float)
    v = df["volume"].to_numpy(float); tb = df["taker_buy_base"].to_numpy(float)
    ntr = df["trades"].to_numpy(float)
    oi = df["sum_open_interest"].to_numpy(float)
    lo_oi = np.log(np.maximum(oi, 1e-9)) if np.isfinite(oi).any() else np.full(len(c), np.nan)
    lsr = df["count_long_short_ratio"].to_numpy(float)
    top = df["sum_toptrader_long_short_ratio"].to_numpy(float)
    tkr = df["sum_taker_long_short_vol_ratio"].to_numpy(float)
    imb = np.where(v > 0, (2 * tb - v) / np.maximum(v, 1e-9), 0.0)

    F: dict[str, np.ndarray] = {}
    for k, t in ((3, "15m"), (12, "1h"), (48, "4h"), (144, "12h"), (288, "1d")):
        dpx, doi = _chg(lc, k), _chg(lo_oi, k)
        F[f"ret_{t}"] = dpx
        F[f"absret_{t}"] = np.abs(dpx)
        F[f"doi_{t}"] = doi
        F[f"oiflow_{t}"] = doi * np.sign(dpx)
        F[f"liqint_{t}"] = _z(np.where(doi < 0, -doi, 0.0) * np.abs(dpx)) * -np.sign(dpx)
        F[f"oidiv_{t}"] = _z(doi) - _z(dpx)
        F[f"dlsr_{t}"] = _chg(np.log(np.maximum(lsr, 1e-9)), k)
        F[f"dtop_{t}"] = _chg(np.log(np.maximum(top, 1e-9)), k)
        F[f"rv_{t}"] = _z(pd.Series(np.abs(_chg(lc, 1))).rolling(k).mean().to_numpy())
        F[f"dvol_{t}"] = _z(np.log(np.maximum(pd.Series(v).rolling(k).sum().to_numpy(), 1e-9)))
        F[f"dtrd_{t}"] = _z(pd.Series(ntr).rolling(k).sum().to_numpy())
        F[f"tkimb_{t}"] = pd.Series(imb).rolling(k).mean().to_numpy()
        F[f"tksign_{t}"] = pd.Series(np.sign(imb)).rolling(k).mean().to_numpy()
        hh = pd.Series(hi).rolling(k).max().to_numpy(); ll = pd.Series(lw).rolling(k).min().to_numpy()
        F[f"pos_{t}"] = np.where(hh > ll, 2 * (c - ll) / np.maximum(hh - ll, 1e-9) - 1, 0.0)
    F["oi_z"] = _z(oi); F["lsr_z"] = _z(lsr); F["top_z"] = _z(top); F["taker_z"] = _z(tkr)
    F["hour_f"] = df["timestamp"].dt.hour.to_numpy(float)

    p = pd.DataFrame(F)
    p["timestamp"] = df["timestamp"]
    p["__lc__"] = lc
    if ticks:
        fs = sorted(AGGFEAT.glob(f"{asset}USDT_*.parquet"))
        if fs:
            tf = pd.concat([pd.read_parquet(f) for f in fs]).sort_index()
            tf = tf[~tf.index.duplicated()].reset_index().rename(columns={"ts": "timestamp"})
            p = p.merge(tf, on="timestamp", how="left")
            for cc in [x for x in tf.columns if x.startswith("tf_") and x != "tf_cvd"]:
                p["z" + cc] = _z(p[cc].to_numpy(float))
            p = p.drop(columns=["tf_cvd"])   # 봉 CVD 와 상관 +1.0000 — 검증 앵커일 뿐 정보 0
    return p.reset_index(drop=True)


_PCACHE: dict[str, pd.DataFrame] = {}


def cached_panel(asset: str) -> pd.DataFrame:
    if asset not in _PCACHE:
        f = OUT / f"panel_{asset}.parquet"
        if f.exists():
            _PCACHE[asset] = pd.read_parquet(f)
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            d = panel(asset)
            d.to_parquet(f)
            _PCACHE[asset] = d
    return _PCACHE[asset]


def featcols(p: pd.DataFrame) -> list[str]:
    skip = {"timestamp", "fwd", "asset"}
    return [c for c in p.columns if c not in skip and not c.startswith("__")
            and pd.api.types.is_numeric_dtype(p[c]) and p[c].notna().mean() > 0.5]


# ── 인과 분위 임계 ────────────────────────────────────────────────────────────
def causal_thresholds(v: np.ndarray, dayno: np.ndarray, qs=QS, cap: int = 50_000):
    """확장창 분위, **당일 제외**(shift 대신 날짜 경계로 자른다), 최소 56일.

    🔴전수 `nanquantile(v, q)` 은 2024년 사건을 2026년 데이터로 판정하는 미래참조다(§5.30 B).
    히스토리는 stride 로 최대 5만점만 쓴다 — 분위 추정 오차는 무시 가능하고 965일 × 60피쳐가
    분 단위로 끝난다. 반환은 각 봉 i 에서 쓸 수 있는 (하위임계[i], 상위임계[i])."""
    D = int(dayno.max()) + 1
    starts = np.searchsorted(dayno, np.arange(D + 1))
    qq = np.array(sorted(set(list(qs) + [1 - q for q in qs])))
    lo = np.full((len(v), len(qs)), np.nan)
    hi = np.full((len(v), len(qs)), np.nan)
    for d in range(MIN_HIST_DAYS, D):
        h = v[:starts[d]]
        h = h[np.isfinite(h)]
        if len(h) < 1000:
            continue
        if len(h) > cap:
            h = h[:: len(h) // cap + 1]
        qv = np.quantile(h, qq)
        s, e = starts[d], starts[d + 1]
        for j, q in enumerate(qs):
            lo[s:e, j] = qv[np.searchsorted(qq, q)]
            hi[s:e, j] = qv[np.searchsorted(qq, 1 - q)]
    return lo, hi


# ── 평가 ─────────────────────────────────────────────────────────────────────
def nonoverlap(idx: np.ndarray, H: int) -> np.ndarray:
    keep, last = [], -10 ** 9
    for i in idx:
        if i - last >= H:
            keep.append(i); last = i
    return np.asarray(keep, dtype=int)


def day_background(fwd_bp: np.ndarray, dayno: np.ndarray) -> np.ndarray:
    """그 날의 배경 롱 평균(bp). §5.30 E ⑥ — 배경을 같은 날로 매칭하지 않으면
    「좋은 날을 고른 것」을 「좋은 타이밍」으로 오독한다."""
    s = pd.Series(fwd_bp).groupby(dayno).transform("mean").to_numpy()
    return s


def year_background(fwd_bp: np.ndarray, yr: np.ndarray, H: int) -> np.ndarray:
    """그 **해**의 배경 롱 평균(bp) — 비겹침 격자에서. 「그냥 롱 홀드보다 나은가」(베타 통제)."""
    ok = np.isfinite(fwd_bp)
    pool = np.flatnonzero(ok)[::H]
    out = np.full(len(fwd_bp), np.nan)
    for y in np.unique(yr):
        out[yr == y] = float(np.nanmean(fwd_bp[pool][yr[pool] == y]))
    return out


def eval_cell(keep, fwd_bp, bgd, bgy, yr, dayno, side: int) -> dict:
    """🔴배경을 **둘** 낸다. 둘은 다른 질문에 답하고 어느 하나가 «정답」이 아니다.

    · `exc`  = 그 **해** 배경 롱 대비 — 「그냥 롱 홀드보다 나은가」(베타 통제).
               §5.30 B-4 의 「초과」 컬럼이 실은 이것이다(실측 ±1bp 일치).
    · `excd` = 그 **날** 배경 롱 대비 — 「그 날 안에서 타이밍이 좋은가」(§5.30 E ⑥).
               🔴이건 **부분적으로 오라클**이다 — 그 날이 좋을 걸 트레이더는 미리 모른다.
               따라서 **기각 관문이 아니라 귀속 진단**으로 쓴다: ≈0 이면 그 규칙의 엣지는
               «좋은 날을 고르는 것»이고(그래도 돈은 번다), 음수면 «그 날 안에서 나쁜 시점»이다.
    · 판정은 `net`(비용 차감 실수익)과 `exc` 로 한다. `net` 은 연도별로도 요구한다."""
    f = fwd_bp[keep]
    eg = side * f - side * bgy[keep]
    ed = side * f - side * bgd[keep]
    net = side * f - COST_BP
    out = {"n": len(keep), "acc": float((side * f > 0).mean()), "E|r|": float(np.abs(f).mean()),
           "net": float(net.mean()), "exc": float(eg.mean()), "excd": float(ed.mean()),
           "z": float(eg.mean() / (eg.std(ddof=1) / np.sqrt(len(eg)))) if len(eg) > 2 and eg.std() > 0 else 0.0}
    for y in YEARS:
        k = yr[keep] == y
        out[f"e{y}"] = float(eg[k].mean()) if k.sum() >= 30 else np.nan
        out[f"net{y}"] = float(net[k].mean()) if k.sum() >= 30 else np.nan
    out["_e"] = eg
    out["_ed"] = ed
    out["_d"] = dayno[keep]
    return out


def dateblock_ci(e: np.ndarray, d: np.ndarray, rng, B: int = 2000) -> tuple[float, float]:
    days = np.unique(d)
    by = {x: e[d == x] for x in days}
    bs = np.empty(B)
    for b in range(B):
        pick = rng.choice(days, len(days), replace=True)
        bs[b] = np.concatenate([by[x] for x in pick]).mean()
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def prep(asset: str, ticks: bool = True):
    p = cached_panel(asset)
    ts = p["timestamp"]
    dayno = (ts.dt.floor("D").astype("int64") // 86_400_000_000_000).to_numpy()
    dayno = dayno - dayno.min()
    return p, p["__lc__"].to_numpy(), ts.dt.year.to_numpy(), dayno


def fwd_of(lc: np.ndarray, H: int) -> np.ndarray:
    f = np.full(len(lc), np.nan); f[:-H] = (lc[H:] - lc[:-H]) * 1e4
    return f


P1 = lambda d: ((d.z > 2.0) & (d.exc > 0) & (d.net > 0)
                & d[[f"e{y}" for y in YEARS]].gt(0).all(axis=1)
                & d[[f"net{y}" for y in YEARS]].gt(0).all(axis=1))


def screen_asset(asset: str, feats: list[str] | None = None, verbose=True) -> pd.DataFrame:
    """인과 전수 스크린 — 피쳐 × 지평 × 분위 × 상/하위 × 두 측면."""
    p, lc, yr, dayno = prep(asset)
    cols = feats or featcols(p)
    rows = []
    TH = {c: causal_thresholds(p[c].to_numpy(float), dayno) for c in cols}
    for hn, H in HS.items():
        fwd = fwd_of(lc, H)
        ok = np.isfinite(fwd)
        bgd = day_background(np.where(ok, fwd, np.nan), dayno)
        bgy = year_background(fwd, yr, H)
        for c in cols:
            v = p[c].to_numpy(float)
            lo, hi = TH[c]
            for j, q in enumerate(QS):
                for cond, mask in (("하위", v <= lo[:, j]), ("상위", v >= hi[:, j])):
                    idx = np.flatnonzero(mask & ok & np.isfinite(bgd) & np.isfinite(lo[:, j]))
                    if len(idx) < 150:
                        continue
                    keep = nonoverlap(idx, H)
                    if len(keep) < 150:
                        continue
                    for side, sl in ((1, "롱"), (-1, "숏")):
                        r = eval_cell(keep, fwd, bgd, bgy, yr, dayno, side)
                        for kk in ("_e", "_ed", "_d"):
                            r.pop(kk)
                        rows.append({"asset": asset, "feat": c, "H": hn, "q": q,
                                     "cond": cond, "side": sl, **r})
    d = pd.DataFrame(rows)
    # 🔴관문 넷을 **동시에** 요구한다(문서 B-4 보다 엄격하다 — 거기엔 연도별 net 이 없었다):
    #   ①전체 순손익>0 ②세 해 각각 순손익>0 ③연배경 초과>0 ④세 해 각각 초과>0, 그리고 z>2.
    #   `net>0` 이 왜 필수인가: 초과에는 비용이 소거된다(양쪽 다 같은 비용). 이 관문을 빼면
    #   `ret_12h` 하위5%→1d 롱이 **같은날 초과 +186bp 인데 순손익 −14.7bp** 로 1위에 오른다.
    d["p1"] = P1(d)
    if verbose:
        print(f"[{asset}] 셀 {len(d):,} · 1차 통과(z>2 & 초과>0 & 세 해 전부) **{int(d.p1.sum())}**")
    return d


def confirm(asset: str, cands: pd.DataFrame, rng, verbose=True) -> pd.DataFrame:
    """생존자만 날짜블록 부트스트랩 CI 로 확정."""
    p, lc, yr, dayno = prep(asset)
    need = sorted(cands.feat.unique())
    TH = {c: causal_thresholds(p[c].to_numpy(float), dayno) for c in need}
    out = []
    for _, x in cands.iterrows():
        H = HS[x["H"]]; fwd = fwd_of(lc, H); ok = np.isfinite(fwd)
        bgd = day_background(np.where(ok, fwd, np.nan), dayno)
        bgy = year_background(fwd, yr, H)
        v = p[x["feat"]].to_numpy(float); lo, hi = TH[x["feat"]]
        j = QS.index(x["q"])
        mask = (v <= lo[:, j]) if x["cond"] == "하위" else (v >= hi[:, j])
        idx = np.flatnonzero(mask & ok & np.isfinite(bgd) & np.isfinite(lo[:, j]))
        keep = nonoverlap(idx, H)
        if len(keep) < 150:
            continue
        side = 1 if x["side"] == "롱" else -1
        r = eval_cell(keep, fwd, bgd, bgy, yr, dayno, side)
        eg, ed, dd = r.pop("_e"), r.pop("_ed"), r.pop("_d")
        lo_, hi_ = dateblock_ci(eg, dd, rng)
        dlo, dhi = dateblock_ci(ed, dd, rng)
        out.append({"asset": asset, "feat": x["feat"], "H": x["H"], "q": x["q"], "cond": x["cond"],
                    "side": x["side"], **r, "lo": lo_, "hi": hi_, "ci": lo_ > 0,
                    "dlo": dlo, "dhi": dhi, "dci": dlo > 0,
                    "y3": bool(all(np.isfinite(r[f"e{y}"]) and r[f"e{y}"] > 0 for y in YEARS))})
    d = pd.DataFrame(out)
    if len(d):
        d["ok"] = d.ci & d.y3
        d["fam"] = d.feat.str.replace(r"_(15m|1h|4h|12h|1d|z)$", "", regex=True).str.replace("^z?tf_", "tf_", regex=True)
        if verbose:
            print(f"[{asset}] 확정(CI 0배제 & 세 해 양수) **{int(d.ok.sum())}** / 후보 {len(d)}")
    return d


def show(d: pd.DataFrame, top=30):
    if not len(d):
        print("  (없음)"); return
    d = d.sort_values("exc", ascending=False).head(top)
    print(f"{'피쳐':>14} {'지평':>4} {'분위':>6} {'조건':>4} {'방향':>4} {'n':>5} {'적중':>6} "
          f"{'E|r|':>7} {'순bp':>8} {'연배경초과':>9} {'날짜블록CI':>20} {'같은날':>8} "
          f"{'net 24/25/26':>27}")
    for _, x in d.iterrows():
        print(f"{x['feat']:>14} {x['H']:>4} {x['q']:>6.1%} {x['cond']:>4} {x['side']:>4} {x['n']:>5} "
              f"{x['acc']:>6.1%} {x['E|r|']:>7.1f} {x['net']:>+8.2f} {x['exc']:>+9.2f} "
              f"[{x['lo']:>+7.1f},{x['hi']:>+7.1f}] {x['excd']:>+8.2f} "
              + "".join(f"{x['net'+str(y)]:>+9.2f}" for y in YEARS))


def stage_screen(a):
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "eth_screen.csv"
    if a.reuse and cache.exists():
        d = pd.read_csv(cache)
        d["p1"] = P1(d)
        print(f"[ETH] 캐시 재사용 셀 {len(d):,} · 1차 통과 **{int(d.p1.sum())}**")
    else:
        d = screen_asset("ETH")
    d.to_csv(cache, index=False)
    c = confirm("ETH", d[d.p1], rng)
    c.to_csv(OUT / "eth_confirm.csv", index=False)
    fin = c[c.ok]
    print(f"\n=== ETH 확정 {len(fin)}셀 · 규칙군 {fin.fam.nunique()}개 ===")
    print(fin.groupby("fam").size().sort_values(ascending=False).to_string())
    print()
    show(fin)
    # 대조: 이전 세션이 남긴 23셀과 겹치는가
    ref = ROOT / "data/research/eth_past_failures_events_20260915/eth_final_clean.csv"
    if ref.exists():
        r = pd.read_csv(ref)
        key = lambda df: set(zip(df.feat, df.H, df.q.round(4), df.cond, df.side))
        A, B = key(fin), key(r)
        print(f"\n🔴이전 세션 23셀 대조: 겹침 **{len(A & B)}** / 이전 {len(B)} · 이번 {len(A)}")
        print(f"   이번에만: {sorted(x[0] for x in A - B)[:12]}")
        print(f"   이전에만: {sorted(x[0] for x in B - A)[:12]}")
    print(f"\n저장: {OUT/'eth_screen.csv'} · {OUT/'eth_confirm.csv'}")


# ── 축 1a. 크로스자산 트리거 ─────────────────────────────────────────────────
def stage_cross(a):
    """**BTC/SOL 의 사건이 ETH 방향을 예측하는가** — 이 저장소가 한 번도 안 한 축.

    근거: 호메로스 §5.29 가 「ETH 단일 자산 안의 축은 전부 소진, 유일한 후보는 자산 수」로
    닫혔다. 그때의 «자산 수»는 **같은 규칙을 여러 자산에 거는 것**이었지 **한 자산의 사건으로
    다른 자산을 거래하는 것**이 아니다. 09-08 의 「돌파/되돌림 = BTC 동조가 거의 전부」
    ([[eth_breakout_reversal_btc_comove_axis_20260908]])가 이 방향을 가리킨다.

    🔴관문은 자기자산 판과 **똑같다**(더 느슨하게 하지 않는다). 추가로 «자기자산 같은 규칙보다
    나은가»를 병기한다 — 안 그러면 그냥 BTC≈ETH 상관을 재확인하는 것이다."""
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [(s_, t_) for s_ in ("BTC", "SOL", "ETH") for t_ in ("ETH", "BTC", "SOL") if s_ != t_]
    alls = []
    for src, tgt in pairs:
        ps, _, _, _ = prep(src)
        pt, lct, yrt, dayt = prep(tgt)
        cols = featcols(ps)
        j = pt[["timestamp"]].merge(ps[["timestamp"] + cols], on="timestamp", how="left")
        rows = []
        TH = {c: causal_thresholds(j[c].to_numpy(float), dayt) for c in cols}
        for hn, H in HS.items():
            fwd = fwd_of(lct, H); ok = np.isfinite(fwd)
            bgd = day_background(np.where(ok, fwd, np.nan), dayt)
            bgy = year_background(fwd, yrt, H)
            for c in cols:
                v = j[c].to_numpy(float); lo, hi = TH[c]
                for q_i, q in enumerate(QS):
                    for cond, mask in (("하위", v <= lo[:, q_i]), ("상위", v >= hi[:, q_i])):
                        idx = np.flatnonzero(mask & ok & np.isfinite(bgd) & np.isfinite(lo[:, q_i]))
                        if len(idx) < 150:
                            continue
                        keep = nonoverlap(idx, H)
                        if len(keep) < 150:
                            continue
                        for side, sl in ((1, "롱"), (-1, "숏")):
                            r = eval_cell(keep, fwd, bgd, bgy, yrt, dayt, side)
                            for kk in ("_e", "_ed", "_d"):
                                r.pop(kk)
                            rows.append({"src": src, "tgt": tgt, "feat": c, "H": hn, "q": q,
                                         "cond": cond, "side": sl, **r})
        d = pd.DataFrame(rows); d["p1"] = P1(d)
        print(f"[{src}→{tgt}] 셀 {len(d):,} · 1차 통과 **{int(d.p1.sum())}**")
        alls.append(d)
    D = pd.concat(alls, ignore_index=True)
    D.to_csv(OUT / "cross_screen.csv", index=False)
    # 확정: 생존자만 날짜블록 CI
    outs = []
    for (src, tgt), g in D[D.p1].groupby(["src", "tgt"]):
        ps, _, _, _ = prep(src); pt, lct, yrt, dayt = prep(tgt)
        cols = sorted(g.feat.unique())
        j = pt[["timestamp"]].merge(ps[["timestamp"] + cols], on="timestamp", how="left")
        TH = {c: causal_thresholds(j[c].to_numpy(float), dayt) for c in cols}
        for _, x in g.iterrows():
            H = HS[x["H"]]; fwd = fwd_of(lct, H); ok = np.isfinite(fwd)
            bgd = day_background(np.where(ok, fwd, np.nan), dayt); bgy = year_background(fwd, yrt, H)
            v = j[x["feat"]].to_numpy(float); lo, hi = TH[x["feat"]]; q_i = QS.index(x["q"])
            mask = (v <= lo[:, q_i]) if x["cond"] == "하위" else (v >= hi[:, q_i])
            keep = nonoverlap(np.flatnonzero(mask & ok & np.isfinite(bgd) & np.isfinite(lo[:, q_i])), H)
            side = 1 if x["side"] == "롱" else -1
            r = eval_cell(keep, fwd, bgd, bgy, yrt, dayt, side)
            eg, ed, dd = r.pop("_e"), r.pop("_ed"), r.pop("_d")
            l_, h_ = dateblock_ci(eg, dd, rng)
            outs.append({"src": src, "tgt": tgt, "feat": x["feat"], "H": x["H"], "q": x["q"],
                         "cond": x["cond"], "side": x["side"], **r, "lo": l_, "hi": h_, "ci": l_ > 0})
    C = pd.DataFrame(outs)
    if len(C):
        C["ok"] = C.ci
        C.to_csv(OUT / "cross_confirm.csv", index=False)
        fin = C[C.ok].sort_values("exc", ascending=False)
        print(f"\n=== 크로스자산 확정 {len(fin)}셀 ===")
        print(f"{'출처→대상':>10} {'피쳐':>14} {'지평':>4} {'분위':>6} {'조건':>4} {'방향':>4} {'n':>5} "
              f"{'순bp':>8} {'초과':>8} {'날짜블록CI':>20} {'net 24/25/26':>27}")
        for _, x in fin.head(40).iterrows():
            print(f"{x['src']+'→'+x['tgt']:>10} {x['feat']:>14} {x['H']:>4} {x['q']:>6.1%} {x['cond']:>4} "
                  f"{x['side']:>4} {x['n']:>5} {x['net']:>+8.2f} {x['exc']:>+8.2f} "
                  f"[{x['lo']:>+7.1f},{x['hi']:>+7.1f}] "
                  + "".join(f"{x['net'+str(y)]:>+9.2f}" for y in YEARS))
        print(f"\n출처별 확정 수:"); print(fin.groupby(["src", "tgt"]).size().to_string())
    else:
        print("\n크로스자산 확정 0셀")


# ── 규칙 → 사건 인덱스 ────────────────────────────────────────────────────────
def rule_events(p, dayno, feat: str, H: int, q: float, cond: str, ok: np.ndarray) -> np.ndarray:
    v = p[feat].to_numpy(float)
    lo, hi = causal_thresholds(v, dayno, qs=(q,))
    mask = (v <= lo[:, 0]) if cond == "하위" else (v >= hi[:, 0])
    return nonoverlap(np.flatnonzero(mask & ok & np.isfinite(lo[:, 0])), H)


def load_rules(kind: str = "rep") -> pd.DataFrame:
    """확정 규칙. kind='rep' 은 규칙군마다 순손익 최고 1개(대표군), 'all' 은 전부."""
    c = pd.read_csv(OUT / "eth_confirm.csv")
    c = c[c.ok].copy()
    if kind == "rep":
        c = c.sort_values("net", ascending=False).groupby("fam", as_index=False).head(1)
    return c.reset_index(drop=True)


# ── 축 4. 포트폴리오 — 노출 상한 + 가중 = «진짜 숫자» ────────────────────────
def simulate(trades, n: int, r: np.ndarray, cap: float, cost_bp: float = COST_BP):
    """봉 단위 노출 시뮬. trades = [(진입봉 i, 측면 ±1, 목표비중 w, 보유 H)].

    🔴§5.30 B-5 ③ 이 「봉당 평균 노출 2.36단위(236%)·최대 7」이라 적고 「상한을 걸면 +13.94bp 도
    비례해 준다 — **상한 후 숫자가 진짜 숫자**」라고 스스로 경고했다. 여기서 그 숫자를 낸다.
    상한 처리: 진입 시점에 **보유 구간 전체의 최대 총노출**을 보고 여유만큼만 넣는다(줄이기).
    여유가 목표의 10% 미만이면 건너뛴다. 비용은 진입 시점에 왕복분을 한 번에 뺀다."""
    gross = np.zeros(n + 1); expo = np.zeros(n + 1); cost = np.zeros(n + 1)
    took = 0
    for i, side, w, H in sorted(trades, key=lambda t: t[0]):
        a, b = i + 1, min(i + 1 + H, n)
        if a >= b:
            continue
        head = cap - float(gross[a:b].max())
        ww = min(w, head)
        if ww < 0.1 * w:
            continue
        gross[a:b] += ww; expo[a:b] += side * ww
        cost[i] += cost_bp * 1e-4 * ww
        took += 1
    pr = expo[:n] * r - cost[:n]
    return pr, expo[:n], gross[:n], took


def random_entry_null(trades, n, r, cap, ts, rng, B=100):
    """**모든 진입을 같은 오프셋만큼 통째로 회전**시킨다(순환이동).

    🔴처음엔 진입 봉을 i.i.d. 로 무작위 추출했는데 **편향된 귀무**였다: 실제 사건은 군집해
    노출 상한에 자주 걸려 일부가 버려지는데(실측 후보 4,300 중 체결 1,349), 흩어진 무작위
    진입은 거의 다 체결돼 **비용을 훨씬 많이 낸다**. 그래서 귀무가 −11~−22bp 로 내려앉고
    모든 자산이 백분위 100% 라는 «너무 완전한 통과»가 나왔다(§5.30 E T1).
    통째 회전은 군집성·측면구성·상한 충돌·체결 수를 **그대로 보존**하고 수익과의 정렬만 깬다."""
    day = ts.dt.floor("D").values
    out = np.empty(B); tk = np.empty(B)
    maxH = max(t[3] for t in trades)
    for b in range(B):
        sh = int(rng.integers(n // 20, n - n // 20))
        rot = [((i + sh) % (n - maxH - 2), sd, w, H) for i, sd, w, H in trades]
        pr, _, _, took = simulate(rot, n, r, cap)
        out[b] = pd.Series(pr).groupby(day).sum().mean() * 1e4
        tk[b] = took
    return out, float(tk.mean())


def perf(pr, ts, yr, rng, label, expo=None, gross=None, took=None, ntr=None):
    d = pd.DataFrame({"pr": pr, "day": ts.dt.floor("D").values, "y": yr})
    g = d.groupby("day").agg(pr=("pr", "sum"), y=("y", "first"))
    dm = g.pr.mean() * 1e4; sd = g.pr.std()
    sharpe = float(g.pr.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0
    eq = g.pr.cumsum().to_numpy()
    mdd = float((eq - np.maximum.accumulate(eq)).min())
    days = g.index.to_numpy()
    bs = np.array([g.pr.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4 for _ in range(1500)])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    ys = [float(g[g.y == y].pr.mean() * 1e4) for y in YEARS]
    row = {"구성": label, "일평균bp": dm, "샤프": sharpe, "MDD%": mdd * 100,
           "누적%": float(eq[-1] * 100), "CI_lo": lo, "CI_hi": hi, "0배제": lo > 0,
           "e24": ys[0], "e25": ys[1], "e26": ys[2]}
    if expo is not None:
        row |= {"평균노출": float(gross.mean()), "p99노출": float(np.quantile(gross, 0.99)),
                "최대노출": float(gross.max()), "거래": took, "후보": ntr,
                "거래당bp": dm * len(g) / max(took, 1)}
    return row


def stage_port(a):
    rng = np.random.default_rng(SEED)
    p, lc, yr, dayno = prep("ETH")
    ts = p["timestamp"]
    r = np.zeros(len(lc)); r[1:] = np.diff(lc)
    rules = load_rules("all")
    reps = load_rules("rep")
    print(f"확정 규칙 {len(rules)}셀 · 대표군 {len(reps)}개\n")
    sets = {"대표군": reps, "전체": rules}
    out = []
    for sname, R in sets.items():
        base = []
        for _, x in R.iterrows():
            H = HS[x["H"]]; fwd = fwd_of(lc, H)
            keep = rule_events(p, dayno, x["feat"], H, x["q"], x["cond"], np.isfinite(fwd))
            side = 1 if x["side"] == "롱" else -1
            base.append((keep, side, H, float(x["exc"]), len(keep)))
        tot_exc = sum(b[3] for b in base); tot_n = sum(b[4] for b in base)
        for wname in ("동일", "초과가중", "빈도역가중"):
            trades = []
            for keep, side, H, exc, nn in base:
                if wname == "동일":
                    w = 1.0
                elif wname == "초과가중":
                    w = exc / (tot_exc / len(base))
                else:
                    w = (1.0 / nn) / (sum(1.0 / b[4] for b in base) / len(base))
                trades += [(int(i), side, w, H) for i in keep]
            for cap in (0.5, 1.0, 2.0, 99.0):
                pr, expo, gross, took = simulate(trades, len(lc), r, cap)
                out.append(perf(pr, ts, yr, rng, f"{sname}/{wname}/상한{cap if cap<9 else '∞'}",
                                expo, gross, took, len(trades)))
    # 벤치마크: 같은 평균 노출의 ETH 롱홀드
    ref = [o for o in out if o["구성"].startswith("대표군/동일/상한1.0")][0]
    bh = perf(r * ref["평균노출"], ts, yr, rng, f"[벤치] 롱홀드 노출{ref['평균노출']:.2f}")
    out.append(bh)
    D = pd.DataFrame(out)
    D.to_csv(OUT / "portfolio.csv", index=False)
    cols = ["구성", "일평균bp", "샤프", "MDD%", "누적%", "CI_lo", "CI_hi", "0배제",
            "e24", "e25", "e26", "평균노출", "최대노출", "거래", "거래당bp"]
    print(D[[c for c in cols if c in D.columns]].to_string(
        index=False, float_format=lambda x: f"{x:9.2f}"))
    print(f"\n저장: {OUT/'portfolio.csv'}")


# ── 축 2. 다중검정 보정 ──────────────────────────────────────────────────────
def stage_fdr(a):
    """BH-FDR — §5.30 B-4 «정직한 경계 ①»(다중검정 보정 없음)을 닫는다."""
    from scipy import stats
    d = pd.read_csv(OUT / "eth_screen.csv")
    d = d[np.isfinite(d.z)].copy()
    d["p"] = 2 * stats.norm.sf(np.abs(d.z))
    for qlev in (0.05, 0.10, 0.20):
        pv = np.sort(d.p.to_numpy()); m = len(pv)
        thr = pv[pv <= qlev * np.arange(1, m + 1) / m]
        cut = thr.max() if len(thr) else 0.0
        k = int((d.p <= cut).sum())
        print(f"BH q={qlev:.2f}: 임계 p≤{cut:.2e} · 통과 **{k}** / {m:,}셀 (귀무 기대 {qlev*max(k,1):.1f} 거짓)")
    pv = np.sort(d.p.to_numpy()); m = len(pv)
    thr = pv[pv <= 0.10 * np.arange(1, m + 1) / m]
    cut = thr.max() if len(thr) else 0.0
    conf = pd.read_csv(OUT / "eth_confirm.csv"); conf = conf[conf.ok]
    k = ["feat", "H", "q", "cond", "side"]
    mm = conf[k].merge(d[k + ["p", "z", "exc"]], on=k, how="left")
    mm["BH10"] = mm.p <= cut
    print(f"\n확정 {len(conf)}셀 중 BH q=0.10 통과 **{int(mm.BH10.sum())}**")
    print(mm.sort_values("p").to_string(index=False, float_format=lambda x: f"{x:9.3g}"))
    # ── 가족 단위 검정: 「19개 통과」 자체가 우연인가 (순환이동 귀무) ──────────
    print("\n=== ⭐파이프라인 통과 수의 순환이동 귀무 ===")
    print("  개별 셀이 BH 를 못 넘는 것과 「이만큼 통과한 게 우연인가」는 다른 질문이다."
          "\n  신호(사건)는 그대로 두고 **수익만 무작위 오프셋만큼 회전**시킨다 — 사건의 군집성·"
          "\n  롱비율·자기상관은 보존되고 수익과의 **정렬만** 깨진다.")
    pp, lc, yr, dayno, cells = materialize("ETH")
    rng2 = np.random.default_rng(SEED)
    FW = {hn: fwd_of(lc, H) for hn, H in HS.items()}
    n = len(lc)

    def count_pass(shift: int) -> int:
        cnt = 0
        for (c, hn, q, cond, side), (keep, _, _, H) in cells.items():
            fw = np.roll(FW[hn], shift)
            f = fw[keep]
            if not np.isfinite(f).all():
                continue
            bgy = year_background(fw, yr, H)
            pn = side * f - COST_BP
            ex = side * f - side * bgy[keep]
            if pn.mean() <= 0 or ex.mean() <= 0:
                continue
            sd = ex.std(ddof=1)
            if sd <= 0 or ex.mean() / (sd / np.sqrt(len(ex))) <= 2.0:
                continue
            ys = yr[keep]
            if all((ys == y).sum() >= 30 and pn[ys == y].mean() > 0 and ex[ys == y].mean() > 0
                   for y in YEARS):
                cnt += 1
        return cnt

    act = count_pass(0)
    nulls = np.array([count_pass(int(rng2.integers(n // 20, n - n // 20))) for _ in range(a.nperm)])
    print(f"  실제 통과 **{act}** · 귀무 평균 {nulls.mean():.1f} · 중앙 {np.median(nulls):.0f} · "
          f"95분위 {np.quantile(nulls,0.95):.0f} · 최대 {nulls.max()}")
    print(f"  **p = {float((nulls >= act).mean()):.4f}** (B={a.nperm})")
    print("\n🔴한계: 셀끼리 독립이 아니다(같은 피쳐군·인접 분위·겹치는 사건). BH 는 양의 의존"
          "\n   아래서도 유효하지만(Benjamini-Yekutieli 보다 느슨), 「몇 개가 진짜인가」의 상한이"
          "\n   아니라 「이 정도 z 는 우연으로 설명 안 된다」의 하한으로만 읽는다.")


# ── 축 2b. OI 계열 상호 통제 ────────────────────────────────────────────────
def stage_oictl(a):
    """`doi`·`oiflow`·`oi_z`·`liqint`·`oidiv` 가 **같은 사건**인가 — §5.30 B-4 경계 ②."""
    p, lc, yr, dayno = prep("ETH")
    rules = load_rules("all")
    ev = {}
    for _, x in rules.iterrows():
        H = HS[x["H"]]
        keep = rule_events(p, dayno, x["feat"], H, x["q"], x["cond"], np.isfinite(fwd_of(lc, H)))
        ev[(x["feat"], x["H"], x["q"], x["cond"], x["side"])] = (set(keep.tolist()), x["fam"], H)
    ks = list(ev)
    print("① 사건 자체의 겹침 (Jaccard, ±H 봉 허용 안 함 — 정확 일치 봉)")
    fams = sorted({ev[k][1] for k in ks})
    M = pd.DataFrame(index=fams, columns=fams, dtype=float)
    for f1 in fams:
        s1 = set().union(*[ev[k][0] for k in ks if ev[k][1] == f1])
        for f2 in fams:
            s2 = set().union(*[ev[k][0] for k in ks if ev[k][1] == f2])
            M.loc[f1, f2] = len(s1 & s2) / max(len(s1 | s2), 1)
    print(M.to_string(float_format=lambda x: f"{x:5.2f}"))
    print("\n② **배타 부분집합** — 다른 군이 하나도 안 걸린 사건만 남기면 초과가 남는가")
    rng = np.random.default_rng(SEED)
    rows = []
    for f1 in fams:
        own = set().union(*[ev[k][0] for k in ks if ev[k][1] == f1])
        others = set().union(*[ev[k][0] for k in ks if ev[k][1] != f1])
        # 다른 군 사건의 ±2시간 안에 있는 것도 «겹침»으로 본다
        wide = set()
        for i in others:
            wide.update(range(i - 24, i + 25))
        for k in [k for k in ks if ev[k][1] == f1]:
            keep = np.array(sorted(ev[k][0]))
            excl = np.array([i for i in keep if i not in wide])
            H = ev[k][2]; fwd = fwd_of(lc, H); okm = np.isfinite(fwd)
            bgd = day_background(np.where(okm, fwd, np.nan), dayno); bgy = year_background(fwd, yr, H)
            side = 1 if k[4] == "롱" else -1
            r_all = eval_cell(keep, fwd, bgd, bgy, yr, dayno, side)
            row = {"fam": f1, "규칙": f"{k[0]} {k[1]} {k[2]:.1%}{k[3]} {k[4]}",
                   "n": len(keep), "초과": r_all["exc"], "n배타": len(excl)}
            if len(excl) >= 40:
                r_ex = eval_cell(excl, fwd, bgd, bgy, yr, dayno, side)
                l_, h_ = dateblock_ci(r_ex["_e"], r_ex["_d"], rng, B=1000)
                row |= {"배타초과": r_ex["exc"], "배타net": r_ex["net"], "lo": l_, "hi": h_,
                        "잔존%": 100 * r_ex["exc"] / r_all["exc"] if r_all["exc"] else np.nan}
            rows.append(row)
    D = pd.DataFrame(rows).sort_values("초과", ascending=False)
    D.to_csv(OUT / "oi_control.csv", index=False)
    print(D.to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    print("\n⭐읽는 법: 배타 부분집합에서 **잔존% 가 낮으면** 그 군의 엣지는 다른 군과 «같은 사건»이다."
          "\n   높으면 독립 발견이다. n배타 가 너무 줄면 판정 유보(검정력 없음).")


# ── 축 1b. 조합(AND) 트리거 ─────────────────────────────────────────────────
def stage_conj(a):
    """**두 트리거의 교집합** + **레짐 필터**. §5.30 G 「연속 트리거 강도」의 이산판.

    🔴다중검정이 커지는 축이라 사전에 좁힌다: 2차 조건은 확정 규칙에서만(같은 지평·같은 측면),
    레짐 필터는 **세 개만** 미리 고정한다(변동성 분위·시각대·일간 추세 부호). 결과는 BH 로 잰다.
    2차 선택이 1차와 같은 데이터 위라는 것도 명시한다 — 이건 «새 발견»이 아니라 «정교화»다."""
    rng = np.random.default_rng(SEED)
    p, lc, yr, dayno = prep("ETH")
    R = load_rules("all")
    print(f"기반 규칙 {len(R)}셀\n")
    base = {}
    for _, x in R.iterrows():
        H = HS[x["H"]]
        fwd = fwd_of(lc, H)
        v = p[x["feat"]].to_numpy(float)
        lo, hi = causal_thresholds(v, dayno, qs=(x["q"],))
        m = (v <= lo[:, 0]) if x["cond"] == "하위" else (v >= hi[:, 0])
        base[tuple(x[["feat", "H", "q", "cond", "side"]])] = (m & np.isfinite(lo[:, 0]), H,
                                                              1 if x["side"] == "롱" else -1, x["fam"])
    # 레짐 필터 셋(인과) — 사전 고정
    rvq = pd.Series(p["rv_1d"].to_numpy(float)).expanding(288 * MIN_HIST_DAYS).rank(pct=True).shift(1).to_numpy()
    hour = p["timestamp"].dt.hour.to_numpy()
    trend = np.sign(p["ret_1d"].to_numpy(float))
    FILTERS = {"저변동(rv하위50%)": rvq <= 0.5, "고변동(rv상위50%)": rvq > 0.5,
               "아시아(0-8h)": hour < 8, "유럽(8-16h)": (hour >= 8) & (hour < 16),
               "미국(16-24h)": hour >= 16, "상승일": trend > 0, "하락일": trend < 0}
    rows = []
    keys = list(base)
    for k1 in keys:
        m1, H, side, fam1 = base[k1]
        fwd = fwd_of(lc, H); okm = np.isfinite(fwd)
        bgd = day_background(np.where(okm, fwd, np.nan), dayno); bgy = year_background(fwd, yr, H)
        b_keep = nonoverlap(np.flatnonzero(m1 & okm), H)
        b = eval_cell(b_keep, fwd, bgd, bgy, yr, dayno, side)
        b_exc, b_net = b["exc"], b["net"]
        cands = [(f"[필터] {fn}", fm) for fn, fm in FILTERS.items()]
        for k2 in keys:
            if k2 == k1 or base[k2][1] != H or base[k2][2] != side or base[k2][3] == fam1:
                continue
            m2 = base[k2][0]
            recent = pd.Series(m2.astype(float)).rolling(max(H // 2, 1), min_periods=1).max().to_numpy() > 0
            cands.append((f"[AND] {k2[0]} {k2[2]:.1%}{k2[3]}", recent))
        for cname, cm in cands:
            keep = nonoverlap(np.flatnonzero(m1 & cm & okm), H)
            if len(keep) < 120:
                continue
            r = eval_cell(keep, fwd, bgd, bgy, yr, dayno, side)
            for kk in ("_e", "_ed", "_d"):
                r.pop(kk)
            rows.append({"기반": f"{k1[0]} {k1[1]} {k1[2]:.1%}{k1[3]} {k1[4]}", "조건": cname,
                         "기반초과": b_exc, "기반net": b_net, **r, "Δ초과": r["exc"] - b_exc,
                         "Δnet": r["net"] - b_net})
    D = pd.DataFrame(rows)
    D["p1"] = P1(D) & (D["Δ초과"] > 0)
    D.to_csv(OUT / "conj.csv", index=False)
    print(f"조합 셀 {len(D):,} · 1차 통과(기본 관문 + 기반보다 개선) **{int(D.p1.sum())}**")
    from scipy import stats
    D["p"] = 2 * stats.norm.sf(np.abs(D.z))
    pv = np.sort(D.p.dropna().to_numpy()); m = len(pv)
    thr = pv[pv <= 0.10 * np.arange(1, m + 1) / m]
    cut = thr.max() if len(thr) else 0.0
    top = D[D.p1].sort_values("Δnet", ascending=False)
    top["BH10"] = top.p <= cut
    print(f"BH q=0.10 임계 p≤{cut:.2e}\n")
    if len(top):
        print(f"{'기반':>34} {'조건':>30} {'n':>5} {'기반net':>8} {'net':>8} {'Δnet':>8} "
              f"{'초과':>8} {'Δ초과':>8} {'BH':>3} {'net 24/25/26':>27}")
        for _, x in top.head(30).iterrows():
            print(f"{x['기반']:>34} {x['조건']:>30} {x['n']:>5} {x['기반net']:>+8.2f} {x['net']:>+8.2f} "
                  f"{x['Δnet']:>+8.2f} {x['exc']:>+8.2f} {x['Δ초과']:>+8.2f} {'✅' if x['BH10'] else '':>3} "
                  + "".join(f"{x['net'+str(y)]:>+9.2f}" for y in YEARS))
    print(f"\n저장: {OUT/'conj.csv'}")


# ── 축 3. 사건별 크기 모델 (ML) ──────────────────────────────────────────────
def stage_size(a):
    """**크기 축** — §5.30 G 1순위. 방향 모델은 §5.30 C 에서 «유의하게 해롭다」로 기각됐다.

    ⭐왜 크기는 다른가: 방향은 (2a−1) 을 올려야 하는데 사건 트리거가 이미 그 정보를 다 썼다.
    크기는 **분모**를 다룬다 — 로그성장 최대화는 비중 ∝ 엣지/분산 이고, 사건마다 E|r| 이
    100~277bp 로 3배 차이 나므로 **같은 크기로 넣는 지금이 명백히 최적이 아니다**.

    🔴규약: ①타깃은 log|수익|(방향 아님) ②월 확장 워크포워드·엠바고 ③**크기 맞춤 비교**
    ([[feedback_size_matched_comparison_and_log_growth_for_capital_structure]]) — 평균 노출을
    동일하게 맞춰 비교하지 않으면 「크게 걸어서 더 벌었다」를 실력으로 오독한다
    ④판정은 **로그성장과 MDD 를 함께** (§5.30 2958행의 분산 드래그 교훈)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(SEED)
    p, lc, yr, dayno = prep("ETH")
    ts = p["timestamp"]
    R = load_rules("rep")
    feats = [c for c in featcols(p) if c != "hour_f"]
    recs = []
    for ri, (_, x) in enumerate(R.iterrows()):
        H = HS[x["H"]]; fwd = fwd_of(lc, H)
        keep = rule_events(p, dayno, x["feat"], H, x["q"], x["cond"], np.isfinite(fwd))
        side = 1 if x["side"] == "롱" else -1
        for i in keep:
            recs.append((ts.iloc[i], int(i), ri, side, H, float(fwd[i])))
    E = pd.DataFrame(recs, columns=["ts", "i", "rule", "side", "H", "fwd"]).sort_values("ts")
    E["pnl"] = E.side * E.fwd - COST_BP
    X = p.iloc[E.i][feats].to_numpy(np.float32)
    X = np.hstack([X, E[["rule", "H"]].to_numpy(np.float32)])
    y = np.log(np.maximum(np.abs(E.fwd.to_numpy()), 1.0))
    print(f"사건 {len(E):,}건 · 대표군 {len(R)} · 피쳐 {X.shape[1]}\n")
    pred = np.full(len(E), np.nan)
    for m0 in pd.date_range("2024-07-01", "2026-08-01", freq="MS"):
        m1 = m0 + pd.DateOffset(months=1)
        tr = (E.ts < m0).to_numpy(); te = ((E.ts >= m0) & (E.ts < m1)).to_numpy()
        if tr.sum() < 200 or te.sum() < 1:
            continue
        mdl = [HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                             l2_regularization=3.0, random_state=sd).fit(X[tr], y[tr])
               for sd in (11, 907, 4231)]
        pred[te] = np.mean([m.predict(X[te]) for m in mdl], axis=0)
    ok = np.isfinite(pred)
    E2 = E[ok].copy(); E2["ehat"] = np.exp(pred[ok]); E2["absr"] = np.abs(E2.fwd)
    ic = float(pd.Series(E2.ehat).corr(pd.Series(E2.absr), method="spearman"))
    print(f"워크포워드 예측 {len(E2):,}건 · **E|r| 예측 스피어만 IC {ic:+.3f}**")
    qs = pd.qcut(E2.ehat, 5, labels=False)
    print("\n예측 E|r| 오분위별:")
    print(f"{'분위':>4} {'n':>5} {'예측E|r|':>9} {'실제E|r|':>9} {'적중':>7} {'평균net':>9}")
    for g in range(5):
        k = qs == g
        print(f"{g+1:>4} {int(k.sum()):>5} {E2.ehat[k].mean():>9.1f} {E2.absr[k].mean():>9.1f} "
              f"{float((E2.pnl[k] + COST_BP > 0).mean()):>7.1%} {E2.pnl[k].mean():>+9.2f}")
    # 사이징 팔 — 전부 평균 비중 1.0 으로 맞춘다(크기 맞춤)
    eh = E2.ehat.to_numpy()
    arms = {"동일": np.ones(len(E2)),
            "∝E|r| (변동성 추종)": eh / eh.mean(),
            "∝1/E|r| (리스크 패리티)": (1 / eh) / (1 / eh).mean(),
            "∝1/E|r|² (로그성장)": (1 / eh ** 2) / (1 / eh ** 2).mean()}
    r = np.zeros(len(lc)); r[1:] = np.diff(lc)
    print("\n크기 맞춤 비교(평균 비중 1.0 고정 · 노출 상한 1.0 · 봉단위 시뮬):")
    rows = []
    for nm, w in arms.items():
        w = np.clip(w, 0.1, 5.0); w = w / w.mean()
        trades = [(int(i), int(sd), float(ww), int(h))
                  for i, sd, ww, h in zip(E2.i, E2.side, w, E2.H)]
        pr, expo, gross, took = simulate(trades, len(lc), r, 1.0)
        rows.append(perf(pr, ts, yr, rng, nm, expo, gross, took, len(trades)))
    D = pd.DataFrame(rows)
    print(D[["구성", "일평균bp", "샤프", "MDD%", "누적%", "CI_lo", "CI_hi", "e24", "e25", "e26",
             "평균노출", "거래"]].to_string(index=False, float_format=lambda x: f"{x:9.2f}"))
    E2.to_csv(OUT / "size_events.csv", index=False)
    D.to_csv(OUT / "size_arms.csv", index=False)
    print(f"\n저장: {OUT/'size_arms.csv'}")


# ── 축 4b. ⭐워크포워드 규칙 선택 — 유일하게 정직한 수익성 증명 ──────────────
def materialize(asset: str = "ETH"):
    """전 셀의 (사건 인덱스·측면·지평·수익) 을 한 번만 만든다. 워크포워드가 이 위에서 돈다."""
    p, lc, yr, dayno = prep(asset)
    cols = featcols(p)
    TH = {c: causal_thresholds(p[c].to_numpy(float), dayno) for c in cols}
    cells = {}
    for hn, H in HS.items():
        fwd = fwd_of(lc, H); ok = np.isfinite(fwd)
        bgy = year_background(fwd, yr, H)
        for c in cols:
            v = p[c].to_numpy(float); lo, hi = TH[c]
            for j, q in enumerate(QS):
                for cond, mask in (("하위", v <= lo[:, j]), ("상위", v >= hi[:, j])):
                    keep = nonoverlap(np.flatnonzero(mask & ok & np.isfinite(lo[:, j])), H)
                    if len(keep) < 150:
                        continue
                    for side in (1, -1):
                        cells[(c, hn, q, cond, side)] = (keep, side * fwd[keep] - COST_BP,
                                                         side * fwd[keep] - side * bgy[keep], H)
    return p, lc, yr, dayno, cells


def stage_wf(a):
    """⭐**규칙 선택까지 워크포워드**. 위 `--stage port` 는 규칙을 2024~2026 전체로 골랐으므로
    표본내다 — 세 해 양수 관문 자체가 선택에 쓰였다. 여기서는 **매월 그 시점까지의 데이터만으로**
    규칙을 고르고 다음 달을 거래한다. 이게 §5.30 이 아직 한 번도 안 한 검사다.

    선택 규칙(사전 고정, 매월 동일): 그 시점까지 사건 ≥`--minn` · 순손익>0 · 연배경 초과>0 ·
    초과 t>`--tsel` · **직전 두 개의 완결 연도 각각 순손익>0**. 최소 12개월 이력 요구."""
    rng = np.random.default_rng(SEED)
    p, lc, yr, dayno, cells = materialize("ETH")
    ts = p["timestamp"]; tsv = ts.to_numpy()
    r = np.zeros(len(lc)); r[1:] = np.diff(lc)
    print(f"셀 {len(cells):,} 개 재료화 완료")
    months = pd.date_range(a.start, "2026-08-01", freq="MS")
    picked_hist, trades = [], []
    for m0 in months:
        m1 = m0 + pd.DateOffset(months=1)
        sel = []
        for key, (keep, pnl, exc, H) in cells.items():
            tk = tsv[keep]
            tr = tk < np.datetime64(m0)
            if tr.sum() < a.minn:
                continue
            pn, ex = pnl[tr], exc[tr]
            if pn.mean() <= 0 or ex.mean() <= 0:
                continue
            sd = ex.std(ddof=1)
            if sd <= 0 or ex.mean() / (sd / np.sqrt(len(ex))) < a.tsel:
                continue
            yrs = pd.Series(tk[tr]).dt.year.to_numpy()
            done = [y for y in np.unique(yrs) if y < m0.year and (yrs == y).sum() >= 30]
            if len(done) < 1 or not all(pn[yrs == y].mean() > 0 for y in done[-2:]):
                continue
            sel.append((key, float(ex.mean()), int(len(ex))))
        # 🔴표본내 판(대표군 8개)과 **거래 수를 맞춘다**. 안 맞추면 비용 차이가 결론을 만든다:
        #   무제한 선택은 월 42규칙 → 12,901거래 → 비용만 −26bp/일 이라 부호가 비용으로 결정된다.
        if a.dedup:
            fam = lambda k: k[0].split("_")[0]
            best = {}
            for it in sorted(sel, key=lambda x: -x[1]):
                best.setdefault(fam(it[0]), it)
            sel = list(best.values())
        sel = sorted(sel, key=lambda x: -x[1])[:a.top]
        picked_hist.append((m0, len(sel)))
        if not sel:
            continue
        mexc = np.mean([x[1] for x in sel])
        for key, ex, nn in sel:
            keep, pnl, _, H = cells[key]
            te = keep[(tsv[keep] >= np.datetime64(m0)) & (tsv[keep] < np.datetime64(m1))]
            w = {"동일": 1.0, "초과가중": ex / mexc,
                 "빈도역가중": (1.0 / nn) / np.mean([1.0 / x[2] for x in sel])}
            for i in te:
                trades.append((int(i), key[4], w, H))
    ph = pd.DataFrame(picked_hist, columns=["월", "선택규칙수"])
    print(f"\n월별 선택 규칙 수: 중앙 {ph.선택규칙수.median():.0f} · 최소 {ph.선택규칙수.min()} · "
          f"최대 {ph.선택규칙수.max()} · 0개월 {int((ph.선택규칙수==0).sum())}")
    print(ph.set_index("월").T.to_string())
    first = pd.Timestamp(a.start)
    mask = (ts >= first).to_numpy()
    rows = []
    for wname in ("동일", "초과가중", "빈도역가중"):
        for cap in (0.5, 1.0, 2.0, 99.0):
            tl = [(i, sd, wd[wname], H) for i, sd, wd, H in trades]
            pr, expo, gross, took = simulate(tl, len(lc), r, cap)
            rows.append(perf(pr[mask], ts[mask], yr[mask], rng,
                             f"WF/{wname}/상한{cap if cap<9 else '∞'}",
                             expo[mask], gross[mask], took, len(tl)))
    bh = perf(r[mask] * rows[1]["평균노출"], ts[mask], yr[mask], rng,
              f"[벤치] 롱홀드 노출{rows[1]['평균노출']:.2f}")
    rows.append(bh)
    # 같은 노출 프로파일·무작위 부호 귀무 (타이밍만 파괴)
    tl = [(i, sd, wd["동일"], H) for i, sd, wd, H in trades]
    nulls = []
    for b in range(200):
        rnd = [(i, int(rng.choice([-1, 1])), w, H) for i, sd, w, H in tl]
        pr, _, _, _ = simulate(rnd, len(lc), r, 1.0)
        nulls.append(pd.Series(pr[mask]).groupby(ts[mask].dt.floor("D").values).sum().mean() * 1e4)
    nulls = np.array(nulls)
    act = [x for x in rows if x["구성"] == "WF/동일/상한1.0"][0]["일평균bp"]
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "walkforward.csv", index=False)
    print()
    print(D[["구성", "일평균bp", "샤프", "MDD%", "누적%", "CI_lo", "CI_hi", "0배제", "e24", "e25",
             "e26", "평균노출", "최대노출", "거래", "거래당bp"]].to_string(
        index=False, float_format=lambda x: f"{x:9.2f}"))
    print(f"\n🔴같은 노출·무작위 부호 귀무(B=200): 평균 {nulls.mean():+.2f}bp · "
          f"95분위 {np.quantile(nulls,0.95):+.2f} · 실제 {act:+.2f} · **백분위 {(nulls<act).mean():.1%}**")
    print(f"\n저장: {OUT/'walkforward.csv'}")


# ── 축 4c. ⭐⭐9자산 워크포워드 — 검정력을 정당하게 늘리는 유일한 길 ──────────
def wf_asset(asset: str, a, rng):
    """한 자산의 워크포워드 원장. 규칙 선택도 그 자산 자기 이력만으로 한다."""
    p, lc, yr, dayno, cells = materialize(asset)
    ts = p["timestamp"]; tsv = ts.to_numpy()
    r = np.zeros(len(lc)); r[1:] = np.diff(lc)
    trades, nsel = [], []
    for m0 in pd.date_range(a.start, "2026-08-01", freq="MS"):
        m1 = m0 + pd.DateOffset(months=1)
        sel = []
        for key, (keep, pnl, exc, H) in cells.items():
            tk = tsv[keep]; tr = tk < np.datetime64(m0)
            if tr.sum() < a.minn:
                continue
            pn, ex = pnl[tr], exc[tr]
            if pn.mean() <= 0 or ex.mean() <= 0:
                continue
            sd = ex.std(ddof=1)
            if sd <= 0 or ex.mean() / (sd / np.sqrt(len(ex))) < a.tsel:
                continue
            yrs = pd.Series(tk[tr]).dt.year.to_numpy()
            done = [y for y in np.unique(yrs) if y < m0.year and (yrs == y).sum() >= 30]
            if len(done) < 1 or not all(pn[yrs == y].mean() > 0 for y in done[-2:]):
                continue
            sel.append((key, float(ex.mean()), int(len(ex))))
        sel = sorted(sel, key=lambda x: -x[1])[:a.top]
        nsel.append(len(sel))
        for key, ex, nn in sel:
            keep, pnl, _, H = cells[key]
            te = keep[(tsv[keep] >= np.datetime64(m0)) & (tsv[keep] < np.datetime64(m1))]
            trades += [(int(i), key[4], 1.0, H) for i in te]
    pr, expo, gross, took = simulate(trades, len(lc), r, a.cap)
    mask = (ts >= pd.Timestamp(a.start)).to_numpy()
    g = pd.DataFrame({"pr": pr[mask], "day": ts[mask].dt.floor("D").values,
                      "y": yr[mask]}).groupby("day").agg(pr=("pr", "sum"), y=("y", "first"))
    return g, took, float(np.median(nsel)), float(gross[mask].mean())


def stage_wfmulti(a):
    """**9자산 워크포워드 합산.** 효과 크기를 키우는 조작이 아니라 **관측 수를 늘리는 것**이다
    (§5.30 B-2 가 `doi_12h` 하나에 대해 한 것을 파이프라인 전체에 한다).
    🔴크립토 자산은 베타로 상관돼 **독립 9개가 아니다** — 합산 CI 는 날짜블록으로 내되
    「독립 9자산」이라고 주장하지 않는다. 그리고 9자산 전부 **현재 상장 중**이라 생존편향이 있다."""
    rng = np.random.default_rng(SEED)
    ASSETS = ["ETH", "BTC", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK"]
    books, rows = {}, []
    for A in ASSETS:
        try:
            g, took, nsel, mexp = wf_asset(A, a, rng)
        except Exception as e:  # 데이터 없는 자산은 건너뛰되 조용히 넘기지 않는다
            print(f"  [{A}] 건너뜀: {type(e).__name__} {e}")
            continue
        books[A] = g
        days = g.index.to_numpy()
        bs = np.array([g.pr.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4
                       for _ in range(1200)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        ys = {y: float(g[g.y == y].pr.mean() * 1e4) if (g.y == y).sum() > 20 else np.nan
              for y in YEARS}
        rows.append({"자산": A, "일평균bp": g.pr.mean() * 1e4,
                     "샤프": float(g.pr.mean() / g.pr.std() * np.sqrt(365)),
                     "MDD%": float((g.pr.cumsum() - g.pr.cumsum().cummax()).min() * 100),
                     "누적%": float(g.pr.sum() * 100), "CI_lo": lo, "CI_hi": hi, "0배제": lo > 0,
                     "e25": ys[2025], "e26": ys[2026], "거래": took, "월선택": nsel, "평균노출": mexp})
        print(f"  [{A}] 완료 · 일 {rows[-1]['일평균bp']:+.2f}bp · 거래 {took}")
    D = pd.DataFrame(rows)
    # 합산 북: 자산 동일가중
    allday = sorted(set().union(*[set(g.index) for g in books.values()]))
    P = pd.DataFrame(index=allday)
    for A, g in books.items():
        P[A] = g.pr.reindex(allday).fillna(0.0)
    P["port"] = P[list(books)].mean(axis=1)
    days = P.index.to_numpy()
    bs = np.array([P.port.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4
                   for _ in range(3000)])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    yrv = pd.Series(pd.to_datetime(P.index).year, index=P.index)
    print("\n=== 자산별 ===")
    print(D.to_string(index=False, float_format=lambda x: f"{x:9.2f}"))
    print(f"\n=== ⭐9자산 동일가중 합산 (상한 {a.cap} · 비용 {COST_BP}bp · 시작 {a.start}) ===")
    print(f"  일평균 **{P.port.mean()*1e4:+.2f}bp** · 샤프 **{P.port.mean()/P.port.std()*np.sqrt(365):.2f}** · "
          f"MDD {(P.port.cumsum()-P.port.cumsum().cummax()).min()*100:.2f}% · 누적 {P.port.sum()*100:+.2f}%")
    print(f"  날짜블록 CI [{lo:+.2f}, {hi:+.2f}] · **0배제 {'✅' if lo>0 else '❌'}**")
    for y in YEARS:
        k = (yrv == y).to_numpy()
        if k.sum() > 20:
            print(f"  {y}: {P.port[k].mean()*1e4:+.2f}bp/일 ({int(k.sum())}일)")
    from scipy.stats import binomtest
    npos = int((D.일평균bp > 0).sum())
    print(f"  양수 자산 {npos}/{len(D)} · 부호검정 p "
          f"{binomtest(npos, len(D), 0.5, 'greater').pvalue:.4f}  "
          f"(🔴자산 상관 때문에 낙관적이다 — 독립 9개가 아니다)")
    P.to_csv(OUT / f"wfmulti_daily_top{a.top}_cap{a.cap}.csv")
    D.to_csv(OUT / f"wfmulti_assets_top{a.top}_cap{a.cap}.csv", index=False)
    print(f"\n저장: {OUT/f'wfmulti_assets_top{a.top}_cap{a.cap}.csv'}")


# ── 축 4d. ⭐⭐⭐자산 제외 선택 (leave-one-asset-out) ─────────────────────────
ASSETS9 = ["ETH", "BTC", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK"]


def materialize_light(asset: str, cols: list[str]):
    """워크포워드용 최소 재료. 패널은 버리고 셀별 (봉인덱스·시각·순손익·초과) 만 남긴다."""
    p, lc, yr, dayno = prep(asset)
    ts = p["timestamp"].to_numpy()
    TH = {c: causal_thresholds(p[c].to_numpy(float), dayno) for c in cols if c in p.columns}
    cells = {}
    for hn, H in HS.items():
        fwd = fwd_of(lc, H); ok = np.isfinite(fwd)
        bgy = year_background(fwd, yr, H)
        for c in TH:
            v = p[c].to_numpy(float); lo, hi = TH[c]
            for j, q in enumerate(QS):
                for cond, mask in (("하위", v <= lo[:, j]), ("상위", v >= hi[:, j])):
                    keep = nonoverlap(np.flatnonzero(mask & ok & np.isfinite(lo[:, j])), H)
                    if len(keep) < 150:
                        continue
                    for side in (1, -1):
                        cells[(c, hn, q, cond, side)] = (
                            keep.astype(np.int32), ts[keep],
                            (side * fwd[keep] - COST_BP).astype(np.float32),
                            (side * fwd[keep] - side * bgy[keep]).astype(np.float32), H)
    r = np.zeros(len(lc)); r[1:] = np.diff(lc)
    return cells, p["timestamp"], r, yr, len(lc)


def stage_loao(a):
    """⭐**규칙을 «다른 자산들」로 고르고 이 자산에서 거래한다.**

    왜 이게 남은 유일한 정당한 설계인가: `--stage wf` 는 4,030셀에서 그 자산 자기 과거 성적으로
    상위 8개를 골랐다 — 그 자체가 과적합 기계이고 9자산 합산이 **−10.11bp/일 · 양수 2/9** 로
    그걸 확인했다. 자산을 빼고 고르면 선택이 **시간축으로도 자산축으로도** 표본외가 된다.
    한 자산에서만 통하는 규칙은 아예 뽑히지 않는다.
    🔴크립토 자산은 상관돼 있어 완전한 독립은 아니다 — 그래도 「그 자산의 과거 성적」보다는
    훨씬 강한 관문이다. 그리고 이 설계는 §5.30 B-2 의 9자산 이식을 **선택 단계로** 옮긴 것이다."""
    rng = np.random.default_rng(SEED)
    base = cached_panel("XRP")           # 틱 피쳐 없는 자산 기준 = 9자산 공통 피쳐
    cols = [c for c in featcols(base) if c != "hour_f"]
    print(f"9자산 공통 피쳐 {len(cols)}개")
    M = {}
    for A in ASSETS9:
        M[A] = materialize_light(A, cols)
        print(f"  [{A}] 셀 {len(M[A][0]):,}")
    keys = sorted(set.intersection(*[set(M[A][0]) for A in ASSETS9]), key=str)
    print(f"공통 셀 {len(keys):,}\n")
    books, rows = {}, []
    for tgt in ASSETS9:
        trades, nsel = [], []
        for m0 in pd.date_range(a.start, "2026-08-01", freq="MS"):
            m1 = m0 + pd.DateOffset(months=1)
            sel = []
            for key in keys:
                pn, ex, nn, npos = [], [], 0, 0
                for A in ASSETS9:
                    if A == tgt:
                        continue
                    _, tk, p_, e_, _ = M[A][0][key]
                    tr = tk < np.datetime64(m0)
                    if tr.sum() < 60:
                        continue
                    pn.append(p_[tr]); ex.append(e_[tr]); nn += int(tr.sum())
                    npos += int(p_[tr].mean() > 0)
                if len(pn) < 6 or nn < a.minn:
                    continue
                PN = np.concatenate(pn); EX = np.concatenate(ex)
                if PN.mean() <= 0 or EX.mean() <= 0 or npos < a.minpos:
                    continue
                sd = EX.std(ddof=1)
                if sd <= 0 or EX.mean() / (sd / np.sqrt(len(EX))) < a.tsel:
                    continue
                sel.append((key, float(EX.mean()), float(PN.mean()), npos))
            sel = sorted(sel, key=lambda x: -x[1])[:a.top]
            nsel.append(len(sel))
            idx, tk, _, _, H = None, None, None, None, None
            for key, ex, pnm, npos in sel:
                idx, tk, _, _, H = M[tgt][0][key]
                te = idx[(tk >= np.datetime64(m0)) & (tk < np.datetime64(m1))]
                trades += [(int(i), key[4], 1.0, H) for i in te]
        _, ts, r, yr, n = M[tgt]
        pr, expo, gross, took = simulate(trades, n, r, a.cap)
        mask = (ts >= pd.Timestamp(a.start)).to_numpy()
        g = pd.DataFrame({"pr": pr[mask], "day": ts[mask].dt.floor("D").values,
                          "y": yr[mask]}).groupby("day").agg(pr=("pr", "sum"), y=("y", "first"))
        books[tgt] = g
        days = g.index.to_numpy()
        bs = np.array([g.pr.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4
                       for _ in range(1200)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        rows.append({"자산": tgt, "일평균bp": g.pr.mean() * 1e4,
                     "샤프": float(g.pr.mean() / g.pr.std() * np.sqrt(365)),
                     "MDD%": float((g.pr.cumsum() - g.pr.cumsum().cummax()).min() * 100),
                     "누적%": float(g.pr.sum() * 100), "CI_lo": lo, "CI_hi": hi, "0배제": lo > 0,
                     "e25": float(g[g.y == 2025].pr.mean() * 1e4),
                     "e26": float(g[g.y == 2026].pr.mean() * 1e4),
                     "거래": took, "월선택": float(np.median(nsel)), "평균노출": float(gross[mask].mean())})
        print(f"  [{tgt}] 일 {rows[-1]['일평균bp']:+.2f}bp · 거래 {took} · 월선택중앙 {np.median(nsel):.0f}")
    D = pd.DataFrame(rows)
    allday = sorted(set().union(*[set(g.index) for g in books.values()]))
    P = pd.DataFrame(index=allday)
    for A, g in books.items():
        P[A] = g.pr.reindex(allday).fillna(0.0)
    P["port"] = P[ASSETS9].mean(axis=1)
    days = P.index.to_numpy()
    bs = np.array([P.port.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4 for _ in range(3000)])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    yrv = pd.Series(pd.to_datetime(P.index).year, index=P.index)
    print("\n=== 자산별 (규칙은 나머지 8자산으로 선택) ===")
    print(D.to_string(index=False, float_format=lambda x: f"{x:9.2f}"))
    from scipy.stats import binomtest
    npos = int((D.일평균bp > 0).sum())
    print(f"\n=== ⭐⭐9자산 동일가중 합산 (상한 {a.cap} · 비용 {COST_BP}bp · top{a.top}) ===")
    print(f"  일평균 **{P.port.mean()*1e4:+.2f}bp** · 샤프 **{P.port.mean()/P.port.std()*np.sqrt(365):.2f}** · "
          f"MDD {(P.port.cumsum()-P.port.cumsum().cummax()).min()*100:.2f}% · 누적 {P.port.sum()*100:+.2f}%")
    print(f"  날짜블록 CI [{lo:+.2f}, {hi:+.2f}] · **0배제 {'✅' if lo>0 else '❌'}**")
    for y in YEARS:
        k = (yrv == y).to_numpy()
        if k.sum() > 20:
            print(f"  {y}: {P.port[k].mean()*1e4:+.2f}bp/일 ({int(k.sum())}일)")
    print(f"  양수 자산 {npos}/9 · 부호검정 p {binomtest(npos, 9, 0.5, 'greater').pvalue:.4f}")
    P.to_csv(OUT / f"loao_daily_top{a.top}.csv"); D.to_csv(OUT / f"loao_assets_top{a.top}.csv", index=False)
    print(f"\n저장: {OUT/f'loao_assets_top{a.top}.csv'}")


# ── 축 4e. ⭐⭐⭐고정 규칙 이식 — 문서의 «실제 주장»을 정면으로 ──────────────
def stage_fixed(a):
    """**ETH 에서 찾은 규칙을 한 글자도 안 바꾸고 나머지 8자산에 건다.**

    왜 이게 따로 필요한가: `--stage wf/loao` 는 **매달 3,400셀에서 고르는 파이프라인**을 검정했고
    그건 문서의 주장이 아니다. §5.30 B-2 의 주장은 「**이 한 규칙**이 9자산에서 부호가 일관된다」
    이다. 선택 잡음이 0인 고정 규칙은 완전히 다른 물건이다.
    ⭐이 설계에서 **ETH 는 표본내**(규칙을 거기서 찾았다)이고 **나머지 8자산은 표본외**다 —
    자산축 표본외가 이 데이터에서 얻을 수 있는 유일한 진짜 표본외다. 둘을 갈라 싣는다."""
    rng = np.random.default_rng(SEED)
    src = {"doc": ROOT / "data/research/eth_past_failures_events_20260915/eth_final_clean.csv",
           "mine": OUT / "eth_confirm.csv"}[a.rules]
    R = pd.read_csv(src)
    if a.rules == "mine":
        R = R[R.ok]
    if a.only:
        R = R[R.feat == a.only]
    if a.fam1:
        R["fam2"] = R.feat.str.replace(r"_(15m|1h|4h|12h|1d|z)$", "", regex=True)
        R = R.sort_values("exc" if "exc" in R else "net", ascending=False).groupby("fam2").head(1)
    print(f"고정 규칙 {len(R)}개 (출처 {a.rules}{' · 군당 1개' if a.fam1 else ''})")
    print(R[["feat", "H", "q", "cond", "side"]].to_string(index=False))
    books, rows = {}, []
    for A in ASSETS9:
        p, lc, yr, dayno = prep(A)
        ts = p["timestamp"]
        r = np.zeros(len(lc)); r[1:] = np.diff(lc)
        trades = []
        for _, x in R.iterrows():
            if x["feat"] not in p.columns:
                continue
            H = HS[x["H"]]
            keep = rule_events(p, dayno, x["feat"], H, float(x["q"]), x["cond"], np.isfinite(fwd_of(lc, H)))
            side = 1 if x["side"] == "롱" else -1
            trades += [(int(i), side, 1.0, H) for i in keep]
        pr, expo, gross, took = simulate(trades, len(lc), r, a.cap)
        g = pd.DataFrame({"pr": pr, "day": ts.dt.floor("D").values, "y": yr}
                         ).groupby("day").agg(pr=("pr", "sum"), y=("y", "first"))
        books[A] = g
        days = g.index.to_numpy()
        bs = np.array([g.pr.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4
                       for _ in range(1500)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        nl, nltook = random_entry_null(trades, len(lc), r, a.cap, ts, rng, B=a.nullb)
        act = g.pr.mean() * 1e4
        bh = pd.Series(r * gross.mean()).groupby(ts.dt.floor("D").values).sum().mean() * 1e4
        rows.append({"자산": A, "표본": "내(규칙출처)" if A == "ETH" else "외",
                     "일평균bp": act, "무작위시점귀무": float(nl.mean()),
                     "초과": act - float(nl.mean()), "백분위": float((nl < act).mean()),
                     "롱홀드동노출": bh, "후보": len(trades), "귀무체결": nltook,
                     "샤프": float(g.pr.mean() / g.pr.std() * np.sqrt(365)),
                     "MDD%": float((g.pr.cumsum() - g.pr.cumsum().cummax()).min() * 100),
                     "누적%": float(g.pr.sum() * 100), "CI_lo": lo, "CI_hi": hi, "0배제": lo > 0,
                     **{f"e{y}": float(g[g.y == y].pr.mean() * 1e4) for y in YEARS},
                     "거래": took, "평균노출": float(gross.mean())})
        print(f"  [{A}] {act:+.2f}bp/일 · 무작위시점 {nl.mean():+.2f} · 초과 {act-nl.mean():+.2f} "
              f"· 백분위 {(nl<act).mean():.0%} · 체결 {took}/{len(trades)} (귀무 {nltook:.0f})")
    D = pd.DataFrame(rows)
    allday = sorted(set().union(*[set(g.index) for g in books.values()]))
    P = pd.DataFrame(index=allday)
    for A, g in books.items():
        P[A] = g.pr.reindex(allday).fillna(0.0)
    oos = [A for A in ASSETS9 if A != "ETH"]
    P["oos8"] = P[oos].mean(axis=1); P["all9"] = P[ASSETS9].mean(axis=1)
    yrv = pd.Series(pd.to_datetime(P.index).year, index=P.index)
    print("\n=== 자산별 (규칙 고정 · 재선택 없음) ===")
    print(D.to_string(index=False, float_format=lambda x: f"{x:9.2f}"))
    from scipy.stats import binomtest
    days = P.index.to_numpy()
    for nm, col, n_ in (("⭐표본외 8자산 합산", "oos8", 8), ("전체 9자산 합산", "all9", 9)):
        bs = np.array([P[col].loc[rng.choice(days, len(days), replace=True)].mean() * 1e4
                       for _ in range(3000)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        sub = D[D.자산 != "ETH"] if col == "oos8" else D
        npos = int((sub.일평균bp > 0).sum())
        print(f"\n=== {nm} (상한 {a.cap} · 비용 {COST_BP}bp) ===")
        print(f"  일평균 **{P[col].mean()*1e4:+.2f}bp** · 샤프 **{P[col].mean()/P[col].std()*np.sqrt(365):.2f}**"
              f" · MDD {(P[col].cumsum()-P[col].cumsum().cummax()).min()*100:.2f}% · 누적 {P[col].sum()*100:+.2f}%")
        print(f"  날짜블록 CI [{lo:+.2f}, {hi:+.2f}] · **0배제 {'✅' if lo>0 else '❌'}**")
        for y in YEARS:
            k = (yrv == y).to_numpy()
            print(f"  {y}: {P[col][k].mean()*1e4:+.2f}bp/일")
        print(f"  양수 자산 {npos}/{n_} · 부호검정 p {binomtest(npos, n_, 0.5, 'greater').pvalue:.4f}")
    tag = f"{a.rules}{'_fam1' if a.fam1 else ''}_cap{a.cap}"
    P.to_csv(OUT / f"fixed_daily_{tag}.csv"); D.to_csv(OUT / f"fixed_assets_{tag}.csv", index=False)
    print(f"\n저장: {OUT/f'fixed_assets_{tag}.csv'}")


# ── 축 1a-2. 크로스자산 규칙의 표본외 이식 ──────────────────────────────────
def stage_crossfix(a):
    """BTC 사건으로 ETH 를 거래하는 규칙(표본내)을 **그대로 나머지 8자산에** 건다.

    크로스자산은 이 저장소 초행 축이라 그 자체로 가치가 있지만, 확정 75셀은 자산쌍마다
    ~4,000셀 탐색의 산물이다. 유일하게 정직한 확인은 **트리거 자산은 고정(BTC)하고 대상만
    바꾸는 것** — BTC→ETH 에서 고른 규칙이 BTC→{BNB,XRP,…} 에서도 사는가."""
    rng = np.random.default_rng(SEED)
    C = pd.read_csv(OUT / "cross_confirm.csv")
    C = C[(C.ok) & (C.src == a.src) & (C.tgt == "ETH")]
    if a.fam1:
        C["fam2"] = C.feat.str.replace(r"_(15m|1h|4h|12h|1d|z)$", "", regex=True)
        C = C.sort_values("exc", ascending=False).groupby("fam2").head(1)
    print(f"{a.src}→ETH 확정 규칙 {len(C)}개를 9자산에 그대로 이식")
    print(C[["feat", "H", "q", "cond", "side"]].to_string(index=False))
    ps, _, _, _ = prep(a.src)
    books, rows = {}, []
    for A in ASSETS9:
        pt, lct, yrt, dayt = prep(A)
        j = pt[["timestamp"]].merge(ps[["timestamp"] + sorted(C.feat.unique())],
                                    on="timestamp", how="left")
        r = np.zeros(len(lct)); r[1:] = np.diff(lct)
        trades = []
        for _, x in C.iterrows():
            H = HS[x["H"]]
            keep = rule_events(j, dayt, x["feat"], H, float(x["q"]), x["cond"],
                               np.isfinite(fwd_of(lct, H)))
            side = 1 if x["side"] == "롱" else -1
            trades += [(int(i), side, 1.0, H) for i in keep]
        pr, expo, gross, took = simulate(trades, len(lct), r, a.cap)
        ts = pt["timestamp"]
        g = pd.DataFrame({"pr": pr, "day": ts.dt.floor("D").values, "y": yrt}
                         ).groupby("day").agg(pr=("pr", "sum"), y=("y", "first"))
        books[A] = g
        days = g.index.to_numpy()
        bs = np.array([g.pr.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4
                       for _ in range(1500)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        nl, nltook = random_entry_null(trades, len(lct), r, a.cap, ts, rng, B=a.nullb)
        act = g.pr.mean() * 1e4
        bh = pd.Series(r * gross.mean()).groupby(ts.dt.floor("D").values).sum().mean() * 1e4
        rows.append({"대상": A, "표본": "내" if A == "ETH" else "외", "일평균bp": act,
                     "무작위시점귀무": float(nl.mean()), "초과": act - float(nl.mean()),
                     "백분위": float((nl < act).mean()), "롱홀드동노출": bh,
                     "후보": len(trades), "귀무체결": nltook,
                     "샤프": float(g.pr.mean() / g.pr.std() * np.sqrt(365)),
                     "MDD%": float((g.pr.cumsum() - g.pr.cumsum().cummax()).min() * 100),
                     "CI_lo": lo, "CI_hi": hi, "0배제": lo > 0,
                     **{f"e{y}": float(g[g.y == y].pr.mean() * 1e4) for y in YEARS}, "거래": took})
        print(f"  [{a.src}→{A}] {act:+.2f}bp/일 · 무작위시점 {nl.mean():+.2f} · "
              f"초과 {act-nl.mean():+.2f} · 백분위 {(nl<act).mean():.0%} · "
              f"롱홀드(동노출) {bh:+.2f} · 체결 {took}/{len(trades)} (귀무체결 {nltook:.0f})")
    D = pd.DataFrame(rows)
    allday = sorted(set().union(*[set(g.index) for g in books.values()]))
    P = pd.DataFrame(index=allday)
    for A, g in books.items():
        P[A] = g.pr.reindex(allday).fillna(0.0)
    oos = [A for A in ASSETS9 if A != "ETH"]
    P["oos8"] = P[oos].mean(axis=1)
    print("\n=== 대상별 ===")
    print(D.to_string(index=False, float_format=lambda x: f"{x:9.2f}"))
    from scipy.stats import binomtest
    days = P.index.to_numpy()
    bs = np.array([P.oos8.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4 for _ in range(3000)])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    npos = int((D[D.대상 != "ETH"].일평균bp > 0).sum())
    yrv = pd.Series(pd.to_datetime(P.index).year, index=P.index)
    print(f"\n=== ⭐표본외 8대상 합산 ({a.src} 트리거 고정) ===")
    print(f"  일평균 **{P.oos8.mean()*1e4:+.2f}bp** · 샤프 {P.oos8.mean()/P.oos8.std()*np.sqrt(365):.2f} · "
          f"CI [{lo:+.2f}, {hi:+.2f}] · **0배제 {'✅' if lo>0 else '❌'}**")
    for y in YEARS:
        print(f"  {y}: {P.oos8[(yrv==y).to_numpy()].mean()*1e4:+.2f}bp/일")
    print(f"  양수 대상 {npos}/8 · 부호검정 p {binomtest(npos, 8, 0.5, 'greater').pvalue:.4f}")
    D.to_csv(OUT / f"crossfix_{a.src}.csv", index=False)
    print(f"\n저장: {OUT/f'crossfix_{a.src}.csv'}")


# ── 축 3b. ⭐⭐E|r| 모델을 **표본외 설계 위에** 필터로 — ML 의 진짜 시험 ────────
def stage_fixedml(a):
    """`--stage size` 가 찾은 것: 사건에서 **예측 E|r| 이 방향 적중까지 단조로 예측**한다
    (오분위 적중 47.9%→56.1% · 워크포워드 IC +0.217). 그게 표본내 규칙 위에서만 통하는
    장식인지, **표본외 8자산에서도 손익을 바꾸는지**가 진짜 시험이다.

    설계: 규칙은 `--stage fixed` 와 **완전히 동일**(ETH 에서 찾은 고정 규칙, 재선택 없음).
    E|r| 모델만 얹는다 — 월 확장 워크포워드로 9자산 사건을 풀링 학습(인과), 예측 E|r| 이
    **하위 `--drop` 분위면 거래하지 않는다**. 비교는 같은 규칙·같은 비용·같은 상한이다.
    🔴ETH 는 규칙이 표본내라 여전히 표본내다. 판정은 **표본외 8자산**으로만 한다."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(SEED)
    R = pd.read_csv(ROOT / "data/research/eth_past_failures_events_20260915/eth_final_clean.csv")
    R["fam2"] = R.feat.str.replace(r"_(15m|1h|4h|12h|1d|z)$", "", regex=True)
    R = R.sort_values("exc", ascending=False).groupby("fam2").head(1)
    base = cached_panel("XRP")
    fcols = [c for c in featcols(base) if c != "hour_f"]
    print(f"고정 규칙 {len(R)}개 · 모델 피쳐 {len(fcols)}개 (9자산 공통)")
    D = {}
    for A in ASSETS9:
        p, lc, yr, dayno = prep(A)
        ts = p["timestamp"]
        r = np.zeros(len(lc)); r[1:] = np.diff(lc)
        recs = []
        for ri, (_, x) in enumerate(R.iterrows()):
            if x["feat"] not in p.columns:
                continue
            H = HS[x["H"]]; fwd = fwd_of(lc, H)
            keep = rule_events(p, dayno, x["feat"], H, float(x["q"]), x["cond"], np.isfinite(fwd))
            side = 1 if x["side"] == "롱" else -1
            for i in keep:
                recs.append((ts.iloc[i], int(i), ri, side, H, float(fwd[i])))
        E = pd.DataFrame(recs, columns=["ts", "i", "rule", "side", "H", "fwd"]).sort_values("ts")
        X = np.hstack([p.iloc[E.i][fcols].to_numpy(np.float32),
                       E[["rule", "H"]].to_numpy(np.float32)])
        D[A] = dict(E=E, X=X, y=np.log(np.maximum(np.abs(E.fwd.to_numpy()), 1.0)),
                    ts=ts, r=r, yr=yr, n=len(lc))
        print(f"  [{A}] 사건 {len(E):,}")
    for A in ASSETS9:
        D[A]["pred"] = np.full(len(D[A]["E"]), np.nan)
    for m0 in pd.date_range("2024-07-01", "2026-08-01", freq="MS"):
        m1 = m0 + pd.DateOffset(months=1)
        Xs, ys = [], []
        for A in ASSETS9:
            tr = (D[A]["E"].ts < m0).to_numpy()
            if tr.sum():
                Xs.append(D[A]["X"][tr]); ys.append(D[A]["y"][tr])
        Xt = np.vstack(Xs); yt = np.concatenate(ys)
        if len(yt) < 500:
            continue
        mdl = [HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                             l2_regularization=3.0, random_state=sd).fit(Xt, yt)
               for sd in (11, 907)]
        for A in ASSETS9:
            te = ((D[A]["E"].ts >= m0) & (D[A]["E"].ts < m1)).to_numpy()
            if te.sum():
                D[A]["pred"][te] = np.mean([m.predict(D[A]["X"][te]) for m in mdl], axis=0)
    # 예측 분위는 **학습 시점까지의 예측 분포**로 자른다(전수 분위는 미래참조)
    books = {}
    rows = []
    for A in ASSETS9:
        E = D[A]["E"].copy(); E["pred"] = D[A]["pred"]
        ok = np.isfinite(E.pred.to_numpy())
        E = E[ok]
        thr = pd.Series(E.pred.to_numpy()).expanding(200).quantile(a.drop).shift(1).to_numpy()
        take = np.isfinite(thr) & (E.pred.to_numpy() > thr)
        for nm, sel in (("전량(ML 없음)", np.isfinite(thr)), (f"ML 하위{a.drop:.0%} 제외", take)):
            # 🔴**건당 비중 w 는 상한과 함께 정해야 한다.** w=1.0·상한 1.0 이면 겹치는 진입이
            #   통째로 거절된다 — 선택 사건의 **49.8%가 직전 보유구간과 겹치므로** 좋은 구간
            #   (사건은 변동성 구간에 뭉친다)을 골라서 버린다. 실측: 상한 없는 이론값
            #   +26.95bp/일 vs w=1.0 상한1.0 측정 +3.14bp/일 = **8.6배 손실**. w 를 나누면
            #   같은 평균 노출에서 동시 보유가 가능해진다.
            trades = [(int(i), int(sd), a.w, int(h))
                      for i, sd, h, k in zip(E.i, E.side, E.H, sel) if k]
            pr, expo, gross, took = simulate(trades, D[A]["n"], D[A]["r"], a.cap)
            ts, yr = D[A]["ts"], D[A]["yr"]
            g = pd.DataFrame({"pr": pr, "day": ts.dt.floor("D").values, "y": yr}
                             ).groupby("day").agg(pr=("pr", "sum"), y=("y", "first"))
            books[(A, nm)] = g
            rows.append({"자산": A, "구성": nm, "표본": "내" if A == "ETH" else "외",
                         "일평균bp": g.pr.mean() * 1e4,
                         "샤프": float(g.pr.mean() / g.pr.std() * np.sqrt(365)),
                         "거래": took})
        print(f"  [{A}] 전량 {rows[-2]['일평균bp']:+.2f} → ML {rows[-1]['일평균bp']:+.2f} bp/일")
    # ⭐표본외 8자산에서 «예측 E|r| 오분위별 방향 적중»이 유지되는가 — ETH 표본내에서 본
    #   단조 상승(47.9%→56.1%)이 규칙이 표본외인 자산에서도 사는지가 이 모델의 진짜 시험이다.
    EV = []
    for A in ASSETS9:
        e = D[A]["E"].copy(); e["pred"] = D[A]["pred"]; e["asset"] = A
        EV.append(e[np.isfinite(e.pred)])
    EV = pd.concat(EV, ignore_index=True)
    EV["ehat"] = np.exp(EV.pred); EV["absr"] = EV.fwd.abs()
    EV["pnl"] = EV.side * EV.fwd - COST_BP
    EV.to_csv(OUT / "fixedml_events.csv", index=False)
    for lab, sub in (("ETH(규칙 표본내)", EV[EV.asset == "ETH"]),
                     ("표본외 8자산", EV[EV.asset != "ETH"])):
        qn = pd.qcut(sub.ehat, 5, labels=False)
        ic = float(pd.Series(sub.ehat.to_numpy()).corr(pd.Series(sub.absr.to_numpy()), method="spearman"))
        print(f"\n  [{lab}] n={len(sub):,} · E|r| 예측 스피어만 IC **{ic:+.3f}**")
        print(f"  {'분위':>4} {'n':>6} {'예측E|r|':>9} {'실제E|r|':>9} {'방향적중':>8} {'평균net':>9}")
        for gq in range(5):
            k = (qn == gq).to_numpy()
            print(f"  {gq+1:>4} {int(k.sum()):>6} {sub.ehat[k].mean():>9.1f} {sub.absr[k].mean():>9.1f} "
                  f"{float(((sub.side * sub.fwd)[k] > 0).mean()):>8.1%} {sub.pnl[k].mean():>+9.2f}")
    T = pd.DataFrame(rows)
    print("\n=== 자산별 ===")
    print(T.pivot_table(index="자산", columns="구성", values=["일평균bp", "샤프", "거래"]).to_string(
        float_format=lambda x: f"{x:8.2f}"))
    oos = [A for A in ASSETS9 if A != "ETH"]
    from scipy.stats import binomtest
    for nm in ("전량(ML 없음)", f"ML 하위{a.drop:.0%} 제외"):
        allday = sorted(set().union(*[set(books[(A, nm)].index) for A in ASSETS9]))
        P = pd.DataFrame(index=allday)
        for A in oos:
            P[A] = books[(A, nm)].pr.reindex(allday).fillna(0.0)
        port = P[oos].mean(axis=1)
        days = P.index.to_numpy()
        bs = np.array([port.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4
                       for _ in range(3000)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        sub = T[(T.구성 == nm) & (T.자산 != "ETH")]
        npos = int((sub.일평균bp > 0).sum())
        yrv = pd.Series(pd.to_datetime(P.index).year, index=P.index)
        print(f"\n=== ⭐표본외 8자산 · {nm} ===")
        print(f"  일평균 **{port.mean()*1e4:+.2f}bp** · 샤프 {port.mean()/port.std()*np.sqrt(365):.2f} · "
              f"MDD {(port.cumsum()-port.cumsum().cummax()).min()*100:.2f}% · "
              f"CI [{lo:+.2f}, {hi:+.2f}] **{'0배제 ✅' if lo>0 else '0포함 ❌'}**")
        print("  " + " · ".join(f"{y}: {port[(yrv==y).to_numpy()].mean()*1e4:+.2f}" for y in YEARS))
        print(f"  양수 자산 {npos}/8 · 부호검정 p {binomtest(npos, 8, 0.5, 'greater').pvalue:.4f}")
    T.to_csv(OUT / f"fixedml_drop{int(a.drop*100)}_w{a.w}_cap{a.cap}.csv", index=False)
    print(f"\n저장: {OUT/f'fixedml_drop{int(a.drop*100)}_w{a.w}_cap{a.cap}.csv'}")


# ── 축 1a-3. ⭐⭐⭐크로스자산 × 시간 워크포워드 = 마지막 관문 ────────────────
def stage_crosswf(a):
    """**트리거는 BTC 고정, 규칙 선택은 「다른 대상들 × 그 시점까지」.**

    `--stage crossfix` 가 표본외 8대상에서 +21.08bp/일(8/8·CI 0배제)을 냈지만 그건
    **시간축으로는 표본내**다 — 규칙을 ETH 의 2024~2026 수익으로 골랐고, 8개 «표본외» 대상은
    ETH 와 0.8 이상 상관이라 사실상 같은 기간·같은 신호를 다시 본 것이다.
    여기서 선택을 **시간축으로도** 밀어낸다: 매달, 대상 자산을 빼고, 그 시점까지의 자료만으로
    BTC 트리거 규칙을 고르고 다음 달을 거래한다. 이게 이 축의 마지막 관문이다."""
    rng = np.random.default_rng(SEED)
    base = cached_panel("XRP")
    cols = [c for c in featcols(base) if c != "hour_f"]
    ps, _, _, _ = prep(a.src)
    print(f"트리거 {a.src} · 공통 피쳐 {len(cols)}개")
    M = {}
    for A in ASSETS9:
        pt, lct, yrt, dayt = prep(A)
        j = pt[["timestamp"]].merge(ps[["timestamp"] + cols], on="timestamp", how="left")
        ts = pt["timestamp"].to_numpy()
        cells = {}
        TH = {c: causal_thresholds(j[c].to_numpy(float), dayt) for c in cols}
        for hn, H in HS.items():
            fwd = fwd_of(lct, H); okm = np.isfinite(fwd); bgy = year_background(fwd, yrt, H)
            for c in cols:
                v = j[c].to_numpy(float); lo, hi = TH[c]
                for q_i, q in enumerate(QS):
                    for cond, mask in (("하위", v <= lo[:, q_i]), ("상위", v >= hi[:, q_i])):
                        keep = nonoverlap(np.flatnonzero(mask & okm & np.isfinite(lo[:, q_i])), H)
                        if len(keep) < 150:
                            continue
                        for side in (1, -1):
                            cells[(c, hn, q, cond, side)] = (
                                keep.astype(np.int32), ts[keep],
                                (side * fwd[keep] - COST_BP).astype(np.float32),
                                (side * fwd[keep] - side * bgy[keep]).astype(np.float32), H)
        r = np.zeros(len(lct)); r[1:] = np.diff(lct)
        M[A] = (cells, pt["timestamp"], r, yrt, len(lct))
        print(f"  [{A}] 셀 {len(cells):,}")
    keys = sorted(set.intersection(*[set(M[A][0]) for A in ASSETS9]), key=str)
    print(f"공통 셀 {len(keys):,}\n")
    books, rows = {}, []
    for tgt in ASSETS9:
        trades, nsel = [], []
        for m0 in pd.date_range(a.start, "2026-08-01", freq="MS"):
            m1 = m0 + pd.DateOffset(months=1)
            sel = []
            for key in keys:
                pn, ex, npos = [], [], 0
                for A in ASSETS9:
                    if A == tgt:
                        continue
                    _, tk, p_, e_, _ = M[A][0][key]
                    tr = tk < np.datetime64(m0)
                    if tr.sum() < 60:
                        continue
                    pn.append(p_[tr]); ex.append(e_[tr]); npos += int(p_[tr].mean() > 0)
                if len(pn) < 6:
                    continue
                PN = np.concatenate(pn); EX = np.concatenate(ex)
                if len(PN) < a.minn or PN.mean() <= 0 or EX.mean() <= 0 or npos < a.minpos:
                    continue
                sd = EX.std(ddof=1)
                if sd <= 0 or EX.mean() / (sd / np.sqrt(len(EX))) < a.tsel:
                    continue
                sel.append((key, float(EX.mean())))
            sel = sorted(sel, key=lambda x: -x[1])[:a.top]
            nsel.append(len(sel))
            for key, _ in sel:
                idx, tk, _, _, H = M[tgt][0][key]
                te = idx[(tk >= np.datetime64(m0)) & (tk < np.datetime64(m1))]
                trades += [(int(i), key[4], 1.0, H) for i in te]
        cells, ts, r, yr, n = M[tgt]
        pr, expo, gross, took = simulate(trades, n, r, a.cap)
        mask = (ts >= pd.Timestamp(a.start)).to_numpy()
        g = pd.DataFrame({"pr": pr[mask], "day": ts[mask].dt.floor("D").values, "y": yr[mask]}
                         ).groupby("day").agg(pr=("pr", "sum"), y=("y", "first"))
        books[tgt] = g
        nl, nltook = random_entry_null(trades, n, r, a.cap, ts, rng, B=max(a.nullb // 3, 30))
        act = g.pr.mean() * 1e4
        rows.append({"대상": tgt, "일평균bp": act, "순환귀무": float(nl.mean()),
                     "초과": act - float(nl.mean()), "백분위": float((nl < act).mean()),
                     "샤프": float(g.pr.mean() / g.pr.std() * np.sqrt(365)),
                     "MDD%": float((g.pr.cumsum() - g.pr.cumsum().cummax()).min() * 100),
                     "e25": float(g[g.y == 2025].pr.mean() * 1e4),
                     "e26": float(g[g.y == 2026].pr.mean() * 1e4),
                     "거래": took, "월선택": float(np.median(nsel))})
        print(f"  [{a.src}→{tgt}] {act:+.2f}bp/일 · 순환귀무 {nl.mean():+.2f} · "
              f"초과 {act-nl.mean():+.2f} · 백분위 {(nl<act).mean():.0%} · 거래 {took}")
    D = pd.DataFrame(rows)
    allday = sorted(set().union(*[set(g.index) for g in books.values()]))
    P = pd.DataFrame(index=allday)
    for A, g in books.items():
        P[A] = g.pr.reindex(allday).fillna(0.0)
    P["port"] = P[ASSETS9].mean(axis=1)
    days = P.index.to_numpy()
    bs = np.array([P.port.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4 for _ in range(3000)])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    yrv = pd.Series(pd.to_datetime(P.index).year, index=P.index)
    from scipy.stats import binomtest
    npos = int((D.일평균bp > 0).sum())
    print("\n=== 대상별 (규칙 = 나머지 8대상 × 그 시점까지) ===")
    print(D.to_string(index=False, float_format=lambda x: f"{x:9.2f}"))
    print(f"\n=== ⭐⭐{a.src} 트리거 · 9대상 합산 · 시간+자산 이중 표본외 ===")
    print(f"  일평균 **{P.port.mean()*1e4:+.2f}bp** · 샤프 **{P.port.mean()/P.port.std()*np.sqrt(365):.2f}** · "
          f"MDD {(P.port.cumsum()-P.port.cumsum().cummax()).min()*100:.2f}% · 누적 {P.port.sum()*100:+.2f}%")
    print(f"  날짜블록 CI [{lo:+.2f}, {hi:+.2f}] · **0배제 {'✅' if lo>0 else '❌'}**")
    for y in YEARS:
        k = (yrv == y).to_numpy()
        if k.sum() > 20:
            print(f"  {y}: {P.port[k].mean()*1e4:+.2f}bp/일 ({int(k.sum())}일)")
    print(f"  양수 대상 {npos}/9 · 부호검정 p {binomtest(npos, 9, 0.5, 'greater').pvalue:.4f}")
    print(f"  순환귀무 대비 초과 평균 {D.초과.mean():+.2f}bp · 백분위 중앙 {D.백분위.median():.0%}")
    P.to_csv(OUT / f"crosswf_daily_{a.src}.csv"); D.to_csv(OUT / f"crosswf_{a.src}.csv", index=False)
    print(f"\n저장: {OUT/f'crosswf_{a.src}.csv'}")


# ── 축 5. ⭐⭐⭐⭐완전 표본외 — LOAO 규칙선택 × 인과 E|r| 게이트 × 분할 비중 ─────
def stage_wfml(a):
    """**이 파일의 결론 실험.** 세 조각을 전부 표본외로 놓고 합친다.

    ① **규칙 선택** — 매월, 대상 자산을 빼고, 그 시점까지의 자료만으로 (시간+자산 이중 표본외)
    ② **E|r| 게이트** — 「지금 큰 움직임이 예상되는가」. 🔴타깃이 |수익| 이라 **규칙과 무관**하다
       ⇒ 사건이 아니라 **전 봉**에서 학습한다(자산 풀링·월 확장 WF·시간당 1봉 표집).
       그래서 이 게이트는 규칙 선택의 과적합을 물려받지 않는다.
       임계는 **인과 확장창 분위**(전수 분위는 미래참조).
    ③ **건당 비중 w** — w=1.0·상한 1.0 이면 겹치는 진입이 통째로 거절된다. 선택 사건의
       **49.8%가 직전 보유구간과 겹치므로**(사건은 변동성 구간에 뭉친다) 좋은 구간을 골라
       버린다: 상한 무시 이론값 +26.95bp/일 vs w=1.0 측정 +3.14bp/일. w 를 나누면 같은
       평균 노출로 동시 보유가 가능해진다.

    대조군 둘을 **반드시 같이** 낸다: 순환이동 귀무(체결 수 보존)와 같은 노출 롱홀드.
    게이트 없는 판도 같이 내서 **게이트의 증분**을 짝지어 잰다."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(SEED)
    base = cached_panel("XRP")
    cols = [c for c in featcols(base) if c != "hour_f"]
    print(f"9자산 공통 피쳐 {len(cols)}개 · 건당비중 {a.w} · 상한 {a.cap} · 게이트 하위{a.drop:.0%} 제외")
    M, PAN = {}, {}
    for A in ASSETS9:
        M[A] = materialize_light(A, cols)
        p, lc, yr, dayno = prep(A)
        PAN[A] = (p[cols].to_numpy(np.float32), lc, yr, p["timestamp"])
        print(f"  [{A}] 셀 {len(M[A][0]):,}")
    keys = sorted(set.intersection(*[set(M[A][0]) for A in ASSETS9]), key=str)
    print(f"공통 셀 {len(keys):,}\n")

    # ── ② E|r| 게이트: 전 봉 학습(시간당 1봉), 지평별 모델, 월 확장 WF ──────────
    PRED = {A: {hn: np.full(len(PAN[A][1]), np.nan) for hn in HS} for A in ASSETS9}
    STRIDE = 12
    for hn, H in HS.items():
        Y = {A: np.log(np.maximum(np.abs(fwd_of(PAN[A][1], H)), 1.0)) for A in ASSETS9}
        for m0 in pd.date_range("2024-07-01", "2026-08-01", freq="MS"):
            m1 = m0 + pd.DateOffset(months=1)
            Xs, ys = [], []
            for A in ASSETS9:
                X, lc, yr, ts = PAN[A]
                tr = np.flatnonzero(((ts < m0 - pd.Timedelta(minutes=5 * H)).to_numpy())
                                    & np.isfinite(Y[A]))[::STRIDE]
                if len(tr):
                    Xs.append(X[tr]); ys.append(Y[A][tr])
            if not Xs:
                continue
            Xt = np.vstack(Xs); yt = np.concatenate(ys)
            if len(yt) < 2000:
                continue
            mdl = [HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, max_depth=4,
                                                 l2_regularization=3.0, random_state=sd).fit(Xt, yt)
                   for sd in (11, 907)]
            for A in ASSETS9:
                X, lc, yr, ts = PAN[A]
                te = np.flatnonzero(((ts >= m0) & (ts < m1)).to_numpy())
                if len(te):
                    PRED[A][hn][te] = np.mean([m.predict(X[te]) for m in mdl], axis=0)
        print(f"  [E|r| 게이트 {hn}] 학습 완료")

    # ── ① LOAO 규칙선택 + ③ 분할 비중 ─────────────────────────────────────────
    books, rows = {}, []
    for tgt in ASSETS9:
        cand = []
        for m0 in pd.date_range(a.start, "2026-08-01", freq="MS"):
            m1 = m0 + pd.DateOffset(months=1)
            sel = []
            for key in keys:
                pn, ex, npos = [], [], 0
                for A in ASSETS9:
                    if A == tgt:
                        continue
                    _, tk, p_, e_, _ = M[A][0][key]
                    tr = tk < np.datetime64(m0)
                    if tr.sum() < 60:
                        continue
                    pn.append(p_[tr]); ex.append(e_[tr]); npos += int(p_[tr].mean() > 0)
                if len(pn) < 6:
                    continue
                PN = np.concatenate(pn); EX = np.concatenate(ex)
                if len(PN) < a.minn or PN.mean() <= 0 or EX.mean() <= 0 or npos < a.minpos:
                    continue
                sd_ = EX.std(ddof=1)
                if sd_ <= 0 or EX.mean() / (sd_ / np.sqrt(len(EX))) < a.tsel:
                    continue
                sel.append((key, float(EX.mean())))
            for key, _ in sorted(sel, key=lambda x: -x[1])[:a.top]:
                idx, tk, _, _, H = M[tgt][0][key]
                m = (tk >= np.datetime64(m0)) & (tk < np.datetime64(m1))
                for i in idx[m]:
                    cand.append((int(i), key[4], H, key[1]))
        cand.sort()
        cells, ts, r, yr, n = M[tgt]
        pr_ = np.array([PRED[tgt][hn][i] for i, _, _, hn in cand])
        thr = pd.Series(pr_).expanding(200).quantile(a.drop).shift(1).to_numpy()
        gate = np.isfinite(thr) & (pr_ > thr)
        for nm, sel_ in (("게이트 없음", np.isfinite(thr)), (f"E|r| 게이트", gate)):
            trades = [(i, sd_, a.w, H) for (i, sd_, H, _), k in zip(cand, sel_) if k]
            if not trades:
                continue
            prs, expo, gross, took = simulate(trades, n, r, a.cap)
            mask = (ts >= pd.Timestamp(a.start)).to_numpy()
            g = pd.DataFrame({"pr": prs[mask], "day": ts[mask].dt.floor("D").values,
                              "y": yr[mask]}).groupby("day").agg(pr=("pr", "sum"), y=("y", "first"))
            books[(tgt, nm)] = g
            nl, nltook = random_entry_null(trades, n, r, a.cap, ts, rng, B=a.nullb // 2)
            act = g.pr.mean() * 1e4
            bh = pd.Series(r[mask] * gross[mask].mean()).groupby(
                ts[mask].dt.floor("D").values).sum().mean() * 1e4
            rows.append({"자산": tgt, "구성": nm, "일평균bp": act, "순환귀무": float(nl.mean()),
                         "초과": act - float(nl.mean()), "백분위": float((nl < act).mean()),
                         "롱홀드동노출": bh,
                         "샤프": float(g.pr.mean() / g.pr.std() * np.sqrt(365)),
                         "MDD%": float((g.pr.cumsum() - g.pr.cumsum().cummax()).min() * 100),
                         "e25": float(g[g.y == 2025].pr.mean() * 1e4),
                         "e26": float(g[g.y == 2026].pr.mean() * 1e4),
                         "체결": took, "후보": len(trades), "평균노출": float(gross[mask].mean())})
        print(f"  [{tgt}] 게이트없음 {rows[-2]['일평균bp']:+.2f} → 게이트 {rows[-1]['일평균bp']:+.2f} bp/일 "
              f"· 귀무 {rows[-1]['순환귀무']:+.2f} · 체결 {rows[-1]['체결']}")
    T = pd.DataFrame(rows)
    T.to_csv(OUT / f"wfml_top{a.top}_w{a.w}_cap{a.cap}_drop{int(a.drop*100)}.csv", index=False)
    print("\n=== 자산별 ===")
    print(T.to_string(index=False, float_format=lambda x: f"{x:9.2f}"))
    from scipy.stats import binomtest
    daily = {}
    for nm in ("게이트 없음", "E|r| 게이트"):
        allday = sorted(set().union(*[set(books[(A, nm)].index) for A in ASSETS9 if (A, nm) in books]))
        P = pd.DataFrame(index=allday)
        for A in ASSETS9:
            if (A, nm) in books:
                P[A] = books[(A, nm)].pr.reindex(allday).fillna(0.0)
        port = P.mean(axis=1); daily[nm] = port
        days = P.index.to_numpy()
        bs = np.array([port.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4
                       for _ in range(3000)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        sub = T[T.구성 == nm]
        npos = int((sub.일평균bp > 0).sum())
        yrv = pd.Series(pd.to_datetime(P.index).year, index=P.index)
        print(f"\n=== ⭐9자산 합산 · {nm} (시간+자산 이중 표본외) ===")
        print(f"  일평균 **{port.mean()*1e4:+.2f}bp** · 샤프 **{port.mean()/port.std()*np.sqrt(365):.2f}** · "
              f"MDD {(port.cumsum()-port.cumsum().cummax()).min()*100:.2f}% · 누적 {port.sum()*100:+.2f}%")
        print(f"  날짜블록 CI [{lo:+.2f}, {hi:+.2f}] · **0배제 {'✅' if lo>0 else '❌'}**")
        print("  " + " · ".join(f"{y}: {port[(yrv==y).to_numpy()].mean()*1e4:+.2f}" for y in YEARS
                                if (yrv == y).sum() > 20))
        print(f"  양수 자산 {npos}/9 · 부호검정 p {binomtest(npos, 9, 0.5, 'greater').pvalue:.4f} · "
              f"순환귀무 대비 초과 평균 {sub.초과.mean():+.2f}bp · 백분위 중앙 {sub.백분위.median():.0%}")
    d = daily["E|r| 게이트"] - daily["게이트 없음"]
    days = d.index.to_numpy()
    bs = np.array([d.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4 for _ in range(3000)])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    print(f"\n⭐**게이트의 짝지은 증분**: {d.mean()*1e4:+.2f}bp/일 · CI [{lo:+.2f}, {hi:+.2f}] "
          f"**{'0배제 ✅' if lo>0 else '0포함 ❌'}**")
    print(f"\n저장: {OUT/f'wfml_top{a.top}_w{a.w}_cap{a.cap}_drop{int(a.drop*100)}.csv'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="저장된 스크린 CSV 재사용")
    ap.add_argument("--stage", required=True,
                    choices=["screen", "cross", "conj", "fdr", "oictl", "port", "size", "wf", "wfmulti", "loao", "fixed", "crossfix", "fixedml", "crosswf", "wfml"])
    ap.add_argument("--minn", type=int, default=120, help="WF 선택 최소 사건 수")
    ap.add_argument("--tsel", type=float, default=2.0, help="WF 선택 초과 t 임계")
    ap.add_argument("--start", default="2025-01-01", help="WF 거래 시작 월")
    ap.add_argument("--dedup", action="store_true", help="WF 선택에서 피쳐군마다 1개만")
    ap.add_argument("--top", type=int, default=999, help="WF 선택 상위 K")
    ap.add_argument("--cap", type=float, default=1.0, help="총 노출 상한")
    ap.add_argument("--nperm", type=int, default=200, help="순환이동 귀무 반복")
    ap.add_argument("--rules", default="doc", choices=["doc", "mine"], help="고정 규칙 출처")
    ap.add_argument("--fam1", action="store_true", help="피쳐군마다 1개만")
    ap.add_argument("--only", default=None, help="이 피쳐 규칙만")
    ap.add_argument("--src", default="BTC", help="크로스자산 트리거 출처")
    ap.add_argument("--nullb", type=int, default=100, help="무작위 시점 귀무 반복")
    ap.add_argument("--w", type=float, default=1.0, help="건당 비중(상한 대비)")
    ap.add_argument("--drop", type=float, default=0.4, help="ML: 예측 E|r| 하위 이 분위는 거래 안 함")
    ap.add_argument("--minpos", type=int, default=6, help="LOAO: 몇 개 자산에서 양수여야 하나(/8)")
    a = ap.parse_args()
    {"screen": stage_screen, "cross": stage_cross, "port": stage_port,
     "fdr": stage_fdr, "oictl": stage_oictl, "conj": stage_conj,
     "size": stage_size, "wf": stage_wf, "wfmulti": stage_wfmulti,
     "loao": stage_loao, "fixed": stage_fixed, "crossfix": stage_crossfix,
     "fixedml": stage_fixedml, "crosswf": stage_crosswf,
     "wfml": stage_wfml}[a.stage](a)
    return 0


def _selfcheck():
    """자명하지 않은 조각 셋: 인과 임계가 미래를 안 보는가 · 비겹침 · 같은날 배경."""
    rng = np.random.default_rng(0)
    n = 288 * 200
    dayno = np.arange(n) // 288
    v = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(50, 1, n - n // 2)])
    lo, hi = causal_thresholds(v, dayno, qs=(0.10,))
    mid = n // 2 + 288          # 레짐이 바뀐 다음날
    assert lo[mid, 0] < 5, f"인과 임계가 미래(평균 50) 를 봤다: {lo[mid,0]:.2f}"
    assert not np.isfinite(lo[:288 * MIN_HIST_DAYS, 0]).any(), "웜업 구간에 임계가 생겼다"
    k = nonoverlap(np.arange(100), 10)
    assert len(k) == 10 and k[1] - k[0] == 10, k
    fwd = np.where(dayno % 2 == 0, 100.0, -100.0)
    bg = day_background(fwd, dayno)
    assert abs(bg[0] - 100) < 1e-9 and abs(bg[288] + 100) < 1e-9, "같은날 배경이 틀렸다"
    e = np.array([1.0, 2.0, 3.0, 4.0]); dd = np.array([0, 0, 1, 1])
    l, h = dateblock_ci(e, dd, np.random.default_rng(1), B=500)
    assert l <= 2.5 <= h, (l, h)

    # ⭐이번에 실제로 틀렸던 두 곳 — 고친 뒤 다시 틀리면 여기서 걸린다.
    # ① 순환이동 귀무가 **체결 수를 보존**해야 한다. i.i.d. 무작위 진입은 군집이 풀려 상한에
    #    덜 걸리고 체결이 늘어 비용을 더 낸다 ⇒ 귀무가 부당하게 나빠지고 「백분위 100%」가 나온다.
    n = 20_000
    rr = rng.normal(0, 3e-4, n)
    tss = pd.Series(pd.date_range("2024-01-01", periods=n, freq="5min"))
    clustered = [(int(i), 1, 1.0, 48) for blk in range(0, n - 500, 2000)
                 for i in range(blk, blk + 300, 10)]
    _, _, _, took0 = simulate(clustered, n, rr, 1.0)
    _, tk = random_entry_null(clustered, n, rr, 1.0, tss, np.random.default_rng(3), B=20)
    assert abs(tk - took0) <= max(2.0, 0.02 * took0), f"귀무 체결 {tk:.0f} vs 실제 {took0}"
    assert took0 < len(clustered), "군집 거래인데 상한이 하나도 안 걸렸다 — 시험이 무의미"

    # ② `net > 0` 관문. 초과는 비용이 소거되므로 초과만 보면 «순손익 음수 1위»가 생긴다.
    d = pd.DataFrame({"z": [3.0], "exc": [50.0], "net": [-5.0],
                      **{f"e{y}": [10.0] for y in YEARS},
                      **{f"net{y}": [-1.0] for y in YEARS}})
    assert not bool(P1(d).iloc[0]), "순손익 음수 셀이 1차 관문을 통과했다"
    print("자체점검 통과")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck(); raise SystemExit(0)
    raise SystemExit(main())
