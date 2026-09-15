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
YEARS_ALL = (2022, 2023, 2024, 2025, 2026)
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


BV_PANEL = ROOT / "data/binance_vision/panel"
ASSETS20 = ["ETH", "BTC", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK",
            "LTC", "DOT", "TRX", "ATOM", "NEAR", "FIL", "APT", "ARB", "OP", "INJ", "SUI"]


def panel(asset: str, ticks: bool = True, since: str = "2024-01-01") -> pd.DataFrame:
    """봉 + 메트릭(OI/LSR) + (있으면) 틱 주문흐름 → 사건 트리거 후보 피쳐 패널.

    `data/binance_vision/panel/{SYM}.parquet` 가 있으면 그걸 쓴다(2022-01~, 20자산 ·
    `build_binance_vision_panel_20260915.py` 산출). 없으면 저장소의 CSV 조합으로 떨어진다."""
    bv = BV_PANEL / f"{asset}USDT.parquet"
    if bv.exists():
        df = pd.read_parquet(bv)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop_duplicates("timestamp").sort_values("timestamp")
        return _features(df, asset, ticks, since)
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
    return _features(df, asset, ticks, since)


def _features(df: pd.DataFrame, asset: str, ticks: bool, since: str) -> pd.DataFrame:
    df = df[df["timestamp"] >= since].reset_index(drop=True)

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


SINCE = "2024-01-01"          # stage 가 바꾼다. 캐시 키에 들어간다.


def cached_panel(asset: str) -> pd.DataFrame:
    key = f"{asset}@{SINCE}"
    if key not in _PCACHE:
        f = OUT / f"panel_{asset}_{SINCE}.parquet"
        if f.exists():
            _PCACHE[key] = pd.read_parquet(f)
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            d = panel(asset, since=SINCE)
            d.to_parquet(f)
            _PCACHE[key] = d
    return _PCACHE[key]


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


# ── 축 6. ⭐⭐⭐⭐⭐결론 증명 설계 — 선택창 2022~23 / 시험창 2024~26 / 20자산 ─────
def stage_proof(a):
    """**이 파일의 최종 설계.** 앞의 모든 판이 가진 약점을 데이터로 없앤다.

    앞 판들의 병목은 **검정력**이었다(완전 표본외 게이트 판: +1.75bp/일 · 7/9 · CI 0포함 ·
    자산당 체결 69~133건). 실측으로 늘릴 수 있는 것 둘을 확인해 받았다
    (`build_binance_vision_panel_20260915.py`): **metrics 가 2022-01 부터** 존재(965→1,695일)
    하고 **신규 11자산** 전부 사용 가능(→ 20자산 · 9,300,939봉).

    ⭐그래서 **월별 재선택도 LOAO 도 필요 없는 설계**가 된다:
      · **선택창 = 2022~2023** (24개월 · 20자산 풀링) — 이 세션에서 한 번도 안 쓴 데이터
      · **시험창 = 2024-01 ~ 2026-08** (32개월 · 20자산) — **재선택 없음, 규칙 동결**
      · 선택은 **풀링**으로 한 번만 ⇒ 월별 재선택 잡음도 자산별 과적합도 없다

    🔴사전 등록(실행 전 고정): 관문 `풀링 n≥1000 & 자산 8개 이상 & net>0 & 연배경초과>0 &
    t>2 & 2022>0 & 2023>0`, **top-K 없음**(그건 손잡이였다) · 게이트 하위 **80%** 제외 ·
    **w=0.25 · 상한 1.0 · 비용 10bp** · 대조군은 순환이동 귀무와 같은 노출 롱홀드.
    게이트 모델은 **분기 재학습** · 씨드 1 · 3시간당 1봉 표집 — 인과성은 같고 비용 결정이다.

    🔴**메모리**: 20자산 × 3,400셀의 사건 배열을 들면 13GB 로 터진다(실측 8자산 7GB). 선택에
    필요한 건 셀별 **요약통계 8개**뿐이라(개수·합·초과합·초과제곱합·연도별 개수/합) 1패스는
    그것만 모으고(3.8MB), 동결 후 **선택된 셀만** 2패스에서 다시 편다.
    🔴**루프 순서**: 임계는 지평과 무관하니 피쳐를 바깥에 둔다(안쪽이면 피쳐당 3회 = 3배).
    전 피쳐 임계를 미리 만드는 건 2.3GB 라 안 된다 — 한 피쳐씩 계산해 쓰고 버린다."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    global SINCE
    SINCE = "2022-01-01"
    rng = np.random.default_rng(SEED)
    assets = [A for A in ASSETS20 if (BV_PANEL / f"{A}USDT.parquet").exists()]
    # 🔴이 설계는 **편향을 하나 없애고 다른 하나를 들인다**: 선택창 2022~23 은 선택 편향을
    #   제거하지만 **레짐이 다르다**(§5.30 이 「23년 이전은 레짐이 달라 제외」라고 적은 그 구간).
    #   ⇒ 양수면 매우 강한 증거지만 **음수는 「신호 없음」과 「레짐 불일치」를 못 가른다.**
    #   그래서 `--selend 2025-01-01`(선택 2024 · 시험 2025~26, 같은 레짐 계열)을 함께 돌려
    #   둘을 나란히 읽는다.
    SEL_END = np.datetime64(a.selend)
    TEST_START = pd.Timestamp(a.selend)
    print(f"자산 {len(assets)} · 선택창 {SINCE}~{a.selend} · 시험창 {a.selend}~2026-08 · "
          f"w={a.w} 상한={a.cap} 게이트 하위{a.drop:.0%} 제외 · t>{a.tsel}", flush=True)

    cols, STAT, TRAIN, META = None, {}, {}, {}
    for A in assets:
        p = panel(A, since=SINCE)
        if cols is None:
            cols = [c for c in featcols(p) if c != "hour_f" and not c.startswith("tf_")
                    and not c.startswith("ztf_")]
            print(f"공통 피쳐 {len(cols)}개", flush=True)
        ts = p["timestamp"]; tsv = ts.to_numpy(); lc = p["__lc__"].to_numpy()
        yr = ts.dt.year.to_numpy()
        dayno = (ts.dt.floor("D").astype("int64") // 86_400_000_000_000).to_numpy()
        dayno = dayno - dayno.min()
        X = p[cols].to_numpy(np.float32)
        FW = {}
        for hn, H in HS.items():
            f_ = fwd_of(lc, H)
            FW[hn] = (f_, np.isfinite(f_), year_background(f_, yr, H))
        st = {}
        for c in cols:
            v = p[c].to_numpy(float)
            lo, hi = causal_thresholds(v, dayno)
            for hn, H in HS.items():
                fwd, ok, bgy = FW[hn]
                for j, q in enumerate(QS):
                    for cond, mask in (("하위", v <= lo[:, j]), ("상위", v >= hi[:, j])):
                        keep = nonoverlap(np.flatnonzero(
                            mask & ok & np.isfinite(lo[:, j]) & (tsv < SEL_END)), H)
                        if len(keep) < 60:
                            continue
                        f = fwd[keep]; b = bgy[keep]; yy = yr[keep]
                        for side in (1, -1):
                            pn = side * f - COST_BP
                            ex = side * f - side * b
                            st[(c, hn, q, cond, side)] = (
                                len(pn), float(pn.sum()), float(ex.sum()), float((ex ** 2).sum()),
                                int((yy == 2022).sum()), float(pn[yy == 2022].sum()),
                                int((yy == 2023).sum()), float(pn[yy == 2023].sum()))
        STAT[A] = st
        sub = {}
        for hn, H in HS.items():
            y = np.log(np.maximum(np.abs(FW[hn][0]), 1.0))
            idx = np.flatnonzero(np.isfinite(y))[::36]
            sub[hn] = (idx, y[idx])
        un = np.unique(np.concatenate([sub[h][0] for h in HS]))
        TRAIN[A] = (X[un], sub, un)
        r = np.zeros(len(lc)); r[1:] = np.diff(lc)
        META[A] = (ts, r, yr, len(lc))
        print(f"  [{A}] 봉 {len(lc):,} · 선택창 셀 {len(st):,}", flush=True)
        del p, X, FW

    keys = sorted(set.intersection(*[set(STAT[A]) for A in assets]), key=str)
    print(f"\n공통 셀 {len(keys):,}", flush=True)

    # ── ① 선택: 2022~2023 풀링, 한 번, 동결 ──────────────────────────────────
    sel = []
    for key in keys:
        n = na = n22 = n23 = 0
        sn = se = se2 = s22 = s23 = 0.0
        for A in assets:
            t = STAT[A].get(key)
            if t is None or t[0] < 30:
                continue
            na += 1
            n += t[0]; sn += t[1]; se += t[2]; se2 += t[3]
            n22 += t[4]; s22 += t[5]; n23 += t[6]; s23 += t[7]
        if na < 8 or n < 1000 or n22 < 200 or n23 < 200:
            continue
        mn, me = sn / n, se / n
        if mn <= 0 or me <= 0 or s22 / n22 <= 0 or s23 / n23 <= 0:
            continue
        var = max(se2 / n - me ** 2, 0.0) * n / max(n - 1, 1)
        if var <= 0 or me / np.sqrt(var / n) < a.tsel:
            continue
        sel.append((key, me, mn, n))
    print(f"⭐**선택창 통과 {len(sel)}셀** / {len(keys):,}  (재선택 없음 · 여기서 동결)", flush=True)
    if not sel:
        print("통과 0 — 시험 불가")
        return
    print("규칙군:", pd.Series([k[0].split("_")[0] for k, *_ in sel]).value_counts().to_dict())
    print("측면:", pd.Series(["롱" if k[4] == 1 else "숏" for k, *_ in sel]).value_counts().to_dict())
    print(f"\n{'피쳐':>14} {'지평':>4} {'분위':>6} {'조건':>4} {'방향':>4} {'풀n':>7} "
          f"{'선택창net':>9} {'선택창초과':>10}")
    for key, ex, pnm, n in sorted(sel, key=lambda x: -x[1])[:12]:
        print(f"{key[0]:>14} {key[1]:>4} {key[2]:>6.1%} {key[3]:>4} "
              f"{'롱' if key[4] == 1 else '숏':>4} {n:>7} {pnm:>+9.2f} {ex:>+10.2f}")
    del STAT

    # ── ② 게이트: 전 봉 학습 · 분기 재학습 · 인과 ───────────────────────────
    MODELS = {}
    for hn, H in HS.items():
        for q0 in pd.date_range("2024-01-01", "2026-07-01", freq="QS"):
            Xs, ys = [], []
            for A in assets:
                Xa, sub, un = TRAIN[A]
                idx, y = sub[hn]
                m = META[A][0].to_numpy()[idx] < np.datetime64(q0 - pd.Timedelta(minutes=5 * H))
                if m.sum() < 100:
                    continue
                Xs.append(Xa[np.searchsorted(un, idx[m])]); ys.append(y[m])
            if not Xs:
                continue
            Xt = np.vstack(Xs); yt = np.concatenate(ys)
            MODELS[(hn, q0)] = HistGradientBoostingRegressor(
                max_iter=120, learning_rate=0.06, max_depth=4,
                l2_regularization=3.0, random_state=11).fit(Xt, yt)
            del Xt, yt
        print(f"  [게이트 {hn}] 분기 모델 {sum(1 for k in MODELS if k[0] == hn)}개", flush=True)
    del TRAIN

    # ── ③ 시험: 선택된 셀만 다시 펴서 2024-01~2026-08 거래 ─────────────────
    need = sorted({k[0] for k, *_ in sel})
    books, rows = {}, []
    for A in assets:
        p = panel(A, since=SINCE)
        ts = p["timestamp"]; tsv = ts.to_numpy(); lc = p["__lc__"].to_numpy()
        dayno = (ts.dt.floor("D").astype("int64") // 86_400_000_000_000).to_numpy()
        dayno = dayno - dayno.min()
        Xf = p[cols].to_numpy(np.float32)
        PR = {}
        for (hn, q0), mdl in MODELS.items():
            te = np.flatnonzero(((ts >= q0) & (ts < q0 + pd.DateOffset(months=3))).to_numpy())
            if len(te):
                PR.setdefault(hn, np.full(len(lc), np.nan))[te] = mdl.predict(Xf[te])
        TH = {c: causal_thresholds(p[c].to_numpy(float), dayno) for c in need}
        cand = []
        for key, *_ in sel:
            c, hn, q, cond, side = key
            H = HS[hn]; fwd = fwd_of(lc, H)
            v = p[c].to_numpy(float); lo, hi = TH[c]; j = QS.index(q)
            base = (v <= lo[:, j]) if cond == "하위" else (v >= hi[:, j])
            m = (base & np.isfinite(fwd) & np.isfinite(lo[:, j])
                 & (tsv >= np.datetime64(TEST_START)))
            for i in nonoverlap(np.flatnonzero(m), H):
                cand.append((int(i), side, H, hn))
        del p, Xf, TH
        if len(cand) < 50:
            print(f"  [{A}] 후보 {len(cand)} — 건너뜀", flush=True)
            continue
        cand.sort()
        _, r, yr, n = META[A]
        pr_ = np.array([PR[hn][i] if hn in PR else np.nan for i, _, _, hn in cand])
        thr = pd.Series(pr_).expanding(200).quantile(a.drop).shift(1).to_numpy()
        gate = np.isfinite(thr) & (pr_ > thr)
        for nm, sel_ in (("게이트 없음", np.isfinite(thr)), ("E|r| 게이트", gate)):
            trades = [(i, sd_, a.w, H) for (i, sd_, H, _), k in zip(cand, sel_) if k]
            if len(trades) < 20:
                continue
            prs, expo, gross, took = simulate(trades, n, r, a.cap)
            mask = (ts >= TEST_START).to_numpy()
            g = pd.DataFrame({"pr": prs[mask], "day": ts[mask].dt.floor("D").values,
                              "y": yr[mask]}).groupby("day").agg(pr=("pr", "sum"), y=("y", "first"))
            books[(A, nm)] = g
            nl, _ = random_entry_null(trades, n, r, a.cap, ts, rng, B=40)
            act = g.pr.mean() * 1e4
            rows.append({"자산": A, "구성": nm, "일평균bp": act, "순환귀무": float(nl.mean()),
                         "초과": act - float(nl.mean()), "백분위": float((nl < act).mean()),
                         "롱홀드동노출": float(pd.Series(r[mask] * gross[mask].mean()).groupby(
                             ts[mask].dt.floor("D").values).sum().mean() * 1e4),
                         "샤프": float(g.pr.mean() / g.pr.std() * np.sqrt(365)),
                         "MDD%": float((g.pr.cumsum() - g.pr.cumsum().cummax()).min() * 100),
                         **{f"e{y}": (float(g[g.y == y].pr.mean() * 1e4)
                                      if (g.y == y).sum() > 20 else np.nan) for y in YEARS},
                         "체결": took, "후보": len(trades), "평균노출": float(gross[mask].mean())})
        if (A, "E|r| 게이트") in books:
            print(f"  [{A}] 게이트없음 {rows[-2]['일평균bp']:+.2f} → 게이트 "
                  f"{rows[-1]['일평균bp']:+.2f} bp/일 · 체결 {rows[-1]['체결']}/{rows[-1]['후보']}",
                  flush=True)
    T = pd.DataFrame(rows)
    tag = f"sel{a.selend[:7]}_w{a.w}_cap{a.cap}_drop{int(a.drop * 100)}"
    T.to_csv(OUT / f"proof_{tag}.csv", index=False)
    print(f"\n=== 자산별 (시험창 {a.selend}~2026-08 · 규칙 동결) ===")
    print(T.to_string(index=False, float_format=lambda x: f"{x:9.2f}"))
    from scipy.stats import binomtest
    daily = {}
    for nm in ("게이트 없음", "E|r| 게이트"):
        aa = [A for A in assets if (A, nm) in books]
        allday = sorted(set().union(*[set(books[(A, nm)].index) for A in aa]))
        P = pd.DataFrame({A: books[(A, nm)].pr.reindex(allday).fillna(0.0) for A in aa}, index=allday)
        port = P.mean(axis=1); daily[nm] = port
        days = P.index.to_numpy()
        bs = np.array([port.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4
                       for _ in range(4000)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        sub = T[T.구성 == nm]
        npos = int((sub.일평균bp > 0).sum())
        yrv = pd.Series(pd.to_datetime(P.index).year, index=P.index)
        print(f"\n=== ⭐⭐{len(aa)}자산 합산 · {nm} (선택 ~{a.selend} / 시험 {a.selend}~) ===")
        print(f"  일평균 **{port.mean() * 1e4:+.2f}bp** · 샤프 "
              f"**{port.mean() / port.std() * np.sqrt(365):.2f}** · "
              f"MDD {(port.cumsum() - port.cumsum().cummax()).min() * 100:.2f}% · "
              f"누적 {port.sum() * 100:+.2f}%")
        print(f"  날짜블록 CI [{lo:+.2f}, {hi:+.2f}] · **0배제 {'✅' if lo > 0 else '❌'}**")
        print("  " + " · ".join(f"{y}: {port[(yrv == y).to_numpy()].mean() * 1e4:+.2f}"
                                for y in YEARS if (yrv == y).sum() > 20))
        print(f"  양수 자산 {npos}/{len(aa)} · 부호검정 p "
              f"{binomtest(npos, len(aa), 0.5, 'greater').pvalue:.4f} · 순환귀무 초과 평균 "
              f"{sub.초과.mean():+.2f}bp · 백분위 중앙 {sub.백분위.median():.0%}")
    d = daily["E|r| 게이트"] - daily["게이트 없음"]
    days = d.index.to_numpy()
    bs = np.array([d.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4 for _ in range(4000)])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    print(f"\n⭐**게이트의 짝지은 증분**: {d.mean() * 1e4:+.2f}bp/일 · CI [{lo:+.2f}, {hi:+.2f}] "
          f"**{'0배제 ✅' if lo > 0 else '0포함 ❌'}**")
    pd.DataFrame(daily).to_csv(OUT / f"proof_daily_{tag}.csv")
    print(f"\n저장: {OUT / f'proof_{tag}.csv'}")


def stage_highvol(a):
    """⭐**규칙을 버리고 «벽이 가장 낮은 자리»에서 방향을 직접 배운다.**

    왜 여기인가 — 벽 식이 자리를 지목한다: 수익벽 = 0.5 + 비용/(2·E|r|).
    `--stage proof` 실측에서 게이트 최상위 오분위의 **E|r| = 321.5bp** 였고 그 자리의 벽은
    0.5 + 10/(2·321.5) = **51.6%** 인데 관측 적중이 **54.2%** 였다 — **2.6pp 여유가 실재한다.**
    지금까지 그 방향은 **사건 규칙**이 정해줬고, 규칙은 두 독립 설계에서 −5.83 / −9.17bp/일 ·
    3/20 / 2/20 으로 **무가치가 확정**됐다. 그러면 남는 질문은 하나다:
    **그 자리에서 방향을 직접 배우면 되는가.**

    ⭐부수 효과가 검정력이다: 모집단이 «규칙 발동 봉」에서 **«전 봉의 상위 20%»** 로 바뀐다
    (자산당 ~400 → ~수천). 앞 절들의 병목이 정확히 표본 수였다.

    🔴사전 등록(실행 전 고정): 모집단 = 예측 E|r| **인과 확장창 상위 20%** · 비겹침 H 간격 ·
    방향 모델 = HGB 분류 · 타깃 `sign(fwd)` · **가중 |fwd|**(정확도 아닌 손익 목적함수
    [[direction_pnl_weighted_12h_candidate_20260915]]) · 20자산 풀링 · 분기 확장 WF ·
    시험 2024-01~2026-08 · 대조군 **셋**(무조건 롱 · 순환이동 귀무 · 같은 노출 롱홀드) ·
    w=0.25 · 상한 1.0 · 비용 10bp. 판정 = **무조건 롱 대비 짝지은 증분 CI 0배제 AND
    절대 수준 CI 0배제** — 둘 다 요구한다(앞 절들이 하나만 만족했다).
    🔴다중성 고지: 이 축은 이 세션에서 **여덟 번째** 설계다. 통과해도 그 사실과 함께 읽는다."""
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    global SINCE
    SINCE = "2022-01-01"
    rng = np.random.default_rng(SEED)
    assets = [A for A in ASSETS20 if (BV_PANEL / f"{A}USDT.parquet").exists()]
    TEST_START = pd.Timestamp("2024-01-01")
    QUARTERS = pd.date_range("2024-01-01", "2026-07-01", freq="QS")
    print(f"자산 {len(assets)} · 상위 {1-a.drop:.0%} 봉 · w={a.w} 상한={a.cap} · 비용 {COST_BP}bp",
          flush=True)

    # ── 1패스: E|r| 학습 재료만 모은다(패널은 버린다) ───────────────────────
    cols, TRAIN, META = None, {}, {}
    for A in assets:
        p = panel(A, since=SINCE)
        if cols is None:
            cols = [c for c in featcols(p) if c != "hour_f" and not c.startswith("tf_")
                    and not c.startswith("ztf_")]
            print(f"공통 피쳐 {len(cols)}개", flush=True)
        lc = p["__lc__"].to_numpy(); ts = p["timestamp"]
        X = p[cols].to_numpy(np.float32)
        sub = {}
        for hn, H in HS.items():
            y = np.log(np.maximum(np.abs(fwd_of(lc, H)), 1.0))
            idx = np.flatnonzero(np.isfinite(y))[::36]
            sub[hn] = (idx, y[idx])
        un = np.unique(np.concatenate([sub[h][0] for h in HS]))
        TRAIN[A] = (X[un], sub, un)
        r = np.zeros(len(lc)); r[1:] = np.diff(lc)
        META[A] = (ts, r, ts.dt.year.to_numpy(), len(lc))
        print(f"  [{A}] 봉 {len(lc):,}", flush=True)
        del p, X

    # ── E|r| 모델(분기 · 인과) ──────────────────────────────────────────────
    EMOD = {}
    for hn, H in HS.items():
        for q0 in QUARTERS:
            Xs, ys = [], []
            for A in assets:
                Xa, sub, un = TRAIN[A]
                idx, y = sub[hn]
                m = META[A][0].to_numpy()[idx] < np.datetime64(q0 - pd.Timedelta(minutes=5 * H))
                if m.sum() < 100:
                    continue
                Xs.append(Xa[np.searchsorted(un, idx[m])]); ys.append(y[m])
            if not Xs:
                continue
            EMOD[(hn, q0)] = HistGradientBoostingRegressor(
                max_iter=120, learning_rate=0.06, max_depth=4, l2_regularization=3.0,
                random_state=11).fit(np.vstack(Xs), np.concatenate(ys))
        print(f"  [E|r| {hn}] 분기 모델 {sum(1 for k in EMOD if k[0] == hn)}개", flush=True)
    del TRAIN

    # ── 2패스: 상위 분위 봉 = 후보. 피쳐·라벨만 남긴다 ──────────────────────
    CAND = {}
    for A in assets:
        p = panel(A, since=SINCE)
        ts = p["timestamp"]; lc = p["__lc__"].to_numpy()
        Xf = p[cols].to_numpy(np.float32)
        for hn, H in HS.items():
            pred = np.full(len(lc), np.nan)
            for q0 in QUARTERS:
                mdl = EMOD.get((hn, q0))
                if mdl is None:
                    continue
                te = np.flatnonzero(((ts >= q0) & (ts < q0 + pd.DateOffset(months=3))).to_numpy())
                if len(te):
                    pred[te] = mdl.predict(Xf[te])
            fwd = fwd_of(lc, H)
            ok = np.isfinite(pred) & np.isfinite(fwd)
            oi = np.flatnonzero(ok)
            # 인과 확장창 분위 — 예측 분포도 미래를 보면 안 된다
            thr = pd.Series(pred[oi]).expanding(500).quantile(a.drop).shift(1).to_numpy()
            hi = oi[np.isfinite(thr) & (pred[oi] > thr)]
            keep = nonoverlap(hi, H)
            if len(keep) < 50:
                continue
            CAND[(A, hn)] = (keep.astype(np.int32), ts.to_numpy()[keep],
                             Xf[keep], fwd[keep].astype(np.float32))
        print(f"  [{A}] 후보 " + " · ".join(
            f"{hn} {len(CAND[(A, hn)][0]) if (A, hn) in CAND else 0}" for hn in HS), flush=True)
        del p, Xf
    del EMOD

    # ── 방향 모델: 후보 위에서만, 분기 확장 WF, |수익| 가중 ─────────────────
    DIR = {A: {} for A in assets}
    for hn, H in HS.items():
        for q0 in QUARTERS:
            Xs, ys, ws = [], [], []
            for A in assets:
                if (A, hn) not in CAND:
                    continue
                _, tk, Xc, fw = CAND[(A, hn)]
                m = tk < np.datetime64(q0 - pd.Timedelta(minutes=5 * H))
                if m.sum() < 50:
                    continue
                Xs.append(Xc[m]); ys.append((fw[m] > 0).astype(int)); ws.append(np.abs(fw[m]))
            if not Xs:
                continue
            X = np.vstack(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
            if len(y) < 500 or y.mean() in (0.0, 1.0):
                continue
            w = w / w.mean()
            clfs = [HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_depth=3, l2_regularization=3.0,
                random_state=sd).fit(X, y, sample_weight=w) for sd in (11, 907, 4231)]
            for A in assets:
                if (A, hn) not in CAND:
                    continue
                _, tk, Xc, _ = CAND[(A, hn)]
                te = (tk >= np.datetime64(q0)) & (tk < np.datetime64(q0 + pd.DateOffset(months=3)))
                if te.sum():
                    DIR[A].setdefault(hn, {})[q0] = (
                        np.flatnonzero(te),
                        np.mean([c.predict_proba(Xc[te])[:, 1] for c in clfs], axis=0))
        print(f"  [방향 {hn}] 분기 학습 완료", flush=True)

    # ── 시험 ────────────────────────────────────────────────────────────────
    books, rows, accs = {}, [], []
    for A in assets:
        ts, r, yr, n = META[A]
        arms = {"무조건 롱": [], "방향 모델": [], "방향+확신": []}
        for hn, H in HS.items():
            if (A, hn) not in CAND or hn not in DIR[A]:
                continue
            keep, tk, _, fw = CAND[(A, hn)]
            for q0, (pos, pr) in DIR[A][hn].items():
                for k, prob in zip(pos, pr):
                    if tk[k] < np.datetime64(TEST_START):
                        continue
                    i = int(keep[k])
                    arms["무조건 롱"].append((i, 1, a.w, H))
                    arms["방향 모델"].append((i, 1 if prob > 0.5 else -1, a.w, H))
                    arms["방향+확신"].append((i, 1 if prob > 0.5 else -1, a.w, H, abs(prob - 0.5)))
                    accs.append((A, hn, float(prob), float(fw[k])))
        conf = [t[4] for t in arms["방향+확신"]]
        if conf:
            cut = float(np.median(conf))
            arms["방향+확신"] = [t[:4] for t in arms["방향+확신"] if t[4] >= cut]
        for nm, tr in arms.items():
            if len(tr) < 50:
                continue
            prs, expo, gross, took = simulate(tr, n, r, a.cap)
            mask = (ts >= TEST_START).to_numpy()
            g = pd.DataFrame({"pr": prs[mask], "day": ts[mask].dt.floor("D").values,
                              "y": yr[mask]}).groupby("day").agg(pr=("pr", "sum"), y=("y", "first"))
            books[(A, nm)] = g
            nl, _ = random_entry_null(tr, n, r, a.cap, ts, rng, B=40)
            act = g.pr.mean() * 1e4
            rows.append({"자산": A, "구성": nm, "일평균bp": act, "순환귀무": float(nl.mean()),
                         "초과": act - float(nl.mean()), "백분위": float((nl < act).mean()),
                         "샤프": float(g.pr.mean() / g.pr.std() * np.sqrt(365)),
                         "MDD%": float((g.pr.cumsum() - g.pr.cumsum().cummax()).min() * 100),
                         **{f"e{y}": (float(g[g.y == y].pr.mean() * 1e4)
                                      if (g.y == y).sum() > 20 else np.nan) for y in YEARS},
                         "체결": took, "후보": len(tr), "평균노출": float(gross[mask].mean())})
        print(f"  [{A}] " + " · ".join(
            f"{x['구성']} {x['일평균bp']:+.2f}" for x in rows[-len(arms):]), flush=True)

    AC = pd.DataFrame(accs, columns=["asset", "H", "prob", "fwd"])
    AC = AC[AC.fwd.notna()]
    print(f"\n=== ⭐방향 모델의 적중률 vs 벽 (시험창 전체 · n={len(AC):,}) ===")
    print(f"{'지평':>5} {'n':>7} {'E|r|':>8} {'수익벽':>7} {'무조건롱 적중':>13} {'모델 적중':>10} "
          f"{'모델−벽':>8}")
    for hn in HS:
        s = AC[AC.H == hn]
        if len(s) < 100:
            continue
        ear = float(s.fwd.abs().mean())
        wall = 0.5 + COST_BP / (2 * ear)
        long_acc = float((s.fwd > 0).mean())
        side = np.where(s.prob > 0.5, 1, -1)
        mdl_acc = float((side * s.fwd > 0).mean())
        print(f"{hn:>5} {len(s):>7} {ear:>8.1f} {wall:>7.1%} {long_acc:>13.1%} {mdl_acc:>10.1%} "
              f"{mdl_acc - wall:>+8.1%}")
    T = pd.DataFrame(rows)
    tag = f"highvol_w{a.w}_cap{a.cap}_top{int((1 - a.drop) * 100)}"
    T.to_csv(OUT / f"{tag}.csv", index=False)
    AC.to_csv(OUT / f"{tag}_acc.csv", index=False)
    from scipy.stats import binomtest
    daily = {}
    for nm in ("무조건 롱", "방향 모델", "방향+확신"):
        aa = [A for A in assets if (A, nm) in books]
        if not aa:
            continue
        allday = sorted(set().union(*[set(books[(A, nm)].index) for A in aa]))
        P = pd.DataFrame({A: books[(A, nm)].pr.reindex(allday).fillna(0.0) for A in aa}, index=allday)
        port = P.mean(axis=1); daily[nm] = port
        days = P.index.to_numpy()
        bs = np.array([port.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4
                       for _ in range(4000)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        sub = T[T.구성 == nm]; npos = int((sub.일평균bp > 0).sum())
        yrv = pd.Series(pd.to_datetime(P.index).year, index=P.index)
        print(f"\n=== ⭐⭐{len(aa)}자산 합산 · {nm} ===")
        print(f"  일평균 **{port.mean() * 1e4:+.2f}bp** · 샤프 "
              f"**{port.mean() / port.std() * np.sqrt(365):.2f}** · "
              f"MDD {(port.cumsum() - port.cumsum().cummax()).min() * 100:.2f}% · "
              f"누적 {port.sum() * 100:+.2f}%")
        print(f"  날짜블록 CI [{lo:+.2f}, {hi:+.2f}] · **0배제 {'✅' if lo > 0 else '❌'}**")
        print("  " + " · ".join(f"{y}: {port[(yrv == y).to_numpy()].mean() * 1e4:+.2f}"
                                for y in YEARS if (yrv == y).sum() > 20))
        print(f"  양수 자산 {npos}/{len(aa)} · 부호검정 p "
              f"{binomtest(npos, len(aa), 0.5, 'greater').pvalue:.4f} · 순환귀무 초과 평균 "
              f"{sub.초과.mean():+.2f}bp · 백분위 중앙 {sub.백분위.median():.0%}")
    for nm in ("방향 모델", "방향+확신"):
        if nm not in daily or "무조건 롱" not in daily:
            continue
        d = (daily[nm] - daily["무조건 롱"]).dropna()
        days = d.index.to_numpy()
        bs = np.array([d.loc[rng.choice(days, len(days), replace=True)].mean() * 1e4
                       for _ in range(4000)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        print(f"⭐**{nm} − 무조건 롱 (짝지은 증분)**: {d.mean() * 1e4:+.2f}bp/일 · "
              f"CI [{lo:+.2f}, {hi:+.2f}] **{'0배제 ✅' if lo > 0 else '0포함 ❌'}**")
    pd.DataFrame(daily).to_csv(OUT / f"{tag}_daily.csv")
    print(f"\n저장: {OUT / f'{tag}.csv'}")


def first_touch(hi, lo, ent, tp, sl, side):
    """봉내 고가/저가로 **먼저 닿은 배리어**를 찾는다. 반환 (수익률bp, 종료봉, 사유).

    🔴규약(CLAUDE.md «배리어/청산 판정은 intrabar 고가/저가»): resting TP/SL 은 종가가 아니라
    닿는 즉시 체결되고, 이미 **확정된 봉**만 쓰므로 lookahead 가 아니다.
    🔴동시 터치(한 봉이 TP·SL 을 다 건드림)는 **SL 우선** — 보수적 쪽이다. 반대로 놓으면
    §5.32 분류학 F(봉내 순서 낙관)가 되어 가짜 엣지를 제조한다.
    hi/lo 는 (n, H) 행렬(진입 **다음** 봉부터), ent 는 (n,), tp/sl 은 (n,) 가격변동률."""
    up = hi / ent[:, None] - 1.0
    dn = lo / ent[:, None] - 1.0
    if side == 1:
        win, loss = up >= tp[:, None], dn <= -sl[:, None]
        wret, lret = tp, -sl
    else:
        win, loss = dn <= -tp[:, None], up >= sl[:, None]
        wret, lret = tp, -sl
    H = hi.shape[1]
    iw = np.where(win.any(1), win.argmax(1), H + 1)
    il = np.where(loss.any(1), loss.argmax(1), H + 1)
    out = np.full(len(ent), np.nan); end = np.full(len(ent), H, dtype=int)
    hit_l = il <= iw                                  # 동시 터치면 손절 우선
    m = hit_l & (il <= H)
    out[m] = lret[m] * 1e4; end[m] = il[m]
    m = (~hit_l) & (iw <= H)
    out[m] = wret[m] * 1e4; end[m] = iw[m]
    return out, end


def stage_exit(a):
    """⭐**벽 식의 «가정»을 친다 — 고정지평이 아니라 비대칭 페이오프.**

    지금까지 15 stage 가 전부 **고정지평 종가청산**이었고, 그 위에서만 성립하는 식이
      수익벽 = 0.5 + 비용/(2·E|r|)
    이다(±E|r| 대칭 페이오프 가정). 페이오프가 비대칭이면 필요 적중률은
      a > (L + 비용) / (W + L)
    로 바뀐다 — **H 절에서 잰 50.5% 로도 넘을 수 있는 형태**다.

    §5.30 B-5 가 트레일링을 기각했지만 그건 ①**죽은 규칙 모집단**에 ②**호메로스 ATR 규약**을
    쓴 것이다. 여기서는 ①**고 E|r| 모집단**에 ②**예측 E|r| 로 배리어를 스케일**한다
    (인과 예측값이 이미 있다 — 이 조합은 이 저장소에서 처음이다).

    🔴사전 등록: `TP = k_tp·ê`, `SL = k_sl·ê`, k ∈ {0.5,1.0,1.5,2.0} → 4×4 격자 + 고정지평
    기준선 · **봉내 고가/저가** 판정 · 동시 터치 **SL 우선**(보수적) · 시간상한 = H ·
    **격자는 2024 에서만 고르고 2025~26 은 딱 한 번** · 방향은 ①모델 ②무조건 롱 둘 다 ·
    비용 10bp · 대조군 = 순환이동 귀무.
    🔴다중성: 이 축은 이 세션 **아홉 번째** 설계이고 격자가 16칸이다. 통과해도 함께 읽는다."""
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    global SINCE
    SINCE = "2022-01-01"
    rng = np.random.default_rng(SEED)
    assets = [A for A in ASSETS20 if (BV_PANEL / f"{A}USDT.parquet").exists()]
    QUARTERS = pd.date_range("2024-01-01", "2026-07-01", freq="QS")
    KS = (0.5, 1.0, 1.5, 2.0)
    print(f"자산 {len(assets)} · 상위 {1-a.drop:.0%} 봉 · 격자 {len(KS)}×{len(KS)} · 비용 {COST_BP}bp",
          flush=True)
    cols, TRAIN, META = None, {}, {}
    for A in assets:
        p = panel(A, since=SINCE)
        if cols is None:
            cols = [c for c in featcols(p) if c != "hour_f" and not c.startswith("tf_")
                    and not c.startswith("ztf_")]
        lc = p["__lc__"].to_numpy(); ts = p["timestamp"]
        X = p[cols].to_numpy(np.float32)
        sub = {}
        for hn, H in HS.items():
            y = np.log(np.maximum(np.abs(fwd_of(lc, H)), 1.0))
            idx = np.flatnonzero(np.isfinite(y))[::36]
            sub[hn] = (idx, y[idx])
        un = np.unique(np.concatenate([sub[h][0] for h in HS]))
        TRAIN[A] = (X[un], sub, un)
        META[A] = (ts, lc, ts.dt.year.to_numpy(), len(lc))
        del p, X
    print("  1패스 완료", flush=True)
    EMOD = {}
    for hn, H in HS.items():
        for q0 in QUARTERS:
            Xs, ys = [], []
            for A in assets:
                Xa, sub, un = TRAIN[A]
                idx, y = sub[hn]
                m = META[A][0].to_numpy()[idx] < np.datetime64(q0 - pd.Timedelta(minutes=5 * H))
                if m.sum() < 100:
                    continue
                Xs.append(Xa[np.searchsorted(un, idx[m])]); ys.append(y[m])
            if Xs:
                EMOD[(hn, q0)] = HistGradientBoostingRegressor(
                    max_iter=120, learning_rate=0.06, max_depth=4, l2_regularization=3.0,
                    random_state=11).fit(np.vstack(Xs), np.concatenate(ys))
        print(f"  [E|r| {hn}] 완료", flush=True)
    del TRAIN
    CAND = {}
    for A in assets:
        p = panel(A, since=SINCE)
        ts = p["timestamp"]; lc = p["__lc__"].to_numpy()
        Xf = p[cols].to_numpy(np.float32)
        raw = pd.read_parquet(BV_PANEL / f"{A}USDT.parquet")
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
        raw = raw.drop_duplicates("timestamp").sort_values("timestamp")
        raw = raw[raw.timestamp >= SINCE].reset_index(drop=True)
        hi = raw["high"].to_numpy(float); lw = raw["low"].to_numpy(float)
        cl = raw["close"].to_numpy(float)
        assert len(hi) == len(lc), f"{A} 원시/패널 길이 불일치 {len(hi)} vs {len(lc)}"
        for hn, H in HS.items():
            pred = np.full(len(lc), np.nan)
            for q0 in QUARTERS:
                mdl = EMOD.get((hn, q0))
                if mdl is None:
                    continue
                te = np.flatnonzero(((ts >= q0) & (ts < q0 + pd.DateOffset(months=3))).to_numpy())
                if len(te):
                    pred[te] = mdl.predict(Xf[te])
            fwd = fwd_of(lc, H)
            ok = np.isfinite(pred) & np.isfinite(fwd)
            oi = np.flatnonzero(ok)
            thr = pd.Series(pred[oi]).expanding(500).quantile(a.drop).shift(1).to_numpy()
            keep = nonoverlap(oi[np.isfinite(thr) & (pred[oi] > thr)], H)
            keep = keep[keep + H < len(lc)]
            if len(keep) < 50:
                continue
            W = np.lib.stride_tricks.sliding_window_view(hi, H + 1)[keep + 1 - 1][:, 1:]
            Wl = np.lib.stride_tricks.sliding_window_view(lw, H + 1)[keep + 1 - 1][:, 1:]
            CAND[(A, hn)] = dict(i=keep.astype(np.int32), ts=ts.to_numpy()[keep],
                                 X=Xf[keep], fwd=fwd[keep].astype(np.float32),
                                 ehat=np.exp(pred[keep]).astype(np.float32) / 1e4,
                                 ent=cl[keep], hi=W.astype(np.float32), lo=Wl.astype(np.float32))
        print(f"  [{A}] 후보 " + " · ".join(
            f"{hn} {len(CAND[(A,hn)]['i']) if (A,hn) in CAND else 0}" for hn in HS), flush=True)
        del p, Xf, raw
    del EMOD
    DIRP = {}
    for hn, H in HS.items():
        for q0 in QUARTERS:
            Xs, ys, ws = [], [], []
            for A in assets:
                c = CAND.get((A, hn))
                if c is None:
                    continue
                m = c["ts"] < np.datetime64(q0 - pd.Timedelta(minutes=5 * H))
                if m.sum() < 50:
                    continue
                Xs.append(c["X"][m]); ys.append((c["fwd"][m] > 0).astype(int))
                ws.append(np.abs(c["fwd"][m]))
            if not Xs:
                continue
            X = np.vstack(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
            if len(y) < 500 or y.mean() in (0.0, 1.0):
                continue
            clfs = [HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_depth=3, l2_regularization=3.0,
                random_state=sd).fit(X, y, sample_weight=w / w.mean()) for sd in (11, 907, 4231)]
            for A in assets:
                c = CAND.get((A, hn))
                if c is None:
                    continue
                te = (c["ts"] >= np.datetime64(q0)) & (c["ts"] < np.datetime64(
                    q0 + pd.DateOffset(months=3)))
                if te.sum():
                    DIRP.setdefault((A, hn), np.full(len(c["i"]), np.nan))[te] = np.mean(
                        [cf.predict_proba(c["X"][te])[:, 1] for cf in clfs], axis=0)
        print(f"  [방향 {hn}] 완료", flush=True)
    # ── 격자 평가: 2024 에서 고르고 2025~26 한 번 ──────────────────────────
    recs = []
    for (A, hn), c in CAND.items():
        pr = DIRP.get((A, hn))
        if pr is None:
            continue
        yr = pd.Series(c["ts"]).dt.year.to_numpy()
        eh = np.clip(c["ehat"], 20e-4, 800e-4)
        for dname, side_v in (("모델", np.where(pr > 0.5, 1, -1)),
                              ("무조건 롱", np.ones(len(pr), int))):
            fin = np.isfinite(pr)
            base = side_v * c["fwd"] - COST_BP                    # 고정지평 기준선
            for s in (1, -1):
                sm = fin & (side_v == s)
                if sm.sum() < 30:
                    continue
                for ktp in KS:
                    for ksl in KS:
                        ret, _ = first_touch(c["hi"][sm], c["lo"][sm], c["ent"][sm],
                                             (ktp * eh[sm]).astype(float),
                                             (ksl * eh[sm]).astype(float), s)
                        fb = s * c["fwd"][sm]
                        ret = np.where(np.isfinite(ret), ret, fb)   # 무터치 → 시간상한 종가
                        for j, k in enumerate(np.flatnonzero(sm)):
                            recs.append((A, hn, dname, ktp, ksl, int(yr[k]),
                                         float(ret[j] - COST_BP), float(base[k])))
    R = pd.DataFrame(recs, columns=["asset", "H", "dir", "ktp", "ksl", "year", "pnl", "base"])
    R.to_parquet(OUT / "exit_grid.parquet")
    print(f"\n격자 레코드 {len(R):,}", flush=True)
    sel_yr, test_yr = R.year == 2024, R.year >= 2025
    print(f"\n=== 격자 선택(2024 만) — 구성별 건당 net bp ===")
    print(f"{'방향':>8} {'ktp':>5} {'ksl':>5} {'2024 건당':>10} {'n':>8}")
    best = {}
    for dname in ("모델", "무조건 롱"):
        g = R[sel_yr & (R.dir == dname)].groupby(["ktp", "ksl"]).agg(
            m=("pnl", "mean"), n=("pnl", "size")).reset_index().sort_values("m", ascending=False)
        for _, x in g.head(4).iterrows():
            print(f"{dname:>8} {x['ktp']:>5.1f} {x['ksl']:>5.1f} {x['m']:>+10.2f} {int(x['n']):>8}")
        b = g.iloc[0]
        best[dname] = (float(b.ktp), float(b.ksl))
        bm = R[sel_yr & (R.dir == dname)].groupby(["ktp", "ksl"]).size().index
        print(f"   → 선택 ktp={b.ktp} ksl={b.ksl} (2024 건당 {b.m:+.2f}bp · 격자 {len(bm)}칸 중)")
        print(f"   [참고] 같은 구성 고정지평 2024 건당 "
              f"{R[sel_yr & (R.dir==dname)].base.mean() - COST_BP:+.2f}bp")
    print(f"\n=== ⭐시험(2025~2026, 딱 한 번) ===")
    print(f"{'방향':>8} {'구성':>16} {'건당net':>9} {'n':>7} {'승률':>7} {'평균이익':>9} "
          f"{'평균손실':>9} {'필요적중':>9} {'실제적중':>9}")
    for dname in ("모델", "무조건 롱"):
        ktp, ksl = best[dname]
        sub = R[test_yr & (R.dir == dname) & (R.ktp == ktp) & (R.ksl == ksl)]
        fx = R[test_yr & (R.dir == dname) & (R.ktp == KS[0]) & (R.ksl == KS[0])]
        for lab, v, col in ((f"배리어 {ktp}/{ksl}", sub, "pnl"),
                            ("고정지평", fx, "base")):
            x = v[col].to_numpy() if col == "pnl" else v[col].to_numpy() - COST_BP
            if not len(x):
                continue
            wins = x[x > 0]; loss = x[x <= 0]
            W = wins.mean() if len(wins) else 0.0
            L = -loss.mean() if len(loss) else 0.0
            need = (L + COST_BP) / (W + L) if (W + L) > 0 else np.nan
            print(f"{dname:>8} {lab:>16} {x.mean():>+9.2f} {len(x):>7} {len(wins)/len(x):>7.1%} "
                  f"{W:>+9.1f} {-L:>+9.1f} {need:>9.1%} {len(wins)/len(x):>9.1%}")
    print(f"\n저장: {OUT/'exit_grid.parquet'}")


def stage_wall(a):
    """⭐**벽을 실측 적중률 아래로 끌어내릴 수 있나** — 산술이 지목하는 마지막 자리.

    H 절 실측: 4h 적중 **50.54% ±0.23** vs 벽@**0bp** 50.00% ⇒ **+0.54pp = 2.3σ**.
    즉 **총수익 기준으로는 이미 유의하게 양수**고 문제는 오직 `비용/(2·E|r|)` 이다.
    벽 = 0.5 + 비용/(2·E|r|) 은 **E|r| 가 커지면 계속 내려간다**. 1d 에서 E|r|=345.7 ·
    벽@5.52bp=50.80% · 실측 50.48% — 간극이 **0.32pp** 까지 좁혀져 있었다.

    아직 안 당긴 레버 둘을 여기서 당긴다:
      ① **더 극단 분위** — 지금까지 상위 20% 만 봤다. 10/5/2.5/1% 로 조이면 E|r| 가 오른다.
      ② **더 긴 지평** — 4h/12h/1d 만 봤다. 3d·7d 는 §5.30 A 표에서 E|r| 449·695bp 다.
    🔴**분모를 키우면 표본이 준다** — 적중률 SE 가 같이 커진다. 그래서 «벽 아래로 내려갔다」가
    아니라 **«적중 − 벽» 을 SE 로 나눈 t** 로 판정한다. 그리고 상위 분위는 **인과 확장창**이다.

    계산 절약: 방향 모델은 **상위 20% 모집단에서 한 번만** 학습하고 더 좁은 분위는 그 **부분집합**
    이므로 그대로 적용한다(재학습 없음 = 새 다중검정 없음)."""
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    global SINCE, HS
    SINCE = "2022-01-01"
    HS_OLD = HS
    HS = {"4h": 48, "12h": 144, "1d": 288, "3d": 864, "7d": 2016}
    rng = np.random.default_rng(SEED)
    assets = [A for A in ASSETS20 if (BV_PANEL / f"{A}USDT.parquet").exists()]
    # 🔴`--teststart` 는 **셀을 고정한 채 시험 구간만** 앞으로 미는 용도다. 셀을 늘리면
    #   max-t 귀무가 올라 p 가 나빠지지만, **사전 등록된 단일 셀을 새 기간에 거는 건 다중성이
    #   늘지 않는다**. 2022-07~2023-12 는 이 세션에서 **채점에 한 번도 안 쓴 18개월**이다
    #   (E|r| 모델 학습에만 썼다) — 독립일을 늘리는 유일하게 정직한 길.
    QUARTERS = pd.date_range(a.teststart, "2026-07-01", freq="QS")
    QGRID = (0.80, 0.90, 0.95, 0.975, 0.99)
    COSTS = (10.0, 8.03, 5.52, 0.0)
    print(f"자산 {len(assets)} · 지평 {list(HS)} · 분위 {[f'{1-q:.1%}' for q in QGRID]}", flush=True)
    cols, TRAIN, META = None, {}, {}
    for A in assets:
        p = panel(A, since=SINCE)
        if cols is None:
            cols = [c for c in featcols(p) if c != "hour_f" and not c.startswith("tf_")
                    and not c.startswith("ztf_")]
        lc = p["__lc__"].to_numpy(); ts = p["timestamp"]
        X = p[cols].to_numpy(np.float32)
        sub = {}
        for hn, H in HS.items():
            y = np.log(np.maximum(np.abs(fwd_of(lc, H)), 1.0))
            idx = np.flatnonzero(np.isfinite(y))[::36]
            sub[hn] = (idx, y[idx])
        un = np.unique(np.concatenate([sub[h][0] for h in HS]))
        TRAIN[A] = (X[un], sub, un)
        META[A] = (ts, lc, len(lc))
        del p, X
    print("  1패스 완료", flush=True)
    EMOD = {}
    for hn, H in HS.items():
        for q0 in QUARTERS:
            Xs, ys = [], []
            for A in assets:
                Xa, sub, un = TRAIN[A]
                idx, y = sub[hn]
                m = META[A][0].to_numpy()[idx] < np.datetime64(q0 - pd.Timedelta(minutes=5 * H))
                if m.sum() < 100:
                    continue
                Xs.append(Xa[np.searchsorted(un, idx[m])]); ys.append(y[m])
            if Xs:
                EMOD[(hn, q0)] = HistGradientBoostingRegressor(
                    max_iter=120, learning_rate=0.06, max_depth=4, l2_regularization=3.0,
                    random_state=11).fit(np.vstack(Xs), np.concatenate(ys))
        print(f"  [E|r| {hn}] 완료", flush=True)
    del TRAIN
    CAND = {}
    for A in assets:
        p = panel(A, since=SINCE)
        ts = p["timestamp"]; lc = p["__lc__"].to_numpy()
        Xf = p[cols].to_numpy(np.float32)
        for hn, H in HS.items():
            pred = np.full(len(lc), np.nan)
            for q0 in QUARTERS:
                mdl = EMOD.get((hn, q0))
                if mdl is None:
                    continue
                te = np.flatnonzero(((ts >= q0) & (ts < q0 + pd.DateOffset(months=3))).to_numpy())
                if len(te):
                    pred[te] = mdl.predict(Xf[te])
            fwd = fwd_of(lc, H)
            oi = np.flatnonzero(np.isfinite(pred) & np.isfinite(fwd))
            if len(oi) < 600:
                continue
            # 각 분위의 인과 임계를 한 번에 — 좁은 분위는 넓은 분위의 부분집합이다
            E = pd.Series(pred[oi]).expanding(500)
            TH = {q: E.quantile(q).shift(1).to_numpy() for q in QGRID}
            base = np.isfinite(TH[QGRID[0]]) & (pred[oi] > TH[QGRID[0]])
            keep = nonoverlap(oi[base], H)
            if len(keep) < 40:
                continue
            pos = np.searchsorted(oi, keep)
            CAND[(A, hn)] = dict(i=keep.astype(np.int32), ts=ts.to_numpy()[keep],
                                 X=Xf[keep], fwd=fwd[keep].astype(np.float32),
                                 pred=pred[keep],
                                 th={q: TH[q][pos] for q in QGRID})
        del p, Xf
    del EMOD
    print("  후보 추출 완료 · " + " · ".join(
        f"{hn} {sum(len(CAND[(A,hn)]['i']) for A in assets if (A,hn) in CAND):,}" for hn in HS),
        flush=True)
    DIRP = {}
    for hn, H in HS.items():
        for q0 in QUARTERS:
            Xs, ys, ws = [], [], []
            for A in assets:
                c = CAND.get((A, hn))
                if c is None:
                    continue
                m = c["ts"] < np.datetime64(q0 - pd.Timedelta(minutes=5 * H))
                if m.sum() < 40:
                    continue
                Xs.append(c["X"][m]); ys.append((c["fwd"][m] > 0).astype(int))
                ws.append(np.abs(c["fwd"][m]))
            if not Xs:
                continue
            X = np.vstack(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
            if len(y) < 300 or y.mean() in (0.0, 1.0):
                continue
            clfs = [HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_depth=3, l2_regularization=3.0,
                random_state=sd).fit(X, y, sample_weight=w / w.mean()) for sd in (11, 907, 4231)]
            for A in assets:
                c = CAND.get((A, hn))
                if c is None:
                    continue
                te = (c["ts"] >= np.datetime64(q0)) & (c["ts"] < np.datetime64(
                    q0 + pd.DateOffset(months=3)))
                if te.sum():
                    DIRP.setdefault((A, hn), np.full(len(c["i"]), np.nan))[te] = np.mean(
                        [cf.predict_proba(c["X"][te])[:, 1] for cf in clfs], axis=0)
        print(f"  [방향 {hn}] 완료", flush=True)
    # 🔴건별 레코드를 반드시 남긴다 — 이항 SE 는 **틀린다**. 20자산의 극단-E|r| 봉은 같은 날에
    #   뭉치므로(시장 전체 변동성 급등) 진짜 SE 는 훨씬 크다. 날짜블록 검정을 하려면 원본이 있어야 한다.
    recs = []
    for hn in HS:
        for A in assets:
            c = CAND.get((A, hn)); p_ = DIRP.get((A, hn))
            if c is None or p_ is None:
                continue
            m = np.isfinite(p_)
            if not m.sum():
                continue
            d = {"asset": A, "H": hn, "ts": c["ts"][m], "fwd": c["fwd"][m], "prob": p_[m]}
            for q in QGRID:
                d[f"q{int(q*1000)}"] = (np.isfinite(c["th"][q]) & (c["pred"] > c["th"][q]))[m]
            recs.append(pd.DataFrame(d))
    REC = pd.concat(recs, ignore_index=True)
    REC.to_parquet(OUT / f"wall_records_{a.teststart[:7]}.parquet")
    print(f"  건별 레코드 {len(REC):,} 저장 → wall_records.parquet", flush=True)

    rows = []
    for hn in HS:
        for q in QGRID:
            fw, pr = [], []
            for A in assets:
                c = CAND.get((A, hn)); p_ = DIRP.get((A, hn))
                if c is None or p_ is None:
                    continue
                m = np.isfinite(p_) & np.isfinite(c["th"][q]) & (c["pred"] > c["th"][q])
                if m.sum():
                    fw.append(c["fwd"][m]); pr.append(p_[m])
            if not fw:
                continue
            fw = np.concatenate(fw); pr = np.concatenate(pr)
            if len(fw) < 200:
                continue
            side = np.where(pr > 0.5, 1, -1)
            acc = float((side * fw > 0).mean()); n = len(fw)
            se = float(np.sqrt(acc * (1 - acc) / n))
            ear = float(np.abs(fw).mean())
            r = {"H": hn, "상위%": f"{1-q:.1%}", "n": n, "E|r|": ear,
                 "적중%": acc * 100, "SE": se * 100,
                 "롱적중%": float((fw > 0).mean()) * 100}
            for c_ in COSTS:
                wall = 0.5 + c_ / (2 * ear)
                r[f"벽@{c_:g}"] = (acc - wall) * 100
                r[f"t@{c_:g}"] = (acc - wall) / se if se > 0 else np.nan
                r[f"net@{c_:g}"] = (2 * acc - 1) * ear - c_
            rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / f"wall_sweep_{a.teststart[:7]}.csv", index=False)
    print("\n=== ⭐벽까지의 거리 (적중 − 벽, pp) · 괄호는 t = (적중−벽)/SE ===")
    print(f"{'지평':>5} {'상위':>7} {'n':>7} {'E|r|':>7} {'적중%':>7} {'SE':>5} {'롱적중%':>8} |"
          + "".join(f"{'벽@'+f'{c:g}bp':>17}" for c in COSTS))
    for _, x in D.iterrows():
        print(f"{x['H']:>5} {x['상위%']:>7} {int(x['n']):>7} {x['E|r|']:>7.1f} {x['적중%']:>7.2f} "
              f"{x['SE']:>5.2f} {x['롱적중%']:>8.2f} |"
              + "".join(f"{x[f'벽@{c:g}']:>+9.2f} ({x[f't@{c:g}']:>+4.1f})" for c in COSTS))
    win = D[(D["t@5.52"] > 1.0) | (D["t@10"] > 1.0)]
    print(f"\n🔴**t > 1.0 인 칸: {len(win)} / {len(D)}** "
          + ("" if len(win) else "— 어떤 (지평 × 분위)에서도 벽을 유의하게 넘지 못한다."))
    if len(win):
        print(win[["H", "상위%", "n", "E|r|", "적중%", "SE", "벽@10", "t@10", "벽@5.52",
                   "t@5.52", "net@10", "net@5.52"]].to_string(index=False,
                                                              float_format=lambda v: f"{v:8.2f}"))
    HS = HS_OLD
    print(f"\n저장: {OUT/f'wall_sweep_{a.teststart[:7]}.csv'}")


def stage_wallcheck(a):
    """🔴**`--stage wall` 의 「벽을 넘었다」를 정직하게 다시 잰다.**

    wall 의 SE 는 **이항(iid)** 이라 틀린다: 20자산의 극단-E|r| 봉은 **같은 날에 뭉친다**
    (시장 전체 변동성 급등). 여기서는 전부 **날짜블록 부트스트랩**으로 다시 내고, 동시에
    「모델이 무조건 롱보다 나은가」·「연도별로 사는가」·「자산 부호가 일치하는가」·
    「독립 일수가 몇인가」를 같이 낸다. 다중성(19칸 스윕)도 명시한다."""
    from scipy.stats import binomtest
    rng = np.random.default_rng(SEED)
    R = pd.read_parquet(OUT / "wall_records.parquet")
    R["ts"] = pd.to_datetime(R["ts"])
    R["day"] = R.ts.dt.floor("D")
    R["year"] = R.ts.dt.year
    R["side"] = np.where(R.prob > 0.5, 1, -1)
    QCOLS = [c for c in R.columns if c.startswith("q")]
    print(f"레코드 {len(R):,} · 지평 {sorted(R.H.unique())} · 분위열 {QCOLS}\n")

    def dblock(v, d, B=4000):
        days = np.unique(d)
        by = {x: v[d == x] for x in days}
        bs = np.empty(B)
        for b in range(B):
            pick = rng.choice(days, len(days), replace=True)
            bs[b] = np.concatenate([by[x] for x in pick]).mean()
        return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

    print(f"{'지평':>4} {'상위':>6} {'n':>6} {'독립일':>6} {'E|r|':>7} | "
          f"{'모델 건당net':>25} | {'롱 건당net':>25} | {'모델−롱':>22} | {'24/25/26':>24} {'자산':>6}")
    out = []
    for hn in ("4h", "12h", "1d", "3d", "7d"):
        for qc in QCOLS:
            s = R[(R.H == hn) & R[qc]]
            if len(s) < 150:
                continue
            d = s.day.values
            ear = float(s.fwd.abs().mean())
            mdl = (s.side * s.fwd - COST_BP).to_numpy()
            lng = (s.fwd - COST_BP).to_numpy()
            dif = mdl - lng
            l1, h1 = dblock(mdl, d); l2, h2 = dblock(lng, d); l3, h3 = dblock(dif, d)
            ys = [float((s[s.year == y].side * s[s.year == y].fwd - COST_BP).mean())
                  if (s.year == y).sum() >= 30 else np.nan for y in YEARS]
            per = s.groupby("asset").apply(
                lambda g: float((g.side * g.fwd - COST_BP).mean()), include_groups=False)
            npos = int((per > 0).sum())
            q = 1 - int(qc[1:]) / 1000
            out.append({"H": hn, "상위%": q * 100, "n": len(s), "독립일": len(np.unique(d)),
                        "E|r|": ear, "모델net": mdl.mean(), "모델lo": l1, "모델hi": h1,
                        "롱net": lng.mean(), "롱lo": l2, "롱hi": h2,
                        "증분": dif.mean(), "증분lo": l3, "증분hi": h3,
                        "e24": ys[0], "e25": ys[1], "e26": ys[2],
                        "양수자산": npos, "자산수": len(per),
                        "signp": binomtest(npos, len(per), 0.5, "greater").pvalue})
            print(f"{hn:>4} {1-int(qc[1:])/1000:>6.1%} {len(s):>6} {len(np.unique(d)):>6} "
                  f"{ear:>7.1f} | {mdl.mean():>+8.2f} [{l1:>+7.2f},{h1:>+7.2f}] {'✅' if l1>0 else '  '}"
                  f" | {lng.mean():>+8.2f} [{l2:>+7.2f},{h2:>+7.2f}] {'✅' if l2>0 else '  '}"
                  f" | {dif.mean():>+7.2f} [{l3:>+6.1f},{h3:>+6.1f}] {'✅' if l3>0 else '  '}"
                  f" | " + "".join(f"{v:>+8.1f}" for v in ys) + f" {npos:>3}/{len(per):<3}")
    D = pd.DataFrame(out)
    D.to_csv(OUT / "wall_check.csv", index=False)
    ok = D[(D.모델lo > 0) & (D[["e24", "e25", "e26"]] > 0).all(axis=1) & (D.signp < 0.05)]
    print(f"\n🔴**세 관문 동시 통과**(모델 건당net 날짜블록 CI 0배제 & 세 해 양수 & "
          f"자산 부호검정 p<0.05): **{len(ok)} / {len(D)}칸**")
    if len(ok):
        print(ok[["H", "상위%", "n", "독립일", "E|r|", "모델net", "모델lo", "모델hi", "증분",
                  "증분lo", "e24", "e25", "e26", "양수자산", "자산수", "signp"]].to_string(
            index=False, float_format=lambda v: f"{v:8.2f}"))
    print(f"\n⚠️다중성: 이 표는 **{len(D)}칸** 스윕이다(지평 5 × 분위 5). 통과 칸은 그 사실과 함께 읽는다."
          f"\n⚠️독립일수 열을 보라 — n 이 커도 **같은 날에 뭉쳐 있으면** 실효 표본은 그 날 수다.")
    print(f"\n저장: {OUT/'wall_check.csv'}")


def cluster_se(x: np.ndarray, day: np.ndarray) -> float:
    """**날짜 클러스터 로버스트 SE.** 같은 날 사건들은 독립이 아니다(시장 전체 변동성 급등).

    SE = sqrt( Σ_d ( Σ_{i∈d} (x_i − x̄) )² ) / n — 부트스트랩과 같은 것을 닫힌 형식으로 준다.
    이항/정규 SE 는 이 합을 **날짜 안에서도 독립**이라고 가정해 과소평가한다."""
    n = len(x)
    if n < 3:
        return float("inf")
    r = x - x.mean()
    s = pd.Series(r).groupby(day).sum().to_numpy()
    v = float((s ** 2).sum())
    return float(np.sqrt(v)) / n if v > 0 else float("inf")


def stage_multi(a):
    """⭐**다중성을 가격에 반영한다 — 와일드 클러스터 부호뒤집기 max-t 귀무.**

    12절이 남긴 마지막 미방어: `4h×상위5%` 는 **19칸 스윕(지평5×분위5) 중 1칸**이고
    칸들이 **중첩·상관**이라 단순 본페로니도 BH 도 맞지 않는다. 옳은 방법은
    **스윕 전체의 max-t 분포를 귀무에서 만드는 것**이다.

    귀무 구성: **날짜 단위 부호뒤집기**(wild cluster bootstrap, Rademacher). 각 «날»의 사건
    전체에 같은 부호를 곱한다 ⇒ **날 안의 의존구조와 날별 사건 수가 보존**되고 평균만 0 이 된다.
    같은 뒤집기를 **19칸에 동시에** 적용하므로 칸 간 상관이 자동으로 반영된다.
    통계량은 칸별 t = 평균/클러스터SE, 스윕 통계량은 **max t**.
    🔴이 검정은 **그로스**(비용 0) 기준이다 — 12절에서 0 을 배제한 게 그로스였다.
    비용을 넣으면 귀무 중심이 −비용으로 이동해 부호뒤집기 귀무가 성립하지 않는다."""
    rng = np.random.default_rng(SEED)
    R = pd.read_parquet(OUT / "wall_records.parquet")
    R["ts"] = pd.to_datetime(R["ts"])
    R["day"] = R.ts.dt.floor("D")
    R["side"] = np.where(R.prob > 0.5, 1, -1)
    R["x"] = R.side * R.fwd
    QCOLS = [c for c in R.columns if c.startswith("q")]
    cells = []
    for hn in sorted(R.H.unique()):
        for qc in QCOLS:
            s = R[(R.H == hn) & R[qc]]
            if len(s) < 150:
                continue
            cells.append((hn, 1 - int(qc[1:]) / 1000, s.x.to_numpy(),
                          s.day.values.astype("datetime64[D]").astype(np.int64)))
    print(f"칸 {len(cells)}개 · 총 사건 {sum(len(c[2]) for c in cells):,}\n")
    # 클러스터 SE 가 부트스트랩과 맞는지 먼저 확인한다(검정의 전제)
    print(f"{'지평':>4} {'상위':>6} {'n':>6} {'독립일':>6} {'그로스':>8} {'클러스터SE':>10} "
          f"{'t':>6} {'이항t(참고)':>11}")
    ts_obs = []
    for hn, q, x, d in cells:
        se = cluster_se(x, d)
        t = x.mean() / se
        ts_obs.append(t)
        naive = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        print(f"{hn:>4} {q:>6.1%} {len(x):>6} {len(np.unique(d)):>6} {x.mean():>+8.2f} "
              f"{se:>10.2f} {t:>+6.2f} {naive:>+11.2f}")
    ts_obs = np.array(ts_obs)
    best = int(np.argmax(ts_obs))
    print(f"\n최대 t: **{ts_obs[best]:+.2f}** ({cells[best][0]} 상위{cells[best][1]:.1%})")
    # ── 와일드 클러스터 부호뒤집기 ──────────────────────────────────────────
    alldays = np.unique(np.concatenate([c[3] for c in cells]))
    pos = {d: i for i, d in enumerate(alldays)}
    IDX = [np.array([pos[v] for v in c[3]]) for c in cells]
    maxt = np.empty(a.nperm)
    for b in range(a.nperm):
        flip = rng.choice([-1.0, 1.0], len(alldays))
        m = -1e9
        for (hn, q, x, d), ii in zip(cells, IDX):
            xf = x * flip[ii]
            se = cluster_se(xf, d)
            m = max(m, xf.mean() / se)
        maxt[b] = m
    p = float((maxt >= ts_obs[best]).mean())
    print(f"\n=== ⭐다중성 보정 (와일드 클러스터 부호뒤집기 · B={a.nperm}) ===")
    print(f"  귀무 max-t: 평균 {maxt.mean():+.2f} · 중앙 {np.median(maxt):+.2f} · "
          f"95분위 **{np.quantile(maxt,0.95):+.2f}** · 최대 {maxt.max():+.2f}")
    print(f"  실제 max-t **{ts_obs[best]:+.2f}** ⇒ **스윕 전체 p = {p:.4f}** "
          f"{'✅ 다중성 보정 후에도 유의' if p < 0.05 else '❌ 다중성 보정에서 탈락'}")
    # 칸별 보정 p (max-t 귀무 대비)
    print(f"\n{'지평':>4} {'상위':>6} {'t':>6} {'보정 p':>8}")
    for (hn, q, x, d), t in sorted(zip(cells, ts_obs), key=lambda z: -z[1])[:8]:
        print(f"{hn:>4} {q:>6.1%} {t:>+6.2f} {float((maxt >= t).mean()):>8.4f}")
    pd.DataFrame({"H": [c[0] for c in cells], "상위": [c[1] for c in cells],
                  "n": [len(c[2]) for c in cells], "그로스": [c[2].mean() for c in cells],
                  "t": ts_obs, "보정p": [float((maxt >= t).mean()) for t in ts_obs]}
                 ).to_csv(OUT / "multiplicity.csv", index=False)
    print(f"\n저장: {OUT/'multiplicity.csv'}")


def stage_freeze(a):
    """⭐**후보를 동결된 아티팩트로 만든다** — `4h × 예측 E|r| 상위5%`.

    13절 결론: 다중성 보정 p **0.0610**(임계 0.05). **더 파면 p 가 나빠진다**(max-t 귀무가
    칸 수에 따라 오른다). 남은 길은 **독립 관측 증가**뿐이고 그건 오늘 시계를 시작해야 는다.
    이 stage 는 그 시계를 위한 **서빙 가능한 아티팩트**를 만든다 — 연구 숫자가 아니라 물건이다.

    담는 것: ①E|r| 회귀(4h) ②방향 분류(상위20% 모집단·3씨드) ③피쳐 목록(74, 순서 고정)
    ④**예측 E|r| 이력**(자산별) — 섀도우가 인과 확장창 분위를 **웜스타트**하는 데 필요하다
    (없으면 처음 500건은 판정 불가) ⑤manifest(sha256·학습창·파라미터).
    🔴학습은 **가진 전 구간**(2022-01~데이터 끝)이다. 이건 백테스트가 아니라 **배포용 최종 적합**
    이므로 워크포워드가 아니다 — 그리고 **이 아티팩트로 과거를 다시 채점하면 표본내**다."""
    import hashlib
    import json
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    global SINCE
    SINCE = "2022-01-01"
    HN = a.hz
    H, TOPQ = HS[HN], a.gateq
    assets = [A for A in ASSETS20 if (BV_PANEL / f"{A}USDT.parquet").exists()]
    # 🔴`int((1-0.90)*100)` 은 부동소수 때문에 **9** 가 된다(0.09999…). round 를 쓴다.
    dst = ROOT / f"data/models/direction_{HN}_top{round((1 - TOPQ) * 100)}_20260915"
    dst.mkdir(parents=True, exist_ok=True)
    cols = None
    Xs, ys, hist, last = [], [], {}, {}
    for A in assets:
        p = panel(A, since=SINCE)
        if cols is None:
            cols = [c for c in featcols(p) if c != "hour_f" and not c.startswith("tf_")
                    and not c.startswith("ztf_")]
        lc = p["__lc__"].to_numpy()
        y = np.log(np.maximum(np.abs(fwd_of(lc, H)), 1.0))
        idx = np.flatnonzero(np.isfinite(y))[::36]
        Xs.append(p[cols].to_numpy(np.float32)[idx]); ys.append(y[idx])
        last[A] = str(p["timestamp"].iloc[-1])
        print(f"  [{A}] 학습표본 {len(idx):,}", flush=True)
        del p
    emod = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06, max_depth=4,
                                         l2_regularization=3.0, random_state=11)
    emod.fit(np.vstack(Xs), np.concatenate(ys))
    del Xs, ys
    print("  E|r| 모델 적합 완료", flush=True)
    # 상위20% 모집단에서 방향 학습 + 예측 이력 저장
    DX, DY, DW = [], [], []
    for A in assets:
        p = panel(A, since=SINCE)
        lc = p["__lc__"].to_numpy(); ts = p["timestamp"]
        Xf = p[cols].to_numpy(np.float32)
        pred = emod.predict(Xf)
        fwd = fwd_of(lc, H)
        ok = np.isfinite(fwd)
        hist[A] = {"ts": [str(x) for x in ts.iloc[::12]],
                   "pred": [round(float(v), 6) for v in pred[::12]]}
        thr20 = np.quantile(pred[ok], a.dirpop)
        m = ok & (pred > thr20)
        keep = nonoverlap(np.flatnonzero(m), H)
        DX.append(Xf[keep]); DY.append((fwd[keep] > 0).astype(int)); DW.append(np.abs(fwd[keep]))
        del p, Xf
    X = np.vstack(DX); Y = np.concatenate(DY); W = np.concatenate(DW)
    dmods = [HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3,
                                            l2_regularization=3.0, random_state=sd
                                            ).fit(X, Y, sample_weight=W / W.mean())
             for sd in (11, 907, 4231)]
    print(f"  방향 모델 적합 완료 (표본 {len(Y):,} · 롱비율 {Y.mean():.1%})", flush=True)
    joblib.dump({"evr": emod, "dir": dmods, "cols": cols}, dst / "models.joblib", compress=3)
    (dst / "evr_history.json").write_text(json.dumps(hist))
    sha = hashlib.sha256((dst / "models.joblib").read_bytes()).hexdigest()
    man = {
        "name": dst.name,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "horizon_bars": H, "horizon": HN, "gate_quantile": TOPQ,
        "dir_train_population_quantile": a.dirpop,
        "assets": assets, "n_features": len(cols), "features": cols,
        "train_since": SINCE, "train_until": last,
        "models_sha256": sha,
        "evidence_ref": ("data/research/eth_event_expansion_20260915/wall_sweep_2024-01.csv "
                         "· 판정 기준은 사용자 결정(2024·25·26 각 해 순손익>0 · 메이커 5.52bp)"),
        "status": "USER_APPROVED_2024_2026_ECONOMICS",
        "gates_failed": ["net_dateblock_ci_includes_zero",
                         "multiplicity_p_0.061_above_0.05",
                         "model_minus_long_ci_includes_zero_except_1d_top10"],
        "portfolio_only": ("🔴ETH 단독은 세 해 양수가 아니다 — **20자산 포트폴리오로만 성립한다.** "
                           "사용자 승인(2026-09-15): 다른 자산도 함께 계산해도 된다."),
        "doc": "docs/experiments/direction_event_trigger_expansion_and_oos_audit_20260915.md",
    }
    (dst / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2))
    print(f"\n⭐아티팩트 저장: {dst}")
    print(f"  models.joblib sha256 {sha[:16]}… · evr_history.json (웜스타트) · manifest.json")
    print(f"  status = {man['status']} · 미통과 관문 {len(man['gates_failed'])}개를 manifest 에 기록")


def stage_quiet(a):
    """⭐**「소음이 없는 구간에는 우리 데이터에 답이 있다」를 벽까지의 거리로 잰다** (2026-09-16, 사용자 가설)

    사용자 관찰: *"방향은 한 순간의 체결·고래흐름과 외부 국채/경제 뉴스에 따라 바뀐다. 하지만
    이런 것들이 없는 횡보장에는 우리가 갖고 있는 데이터에 정답이 무조건 있다."*

    🔴선행 결과가 이 가설의 **절반을 이미 확인했다**: 09-14 횡보 조건부 검정에서 「횡보에서 더
    맞는다」가 **통계적으로 참**이었다(DiD +1.6~+3.5pp · 조건 순환이동 귀무 20칸 중 16칸 p<0.05).
    돈이 안 된 이유는 구조적이다 — 벽 = 0.5 + 비용/(2·E|r|) 인데 **조용한 구간은 정의상 E|r| 이
    작아 벽이 가장 높다**(배리어 18bp → 손익분기 78.4%). 배리어를 넓혀 벽을 51.9% 까지 내리면
    **초과가 같이 0 이 됐다**.
    ⇒ 그러므로 이 판의 질문은 「있나 없나」가 **아니라** 「**정확도 상승이 E|r| 하락을 이기는가**」다.

    ⭐09-14 와 다른 점 — **횡보를 변동성 레짐이 아니라 «소음원 끄기»로 정의한다**:
      ① 매크로 창 밖   8:30 ET ±30분(하드데이터) · 14:00 ET ±30분(FOMC) · 평일 (09-11 대리변수,
                       전체 봉의 10.4%이고 그 안에서 롱 −9.2 / 숏 +4.6bp 로 측면이 뒤집힌다)
      ② 고래 조용      |tf_bigimb|(상위1% 대형체결 불균형) **하위 Q**
      ③ 큰 움직임 아님  예측 E|r| **하위 Q**
    교집합 = 「조용한 봉」. 기본 Q=0.30 (사용자 결정 2026-09-16).

    사전 등록:
      자산 ETH(사용자 결정) · 시험 2024-07~2026-09 · 분기 확장 WF · 비겹침 · 3씨드 · 가중 |fwd|
      지평 1h/4h/12h 나란히 · 비용 열 10 / 5.52(메이커) / 3.96(양다리 메이커) / **0**
      팔 셋: quiet(조용) · noisy(거울: 세 조건 모두 반대) · all(전체) — **각 팔은 자기 모집단에서
      학습하고 자기 모집단에서 평가**한다(같은 모델을 부분집합에 적용하면 팔 간 비교가 오염된다).
      판정 = **적중률 − 벽** 을 날짜블록 CI 로 본다. 🔴이항 SE 는 쓰지 않는다(사건이 뭉친다).
    🔴임계는 **그 분기의 학습창 분포**에서만 뽑는다(엠바고 H봉 적용) — 전수 분위는 미래참조다.
    🔴비용 0 열이 이 판의 핵심이다: 넘으면 「집행 문제」, 못 넘으면 「정보 아님」으로 갈린다."""
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from zoneinfo import ZoneInfo
    A, Q = a.asset, a.quiet
    SINCE = "2024-01-01"                       # 틱 피쳐(aggfeat) 커버리지 시작
    HSQ = {"1h": 12, "4h": 48, "12h": 144}
    TRAIN_CAP = 30_000                         # 팔마다 같은 상한 (학습량 교란 차단)
    COSTS = (10.0, 5.52, 3.96, 0.0)
    rng = np.random.default_rng(SEED)

    p = panel(A, ticks=True, since=SINCE)
    if "tf_bigimb" not in p.columns:
        print(f"없음: {A} 틱 피쳐(data/binance_vision/aggfeat) — 고래 컷을 만들 수 없다"); return
    ts = p["timestamp"]; tsv = ts.to_numpy(); lc = p["__lc__"].to_numpy()
    cols = featcols(p)
    X = p[cols].to_numpy(np.float32)
    finite = np.isfinite(X).any(1)

    ny = ts.dt.tz_localize("UTC").dt.tz_convert(ZoneInfo("America/New_York"))
    mm = (ny.dt.hour * 60 + ny.dt.minute).to_numpy()
    macro = ((ny.dt.dayofweek.to_numpy() < 5)
             & (((mm >= 480) & (mm <= 540)) | ((mm >= 810) & (mm <= 870))))
    big = np.abs(p["tf_bigimb"].to_numpy(float))
    print(f"{A} · 봉 {len(p):,} ({ts.iloc[0]:%Y-%m-%d}~{ts.iloc[-1]:%Y-%m-%d}) · 피쳐 {len(cols)} "
          f"· 매크로창 {macro.mean():.1%} · 조용 분위 {Q:.0%}", flush=True)

    QT = list(pd.date_range("2024-07-01", "2026-10-01", freq="QS"))
    recs = []
    for hn, H in HSQ.items():
        fwd = fwd_of(lc, H)
        ylg = np.log(np.maximum(np.abs(fwd), 1.0))
        ok = np.isfinite(fwd) & finite & np.isfinite(big)
        for q0, q1 in zip(QT[:-1], QT[1:]):
            emb = np.datetime64(q0 - pd.Timedelta(minutes=5 * H))
            tr = ok & (tsv < emb)
            te = ok & (tsv >= np.datetime64(q0)) & (tsv < np.datetime64(q1))
            if tr.sum() < 5000 or te.sum() < 200:
                continue
            itr = np.flatnonzero(tr)[::12]                    # 1시간당 1봉 (E|r| 회귀 표집)
            ev = HistGradientBoostingRegressor(max_iter=150, random_state=SEED).fit(X[itr], ylg[itr])
            pr = np.full(len(lc), np.nan)
            mall = np.flatnonzero(ok)
            pr[mall] = ev.predict(X[mall])
            elo, ehi = np.quantile(pr[tr], Q), np.quantile(pr[tr], 1 - Q)
            blo, bhi = np.quantile(big[tr], Q), np.quantile(big[tr], 1 - Q)
            # ⭐컷을 **분해**한다 (2026-09-16 사용자 질문: 「변동성 낮은 걸로만 가면?」).
            #   세 컷을 한 번에 걸면 무엇이 일했는지 알 수 없다. evr_only / big_only 로 가른다.
            evr_lo, big_lo = pr <= elo, big <= blo
            pops = {"evr_only": evr_lo,                        # ⭐저변동만 (= 횡보의 정석 정의)
                    "big_only": big_lo,                        # 고래 조용만
                    "quiet": evr_lo & big_lo & ~macro,         # 셋 다 (사용자 가설 원형)
                    "noisy": (pr >= ehi) & (big >= bhi) & ~macro,   # 두 연속 컷의 거울
                    "all": np.ones(len(lc), bool),                  # 전체
                    "all_sz": np.ones(len(lc), bool)}               # 전체 · 학습 **크기 맞춤**
            nq = int((tr & pops["quiet"]).sum())
            for arm, pop in pops.items():
                itrain = np.flatnonzero(tr & pop)
                itest = np.flatnonzero(te & pop)
                # 🔴크기 맞춘 대조군 — 조용한 팔은 9% 표본으로 학습하는데 전체 팔이 100% 로
                #   학습하면 「레짐」과 「학습량」이 섞인다. all_sz 는 quiet 과 **같은 수**로 자른다.
                cap = nq if arm == "all_sz" else TRAIN_CAP
                if len(itrain) > cap:
                    itrain = itrain[:: len(itrain) // cap + 1]
                if len(itrain) < 500 or len(itest) < 20:
                    continue
                w = np.abs(fwd[itrain]); w = w / max(w.mean(), 1e-9)
                pb = np.zeros(len(itest))
                for sd in range(3):
                    clf = HistGradientBoostingClassifier(max_iter=150, random_state=SEED + sd)
                    clf.fit(X[itrain], (fwd[itrain] > 0).astype(int), sample_weight=w)
                    pb += clf.predict_proba(X[itest])[:, 1]
                recs.append(pd.DataFrame({"i": itest, "ts": tsv[itest], "H": hn, "arm": arm,
                                          "fwd": fwd[itest], "prob": pb / 3}))
        print(f"  {hn} 완료", flush=True)

    R = pd.concat(recs, ignore_index=True)
    R["day"] = pd.to_datetime(R.ts).dt.floor("D")
    R["side"] = np.where(R.prob > 0.5, 1, -1)
    R.to_parquet(OUT / f"quiet_records_{A}_q{round(Q*100)}.parquet", index=False)

    def blk(v, d, B=4000):
        days = np.unique(d); by = {x: v[d == x] for x in days}
        bs = np.array([np.concatenate([by[x] for x in rng.choice(days, len(days), replace=True)]
                                      ).mean() for _ in range(B)])
        return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

    print(f"\n{'지평':>5} {'팔':>12} {'n':>6} {'독립일':>6} {'E|r|':>7} "
          f"{'적중률':>7} {'날짜블록 CI':>16} | " + " ".join(f"{'벽@'+str(c):>8}" for c in COSTS)
          + " | " + " ".join(f"{'거리@'+str(c):>9}" for c in COSTS))
    rows = []
    for hn, H in HSQ.items():
        for arm in ("evr_only", "big_only", "quiet", "noisy", "all", "all_sz"):
            s = R[(R.H == hn) & (R.arm == arm)].sort_values("i")
            if not len(s):
                continue
            keep = set(nonoverlap(s.i.to_numpy(), H).tolist())
            s = s[s.i.isin(keep)]
            hit = ((s.side * s.fwd) > 0).to_numpy(float)
            d = s.day.values
            er = float(s.fwd.abs().mean())
            acc = hit.mean(); alo, ahi = blk(hit, d)
            walls = [0.5 + c / (2 * er) for c in COSTS]
            print(f"{hn:>5} {arm:>12} {len(s):>6} {len(np.unique(d)):>6} {er:>7.1f} "
                  f"{acc:>7.2%} [{alo:>6.2%},{ahi:>6.2%}] | "
                  + " ".join(f"{w:>8.2%}" for w in walls) + " | "
                  + " ".join(f"{(acc-w)*100:>+8.2f}pp" for w in walls))
            g = (s.side * s.fwd).to_numpy()
            r = {"지평": hn, "팔": arm, "n": len(s), "독립일": len(np.unique(d)), "E|r|": er,
                 "적중": acc, "적중lo": alo, "적중hi": ahi, "그로스": g.mean()}
            for c in COSTS:
                lo, hi = blk(g - c, d)
                r |= {f"net@{c}": g.mean() - c, f"net_lo@{c}": lo, f"net_hi@{c}": hi,
                      f"벽@{c}": 0.5 + c / (2 * er)}
            rows.append(r)
    print(f"\n{'지평':>5} {'팔':>12} | " + " ".join(f"{'건당net@'+str(c):>26}" for c in COSTS))
    for r in rows:
        print(f"{r['지평']:>5} {r['팔']:>12} | " + " ".join(
            f"{r[f'net@{c}']:>+9.2f} [{r[f'net_lo@{c}']:>+7.2f},{r[f'net_hi@{c}']:>+7.2f}]"
            f"{'✅' if r[f'net_lo@{c}'] > 0 else '❌'}" for c in COSTS))
    pd.DataFrame(rows).to_csv(OUT / f"quiet_regime_{A}_q{round(Q*100)}.csv", index=False)
    print(f"\n저장: {OUT / f'quiet_regime_{A}_q{round(Q*100)}.csv'}")
    print("🔴판정: **비용 0 열에서도 거리가 음수(SE 안)면 「정보가 아니다」**. "
          "비용 0 은 넘고 5.52 는 못 넘으면 「집행·지평 문제」다.")
    print("🔴이항 SE 를 쓰지 않았다 — 적중률 CI 는 날짜블록 부트다(사건이 하루에 뭉친다).")


def stage_confirm(a):
    """⭐**사전 등록된 단일 셀을 «채점에 한 번도 안 쓴 기간»에 건다.**

    13절: 다중성 보정 p **0.0610**. 그리고 「더 파면 p 가 나빠진다」 — 맞다, **셀을 늘리면**.
    그러나 **셀을 고정한 채 시험 구간만 새로 여는 건 다중성이 늘지 않는다.**
    이 세션의 모든 스윕은 **2024-01 부터만** 채점했고 **2022-07~2023-12 18개월**은 E|r| 모델
    학습에만 쓰였다 — 채점에는 한 번도 안 썼다. 거기 걸면 독립일이 늘고 다중성은 그대로다.

    🔴사전 등록(이 stage 는 **단 하나의 셀**만 본다). 기본은 13절의 `4h × 상위5%` 이고,
    `--hz/--gateq` 로 **다른 사전 등록 셀**을 걸 수 있다 — 2026-09-16 에 `--hz 1d --gateq 0.90`
    으로 **실제 섀도우로 켠 칸**(`data/models/direction_1d_top10_20260915`)을 같은 규율에
    태웠다. 셀을 바꿔 가며 훑는 용도가 아니다: **미리 정한 칸 하나**를 새 구간에 거는 것이고,
    그래서 여기서도 max-t 보정은 필요 없다. ·
    방향 = 상위20% 모집단 학습 3씨드 · w/상한 없음(건당 기준) · 비용 메이커 5.52bp ·
    판정 = **날짜블록 CI 0배제 AND 세 해 양수**. 스윕이 아니므로 max-t 보정은 필요 없다.
    🔴정직한 한계: 2022~23 은 **레짐이 다르고**(§5.30 이 「23년 이전 제외」라 적은 구간),
    초기 분기는 학습 데이터가 얇다(2022-01~07 = 6개월). 음수면 「신호 없음」과 「레짐 불일치」를
    못 가른다 — 그래서 **세 구간을 나란히** 싣는다."""
    from scipy.stats import binomtest
    rng = np.random.default_rng(SEED)
    f_new = OUT / "wall_records_2022-07.parquet"
    f_old = OUT / "wall_records_2024-01.parquet"
    if not f_new.exists():
        print(f"없음: {f_new} — `--stage wall --teststart 2022-07-01` 먼저"); return
    R = pd.read_parquet(f_new)
    R["ts"] = pd.to_datetime(R["ts"]); R["day"] = R.ts.dt.floor("D"); R["year"] = R.ts.dt.year
    R["side"] = np.where(R.prob > 0.5, 1, -1)
    # ⭐사전 등록 셀 하나. q 컬럼은 **round** 로 만든다 -- int() 는 0.90 에서 q899 가 된다
    #   (2026-09-15 freeze 에서 실제로 난 부동소수 버그와 같은 자리).
    qcol = f"q{round(a.gateq * 1000)}"
    if qcol not in R.columns:
        print(f"없음: {qcol} (있는 것: {[c for c in R.columns if c.startswith('q')]})"); return
    R = R[(R.H == a.hz) & R[qcol]]
    print(f"사전 등록 셀 = {a.hz} × 예측 E|r| 상위{(1 - a.gateq) * 100:.1f}% "
          f"· 전체 레코드 {len(R):,}\n")

    def blk(v, d, B=6000):
        days = np.unique(d); by = {x: v[d == x] for x in days}
        bs = np.array([np.concatenate([by[x] for x in rng.choice(days, len(days), replace=True)]
                                      ).mean() for _ in range(B)])
        return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

    def cse(x, d):
        r = x - x.mean(); s = pd.Series(r).groupby(d).sum().to_numpy()
        return float(np.sqrt((s ** 2).sum())) / len(x)

    COST = 5.52
    print(f"{'구간':>22} {'n':>6} {'독립일':>6} {'E|r|':>7} {'그로스':>8} {'클러t':>6} "
          f"{'메이커 건당':>10} {'날짜블록 CI':>21} {'연도별':>34} {'자산':>7} {'모델−롱':>8}")
    rows = []
    segs = [("⭐신규 2022-07~2023-12", (R.ts >= "2022-07-01") & (R.ts < "2024-01-01")),
            ("기존 2024-01~2026-08", R.ts >= "2024-01-01"),
            ("**합산 2022-07~2026-08**", R.ts >= "2022-07-01")]
    for lab, m in segs:
        s = R[m]
        if len(s) < 100:
            print(f"{lab:>22} {len(s):>6}  표본 부족"); continue
        g = (s.side * s.fwd).to_numpy(); d = s.day.values
        net = g - COST
        lo, hi = blk(net, d)
        yrs = sorted(s.year.unique())
        ys = [(y, float((s[s.year == y].side * s[s.year == y].fwd - COST).mean()))
              for y in yrs if (s.year == y).sum() >= 30]
        per = s.groupby("asset").apply(lambda t: float((t.side * t.fwd - COST).mean()),
                                       include_groups=False)
        npos = int((per > 0).sum()); pv = binomtest(npos, len(per), 0.5, "greater").pvalue
        dif = net - (s.fwd - COST).to_numpy()
        dl, dh = blk(dif, d)
        y3 = all(v > 0 for _, v in ys)
        print(f"{lab:>22} {len(s):>6} {len(np.unique(d)):>6} {float(s.fwd.abs().mean()):>7.1f} "
              f"{g.mean():>+8.2f} {g.mean()/cse(g,d):>+6.2f} {net.mean():>+10.2f} "
              f"[{lo:>+8.2f},{hi:>+8.2f}]{'✅' if lo>0 else '❌'} "
              + " ".join(f"{y}:{v:+.0f}" for y, v in ys) + f"{'✅' if y3 else '❌'}"
              + f" {npos:>2}/{len(per):<2} p{pv:.3f} {dif.mean():>+7.2f}{'✅' if dl>0 else '❌'}")
        rows.append({"구간": lab, "n": len(s), "독립일": len(np.unique(d)),
                     "E|r|": float(s.fwd.abs().mean()), "그로스": g.mean(),
                     "클러스터t": g.mean() / cse(g, d), "메이커건당": net.mean(),
                     "CI_lo": lo, "CI_hi": hi, "세해양수": y3, "양수자산": npos,
                     "자산수": len(per), "부호p": pv, "모델−롱": dif.mean(),
                     "증분lo": dl, "증분hi": dh})
    pd.DataFrame(rows).to_csv(OUT / "confirm_prereg_cell.csv", index=False)
    print("\n🔴판정: **날짜블록 CI 0배제 AND 연도 전부 양수** 둘 다여야 통과다. "
          "스윕이 아니므로 다중성 보정은 필요 없다.")
    print("🔴한계: 2022~23 은 레짐이 다르고(§5.30 이 「23년 이전 제외」라 적은 구간) 초기 분기는"
          "\n   학습이 얇다(2022-01~07 = 6개월) ⇒ **음수면 「신호 없음」과 「레짐 불일치」를 못 가른다.**")
    print(f"\n저장: {OUT/'confirm_prereg_cell.csv'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="저장된 스크린 CSV 재사용")
    ap.add_argument("--stage", required=True,
                    choices=["screen", "cross", "conj", "fdr", "oictl", "port", "size", "wf", "wfmulti", "loao", "fixed", "crossfix", "fixedml", "crosswf", "wfml", "proof", "highvol", "exit", "wall", "wallcheck", "multi", "freeze", "confirm", "quiet"])
    ap.add_argument("--minn", type=int, default=120, help="WF 선택 최소 사건 수")
    ap.add_argument("--tsel", type=float, default=2.0, help="WF 선택 초과 t 임계")
    ap.add_argument("--start", default="2025-01-01", help="WF 거래 시작 월")
    ap.add_argument("--dedup", action="store_true", help="WF 선택에서 피쳐군마다 1개만")
    ap.add_argument("--top", type=int, default=999, help="WF 선택 상위 K")
    ap.add_argument("--cap", type=float, default=1.0, help="총 노출 상한")
    ap.add_argument("--nperm", type=int, default=200, help="순환이동 귀무 반복")
    ap.add_argument("--teststart", default="2024-01-01", help="wall: 시험 구간 시작 분기")
    ap.add_argument("--rules", default="doc", choices=["doc", "mine"], help="고정 규칙 출처")
    ap.add_argument("--fam1", action="store_true", help="피쳐군마다 1개만")
    ap.add_argument("--only", default=None, help="이 피쳐 규칙만")
    ap.add_argument("--src", default="BTC", help="크로스자산 트리거 출처")
    ap.add_argument("--nullb", type=int, default=100, help="무작위 시점 귀무 반복")
    ap.add_argument("--selend", default="2024-01-01", help="선택창 끝 = 시험창 시작")
    ap.add_argument("--hz", default="4h", choices=list(HS), help="freeze: 동결할 지평")
    ap.add_argument("--gateq", type=float, default=0.95, help="freeze: 게이트 분위(상위 1-q)")
    ap.add_argument("--dirpop", type=float, default=0.80, help="freeze: 방향 학습 모집단 분위")
    ap.add_argument("--w", type=float, default=1.0, help="건당 비중(상한 대비)")
    ap.add_argument("--drop", type=float, default=0.4, help="ML: 예측 E|r| 하위 이 분위는 거래 안 함")
    ap.add_argument("--minpos", type=int, default=6, help="LOAO: 몇 개 자산에서 양수여야 하나(/8)")
    ap.add_argument("--asset", default="ETH", help="quiet: 대상 자산(단일)")
    ap.add_argument("--quiet", type=float, default=0.30, help="quiet: 「조용함」 분위(하위/상위 각각)")
    a = ap.parse_args()
    {"screen": stage_screen, "cross": stage_cross, "port": stage_port,
     "fdr": stage_fdr, "oictl": stage_oictl, "conj": stage_conj,
     "size": stage_size, "wf": stage_wf, "wfmulti": stage_wfmulti,
     "loao": stage_loao, "fixed": stage_fixed, "crossfix": stage_crossfix,
     "fixedml": stage_fixedml, "crosswf": stage_crosswf,
     "wfml": stage_wfml, "proof": stage_proof,
     "highvol": stage_highvol, "exit": stage_exit,
     "wall": stage_wall, "wallcheck": stage_wallcheck,
     "multi": stage_multi, "freeze": stage_freeze, "quiet": stage_quiet,
     "confirm": stage_confirm}[a.stage](a)
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
    # ③ 배리어 first_touch — 분기·동시터치 규약이 자명하지 않다
    ent = np.array([100.0, 100.0, 100.0, 100.0])
    hi = np.array([[100.5, 102.0, 103.0],      # 롱: 2번째 봉에서 TP(+1%)
                   [100.2, 100.3, 100.4],      # 롱: 아무것도 안 닿음 → NaN
                   [100.1,  99.0,  98.0],      # 롱: 2번째 봉에서 SL(−1%)
                   [101.5,  99.0,  99.0]])     # 롱: 1번째 봉이 TP·SL 동시 → **SL 우선**
    lo = np.array([[99.8, 100.5, 101.0],
                   [99.9,  99.9,  99.9],
                   [99.9,  98.5,  97.5],
                   [98.0,  98.5,  98.5]])
    tp = np.full(4, 0.01); sl = np.full(4, 0.01)
    ret, end = first_touch(hi, lo, ent, tp, sl, 1)
    assert abs(ret[0] - 100) < 1e-6 and end[0] == 1, (ret[0], end[0])
    assert not np.isfinite(ret[1]), ret[1]
    assert abs(ret[2] + 100) < 1e-6 and end[2] == 1, (ret[2], end[2])
    assert abs(ret[3] + 100) < 1e-6 and end[3] == 0, f"동시 터치가 SL 우선이 아니다: {ret[3]}"
    # 숏: 가격이 내리면 이익
    rs, _ = first_touch(hi[[2]], lo[[2]], ent[[2]], tp[[0]], sl[[0]], -1)
    assert abs(rs[0] - 100) < 1e-6, rs[0]
    # ④ 클러스터 SE — 같은 날 사건이 뭉치면 이항/정규 SE 보다 **커야** 한다
    rr = np.random.default_rng(7)
    dayeff = rr.normal(0, 50, 200)                      # 날마다 공통 충격
    xs, ds = [], []
    for k, de in enumerate(dayeff):
        m = 10
        xs.append(de + rr.normal(0, 5, m)); ds.append(np.full(m, k))
    xs = np.concatenate(xs); ds = np.concatenate(ds)
    naive = xs.std(ddof=1) / np.sqrt(len(xs))
    cse = cluster_se(xs, ds)
    assert cse > 2.0 * naive, f"클러스터 SE 가 안 커졌다: {cse:.2f} vs {naive:.2f}"
    indep = rr.normal(0, 5, 2000)
    di = np.arange(2000)                                 # 날마다 1건 = 독립
    assert abs(cluster_se(indep, di) - indep.std(ddof=1) / np.sqrt(2000)) < 0.02, "독립일 땐 같아야"
    print("자체점검 통과")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck(); raise SystemExit(0)
    raise SystemExit(main())
