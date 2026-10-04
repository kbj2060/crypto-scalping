"""볼륨 프로파일 «실제 트레이더 규칙» 다섯 개 검정 (2026-10-04, 사용자 «다섯 개 모두 검정해줘»).

09-30 검정(research_eth_profile_vwap_levels_20260930 · research_eth_lvn_break_20260930)은 «선에 닿으면 반응»만 봤다.
이번엔 트레이더들이 실제로 쓰는 규칙 다섯 개를 «주장한 일이 일어나는가»로 잰다. 매매 수익은 보조.

프로파일 원천(09-30 과 같다): 바이낸스 USDT-M ETHUSDT **1분봉**(data.binance.vision). 체결 테이프 아님.
  가격대별 거래량 = 분마다 대표가 tp=(h+l+c)/3 칸에 그 분 거래량 v 전부를 넣는다(1분 안 가격 분포는 무시).
  칸 = 대시보드 dashboard/market_ctx.py 규약: k = floor(tp/bw), bw = 세션 마지막 종가 × PROFILE_BIN_FRAC(0.05%).
  VA·POC = market_ctx.value_area(70%, POC 에서 큰 쪽부터), HVN/LVN = market_ctx.profile_nodes(직전 24h, 정시 갱신 -- 화면과 같음).
세션: 주 = UTC 00:00~24:00. 보조 = 미국장 13:30 UTC 고정 시작 24h(서머타임 무시 -- 겨울엔 실제 개장 14:30).
기간: 주 판정 2025-01-01 ~ 2026-09-30 · 보조 2021-01-01 ~ 2024-12-31(따로 보인다, 판정에 안 섞는다).
  2021 전체와 2026-09-15~30 은 data.binance.vision 에서 받아 tmp/volume_profile_rules_20261004/extra_1m 에 둔다.
시점 계약: 세션 d 의 프로파일은 d 가 끝난 뒤(d+1 첫 분부터)만 쓴다. 사건은 분 j 종가가 나와야 알고, 결과는 j+1 분부터 잰다.
CI: 일 블록 포아송 부트스트랩 B=1000(사건이 하루 단위면 그 날이 블록).

판정 기준은 아래 CRITERIA 에 결과를 보기 전에 고정했다(2026-10-04).
실행: python scripts/research_eth_volume_profile_rules_20261004.py [--selftest]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "dashboard"))
import market_ctx as MC                                  # noqa: E402  value_area · profile_nodes · PROFILE_BIN_FRAC
import research_eth_fp_event_response_20260930 as E      # noqa: E402  mean_ci
import research_eth_fp_pattern_lookback_20260930 as R    # noqa: E402  default_data_dir
import research_eth_profile_vwap_levels_20260930 as PV   # noqa: E402  rule80 · rule80_match(80% 규칙 09-30 정의 그대로)

OUT = ROOT / "tmp/volume_profile_rules_20261004"
EXTRA = OUT / "extra_1m"
D1 = 1440
US_OFF = 13 * 60 + 30
PERIODS = {"main": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-10-01")),
           "aux": (pd.Timestamp("2021-01-01"), pd.Timestamp("2025-01-01"))}
COST_TAKER_BP, COST_MAKER_BP = 4.0, 0.0

# ---------- 결과 보기 전에 고정한 판정 기준 ----------
CRITERIA = {
    "공통": "주 기간 주 지표가 «주 대조»를 이론 방향으로 이기고 95% CI 가 0 을 배제 → 지지. "
            "반대 방향으로 0 배제 → 근거 없음(반대). CI 가 0 포함이고 반폭 > 0.7×MDE(80% 검정력에 못 미침) → 검정력 부족(필요 표본 = n×(반폭/(0.7×MDE))²). "
            "그 밖 → 근거 없음. 보조 기간·보조 대조·변동성 통제 변형은 보고만(판정 불변).",
    "H1": dict(name="80% 규칙", metric="그날 안 반대편 VA 끝 도달률 − 짝 대조 도달률(쌍별 차)", sign=+1, mde=0.10,
               main_ctl="A: 시가가 VA 안인 같은 기간 날들에서 같은 시각 ±30분·직전 30종가 VA 안·목표까지 VA 폭 비율 ±0.1·"
                        "VA폭/전일 RV(로그) ±0.3 인 점들의 같은 목표 도달률(09-30 rule80_match 그대로 -- 변동성 짝맞춤 포함)",
               aux_ctl="B: 같은 분·같은 거리(bp, 전일 RV 비로 늘이고 줄임)의 무작위 날 도달률 · B0: 거리 그대로 · 1시간 머묾 · 미국장 세션"),
    "H2": dict(name="네이키드 POC 자석", metric="0.5% 안으로 처음 들어온 분 다음 240분 안 터치율 − 대조(변동성 5분위 층화 차)", sign=+1, mde=0.05,
               main_ctl="P2: 같은 세션 고가~저가 사이 균등 무작위 가격 5개(같은 네이키드·접근·터치 절차) · 층 = 직전 240분 RV 5분위(사건 기준)",
               aux_ctl="P1: 7·14·21·28 세션 전 POC 를 세션 종가 비율로 옮긴 위약 · 층화 없는 단순 차 · 미국장 세션"),
    "H3a": dict(name="VA 이동 방향", metric="위/아래 이동일 다음 날 수익 부호 적중률 − 같은 기간 그 부호의 무조건 비율(쌍별)", sign=+1, mde=0.05,
                main_ctl="같은 기간 모든 날의 P(다음 날 부호 = 예측 부호)",
                aux_ctl="같은 분류를 고가~저가 범위로(거래량 없이) · 부호×수익 bp · 미국장 세션",
                define="ov = |VA_d∩VA_{d−1}|/|VA_d∪VA_{d−1}|. 위 = val_d≥val_{d−1} & vah_d>vah_{d−1} & ov<0.5, 아래 대칭, 겹침 = ov≥0.5"),
    "H3b": dict(name="VA 겹침 → 다음 날 횡보", metric="log(RV_{d+1}/RV_d) 겹침일 − 이동일(위·아래) 차, RV_d/직전 30세션 중앙 5분위 층화", sign=-1, mde=0.10,
                main_ctl="이동일(위·아래)", aux_ctl="층화 없는 단순 차"),
    "H4": dict(name="프로파일 모양(P/b)", metric="x=(POC−저가)/(고가−저가) 와 다음 날 수익/RV_d 의 스피어만 ρ", sign=+1, mde=0.10,
               main_ctl="무작위 날 = ρ 0(일 단위 부트스트랩)",
               aux_ctl="종가 위치(CLV) ρ · CLV 통제 편상관 · POC 의 VA 내 위치 · 미국장 세션",
               define="해석 A(판정): P형(POC 위쪽) = 매수 수용 → 다음 날 상승 지속(ρ>0). 해석 B: P형 = 숏커버 → 다음 날 약세(ρ<0) -- "
                      "B 가 맞으면 판정은 «근거 없음(반대)»로 나오고 그대로 보고"),
    "H5": dict(name="손절 위치 HVN 너머 vs LVN 안", metric="같은 진입·같은 거리 손절의 240분 안 터치율: HVN 너머 − LVN 안(방향×RV 5분위 층화)", sign=-1, mde=0.03,
               main_ctl="LVN 안 손절", aux_ctl="중립(둘 다 아님) · 위약 HVN(7일 전 노드 비율 이동) 너머 · 털린 뒤 240분 안 진입가 복귀율",
               define="진입 = 15분 격자 모든 분 종가 · 롱·숏 둘 다 · 손절 거리 D = 0.5%·1.0% 각각 판정(둘 다 지지면 지지, 하나면 «한 거리만»). "
                      "HVN 너머 = 손절가와 진입가 사이 손절가에서 0.15% 안에 HVN. LVN 안 = 손절가 ±0.10% 안에 LVN 이고 HVN 너머 아님. "
                      "노드 = 직전 정시까지 24h 1분 프로파일(market_ctx.profile_nodes, 화면과 같음)."),
}


# ---------- 데이터 ----------
def load_1m() -> pd.DataFrame:
    fs = sorted(glob.glob(str(R.default_data_dir() / "ETHUSDT-1m-*.parquet"))) + sorted(glob.glob(str(EXTRA / "ETHUSDT-1m-*.parquet")))
    d = pd.concat([pd.read_parquet(f, columns=["t", "h", "l", "c", "v"]) for f in fs]).drop_duplicates("t").sort_values("t").set_index("t")
    t0 = d.index.min() // 86_400_000 * 86_400_000
    t1 = (d.index.max() // 86_400_000 + 1) * 86_400_000
    return d.reindex(np.arange(t0, t1, 60_000))


def period_of(ts_ms: np.ndarray) -> np.ndarray:
    tsd = pd.to_datetime(ts_ms, unit="ms"); out = np.full(len(ts_ms), "", dtype=object)
    for k, (a, b) in PERIODS.items():
        out[(tsd >= a) & (tsd < b)] = k
    return out


def session_profile(tp: np.ndarray, v: np.ndarray, last: float) -> dict | None:
    """한 세션의 1분 tp·v → VA(val·poc·vah). 칸 = 마지막 종가 × 0.05%(market_ctx 규약)."""
    ok = np.isfinite(tp) & np.isfinite(v)
    if ok.sum() < 720 or not np.isfinite(last):
        return None
    bw = last * MC.PROFILE_BIN_FRAC
    return MC.value_area(np.floor(tp[ok] / bw).astype(np.int64), v[ok], bw)


def sessions(m1: pd.DataFrame, off: int) -> dict:
    """off 분에서 시작하는 24h 세션들. 행 d 의 va* 는 세션 d **자신**의 프로파일(쓸 때 d+1 에서 shift)."""
    t = m1.index.to_numpy(np.int64)[off:]
    n = len(t) // D1 * D1
    h, l, c, v = (m1[k].to_numpy(float)[off:off + n] for k in ("h", "l", "c", "v"))
    t = t[:n]; ns = n // D1
    tp = (h + l + c) / 3
    C, H, L, TP, V = (x.reshape(ns, D1) for x in (c, h, l, tp, v))
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = np.r_[np.nan, np.diff(np.log(c))]
    last = pd.DataFrame(C.T).ffill().iloc[-1].to_numpy()
    va = np.full((ns, 3), np.nan)
    for d in range(ns):
        p = session_profile(TP[d], V[d], last[d])
        if p:
            va[d] = (p["val"], p["poc"], p["vah"])
    rv = np.sqrt(np.nansum(r1.reshape(ns, D1) ** 2, 1)) * 1e4
    rv[np.isfinite(C).sum(1) < 720] = np.nan
    return dict(t=t, h=h, l=l, c=c, C=C, H=H, L=L, ns=ns, last=last, va=va, rv=rv,
                hi=np.nanmax(H, 1), lo=np.nanmin(L, 1), per=period_of(t[::D1]), start=np.arange(ns) * D1)


def prev(x: np.ndarray) -> np.ndarray:
    """행 d 에 행 d−1 값(세션이 끝난 뒤에만 쓴다)."""
    out = np.full_like(x, np.nan, dtype=float); out[1:] = x[:-1]
    return out


# ---------- 통계 ----------
def sdiff(y, g, day, z=None, nq: int = 5, B: int = 1000, seed: int = 7) -> tuple[float, float, float]:
    """사건(g=1) − 대조(g=0) 평균 차. z 가 있으면 사건 기준 z 분위로 층화(층마다 차 → 사건 비중으로 평균). 일 블록 포아송 부트스트랩."""
    y = np.asarray(y, float); g = np.asarray(g, int); day = np.asarray(day)
    if z is None:
        b, nb = np.zeros(len(y), int), 1
    else:
        z = np.asarray(z, float)
        b, nb = np.searchsorted(np.nanquantile(z[g == 1], np.linspace(0, 1, nq + 1)[1:-1]), z), nq
    _, inv = np.unique(day, return_inverse=True); D = inv.max() + 1
    idx = inv * nb + b
    agg = lambda w: np.bincount(idx, weights=w, minlength=D * nb).reshape(D, nb)   # noqa: E731
    S1, N1, S0, N0 = agg(y * (g == 1)), agg((g == 1).astype(float)), agg(y * (g == 0)), agg((g == 0).astype(float))

    def stat(W):
        s1, n1, s0, n0 = W @ S1, W @ N1, W @ S0, W @ N0
        with np.errstate(invalid="ignore", divide="ignore"):
            d = s1 / n1 - s0 / n0
        ok = (n1 > 0) & (n0 > 0)
        p = np.where(ok, n1, 0); p = p / p.sum(-1, keepdims=True)
        return np.nansum(np.where(ok, p * d, 0), -1)

    W = np.random.default_rng(seed).poisson(1.0, (B, D)).astype(float)
    lo, hi = np.nanpercentile(stat(W), [2.5, 97.5])
    return float(stat(np.ones((1, D)))[0]), float(lo), float(hi)


def spearman_ci(x, y, B: int = 1000, seed: int = 7) -> tuple[float, float, float]:
    from scipy.stats import rankdata
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    f = lambda a, b: float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])   # noqa: E731
    rng = np.random.default_rng(seed)
    bs = [f(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f(x, y), float(lo), float(hi)


def verdict(est: float, lo: float, hi: float, sign: int, mde: float, n: int) -> str:
    hw = (hi - lo) / 2
    if (lo > 0 and sign > 0) or (hi < 0 and sign < 0):
        return "지지"
    if lo > 0 or hi < 0:
        return "근거 없음(반대 방향으로 0 배제)"
    if hw > 0.7 * mde:
        return f"검정력 부족(필요 표본 ≈ {int(np.ceil(n * (hw / (0.7 * mde)) ** 2)):,})"
    return "근거 없음"


def row(name: str, est, lo, hi, n, **kw) -> dict:
    return {"name": name, "est": est, "lo": lo, "hi": hi, "n": int(n), **kw}


# ---------- H1 80% 규칙 ----------
def h1(S: dict, run: int, tag: str) -> list[dict]:
    va = prev(S["va"][:, 0]), prev(S["va"][:, 2])                                     # 전날 VA
    val, vah = va
    opn = prev(S["last"])                                                             # 시가 = 전 세션 마지막 종가
    sig = prev(S["rv"])
    C, Hh, Ll = S["C"], S["H"], S["L"]
    ev, pool = PV.rule80(C, Hh, Ll, vah, val, opn, sig, 0, run)
    sufmin = np.fmin.accumulate(np.where(np.isfinite(Ll), Ll, np.inf)[:, ::-1], 1)[:, ::-1]
    sufmax = np.fmax.accumulate(np.where(np.isfinite(Hh), Hh, -np.inf)[:, ::-1], 1)[:, ::-1]
    out = []
    for p in PERIODS:
        evs = [e for e in ev if S["per"][e[0]] == p]
        y, cr, dd = PV.rule80_match(evs, pool, S["per"] == p)
        est, lo, hi = E.mean_ci(y - cr, dd) if len(y) else (np.nan,) * 3
        # 보조 B: 같은 분 m·같은 bp 거리(전일 RV 비로 스케일)·같은 방향, 같은 기간 다른 날 전부
        pdays = np.flatnonzero((S["per"] == p) & np.isfinite(sig) & np.isfinite(C[:, 0]))
        yb, cb, cb0, rb, db = [], [], [], [], []
        for d, m, T, *_ , ye in evs:
            if m + 1 >= D1 or not np.isfinite(sig[d]):
                continue
            tgt = val[d] if T == 0 else vah[d]; dist = abs(tgt / C[d, m] - 1)
            oth = pdays[pdays != d]
            cm = C[oth, m]
            for scale, acc in ((sig[oth] / sig[d], cb), (1.0, cb0)):
                lvl = cm * (1 - dist * scale) if T == 0 else cm * (1 + dist * scale)
                hit = sufmin[oth, m + 1] <= lvl if T == 0 else sufmax[oth, m + 1] >= lvl
                acc.append(float(np.nanmean(np.where(np.isfinite(cm), hit, np.nan))))
            yb.append(float(ye)); db.append(d)
            # 보조 매매: c[m] 시장가(4bp) 진입, 반대편 VA 끝 지정가(0bp) 또는 세션 마지막 종가 시장가(4bp)
            sgn = -1 if T == 0 else 1
            ex, cost = (tgt, COST_TAKER_BP + COST_MAKER_BP) if ye else (S["last"][d], 2 * COST_TAKER_BP)
            rb.append(sgn * (ex / C[d, m] - 1) * 1e4 - cost)
        yb, cb, cb0, db = map(np.array, (yb, cb, cb0, db))
        b = E.mean_ci(yb - cb, db) if len(yb) else (np.nan,) * 3
        b0 = E.mean_ci(yb - cb0, db) if len(yb) else (np.nan,) * 3
        tr = E.mean_ci(np.array(rb), db) if len(rb) else (np.nan,) * 3
        out.append(row(f"H1 {tag} run{run} {p}", est, lo, hi, len(y), period=p, n_raw=len(evs), rate=float(np.mean(y)) if len(y) else None,
                       rate_ctl=float(np.mean(cr)) if len(cr) else None,
                       ctlB_vol=dict(rate=float(cb.mean()) if len(cb) else None, diff=b), ctlB0_raw=dict(rate=float(cb0.mean()) if len(cb0) else None, diff=b0),
                       trade_bp_net=tr))
    return out


# ---------- H2 네이키드 POC ----------
def naked_events(level: np.ndarray, ns_start: np.ndarray, h, l, c, near: float = 0.005, age: int = 10, T: int = 240):
    """세션 d 의 레벨 level[d] 를 세션 d 끝(=start[d+1])부터 age 세션 동안 추적. 아직 안 닿았고(분 j 까지 [l,h] 에 안 들어옴)
    거리가 near 를 넘다가 처음 near 안으로 들어온 분 j = 사건(하나만). 결과 = j+1..j+T 안 터치(1/0). 반환 (j, hit, 방향(+1 위로))."""
    n = len(c); J, Y, S, LL = [], [], [], []
    for d in range(len(level) - 1):
        L = level[d]
        if not np.isfinite(L):
            continue
        a = ns_start[d + 1]; b = min(n, a + age * D1)
        hh, ll, cc = h[a:b], l[a:b], c[a:b]
        with np.errstate(invalid="ignore"):
            tch = (ll <= L) & (hh >= L)
            dist = np.abs(cc / L - 1)
        tau = int(np.argmax(tch)) if tch.any() else b - a          # 첫 터치(구간 안 상대 위치)
        cp = np.r_[np.abs(c[a - 1] / L - 1) if a >= 1 else np.nan, dist[:-1]]
        with np.errstate(invalid="ignore"):
            cand = np.flatnonzero((dist <= near) & (cp > near))
        cand = cand[cand < tau]
        if not len(cand):
            continue
        j = int(cand[0])
        if a + j + T >= n:
            continue
        J.append(a + j); Y.append(1.0 if tau - j <= T and tau < b - a else 0.0); S.append(1 if L > cc[j] else -1); LL.append(L)
    return np.array(J, np.int64), np.array(Y), np.array(S), np.array(LL)


def h2(S: dict, tag: str) -> list[dict]:
    h, l, c = S["h"], S["l"], S["c"]; n = len(c)
    poc = S["va"][:, 1]; st = S["start"]
    with np.errstate(invalid="ignore"):
        r1 = np.r_[np.nan, np.diff(np.log(c))]
    G = np.r_[0.0, np.cumsum(np.nan_to_num(r1) ** 2)]
    rv240 = np.full(n, np.nan); rv240[240:] = np.sqrt(G[241:] - G[1:-240]) * 1e4
    rng = np.random.default_rng(11)
    sets = {"real": [naked_events(poc, st, h, l, c)]}
    sets["P2"] = [naked_events(np.where(np.isfinite(poc), S["lo"] + rng.random(S["ns"]) * (S["hi"] - S["lo"]), np.nan), st, h, l, c) for _ in range(5)]
    sets["P1"] = []
    for k in (7, 14, 21, 28):
        fk = np.full(S["ns"], np.nan); fk[k:] = poc[:-k] * S["last"][k:] / S["last"][:-k]
        sets["P1"].append(naked_events(fk, st, h, l, c))
    cat = {g: tuple(np.concatenate(x) for x in zip(*v)) for g, v in sets.items()}
    per = period_of(S["t"]); day = S["t"] // 86_400_000
    out = []
    for p in PERIODS:
        for ctl in ("P2", "P1"):
            J = np.r_[cat["real"][0], cat[ctl][0]]; Y = np.r_[cat["real"][1], cat[ctl][1]]
            g = np.r_[np.ones(len(cat["real"][0]), int), np.zeros(len(cat[ctl][0]), int)]
            m = per[J] == p
            st_ = sdiff(Y[m], g[m], day[J[m]], rv240[J[m]])
            raw = sdiff(Y[m], g[m], day[J[m]])
            out.append(row(f"H2 {tag} {p} vs {ctl}", *st_, int((g[m] == 1).sum()), period=p, ctl=ctl, n_ctl=int((g[m] == 0).sum()),
                           rate=float(Y[m][g[m] == 1].mean()), rate_ctl=float(Y[m][g[m] == 0].mean()), unstratified=raw))
        # 보조 매매: c[j] 시장가(4bp)로 POC 쪽 진입, POC 지정가(0bp) 또는 240분 뒤 종가 시장가(4bp)
        J, Y, Sg, Lv = (x[per[cat["real"][0]] == p] for x in cat["real"])
        ex = np.where(Y > 0, Lv, c[J + 240]); cost = np.where(Y > 0, COST_TAKER_BP, 2 * COST_TAKER_BP)
        out[-2]["trade_bp_net"] = E.mean_ci(Sg * (ex / c[J] - 1) * 1e4 - cost, day[J])
    return out


# ---------- H3 · H4 ----------
def h3h4(S: dict, tag: str) -> list[dict]:
    val, poc, vah = S["va"].T
    nxt = lambda x: np.r_[x[1:], np.nan]                                              # noqa: E731
    with np.errstate(invalid="ignore", divide="ignore"):
        ret1 = (nxt(S["last"]) / S["last"] - 1) * 1e4                                   # 세션 d 끝 → d+1 끝
        pv, ph = prev(val), prev(vah)
        inter = np.clip(np.minimum(vah, ph) - np.maximum(val, pv), 0, None)
        ov = inter / (np.maximum(vah, ph) - np.minimum(val, pv))
        up = (val >= pv) & (vah > ph) & (ov < 0.5); dn = (val <= pv) & (vah < ph) & (ov < 0.5); bal = ov >= 0.5
        # 거래량 없는 대조: 같은 분류를 고가~저가로
        rh, rl = S["hi"], S["lo"]; prh, prl = prev(rh), prev(rl)
        ovr = np.clip(np.minimum(rh, prh) - np.maximum(rl, prl), 0, None) / (np.maximum(rh, prh) - np.minimum(rl, prl))
        upr = (rl >= prl) & (rh > prh) & (ovr < 0.5); dnr = (rl <= prl) & (rh < prh) & (ovr < 0.5)
        rvr = np.log(nxt(S["rv"]) / S["rv"])
        rvlvl = S["rv"] / pd.Series(S["rv"]).rolling(30, min_periods=20).median().shift(1).to_numpy()
        x = (poc - rl) / (rh - rl); xva = (poc - val) / (vah - val)
        clv = (S["last"] - rl) / (rh - rl)
        yn = ret1 / S["rv"]
    day = np.arange(S["ns"]); out = []
    for p in PERIODS:
        P = (S["per"] == p) & np.isfinite(ret1) & (ret1 != 0)
        for nm, U, Dn in (("VA", up, dn), ("범위(대조)", upr, dnr)):
            sgn = np.where(U, 1, np.where(Dn, -1, 0)); m = P & (sgn != 0)
            base = {s: float((np.sign(ret1[P]) == s).mean()) for s in (1, -1)}
            hit = (np.sign(ret1[m]) == sgn[m]).astype(float)
            b = np.array([base[s] for s in sgn[m]])
            est = E.mean_ci(hit - b, day[m])
            bp = E.mean_ci(sgn[m] * ret1[m], day[m])
            bpn = E.mean_ci(sgn[m] * ret1[m] - 2 * COST_TAKER_BP, day[m])
            out.append(row(f"H3a {tag} {p} {nm}", *est, int(m.sum()), period=p, hit=float(hit.mean()), base=float(b.mean()),
                           n_up=int((m & (sgn > 0)).sum()), n_dn=int((m & (sgn < 0)).sum()), signed_bp=bp, signed_bp_net_taker=bpn))
        m = P & np.isfinite(rvr) & (bal | up | dn)
        g = bal[m].astype(int)
        out.append(row(f"H3b {tag} {p}", *sdiff(rvr[m], g, day[m], rvlvl[m]), int(g.sum()), period=p, n_ctl=int((g == 0).sum()),
                       mean_bal=float(rvr[m][g == 1].mean()), mean_move=float(rvr[m][g == 0].mean()),
                       unstratified=sdiff(rvr[m], g, day[m]), abs_ret_bal=float(np.abs(ret1[m][g == 1]).mean()),
                       abs_ret_move=float(np.abs(ret1[m][g == 0]).mean())))
        m = P & np.isfinite(x) & np.isfinite(yn)
        rho = spearman_ci(x[m], yn[m])
        from scipy.stats import rankdata
        rx, rc, ry = (rankdata(a[m]) for a in (x, clv, yn))
        res = lambda a: a - np.polyval(np.polyfit(rc, a, 1), rc)                          # noqa: E731
        q = np.quantile(x[m], [1 / 3, 2 / 3])
        top, bot = x[m] >= q[1], x[m] <= q[0]
        out.append(row(f"H4 {tag} {p}", *rho, int(m.sum()), period=p, rho_raw_ret=spearman_ci(x[m], ret1[m]),
                       rho_clv=spearman_ci(clv[m], yn[m]), rho_x_vs_clv=spearman_ci(x[m], clv[m]),
                       partial_rho_given_clv=spearman_ci(res(rx), res(ry)), rho_poc_in_va=spearman_ci(xva[m], yn[m]),
                       bp_top_tercile=float(ret1[m][top].mean()), bp_bottom_tercile=float(ret1[m][bot].mean()),
                       top_minus_bottom=sdiff(ret1[m][top | bot], top[top | bot].astype(int), day[m][top | bot])))
    return out


# ---------- H5 손절 위치 ----------
def hourly_nodes(h, l, c, v) -> tuple[np.ndarray, np.ndarray]:
    """정시 r 마다 [r−24h, r) 1분 tp 프로파일의 HVN·LVN(market_ctx.profile_nodes, 칸 = 직전 종가 × 0.05%). 행 r 은 분 [60r, 60r+60) 에 쓴다."""
    tp = (h + l + c) / 3; nh = len(c) // 60
    HV = np.full((nh, MC.NODE_MAX), np.nan); LV = np.full((nh, MC.NODE_MAX), np.nan)
    for r in range(24, nh):
        a, b = (r - 24) * 60, r * 60
        p, w = tp[a:b], v[a:b]; ok = np.isfinite(p) & np.isfinite(w)
        lastc = c[b - 1]
        if ok.sum() < 720 or not np.isfinite(lastc):
            continue
        bw = lastc * MC.PROFILE_BIN_FRAC
        nd = MC.profile_nodes(np.floor(p[ok] / bw).astype(np.int64), w[ok], bw)
        for M, key in ((HV, "hvn"), (LV, "lvn")):
            xs = [x["px"] for x in nd[key]]; M[r, :len(xs)] = xs
    return HV, LV


def classify_stop(Sp: np.ndarray, dirn: np.ndarray, HVr: np.ndarray, LVr: np.ndarray, c: np.ndarray,
                  d_h: float = 0.0015, d_l: float = 0.0010) -> np.ndarray:
    """손절가 Sp(롱 dirn=+1 이면 진입가 아래) → 1 HVN 너머 · 0 LVN 안 · −1 중립. HVN 너머 = HVN 이 손절가와 진입가 사이, 손절가에서 d_h 안."""
    S_, cc = Sp[:, None], c[:, None]
    with np.errstate(invalid="ignore"):
        hb = np.where(dirn[:, None] > 0, (HVr > S_) & (HVr <= S_ + d_h * cc), (HVr < S_) & (HVr >= S_ - d_h * cc)).any(1)
        li = (np.abs(LVr - S_) <= d_l * cc).any(1)
    return np.where(hb, 1, np.where(li, 0, -1))


def h5(m1: pd.DataFrame) -> list[dict]:
    t = m1.index.to_numpy(np.int64); h, l, c, v = (m1[k].to_numpy(float) for k in ("h", "l", "c", "v")); n = len(c)
    HV, LV = hourly_nodes(h, l, c, v)
    print("H5 노드 준비 끝", flush=True)
    P_hr = np.r_[np.nan, c[np.arange(1, len(HV)) * 60 - 1]]
    fHV = np.full_like(HV, np.nan); k = 24 * 7; fHV[k:] = HV[:-k] * (P_hr[k:] / P_hr[:-k])[:, None]
    with np.errstate(invalid="ignore"):
        r1 = np.r_[np.nan, np.diff(np.log(c))]
    G = np.r_[0.0, np.cumsum(np.nan_to_num(r1) ** 2)]
    rv240 = np.full(n, np.nan); rv240[240:] = np.sqrt(G[241:] - G[1:-240]) * 1e4
    T = 240
    fmin = pd.Series(l).rolling(T, min_periods=200).min().shift(-T).to_numpy()            # 분 j+1..j+240
    fmax = pd.Series(h).rolling(T, min_periods=200).max().shift(-T).to_numpy()
    J = np.arange(0, n - 2 * T - 2, 15)
    J = J[np.isfinite(c[J]) & np.isfinite(rv240[J]) & np.isfinite(HV[J // 60, 0]) & np.isfinite(fmin[J])]
    per = period_of(t[J]); day = t[J] // 86_400_000
    out = []
    for D in (0.005, 0.010):
        for dirn_v in (1, -1):
            dirn = np.full(len(J), dirn_v)
            Sp = c[J] * (1 - dirn * D)
            cls = classify_stop(Sp, dirn, HV[J // 60], LV[J // 60], c[J])
            fcls = classify_stop(Sp, dirn, fHV[J // 60], np.full_like(LV[J // 60], np.nan), c[J])
            hit = np.where(dirn > 0, fmin[J] <= Sp, fmax[J] >= Sp).astype(float)
            if dirn_v == 1:
                acc = dict(J=[J], dir=[dirn], cls=[cls], fcls=[fcls], hit=[hit], Sp=[Sp])
            else:
                for kk, vv in (("J", J), ("dir", dirn), ("cls", cls), ("fcls", fcls), ("hit", hit), ("Sp", Sp)):
                    acc[kk].append(vv)
        A = {kk: np.concatenate(vv) for kk, vv in acc.items()}
        JJ, dd = A["J"], A["dir"]
        # 털린 뒤 진입가 복귀: 첫 터치 분 s 뒤 240분 안에 진입가 c[j] 재도달
        rec = np.full(len(JJ), np.nan)
        for i in np.flatnonzero(A["hit"] > 0):
            j = JJ[i]; seg = l[j + 1:j + 1 + T] if dd[i] > 0 else h[j + 1:j + 1 + T]
            with np.errstate(invalid="ignore"):
                s = j + 1 + int(np.argmax(seg <= A["Sp"][i] if dd[i] > 0 else seg >= A["Sp"][i]))
                back = h[s + 1:s + 1 + T] >= c[j] if dd[i] > 0 else l[s + 1:s + 1 + T] <= c[j]
            rec[i] = float(back.any())
        perA = np.tile(per, 2); dayA = np.tile(day, 2)
        zq = np.tile(rv240[J], 2)
        for p in PERIODS:
            m = (perA == p) & (A["cls"] >= 0)
            g = A["cls"][m]
            strat = (np.searchsorted(np.quantile(zq[m][g == 1], [.2, .4, .6, .8]), zq[m]) * 2 + (dd[m] > 0))
            est = _strat(A["hit"][m], g, dayA[m], strat)
            mr = m & np.isfinite(rec)
            rec_d = _strat(rec[mr], A["cls"][mr], dayA[mr], (np.searchsorted(np.quantile(zq[m][g == 1], [.2, .4, .6, .8]), zq[mr]) * 2 + (dd[mr] > 0)))
            mn = (perA == p) & ((A["cls"] == 1) | (A["cls"] == -1))
            neu = _strat(A["hit"][mn], (A["cls"][mn] == 1).astype(int), dayA[mn],
                             np.searchsorted(np.quantile(zq[mn], [.2, .4, .6, .8]), zq[mn]) * 2 + (dd[mn] > 0))
            yy = np.r_[A["hit"][(perA == p) & (A["cls"] == 1)], A["hit"][(perA == p) & (A["fcls"] == 1)]]
            gg = np.r_[np.ones(((perA == p) & (A["cls"] == 1)).sum(), int), np.zeros(((perA == p) & (A["fcls"] == 1)).sum(), int)]
            zz = np.r_[zq[(perA == p) & (A["cls"] == 1)], zq[(perA == p) & (A["fcls"] == 1)]]
            dy = np.r_[dayA[(perA == p) & (A["cls"] == 1)], dayA[(perA == p) & (A["fcls"] == 1)]]
            fake = sdiff(yy, gg, dy, zz)
            out.append(row(f"H5 D={D:.3f} {p}", *est, int((g == 1).sum()), period=p, D=D, n_lvn=int((g == 0).sum()),
                           n_neutral=int(((perA == p) & (A["cls"] == -1)).sum()),
                           rate_hvn=float(A["hit"][m][g == 1].mean()), rate_lvn=float(A["hit"][m][g == 0].mean()),
                           rate_neutral=float(A["hit"][(perA == p) & (A["cls"] == -1)].mean()),
                           unstratified=sdiff(A["hit"][m], g, dayA[m]), hvn_minus_neutral=neu, hvn_real_minus_fake=fake,
                           recover_hvn=float(np.nanmean(rec[mr & (A["cls"] == 1)])), recover_lvn=float(np.nanmean(rec[mr & (A["cls"] == 0)])),
                           recover_diff=rec_d, n_stopped_hvn=int((mr & (A["cls"] == 1)).sum()), n_stopped_lvn=int((mr & (A["cls"] == 0)).sum())))
    return out


def _strat(y, g, day, cat, B: int = 1000, seed: int = 7) -> tuple[float, float, float]:
    """sdiff 의 범주 층(정수 cat) 판: 층마다 차 → 사건 비중 평균."""
    y = np.asarray(y, float); g = np.asarray(g, int)
    _, inv = np.unique(day, return_inverse=True); D = inv.max() + 1; nb = int(cat.max()) + 1
    idx = inv * nb + cat
    agg = lambda w: np.bincount(idx, weights=w, minlength=D * nb).reshape(D, nb)   # noqa: E731
    S1, N1, S0, N0 = agg(y * (g == 1)), agg((g == 1).astype(float)), agg(y * (g == 0)), agg((g == 0).astype(float))

    def stat(W):
        s1, n1, s0, n0 = W @ S1, W @ N1, W @ S0, W @ N0
        with np.errstate(invalid="ignore", divide="ignore"):
            d = s1 / n1 - s0 / n0
        ok = (n1 > 0) & (n0 > 0)
        p = np.where(ok, n1, 0); p = p / p.sum(-1, keepdims=True)
        return np.nansum(np.where(ok, p * d, 0), -1)

    W = np.random.default_rng(seed).poisson(1.0, (B, D)).astype(float)
    lo, hi = np.nanpercentile(stat(W), [2.5, 97.5])
    return float(stat(np.ones((1, D)))[0]), float(lo), float(hi)


# ---------- 실행 ----------
def judge(rows: list[dict]) -> list[dict]:
    """주 행에 판정을 붙인다(주 기간 · CRITERIA 의 부호·MDE)."""
    for r in rows:
        key = r["name"].split()[0]
        cr = CRITERIA.get(key)
        if cr and np.isfinite(r["est"]):
            r["verdict"] = verdict(r["est"], r["lo"], r["hi"], cr["sign"], cr["mde"], r["n"])
    return rows


def main() -> None:
    m1 = load_1m()
    print(f"1분봉 {len(m1):,} ({pd.to_datetime(m1.index[0], unit='ms')} ~ {pd.to_datetime(m1.index[-1], unit='ms')}) · 결측 {m1['c'].isna().mean():.4%}", flush=True)
    rows = []
    for tag, off in (("UTC", 0), ("US", US_OFF)):
        S = sessions(m1, off)
        print(f"세션 {tag}: {S['ns']} · VA 있음 {np.isfinite(S['va'][:, 0]).sum()}", flush=True)
        for run in ((30, 60) if tag == "UTC" else (30,)):
            rows += h1(S, run, tag)
        rows += h2(S, tag)
        rows += h3h4(S, tag)
        print(f"{tag} H1~H4 끝", flush=True)
    rows += h5(m1)
    rows = judge(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({"criteria": CRITERIA, "rows": rows}, ensure_ascii=False, indent=1, default=float))
    lines = ["# 볼륨 프로파일 규칙 5종 검정 (2026-10-04)", "", "주 판정 = 주 기간(2025-01~2026-09) · UTC 세션 · 주 지표 행. 나머지는 보조.", "",
             "| 행 | n | 추정 | 95% CI | 판정 | 비고 |", "|---|---|---|---|---|---|"]
    for r in rows:
        extra = {k: r[k] for k in ("rate", "rate_ctl", "hit", "base", "rate_hvn", "rate_lvn", "rate_neutral") if k in r and r[k] is not None}
        lines.append(f"| {r['name']} | {r['n']} | {r['est']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r.get('verdict', '')} | "
                     + ", ".join(f"{k}={v:.3f}" for k, v in extra.items()) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def selftest() -> None:
    # 1) VA = market_ctx 그대로: 칸 = floor(tp/bw), 70% 를 POC 에서 큰 쪽부터
    tp = np.r_[np.full(50, 100.02), np.full(30, 100.07), np.full(20, 99.97)]; v = np.ones(100)
    va = session_profile(np.r_[tp, np.full(700, 100.02)], np.r_[v, np.ones(700)], 100.0)
    assert abs(va["poc"] - 100.025) < 1e-9 and abs(va["val"] - 100.0) < 1e-9 and abs(va["vah"] - 100.05) < 1e-9, va   # POC 칸 하나가 70% 를 넘는다
    tp2 = np.r_[np.full(250, 100.02), np.full(300, 100.07), np.full(250, 99.97)]                                    # 거래량 250:300:250(+위 100.12 에 1)
    va = session_profile(np.r_[tp2, 100.12], np.r_[np.ones(800), 1.0], 100.0)                                        # 70% = 561 → POC(300) + 아래(250)=550 < 561 → 다시 큰 쪽(아래 250 > 위 1)
    assert abs(va["poc"] - 100.075) < 1e-9 and abs(va["val"] - 99.95) < 1e-9 and abs(va["vah"] - 100.1) < 1e-9, va
    # 2) 시점: 세션 d 의 VA 는 d+1 행에서만 보인다(prev). 둘째 날에 120 거대 거래량을 넣어도 둘째 날 행에서 쓰는 VA 는 첫날 것
    t0 = 1_735_689_600_000
    idx = t0 + np.arange(3 * D1) * 60_000
    px = 100 + 0.3 * np.sin(np.arange(3 * D1) / 40)
    m1 = pd.DataFrame({"h": px + 0.01, "l": px - 0.01, "c": px, "v": 1.0}, index=idx)
    m1.iloc[D1 + 100:D1 + 900, :3] = 120.0; m1.iloc[D1 + 100:D1 + 900, 3] = 1e6
    S = sessions(m1, 0)
    used = prev(S["va"][:, 2])
    assert np.isnan(used[0]) and used[1] < 101 and used[2] > 119, used
    # 3) 80% 규칙 진입: 시가 102 > VAH 101, 5분부터 VA 안 → 30번째 VA 안 종가 = 분 34 가 확인 분. 결과는 분 35 부터
    C = np.full((1, D1), 100.0); Hh = C + 0.05; Ll = C - 0.05
    C[0, :5] = 102.0; Hh[0, :5] = 102.05; Ll[0, :5] = 101.95
    Ll[0, 34] = 98.0                                       # 확인 분 자신의 저가는 결과에 안 센다
    ev, _ = PV.rule80(C, Hh, Ll, np.array([101.0]), np.array([99.0]), np.array([102.0]), np.array([100.0]))
    assert len(ev) == 1 and ev[0][1] == 34 and ev[0][2] == 0 and not ev[0][5], ev
    Ll[0, 35] = 98.0
    assert PV.rule80(C, Hh, Ll, np.array([101.0]), np.array([99.0]), np.array([102.0]), np.array([100.0]))[0][0][5]
    # 4) 손절 분류: 롱 진입 100, 손절 99.5 · HVN 99.6(손절 위 0.1%) → 너머 · LVN 99.45 → LVN 안 · 둘 다 없으면 중립
    cls = classify_stop(np.array([99.5, 99.5, 99.5]), np.array([1, 1, 1]),
                        np.array([[99.6], [np.nan], [99.4]]), np.array([[np.nan], [99.45], [np.nan]]), np.array([100.0] * 3))
    assert list(cls) == [1, 0, -1], cls
    # 5) 네이키드 POC: 레벨 101, 다음 세션 c 100 → 100.6(0.4% 안) 첫 진입 분이 사건, 그 뒤 터치하면 1
    n = 3 * D1; c = np.full(n, 99.0); h = c + 0.01; l = c - 0.01
    c[D1 + 10:] = 100.6; h[D1 + 10:] = 100.61; l[D1 + 10:] = 100.59; h[D1 + 50] = 101.2
    J, Y, Sg, _ = naked_events(np.array([101.0, np.nan, np.nan]), np.arange(3) * D1, h, l, c)
    assert list(J) == [D1 + 10] and Y[0] == 1 and Sg[0] == 1, (J, Y)
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else main()
