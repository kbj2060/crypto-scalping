#!/usr/bin/env python3
"""증거신호 **앵커 기반 학습 라벨 데이터셋** (2026-09-07).

사용자: *"앵커 이후 지속일지 되돌림일지는 다음 딥러닝 모델을 만들어서 진행할거야. 우선 그 전에
잘만든 앵커를 기준으로 학습라벨 데이터를 만들려고 지금 이 작업을 하고 있는거야."*

앵커 확정 근거는 `docs/experiments/eth_anchor_combination_screen_20260907.md` 부록 B~E.
  확정 셀  bottom/top `any3 / Wc3` -- 8종 중 **3종 이상이 3봉(15분) 안에 발동**, 마지막이 그 봉,
           GAP 12봉 중복제거. 하루 2.7~2.8회. 변형(any3/Wc1 · any4/Wc12 · any2/Wc3)과
           현행 기준선(first_fire_union)을 **같은 표에 담아** ablation 가능하게 한다.

## 이 데이터셋이 지키는 계약
1. **known_ts = 앵커 봉**. 8종 raw 발동은 그 봉 종가에 전부 확정된다(마감봉만 사용).
   따라서 **라벨 시작은 t+1 봉**이고, 모든 라벨을 `open[t+1]` 기준으로도 만든다(`_e1`).
   `close[t]` 기준(`_c0`)은 부록 A~E와의 대조용으로만 병기한다 -- 라이브에서 그 가격엔 못 들어간다.
2. **라벨 패밀리**를 준다. 방향축은 이 저장소에서 10형태가 기각됐으므로(09-06 부록 A),
   한 라벨에 걸지 않고 경로/구조/시차/크기/경제를 모두 담아 모델이 무엇을 배우는지 분해 가능하게 한다.
3. **게이트**: `scripts/gate_eth_entry_layers_20260903.py --pipeline <이 디렉토리>` 의
   L4(known_ts 계약)·L1(발동 인과성)을 통과해야 한다. L2/L2P/L3는 피쳐·라이브 채점기가 생긴 뒤.

## 라벨 (전부 앵커 측면 기준. bottom: 페이드=롱/지속=숏, top: 페이드=숏/지속=롱)
  경로   up_H, dn_H, mfe_fade_H, mfe_cont_H, r1_H=mfe_cont/(mfe_cont+mfe_fade), y_cont_H=r1>0.5
  결단력 f1_H = |up-dn|/(up+dn)          <- 부록 E: 앵커는 이걸 못 움직인다(대조군으로 유지)
  크기   size_H = (up+dn)/atr_pct        <- 부록 E: 이건 움직인다(보조 과제/다중과제용)
  구조   lead_bars_W{24,48} = 측면 극점까지 봉 수(48봉 검열) · ext_rebound_pct_W{24,48} = 그 극점의 반대이탈%
         ⭐이 둘이 원시값이고, y_ext(D,g) = (lead <= D) & (reb >= g) 로 **학습 시점에 D·g 자유 선택**.
         y_ext_W{W}_D{D}_g{g} 고정 컬럼은 편의용 사본(부록 B/C 확정 셀 D=1~2).
  경제   net_fade_bp / net_cont_bp / margin_bp / y_econ_cont
         수정된 트레일링(`sim_exit(infeasible='exit')` 규약, 게이트 `trail(anchor='entry')`와 파리티 검증)
         셀 5.0/1.5/0.1 ATR · 200봉 · 왕복 10bp. ⚠️출구 구조 의존 라벨이므로 1차 목적함수로 쓰지 말 것.

## split (CLAUDE.md Fresh-Forward 기본값)
  TRAIN 2024-01-01~2025-08-31 · VAL 2025-09-01~2025-12-31 · OOS 2026-01-01~2026-03-31
  HOLDOUT 2026-04-01~ (⚠️**이미 소진** -- 다른 실험에 노출됨. 신선한 검정으로 쓰지 말 것)
데이터 상한은 펀딩 커버리지(2026-07-31)와 BTC klines(2026-08-20) 중 이른 쪽.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

OUT = ROOT / "tmp/eth_anchor_label_dataset_20260907"
ETH_KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_KL = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
FUNDING = ROOT / "data/TOTAL_ETHUSDT_fundingRate_2025_2026.csv"
FUNDING_Z_MIN_PERIODS = 30

SIGNALS = ["taker_delta_z_climax", "short_term_return_z", "liquidity_sweep", "orthogonal_combo",
           "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]
ABBR = {"taker_delta_z_climax": "taker", "short_term_return_z": "strz", "liquidity_sweep": "sweep",
        "orthogonal_combo": "orth", "smt_divergence": "smt", "fib_extension_exhaustion": "fib",
        "demarker_extreme": "dem", "kalman_deviation_meanrev": "kal"}
GAP_BARS = 12
ANCHOR_SPECS = [("any3", 3), ("any3", 1), ("any4", 12), ("any2", 3), ("first_fire", None)]
PRIMARY = ("any3", 3)
HS = (12, 24, 48, 96)
EXT_CELLS = [(24, 2, 0.0), (24, 2, 1.0), (48, 1, 0.0), (48, 1, 1.0), (48, 2, 1.0), (48, 3, 1.0)]
LEAD_WS, LEAD_MAX = (24, 48), 48   # 시차는 48봉까지 검열 없이 기록 -> D 를 학습 시점에 자유 선택
EXIT = dict(sl=5.0, arm=1.5, trail=0.1)
ECON_H, COST_BP = 200, 10.0
SPLITS = [("TRAIN", "2024-01-01", "2025-08-31 23:59:59"), ("VAL", "2025-09-01", "2025-12-31 23:59:59"),
          ("OOS", "2026-01-01", "2026-03-31 23:59:59"), ("HOLDOUT_SPENT", "2026-04-01", "2099-01-01")]


# --------------------------------------------------------------------------- 입력

_CACHE: dict = {}


def _load_kl(path: Path) -> pd.DataFrame:
    key = ("kl", str(path))
    if key not in _CACHE:                      # 게이트 L1 은 build_fires 를 수백 번 부른다
        df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"],
                         parse_dates=["timestamp"])
        _CACHE[key] = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return _CACHE[key]


def _load_funding() -> pd.DataFrame:
    """research_eth_funding_crossasset_combo_signal_20260825.load_funding_z 원문 축자 복제
    (그 모듈의 임포트 사슬이 torch를 끌어오는데 로컬에 없다. 함수 자체는 순수 pandas)."""
    if "funding" in _CACHE:
        return _CACHE["funding"]
    f = pd.read_csv(FUNDING, parse_dates=["calc_time"]).sort_values("calc_time").reset_index(drop=True)
    mean = f["last_funding_rate"].rolling(90, min_periods=FUNDING_Z_MIN_PERIODS).mean()
    std = f["last_funding_rate"].rolling(90, min_periods=FUNDING_Z_MIN_PERIODS).std()
    f["funding_z"] = (f["last_funding_rate"] - mean) / std.replace(0.0, np.nan)
    _CACHE["funding"] = f[["calc_time", "funding_z"]]
    return _CACHE["funding"]


# --------------------------------------------------------------------------- 앵커

def _within(fire: np.ndarray, W: int) -> np.ndarray:
    if W == 1:
        return fire
    return np.stack([pd.Series(fire[:, j]).rolling(W, min_periods=1).max().to_numpy() > 0
                     for j in range(fire.shape[1])], axis=1)


def _dedup(idx: np.ndarray, gap: int = GAP_BARS) -> np.ndarray:
    keep, last = [], -10 ** 9
    for i in idx:
        if i - last > gap:
            keep.append(int(i))
        last = i
    return np.array(keep, dtype=int)


def anchor_index(fire: np.ndarray, name: str, wc) -> np.ndarray:
    """뒤만 보는 앵커. name='anyK' = K종 이상이 wc봉 안에 발동(마지막이 그 봉) · 'first_fire' = 신호별 첫발동 합집합."""
    if name == "first_fire":
        out = set()
        for j in range(fire.shape[1]):
            for i in _dedup(np.flatnonzero(fire[:, j])):
                out.add(int(i))
        return np.array(sorted(out), dtype=int)
    k = int(name[3:])
    w = _within(fire, wc)
    return _dedup(np.flatnonzero((w.sum(axis=1) >= k) & fire.any(axis=1)))


def build_fires(kl: pd.DataFrame) -> pd.DataFrame:
    """게이트 L1 계약: fn(kl) -> DataFrame[timestamp, ..., known_ts].

    kl 이 잘린 채 들어와도 **그 시점까지의 정보만** 쓴다 -- BTC/펀딩도 같은 상한으로 자른다.
    known_ts = 앵커 봉 자신: 8종 raw 발동이 그 봉 종가에 전부 확정된다(마감봉만 사용).

    ⚠️게이트는 `kl.iloc[cut-warm:cut].reset_index(drop=True)` 로 **창을 잘라** 부르고,
    반환 프레임의 known_ts 를 뺀 **모든 컬럼**을 발동 식별키로 쓴다. 따라서 행 인덱스처럼
    **절단에 따라 값이 변하는 컬럼을 반환하면 안 된다**(2026-09-07 L1 첫 실행이 bar_idx 때문에
    118/120 missing 으로 FAIL). bar_idx 는 데이터셋 빌더가 timestamp 로 조인해 붙인다.
    """
    kl = kl.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    tmax = kl["timestamp"].max()
    btc = _load_kl(BTC_KL)
    btc = btc[btc["timestamp"] <= tmax]
    fund = _load_funding()
    fund = fund[fund["calc_time"] <= tmax]
    sig = compute_signals(kl, btc_df=btc, funding_df=fund)
    ts = pd.to_datetime(sig["timestamp"].to_numpy())
    rows = []
    for side in ("bottom", "top"):
        fire = np.stack([sig[f"{side}_{s}"].fillna(False).to_numpy(bool) for s in SIGNALS], axis=1)
        for name, wc in ANCHOR_SPECS:
            idx = anchor_index(fire, name, wc)
            if len(idx) == 0:
                continue
            tag = name if name == "first_fire" else f"{name}/Wc{wc}"
            names = ["+".join(ABBR[SIGNALS[j]] for j in np.flatnonzero(fire[i])) for i in idx]
            rows.append(pd.DataFrame({"timestamp": ts[idx], "side": side, "anchor": tag,
                                      "n_signals": fire[idx].sum(axis=1), "signals": names,
                                      "known_ts": ts[idx]}))
    out = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "side", "anchor"])
    out["signal"] = out["side"] + ":" + out["anchor"]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- 라벨

def _fwd(x: np.ndarray, H: int, how: str, shift1: bool) -> np.ndarray:
    """shift1=False: [t, t+H-1] 극값 · True: [t+1, t+H] 극값. 끝은 NaN."""
    n = len(x)
    r = pd.Series(x[::-1]).rolling(H, min_periods=H)
    v = (r.max() if how == "max" else r.min()).to_numpy()[::-1]
    if not shift1:
        return v
    out = np.full(n, np.nan)
    out[:n - 1] = v[1:]
    return out


def path_labels(high, low, close, open_, atr) -> dict:
    """앵커 봉 t 에 대해 두 기준가로 경로 라벨. _c0 = close[t] 기준(부록 A~E 대조용),
    _e1 = open[t+1] 기준(**라이브 실행 가능 규약** -- known_ts 봉 마감에 주문하면 다음 봉에 들어간다)."""
    n = len(close)
    e1 = np.full(n, np.nan); e1[:n - 1] = open_[1:]
    out = {}
    for H in HS:
        # _c0: t+1..t+H · _e1: t+1..t+H (진입가만 다르다)
        hi = _fwd(high, H, "max", shift1=True)
        lo = _fwd(low, H, "min", shift1=True)
        for tag, ref in (("c0", close), ("e1", e1)):
            up = (hi - ref) / ref
            dn = (ref - lo) / ref
            tot = up + dn
            out[f"up_H{H}_{tag}"] = up
            out[f"dn_H{H}_{tag}"] = dn
            out[f"f1_H{H}_{tag}"] = np.where(tot > 0, np.abs(up - dn) / np.maximum(tot, 1e-12), np.nan)
            out[f"size_H{H}_{tag}"] = np.where(atr > 0, tot / np.maximum(atr, 1e-12), np.nan)
    return out


def ext_labels(high, low) -> dict:
    """측면에 맞는 **전방향** W봉 극점이 t..t+D 안에 있는가 (+ 이후 최대 반대이탈 >= g%).
    ⭐중심창(+-W)을 쓰지 않는다 -- sweep/str_z 는 '직전 저점 돌파'가 발동 조건이라 뒤쪽 절반이
    기계적으로 충족된다(2026-09-01 local_extreme 93.4%->23% 붕괴가 그 메커니즘). 부록 C."""
    n = len(low)
    out = {}
    for W, D, g in EXT_CELLS:
        fmin = _fwd(low, W + 1, "min", shift1=False)
        fmax = _fwd(high, W + 1, "max", shift1=False)
        for side, x, is_min in (("bottom", low, True), ("top", high, False)):
            e = np.full(n, np.nan)
            m = n - W
            if is_min:
                ok = (x[:m] <= fmin[:m] + 1e-12) & (((fmax[:m] - x[:m]) / x[:m] * 100) >= g)
            else:
                ok = (x[:m] >= fmax[:m] - 1e-12) & (((x[:m] - fmin[:m]) / x[:m] * 100) >= g)
            e[:m] = ok.astype(float)
            # ⚠️2026-09-07 수정: 이전 판은 `v[:n-D] = a[D:]` 로 라벨을 D만큼 밀어 [t+D, t+2D] 를 봤다.
            a = pd.Series(e[::-1]).rolling(D + 1, min_periods=1).max().to_numpy()[::-1]
            v = a.astype(float).copy(); v[n - D:] = np.nan
            out[f"y_ext_W{W}_D{D}_g{g:g}__{side}"] = v
    # ⭐시차 + 그 극점의 반대이탈%를 **원시로** 기록한다. 그러면 (D, g) 를 학습 시점에 자유롭게 고를 수 있다:
    #     y_ext(D, g) = (lead_bars_W* <= D) & (ext_rebound_pct_W* >= g)
    # 위 고정 y_ext_* 컬럼은 편의용 사본이다. 근거: D 스윕에서 D=1~2 만 앵커 우위가 남고
    # (D=3 부터 현행 first_fire 대비 CI 가 0 을 포함, D>=4 는 초과가 0 으로 수렴),
    # 진입 비용도 D 와 함께 커진다(극점이 +7~12봉이면 진입가부터 역행 중앙 1.06%).
    for W in LEAD_WS:
        fmin = _fwd(low, W + 1, "min", shift1=False)
        fmax = _fwd(high, W + 1, "max", shift1=False)
        for side, x, is_min in (("bottom", low, True), ("top", high, False)):
            e = np.zeros(n, bool); m = n - W
            e[:m] = (x[:m] <= fmin[:m] + 1e-12) if is_min else (x[:m] >= fmax[:m] - 1e-12)
            reb = np.full(n, np.nan)
            if is_min:
                reb[:m] = (fmax[:m] - x[:m]) / x[:m] * 100
            else:
                reb[:m] = (x[:m] - fmin[:m]) / x[:m] * 100
            lead = np.full(n, np.nan); rb = np.full(n, np.nan)
            for j in range(LEAD_MAX, -1, -1):
                idx = np.arange(n - LEAD_MAX)
                hit = idx[e[idx + j]]
                lead[hit] = j; rb[hit] = reb[hit + j]
            out[f"lead_bars_W{W}__{side}"] = lead
            out[f"ext_rebound_pct_W{W}__{side}"] = rb
    return out


def sim_exit(entry, atr, sign, H, L, C, sl, arm, trail):
    """`research_eth_v_rebound_ensemble_portfolio_sim_20260902.sim_exit(infeasible='exit')` 축자 복제.
    (그 모듈의 임포트 사슬이 torch를 끌어온다.) `atr` 은 **절대 ATR**(가격 단위).
    게이트 `trail(anchor='entry')` 와 파리티를 main() 에서 무작위 2,000 경로로 검증한다."""
    n = len(entry)
    stop = entry - sign * sl * atr
    armed = np.zeros(n, bool); best = entry.copy()
    done = np.zeros(n, bool); out = np.zeros(n); ex = np.full(n, H.shape[1] - 1)
    fav = np.where(sign[:, None] > 0, H, L)
    adv = np.where(sign[:, None] > 0, L, H)
    for t in range(H.shape[1]):
        if done.all():
            break
        a_ = adv[:, t]; live = ~done
        hit = live & np.where(sign > 0, a_ <= stop, a_ >= stop)
        out = np.where(hit, sign * (stop - entry) / entry, out)
        ex = np.where(hit, t, ex); done = done | hit
        f_ = fav[:, t]; live = ~done
        imp = live & (sign * (f_ - best) > 0)
        best = np.where(imp, f_, best)
        newly = live & ~armed & (sign * (best - entry) >= arm * atr)
        armed = armed | newly
        ns = best - sign * trail * atr
        u = live & armed & (sign * (ns - stop) > 0)
        bad = u & (sign * (ns - C[:, t]) > 0)
        out = np.where(bad, sign * (C[:, t] - entry) / entry, out)
        ex = np.where(bad, t, ex); done = done | bad
        u = u & ~bad
        stop = np.where(u, ns, stop)
    out = np.where(done, out, sign * (C[:, -1] - entry) / entry)
    return out, ex


def econ_labels(A: pd.DataFrame, high, low, close, open_, atr) -> pd.DataFrame:
    """수정된 트레일링(걸 수 없는 스톱 = 즉시 종가 청산)으로 페이드/지속 양쪽 순손익.
    진입 = open[t+1] (known_ts 봉 마감 주문). ⚠️출구 구조 의존 라벨 -- 1차 목적함수로 쓰지 말 것."""
    n = len(close)
    idx = A["bar_idx"].to_numpy()
    ok = (idx + 1 + ECON_H) < n
    e = np.full(len(A), np.nan); res = {k: np.full(len(A), np.nan) for k in ("net_fade_bp", "net_cont_bp")}
    if ok.sum() == 0:
        return pd.DataFrame(res)
    ii = idx[ok]
    entry = open_[ii + 1]
    win = np.arange(ECON_H)[None, :] + (ii + 1)[:, None]
    Hm, Lm, Cm = high[win], low[win], close[win]
    atr_abs = atr[ii] * entry
    fade_sign = np.where(A["side"].to_numpy()[ok] == "bottom", 1.0, -1.0)
    for tag, sgn in (("net_fade_bp", fade_sign), ("net_cont_bp", -fade_sign)):
        mv, _ = sim_exit(entry, atr_abs, sgn, Hm, Lm, Cm, EXIT["sl"], EXIT["arm"], EXIT["trail"])
        res[tag][ok] = mv * 1e4 - COST_BP
    return pd.DataFrame(res)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/6] 캐노니컬 klines + 발동 ...", flush=True)
    eth = _load_kl(ETH_KL)
    fund = _load_funding()
    btc = _load_kl(BTC_KL)
    tmax = min(eth["timestamp"].max(), btc["timestamp"].max(), fund["calc_time"].max())
    eth = eth[eth["timestamp"] <= tmax].reset_index(drop=True)
    print(f"      상한 {tmax} (펀딩·BTC·ETH 중 이른 쪽) · {len(eth):,}봉", flush=True)
    A = build_fires(eth)
    sig = compute_signals(eth, btc_df=btc[btc["timestamp"] <= tmax], funding_df=fund[fund["calc_time"] <= tmax])
    ts = pd.to_datetime(sig["timestamp"].to_numpy())
    high, low, close, open_ = (sig[c].to_numpy(float) for c in ("high", "low", "close", "open"))
    atr = sig["atr_pct"].to_numpy(float)
    A["bar_idx"] = A["timestamp"].map({t: i for i, t in enumerate(ts)}).astype(int)   # 절단 불변이 아니라 여기서 붙인다

    print("[2/6] 경로·구조 라벨 ...", flush=True)
    P = path_labels(high, low, close, open_, atr)
    E = ext_labels(high, low)
    idx = A["bar_idx"].to_numpy()
    side = A["side"].to_numpy()
    L = {"atr_pct": atr[idx]}
    for k, v in P.items():
        L[k] = v[idx]
    for H in HS:
        for tag in ("c0", "e1"):
            up, dn = L[f"up_H{H}_{tag}"], L[f"dn_H{H}_{tag}"]
            fade = np.where(side == "bottom", up, dn)      # bottom 페이드=롱 -> 유리=상승
            cont = np.where(side == "bottom", dn, up)
            tot = fade + cont
            L[f"mfe_fade_H{H}_{tag}"], L[f"mfe_cont_H{H}_{tag}"] = fade, cont
            r1 = np.where(tot > 0, cont / np.maximum(tot, 1e-12), np.nan)
            L[f"r1_H{H}_{tag}"] = r1
            L[f"y_cont_H{H}_{tag}"] = np.where(np.isfinite(r1), (r1 > 0.5).astype(float), np.nan)
    for key, v in E.items():
        base, sd = key.split("__")
        L[base] = np.where(side == sd, v[idx], L.get(base, np.nan)) if base in L else np.where(side == sd, v[idx], np.nan)
    for key in [k for k in E if k.endswith("__bottom")]:
        base = key.split("__")[0]
        L[base] = np.where(side == "bottom", E[f"{base}__bottom"][idx], E[f"{base}__top"][idx])

    print("[3/6] 경제 라벨(수정 트레일링) ...", flush=True)
    ec = econ_labels(A, high, low, close, open_, atr)
    L.update({c: ec[c].to_numpy() for c in ec.columns})
    L["margin_bp"] = L["net_fade_bp"] - L["net_cont_bp"]
    L["y_econ_cont"] = np.where(np.isfinite(L["margin_bp"]), (L["margin_bp"] < 0).astype(float), np.nan)

    D = pd.concat([A.reset_index(drop=True), pd.DataFrame(L)], axis=1)
    D["split"] = "PRE"
    for nm, a, b in SPLITS:
        D.loc[(D.timestamp >= pd.Timestamp(a)) & (D.timestamp <= pd.Timestamp(b)), "split"] = nm

    print("[4/6] 파리티 검증(sim_exit vs 게이트 trail(anchor='entry'), 2,000 무작위 경로) ...", flush=True)
    import gate_eth_entry_layers_20260903 as GT
    rng = np.random.default_rng(20260907)
    cand = np.flatnonzero(((np.arange(len(close)) + 1 + ECON_H) < len(close)) & np.isfinite(atr))
    smp = rng.choice(cand[cand > 300], 2000, replace=False)
    ent = open_[smp + 1]; aabs = atr[smp] * ent
    win = np.arange(ECON_H)[None, :] + (smp + 1)[:, None]
    sg = rng.choice([1.0, -1.0], len(smp))
    mv, _ = sim_exit(ent, aabs, sg, high[win], low[win], close[win], EXIT["sl"], EXIT["arm"], EXIT["trail"])
    ref = np.array([GT.trail(int(sg[j]), ent[j], atr[smp[j]], high[win[j]], low[win[j]], close[win[j]],
                             EXIT["sl"], EXIT["arm"], EXIT["trail"], anchor="entry") for j in range(len(smp))])
    dmax = float(np.nanmax(np.abs(mv - ref)))
    print(f"      |Δ|max = {dmax:.3e}  ({'일치' if dmax < 1e-9 else '⚠️불일치'})", flush=True)

    print("[5/6] 저장 ...", flush=True)
    D.to_parquet(OUT / "anchors_labels.parquet", index=False)
    prim = D[D.anchor == f"{PRIMARY[0]}/Wc{PRIMARY[1]}"]
    fills = pd.DataFrame({"timestamp": D.timestamp, "signal": D.signal, "sd": np.where(D.side == "bottom", 1, -1),
                          "lim": open_[np.minimum(D.bar_idx + 1, len(close) - 1)], "atr_pct": D.atr_pct,
                          "fi": np.minimum(D.bar_idx + 1, len(close) - 1), "btf": 1,
                          "ei": np.minimum(D.bar_idx + 1 + ECON_H, len(close) - 1),
                          "y": D["y_econ_cont"].fillna(0).astype(int)})
    fills.to_csv(OUT / "fills.csv", index=False)
    pd.DataFrame({"timestamp": ts, "atr_pct": atr, "close": close}).to_parquet(OUT / "bar_features.parquet", index=False)
    cfg = {
        "splits": {"VAL": "2025-09-01", "OOS": "2026-01-01", "HOLDOUT": "2026-04-01"},
        "known_ts": {"assumption": "raw 트리거 -- 8종 증거신호는 마감봉만 보고 계산되므로 앵커 봉 종가에 "
                                   "공동발동이 확정된다. 라벨/진입은 다음 봉(open[t+1])부터 시작한다."},
        "label": {"fills": str((OUT / "fills.csv").relative_to(ROOT)), "ts_col": "timestamp", "y_col": "y",
                  "entry_col": "lim", "side_col": "sd", "atr_col": "atr_pct", "fill_idx_col": "fi",
                  "exit_idx_col": "ei", "bars_to_fill_col": "btf", "signal_col": "signal",
                  "exit": {"sl_atr": EXIT["sl"], "arm_atr": EXIT["arm"], "trail_atr": EXIT["trail"],
                           "trail_anchor": "entry"},
                  "cost_roundtrip": COST_BP / 1e4, "notional": 1.0, "tol_mean_bp": 2.0, "tol_winrate_pp": 2.0},
        "trigger": {"module": "build_eth_anchor_label_dataset_20260907", "fn": "build_fires",
                    "kwargs": {}, "warmup_bars": 8000, "sample_n": 120},
        "features": {"cols": ["atr_pct"], "frame": str((OUT / "bar_features.parquet").relative_to(ROOT))},
        "selection": {"keep_frac": 0.2, "labels": ["y"]},
        "seed": 20260907,
        "_note": "L2/L2P/L3 는 피쳐셋·라이브 채점기가 생긴 뒤에 돌린다. 지금은 L4·L1(인과성)이 목적.",
    }
    (OUT / "gate_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    print("[6/6] 요약", flush=True)
    print(D.groupby(["split", "side"]).size().unstack(fill_value=0).to_string())
    print("\n앵커별 행수:")
    print(D.groupby(["anchor", "split"]).size().unstack(fill_value=0).to_string())
    print(f"\n확정 셀 {PRIMARY[0]}/Wc{PRIMARY[1]}: {len(prim):,}행 · "
          f"라벨 컬럼 {len([c for c in D.columns if c.startswith(('y_','r1_','mfe_','up_','dn_','f1_','size_','lead_','net_','margin'))])}개")
    print(f"저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
