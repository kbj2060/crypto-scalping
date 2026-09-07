#!/usr/bin/env python3
"""앵커 방향 예측 — **피쳐 구성** (2026-09-07).

09-06 페이드/지속 데이터셋(`research_eth_fire_ordering_model_20260906.py`)의
`build_context` / `window_features` 를 **축자 재사용**한다. 같은 질문(페이드 vs 지속)으로
이미 인과성이 검토된 계보이고, 모듈 자체는 `V2.OUT/frame.parquet`(다른 앵커 모집단)에
묶여 있어 함수만 가져온다.

피쳐: 5분봉 기본 + 강제흐름(OI·롱숏비·테이커·펀딩·베이시스·호가깊이·BTC) 36종.
창 [발동−2, 발동] = −10분~0 (인과). 방향 있는 피쳐는 **페이드 방향 부호로 정렬**(측면 대칭).

데이터 완전 구간: 2024-04-20(bookdepth 시작) ~ 2026-03-31(spot/btc/metrics 끝).
=> HOLDOUT_SPENT(2026-04-01~) 는 피쳐가 없어 제외. TRAIN/VAL/OOS 는 완전히 덮인다.

## 자체 검증
  F1 인과성 재계산   앵커에서 데이터를 잘라 다시 만들어도 같은 피쳐값인가 (무작위 40건)
  F2 결측률          피쳐별 NaN 비율 · 전부 NaN 인 열 없음
  F3 누수 탐지       단일 피쳐 AUC ≥ 0.95 면 FAIL (게이트 L3 규약)
  F4 조인 시점       결정 봉 수익률이 **마감봉**과 상관 1.0 인가 (다음 봉이면 한 봉 미래참조)
  F5 측면 대칭       bottom/top 피쳐 분포가 정렬 후 유사한가 (KS 통계량)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


TS = _load("trend_v1_mod", "scripts/research_eth_trend_signals_v1_screen_20260904.py")
LAB = ROOT / "tmp/eth_anchor_labels_v2_20260907/labels_v2.parquet"
OUT = ROOT / "tmp/eth_anchor_features_20260907"
W = 2                              # ±10분 = ±2봉 (09-06 계보 상수)
FEAT_START = pd.Timestamp("2024-04-20")
FEAT_END = pd.Timestamp("2026-03-31 23:55")


def _ns(x):
    return pd.to_datetime(x).astype("datetime64[ns]")


def build_context(seg: pd.DataFrame) -> pd.DataFrame:
    """`research_eth_fire_ordering_model_20260906.build_context` 축자 복제.
    모든 재료를 seg 인덱스에 **인과 조인**(결정 봉 마감 이전 스냅샷만)."""
    ts = seg["timestamp"]; close_ts = _ns(ts + pd.Timedelta(minutes=5))
    X = pd.DataFrame(index=seg.index)
    c, o, h, l, v = (seg[k].to_numpy(float) for k in ("close", "open", "high", "low", "volume"))
    X["close"], X["atr_pct"] = c, pd.Series(h - l).rolling(14).mean().to_numpy() / c
    X["ret1"] = pd.Series(c).pct_change().to_numpy()
    X["delta_z"] = TS.roll_z(pd.Series(2.0 * seg["taker_buy_base"].to_numpy(float) - v)).to_numpy()
    X["vol_z"] = TS.roll_z(pd.Series(v)).to_numpy()
    X["trades_z"] = TS.roll_z(seg["trades"].astype(float)).to_numpy()
    X["range_atr"] = (h - l) / (X["atr_pct"].to_numpy() * c)

    met = TS.load_metrics().copy(); met["ts"] = _ns(met["ts"])
    for tag, at in (("", close_ts), ("_lag1", close_ts - pd.Timedelta(minutes=5))):
        m = pd.merge_asof(pd.DataFrame({"ts": _ns(at)}), met, on="ts", direction="backward",
                          tolerance=pd.Timedelta(minutes=15))
        X["oi" + tag] = m["sum_open_interest_value"].to_numpy(float)
        X["lsr_retail" + tag] = m["count_long_short_ratio"].to_numpy(float)
        X["lsr_top" + tag] = m["sum_toptrader_long_short_ratio"].to_numpy(float)
        X["taker_ratio" + tag] = m["sum_taker_long_short_vol_ratio"].to_numpy(float)

    fu = TS.load_funding().copy(); fu["ts"] = _ns(fu["ts"])
    f = pd.merge_asof(pd.DataFrame({"ts": close_ts}), fu, on="ts", direction="backward")
    X["funding_z"] = f["funding_z"].to_numpy(float)

    spot = TS.load_klines(TS.KL_SPOT, "spot_")
    sc = seg[["timestamp"]].merge(spot[["timestamp", "spot_close"]], on="timestamp", how="left")["spot_close"].ffill().to_numpy(float)
    X["basis"] = c / sc - 1.0

    btc = TS.load_klines(TS.KL_BTC, "btc_")
    bc = seg[["timestamp"]].merge(btc[["timestamp", "btc_close"]], on="timestamp", how="left")["btc_close"].ffill().to_numpy(float)
    X["btc_close"] = bc

    try:
        bd = TS.load_bookdepth()
        bd = bd.reset_index() if bd.index.name else bd
        tcol = next(cc for cc in bd.columns if np.issubdtype(bd[cc].dtype, np.datetime64))
        bd = bd.rename(columns={tcol: "ts"}); bd["ts"] = _ns(bd["ts"]); bd = bd.sort_values("ts")
        cols = [cc for cc in ("up1", "dn1") if cc in bd.columns]
        if len(cols) == 2:
            b = pd.merge_asof(pd.DataFrame({"ts": close_ts}), bd[["ts"] + cols], on="ts",
                              direction="backward", tolerance=pd.Timedelta(minutes=15))
            up, dn = b["up1"].to_numpy(float), b["dn1"].to_numpy(float)
            X["depth_imb"] = (dn - up) / (dn + up)
        else:
            X["depth_imb"] = np.nan
    except Exception as e:                                                     # noqa: BLE001
        print(f"bookDepth 스킵: {type(e).__name__}: {e}"); X["depth_imb"] = np.nan
    return X


def window_features(X: pd.DataFrame, kp: np.ndarray, sgn_fade: np.ndarray, k_end: int = 0) -> pd.DataFrame:
    """`...window_features` 축자 복제. 창 [발동−W, 발동+k_end] 의 변화. k_end=0 = 인과."""
    i0, i1 = kp - W, kp + k_end
    g = lambda col, idx: X[col].to_numpy()[idx]                                # noqa: E731
    F = {}
    c0, c1 = g("close", i0), g("close", i1)
    atr_abs = g("atr_pct", i1) * c1
    F["w_ret_atr"] = sgn_fade * (c1 - c0) / atr_abs
    F["w_ret_raw"] = sgn_fade * (c1 / c0 - 1.0)
    for col in ("delta_z", "vol_z", "trades_z", "range_atr"):
        F[f"w_{col}_end"] = g(col, i1)
        F[f"w_{col}_chg"] = g(col, i1) - g(col, i0)
    F["w_delta_z_aligned"] = sgn_fade * g("delta_z", i1)
    for tag in ("", "_lag1"):
        oi0, oi1 = g("oi" + tag, i0), g("oi" + tag, i1)
        d_oi = oi1 / oi0 - 1.0
        F[f"w_oi_chg{tag}"] = d_oi
        F[f"w_forced_flow{tag}"] = -np.sign(d_oi) * np.abs((c1 / c0 - 1.0)) / (g("atr_pct", i1) + 1e-12)
        F[f"w_oi_x_ret{tag}"] = d_oi * sgn_fade * (c1 / c0 - 1.0) * 1e4
        for col in ("lsr_retail", "lsr_top", "taker_ratio"):
            F[f"w_{col}_chg{tag}"] = g(col + tag, i1) - g(col + tag, i0)
            F[f"w_{col}_end{tag}"] = g(col + tag, i1)
    F["w_funding_z"] = g("funding_z", i1)
    F["w_basis"] = g("basis", i1)
    F["w_basis_chg"] = g("basis", i1) - g("basis", i0)
    F["w_depth_imb"] = sgn_fade * g("depth_imb", i1)
    F["w_depth_imb_chg"] = sgn_fade * (g("depth_imb", i1) - g("depth_imb", i0))
    b0, b1 = g("btc_close", i0), g("btc_close", i1)
    F["w_btc_ret_aligned"] = sgn_fade * (b1 / b0 - 1.0) * 1e4
    F["w_eth_idio"] = sgn_fade * ((c1 / c0 - 1.0) - (b1 / b0 - 1.0)) * 1e4
    return pd.DataFrame(F)


# ------------------------------------------------------------------ 자체 검증

def verify(D, F, seg, X, kp, sgn):
    fails = []
    def chk(name, cond, note=""):
        print(f"  {'PASS' if cond else '🔴FAIL'}  {name}{('  ' + note) if note else ''}", flush=True)
        if not cond:
            fails.append(name)

    cols = list(F.columns)
    # F2 결측
    na = F.isna().mean()
    chk("F2b 비유한값(inf) 없음", bool(np.isfinite(F.to_numpy(float)) | F.isna().to_numpy()).__bool__()
        if False else bool(not np.isinf(F.to_numpy(float)).any()))
    chk("F2 전부 NaN 인 열 없음", bool((na < 1.0).all()), f"최대 결측 {na.max():.3f} ({na.idxmax()})")
    chk("F2 결측률 50% 초과 열 없음", bool((na <= 0.5).all()),
        ", ".join(f"{c} {na[c]:.2f}" for c in na[na > 0.5].index[:5]))

    # F1 인과성: 앵커에서 잘라 재계산해도 같은 값인가
    rng = np.random.default_rng(11)
    pick = rng.choice(np.flatnonzero(kp > 5000), 40, replace=False)
    bad = 0
    for k in pick:
        cut = int(kp[k]) + 1                       # 앵커 봉까지만
        sub = seg.iloc[:cut].reset_index(drop=True)
        Xs = build_context(sub)
        fs = window_features(Xs, np.array([kp[k]]), np.array([sgn[k]]), 0)
        a = F.iloc[k][cols].to_numpy(float); b = fs.iloc[0][cols].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        if not np.allclose(a[m], b[m], rtol=1e-9, atol=1e-12) or (np.isfinite(a) != np.isfinite(b)).sum():
            bad += 1
    chk("F1 앵커 절단 재계산 일치 (40건)", bad == 0, f"불일치 {bad}")

    # F3 누수 탐지 — 단일 피쳐 AUC
    y = D["y2"].to_numpy(float)
    m = np.isfinite(y)
    aucs = {}
    for c in cols:
        v = F[c].to_numpy(float)
        mm = m & np.isfinite(v)
        if mm.sum() < 200 or len(np.unique(y[mm])) < 2:
            continue
        a = roc_auc_score(y[mm], v[mm])
        aucs[c] = max(a, 1 - a)
    top = sorted(aucs.items(), key=lambda x: -x[1])[:5]
    chk("F3 단일피쳐 AUC < 0.95 (누수)", all(a < 0.95 for a in aucs.values()),
        " ".join(f"{c}={a:.3f}" for c, a in top))
    chk("F3 최고 단일피쳐 AUC < 0.70 (강한 누수 의심)", top[0][1] < 0.70 if top else True)

    # F4 조인 시점 — 결정 봉 수익률이 마감봉과 상관 1.0
    c_arr = X["close"].to_numpy()
    ret_close = (c_arr[kp] - c_arr[kp - 1]) / c_arr[kp - 1]
    ret_next = (c_arr[kp + 1] - c_arr[kp]) / c_arr[kp]
    r_now = np.corrcoef(F["w_ret_raw"].to_numpy() * sgn, ret_close)[0, 1]
    r_nxt = np.corrcoef(F["w_ret_raw"].to_numpy() * sgn, ret_next)[0, 1]
    chk("F4 조인 시점: 마감봉 상관 > 다음봉 상관", abs(r_now) > abs(r_nxt),
        f"마감봉 {r_now:+.3f} vs 다음봉 {r_nxt:+.3f}")

    # F5 측면 대칭 — **정렬 피쳐에만** 적용한다.
    # `_end`/`_chg` 계열은 sgn_fade 를 안 곱한 **원시값**이라 측면 비대칭이 정상이다
    # (바닥은 매도 클라이맥스라 delta_z 가 크게 음수, 천장은 반대).
    # ⚠️`w_oi_x_ret*` 는 정렬 피쳐인데도 비대칭인데, 원인은 코드가 아니라 **d_oi 자체의 부호 뒤집힘**이다
    # (2026-09-07 실측: 바닥 발동 OI 중앙 -0.71%(청산·디레버리징) vs 천장 +0.47%(신규 롱)).
    # 실제 시장 비대칭이므로 결함이 아니고, 대신 `side_is_bottom` 을 명시 피쳐로 넣어
    # 모델이 측면을 조건화할 수 있게 한다.
    from scipy.stats import ks_2samp
    ALIGNED = {"w_ret_atr", "w_ret_raw", "w_delta_z_aligned", "w_depth_imb", "w_depth_imb_chg",
               "w_btc_ret_aligned", "w_eth_idio"}
    OI_PROD = {"w_oi_x_ret", "w_oi_x_ret_lag1"}
    b = D["side"].to_numpy() == "bottom"
    ks = {}
    for c in cols:
        v = F[c].to_numpy(float)
        x1, x2 = v[b & np.isfinite(v)], v[~b & np.isfinite(v)]
        if len(x1) > 50 and len(x2) > 50:
            ks[c] = (ks_2samp(x1, x2).statistic, np.median(x1), np.median(x2))
    al = {c: v for c, v in ks.items() if c in ALIGNED}
    chk("F5 정렬 피쳐 측면 대칭 KS < 0.30", all(v[0] < 0.30 for v in al.values()),
        " ".join(f"{c}={v[0]:.2f}" for c, v in sorted(al.items(), key=lambda x: -x[1][0])[:3]))
    chk("F5 정렬 피쳐 측면 중앙 부호 일치",
        all(np.sign(v[1]) == np.sign(v[2]) for v in al.values()))
    print("     (참고) 비정렬·OI곱 피쳐 KS 상위: " +
          " ".join(f"{c}={v[0]:.2f}" for c, v in sorted(
              ((c, v) for c, v in ks.items() if c not in ALIGNED), key=lambda x: -x[1][0])[:4]) +
          "  — d_oi 부호 뒤집힘(바닥 청산 vs 천장 신규롱)에서 오는 실제 비대칭")
    return fails, aucs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(LAB)
    eth = B._load_kl(B.ETH_KL)
    kl = pd.read_csv(B.ETH_KL, usecols=["timestamp", "open", "high", "low", "close", "volume",
                                        "quote_volume", "trades", "taker_buy_base"],
                     parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    seg = kl[kl["timestamp"] <= FEAT_END].reset_index(drop=True)
    pos = {t: i for i, t in enumerate(seg["timestamp"].to_numpy())}
    D = D[D["timestamp"].isin(pos.keys())].reset_index(drop=True)
    D["kp"] = D["timestamp"].map(pos).astype(int)
    D = D[(D["timestamp"] >= FEAT_START) & (D["kp"] > W + 20)].reset_index(drop=True)
    print(f"[1/3] 앵커 {len(D):,}행 (피쳐 구간 {FEAT_START.date()} ~ {FEAT_END.date()})", flush=True)
    print(D.groupby("split").size().to_string(), flush=True)

    print("[2/3] 컨텍스트·창 피쳐 ...", flush=True)
    X = build_context(seg)
    kp = D["kp"].to_numpy()
    sgn = np.where(D["side"].to_numpy() == "bottom", 1.0, -1.0)     # 페이드 방향 정렬
    F = window_features(X, kp, sgn, 0)
    # ⭐±inf -> NaN. 원인은 OI 결측 봉(oi0 == 0)에서 d_oi = oi1/oi0 - 1 = inf.
    # 2026-09-07 실측 6개(3행, 2024-07). 남겨두면 StandardScaler 가 오버플로로 NaN 을 만든다.
    n_inf = int(np.isinf(F.to_numpy(float)).sum())
    F = F.replace([np.inf, -np.inf], np.nan)
    print(f"      피쳐 {F.shape[1]}개 · {F.shape[0]:,}행 · inf->NaN {n_inf}개", flush=True)

    print("\n=== 자체 검증 ===", flush=True)
    fails, aucs = verify(D, F, seg, X, kp, sgn)

    out = pd.concat([D.reset_index(drop=True), F.reset_index(drop=True)], axis=1)
    out["votes"] = out["n_signals"]
    out["side_is_bottom"] = (out["side"] == "bottom").astype(float)   # 측면을 명시 피쳐로(F5 참조)
    out.to_parquet(OUT / "features.parquet", index=False)
    pd.Series(aucs).sort_values(ascending=False).to_csv(OUT / "single_feature_auc.csv")
    (OUT / "meta.json").write_text(json.dumps({
        "n": len(out), "n_features": int(F.shape[1]), "feature_cols": list(F.columns) + ["votes", "atr_pct", "side_is_bottom"],
        "window_bars": W, "k_end": 0, "aligned_by": "fade direction (sgn)",
        "feat_range": [str(FEAT_START), str(FEAT_END)], "self_check_failed": fails,
    }, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT} · 검증 실패 {len(fails)}건")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
