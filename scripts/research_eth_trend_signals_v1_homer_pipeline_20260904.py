#!/usr/bin/env python3
"""추세 신호 v1 -- **호메로스 배포 파이프라인 적용** (반전 8종과 같은 절차, 2026-09-04, 서버 GPU).

사용자: "되돌림 증거신호들의 배포 과정을 호메로스 문서를 보고 따라가줘 — 규칙 기반으로 라벨 데이터를 만들고 TabPFN으로 재학습해 정확도를 올렸다."
템플릿(docs/homer/README.md "재사용 방법론 템플릿") 순서대로:
  1) 피쳐 = Tier0 23 (`research_eth_taker_delta_climax_metalabel_tabpfn_20260829.build_indicator_frame`, 배포 칩과 동일)
  2) 라벨 설계 전 진단: 호라이즌별 raw hit률(K=1.0), MFE/ATR 크기 분포, 발동↔극값 시차, 클러스터링(raw vs 첫발동)
  3) 라벨 = 발동봉 종가 entry → H봉 안 고/저 MFE ≥ K×ATR% (터치) ; K는 TRAIN 50/50 캘리브레이션(K 그리드 0.2~6.0)
     + 5.7절 MAE 캡 변형(hit ∧ MAE < 2K, 정점 이후 분기에서 캘리브레이션)
  4) 모델 사다리: HGB로 H×GAP 그리드(5.5절 8점, 선택 = max min(VAL,OOS), 경계면 표시) → TabPFN 5시드
  5) 검증 3종: 룩어헤드(트리거·라벨 전부 봉 마감 이전/이후로 분리된 배열 연산), permutation importance(VAL), 무작위 봉 기준선(항상-on 모멘텀 방향 규칙)
  ⚠️모집단은 **raw 첫발동(GAP, 인과)** — 반전 8종 원래 배포가 쓴 cluster_dedup 앵커(5.16절 미래참조)는 쓰지 않는다.
  경제성: F0 프레임 두 측면 경제라벨로 메타라벨 상위30% 선별의 신호 vs 반대 순손익(5.8절 방향뒤집기 대조 내장).
산출: data/research/eth_trend_signals_v1_homer_20260904/{report.json, grid_<sig>.csv}, 동결 컨텍스트 data/labels/eth_5m_trend_signals_v1_20260904/<sig>_train_context_frozen_20260904.csv
HOLDOUT(≥2026-04-01) 미접촉.
"""
from __future__ import annotations
import glob, io, json, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import FEATURE_COLUMNS, build_indicator_frame  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score, balanced_accuracy_score  # noqa: E402

KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"; KL_BTC = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"; KL_SPOT = ROOT / "binance_data/klines_spot/ETHUSDT/ETHUSDT-5m-spot.csv"
FRAME = ROOT / "tmp/homer_entry_v2_20260904/frame.parquet"; REG = ROOT / "tmp/eth_entry_oof_regime_20260903/regime_oof_eth.parquet"
BD = ROOT / "data/research/eth_trend_signals_v1_screen_20260904/bookdepth_wide.parquet"
OUT = ROOT / "data/research/eth_trend_signals_v1_homer_20260904"; CTX = ROOT / "data/labels/eth_5m_trend_signals_v1_20260904"
START = pd.Timestamp("2024-04-01"); VAL_START = pd.Timestamp("2025-09-01"); OOS_START = pd.Timestamp("2026-01-01"); HOLDOUT_START = pd.Timestamp("2026-04-01")
H_GRID = [6, 8, 12, 16, 20, 24, 30, 36]; GAP_GRID = [6, 12]; K_GRID = np.round(np.arange(0.2, 6.01, 0.05), 2); SEEDS = [20260829, 141592, 271828, 577215, 20260904]
DIAG_H = [3, 6, 12, 24, 48]
SIGNALS = {   # 이름: 선택 변형 파라미터 (variants 탐색에서 TRAIN 규칙으로 고른 값)
    "regime_pullback_resume": {"bounce": 1.0}, "oi_confirmed_breakout": {"oi": 0.001, "volz": 0.5}, "spot_led_move": {"mv": 1.0, "lead": 0.05},
    "btc_leadlag": {"zb": 2.0, "lag": 0.7}, "liquidity_vacuum": {"mv": 0.5, "w": 0.7},
}


def log(m): print(f"[homer-trend] {m}", flush=True)


def roll_z(x, w=288, minp=144):
    m = x.rolling(w, min_periods=minp).mean(); s = x.rolling(w, min_periods=minp).std(); return (x - m) / s.replace(0, np.nan)


def first_fire(mask, gap):
    keep = np.zeros(len(mask), bool); last = -10**9
    for i in np.flatnonzero(mask):
        if i - last > gap:
            keep[i] = True
        last = i
    return keep


def load_kl(path, prefix=""):
    d = pd.read_csv(path, parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    if d["timestamp"].dt.tz is not None:
        d["timestamp"] = d["timestamp"].dt.tz_localize(None)
    d = d.loc[d["timestamp"] < HOLDOUT_START].reset_index(drop=True)
    return d if not prefix else d.rename(columns={c: f"{prefix}{c}" for c in d.columns if c != "timestamp"})


_AUX = None


def load_aux():
    """kl과 무관한 보조 표(BTC·현물 klines, OI metrics, 레짐 OOF, bookDepth)를 한 번만 로드해 캐시 -- 게이트 L1이 절단 kl로 수백 번 호출한다."""
    global _AUX
    if _AUX is not None:
        return _AUX
    rows = []
    for f in sorted(glob.glob(str(ROOT / "binance_data/metrics/ETHUSDT-metrics-*.zip"))):
        day = f[-14:-4]
        if "2024-03-20" <= day <= "2026-03-31":
            z = zipfile.ZipFile(f); rows.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), usecols=["create_time", "sum_open_interest_value"]))
    met = pd.concat(rows); met["ts"] = pd.to_datetime(met["create_time"]); met = met.sort_values("ts")[["ts", "sum_open_interest_value"]]
    reg = pd.read_parquet(REG); reg["timestamp"] = pd.to_datetime(reg["timestamp"])
    bd = pd.read_parquet(BD).sort_values("ts"); bd["ts"] = pd.to_datetime(bd["ts"])
    _AUX = {"btc": load_kl(KL_BTC, "btc_"), "spot": load_kl(KL_SPOT, "spot_"), "met": met, "reg": reg, "bd": bd}
    return _AUX


def build_triggers(kl, aux=None):
    aux = aux or load_aux()
    n = len(kl); o, h, l, c, v = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
    ts = kl["timestamp"]; close_ts = ts + pd.Timedelta(minutes=5)
    prev = np.r_[np.nan, c[:-1]]; tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev))); atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    vol_z = roll_z(kl["volume"]).to_numpy(); ret3 = c / np.r_[np.nan, np.nan, np.nan, c[:-3]] - 1.0; mv3 = (c - np.r_[np.nan, np.nan, np.nan, c[:-3]]) / atr
    hi48 = pd.Series(h).shift(1).rolling(48).max().to_numpy(); lo48 = pd.Series(l).shift(1).rolling(48).min().to_numpy()
    btc = aux["btc"]; bc = kl[["timestamp"]].merge(btc[["timestamp", "btc_close"]], on="timestamp", how="left")["btc_close"].ffill().to_numpy()
    bret3 = bc / np.r_[np.nan, np.nan, np.nan, bc[:-3]] - 1.0; z_e = roll_z(pd.Series(ret3)).to_numpy(); z_b = roll_z(pd.Series(bret3)).to_numpy()
    spot = aux["spot"]; sc = kl[["timestamp"]].merge(spot[["timestamp", "spot_close"]], on="timestamp", how="left")["spot_close"].ffill().to_numpy(); sret3 = sc / np.r_[np.nan, np.nan, np.nan, sc[:-3]] - 1.0
    spot_lead = (sret3 - ret3) * c / atr
    met = aux["met"]
    oi = pd.merge_asof(pd.DataFrame({"ts": close_ts}), met, on="ts", direction="backward")["sum_open_interest_value"].to_numpy(float); oi_chg6 = oi / np.r_[[np.nan] * 6, oi[:-6]] - 1.0
    reg = aux["reg"]; rg = kl[["timestamp"]].merge(reg[["timestamp", "regime_oof"]], on="timestamp", how="left")["regime_oof"].fillna(-1).astype(int).to_numpy()
    bd = aux["bd"]
    b = pd.merge_asof(pd.DataFrame({"ts": close_ts}), bd, on="ts", direction="backward"); b1h = pd.merge_asof(pd.DataFrame({"ts": close_ts - pd.Timedelta(hours=1)}), bd, on="ts", direction="backward")
    up1, dn1, up1h, dn1h = (x.to_numpy(float) for x in (b["up1"], b["dn1"], b1h["up1"], b1h["dn1"]))
    lo_ref = pd.Series(l).shift(6).rolling(30).min().to_numpy(); hi_ref = pd.Series(h).shift(6).rolling(30).max().to_numpy()
    bounce_up = pd.Series(h).shift(1).rolling(5).max().to_numpy() - lo_ref; bounce_dn = hi_ref - pd.Series(l).shift(1).rolling(5).min().to_numpy()
    P = SIGNALS; T = {}
    T["regime_pullback_resume"] = ((rg == 0) & (bounce_dn >= P["regime_pullback_resume"]["bounce"] * atr) & (c > hi_ref), (rg == 1) & (bounce_up >= P["regime_pullback_resume"]["bounce"] * atr) & (c < lo_ref))
    p = P["oi_confirmed_breakout"]; T["oi_confirmed_breakout"] = ((c > hi48) & (oi_chg6 >= p["oi"]) & (vol_z >= p["volz"]), (c < lo48) & (oi_chg6 >= p["oi"]) & (vol_z >= p["volz"]))
    p = P["spot_led_move"]; T["spot_led_move"] = ((mv3 >= p["mv"]) & (spot_lead >= p["lead"]), (mv3 <= -p["mv"]) & (spot_lead <= -p["lead"]))
    p = P["btc_leadlag"]; T["btc_leadlag"] = ((z_b >= p["zb"]) & (z_e < p["lag"] * z_b), (z_b <= -p["zb"]) & (z_e > p["lag"] * z_b))
    p = P["liquidity_vacuum"]; T["liquidity_vacuum"] = ((mv3 >= p["mv"]) & (up1 <= p["w"] * up1h), (mv3 <= -p["mv"]) & (dn1 <= p["w"] * dn1h))
    T = {k: (np.nan_to_num(u.astype(float)).astype(bool), np.nan_to_num(d.astype(float)).astype(bool)) for k, (u, d) in T.items()}
    return T, atr, mv3


def mfe_mae(h, l, c, idx, H, up):
    """발동봉 종가 대비 H봉 안 유리폭/역행폭(가격비율) -- i+1..i+H (발동봉 제외, 체결봉 크레딧 없음)."""
    ent = c[idx]
    if up:
        fav = np.array([h[i + 1:i + H + 1].max() for i in idx]); adv = np.array([l[i + 1:i + H + 1].min() for i in idx])
        return (fav - ent) / ent, (ent - adv) / ent
    fav = np.array([l[i + 1:i + H + 1].min() for i in idx]); adv = np.array([h[i + 1:i + H + 1].max() for i in idx])
    return (ent - fav) / ent, (adv - ent) / ent


def calibrate_k(peak_tr, mae_tr=None, mult=2.0, target=0.5):
    if mae_tr is None:
        rates = np.array([(peak_tr >= k).mean() for k in K_GRID]); j = int(np.argmin(np.abs(rates - target))); return float(K_GRID[j]), float(rates[j])
    rates = np.array([((peak_tr >= k) & (mae_tr < mult * k)).mean() for k in K_GRID]); jpk = int(np.argmax(rates))
    sub = np.arange(jpk, len(K_GRID)); j = sub[int(np.argmin(np.abs(rates[sub] - target)))]; return float(K_GRID[j]), float(rates[j])


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); CTX.mkdir(parents=True, exist_ok=True)
    kl = load_kl(KL); ind = build_indicator_frame(kl); assert len(ind) == len(kl)
    h, l, c = (kl[x].to_numpy(float) for x in ("high", "low", "close")); ts = kl["timestamp"].to_numpy(); n = len(kl)
    atr_pct = ind["atr_pct"].to_numpy(float); feat_cols = [x for x in FEATURE_COLUMNS if x != "is_bottom"]
    T, atr_abs, mv3 = build_triggers(kl)
    log(f"klines {n:,} · Tier0 {len(feat_cols)}+is_bottom · triggers built ({time.time()-t0:.0f}s)")
    D = pd.read_parquet(FRAME, columns=["timestamp", "is_downside", "split", "net_bp", "net_bp_flip"]); D["timestamp"] = pd.to_datetime(D["timestamp"]); econ = D.set_index(["timestamp", "is_downside"])
    # 무작위 봉 기준선: 항상-on 모멘텀 방향 규칙(직전 15분 이동 부호로 지속 방향) -- 트리거 없이 모든 봉
    rng = np.random.default_rng(20260904)
    report = {"holdout_touched": False, "signals": {}}
    for name, (up_raw, dn_raw) in T.items():
        R = {"params": SIGNALS[name]}
        # ---- 2) 진단 ----
        diag = {"raw_fires": {"up": int(up_raw.sum()), "dn": int(dn_raw.sum())}, "first_fires_gap12": {}, "hit_rate_by_H_K1": {}, "mfe_atr_q_H12": {}, "bars_to_extreme_H24_q": {}}
        ff = {sd: first_fire(m, 12) for sd, m in (("up", up_raw), ("dn", dn_raw))}
        for sd in ("up", "dn"):
            idx = np.flatnonzero(ff[sd]); idx = idx[(idx < n - 48) & (ts[idx] >= np.datetime64(START)) & np.isfinite(atr_pct[idx])]; diag["first_fires_gap12"][sd] = int(len(idx))
            if len(idx) < 30:
                continue
            for H in DIAG_H:
                mfe, _ = mfe_mae(h, l, c, idx, H, sd == "up"); diag["hit_rate_by_H_K1"][f"{sd}_H{H}"] = round(float((mfe / atr_pct[idx] >= 1.0).mean()), 3)
            mfe12, _ = mfe_mae(h, l, c, idx, 12, sd == "up"); diag["mfe_atr_q_H12"][sd] = [round(float(x), 2) for x in np.quantile(mfe12 / atr_pct[idx], [0.25, 0.5, 0.75, 0.9])]
            ext = np.array([(np.argmax(h[i + 1:i + 25]) if sd == "up" else np.argmin(l[i + 1:i + 25])) + 1 for i in idx]); diag["bars_to_extreme_H24_q"][sd] = [int(x) for x in np.quantile(ext, [0.25, 0.5, 0.75])]
        R["diagnostics"] = diag
        # ---- 3)+4) H×GAP 그리드 (K는 TRAIN 50/50), HGB 스크린 ----
        grid = []
        for H in H_GRID:
            for gap in GAP_GRID:
                parts = []
                for sd, m in (("up", up_raw), ("dn", dn_raw)):
                    idx = np.flatnonzero(first_fire(m, gap)); idx = idx[(idx < n - H) & (ts[idx] >= np.datetime64(START)) & np.isfinite(atr_pct[idx])]
                    mfe, mae = mfe_mae(h, l, c, idx, H, sd == "up"); a = atr_pct[idx]
                    df = ind.iloc[idx][["timestamp"] + feat_cols].copy(); df["pos"] = idx; df["side"] = sd; df["is_bottom"] = 1 if sd == "up" else 0
                    df["peak"] = mfe / a; df["mae_m"] = mae / a; parts.append(df)
                F = pd.concat(parts).sort_values("timestamp").reset_index(drop=True); F = F.dropna(subset=feat_cols)
                tsF = pd.to_datetime(F["timestamp"]); tr = (tsF < VAL_START).to_numpy(); va = ((tsF >= VAL_START) & (tsF < OOS_START)).to_numpy(); oo = ((tsF >= OOS_START) & (tsF < HOLDOUT_START)).to_numpy()
                if tr.sum() < 200 or va.sum() < 40 or oo.sum() < 40:
                    grid.append({"H": H, "gap": gap, "skipped": f"n {tr.sum()}/{va.sum()}/{oo.sum()}"}); continue
                for variant in ("plain", "mae_cap"):
                    if variant == "plain":
                        K, rate = calibrate_k(F.loc[tr, "peak"].to_numpy()); y = (F["peak"] >= K).astype(int).to_numpy()
                    else:
                        K, rate = calibrate_k(F.loc[tr, "peak"].to_numpy(), F.loc[tr, "mae_m"].to_numpy()); y = ((F["peak"] >= K) & (F["mae_m"] < 2.0 * K)).astype(int).to_numpy()
                    if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2 or len(np.unique(y[oo])) < 2:
                        continue
                    cols = feat_cols + ["is_bottom"]
                    clf = HistGradientBoostingClassifier(random_state=20260904).fit(F.loc[tr, cols], y[tr])
                    va_auc = roc_auc_score(y[va], clf.predict_proba(F.loc[va, cols])[:, 1]); oo_auc = roc_auc_score(y[oo], clf.predict_proba(F.loc[oo, cols])[:, 1])
                    grid.append({"H": H, "gap": gap, "variant": variant, "K": K, "train_hit": round(rate, 3), "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_oos": int(oo.sum()),
                                 "val_auc": round(float(va_auc), 4), "oos_auc": round(float(oo_auc), 4), "min_val_oos": round(float(min(va_auc, oo_auc)), 4)})
        G = pd.DataFrame(grid); G.to_csv(OUT / f"grid_{name}.csv", index=False)
        Gv = G.dropna(subset=["min_val_oos"]) if "min_val_oos" in G else pd.DataFrame()
        if Gv.empty:
            R["verdict"] = "SKIP(n)"; report["signals"][name] = R; log(f"{name}: grid empty"); continue
        best = Gv.sort_values("min_val_oos", ascending=False).iloc[0]; H, gap, variant, K = int(best["H"]), int(best["gap"]), best["variant"], float(best["K"])
        R["grid_best"] = {k: (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else v)) for k, v in best.items()}
        R["grid_boundary"] = bool(H in (H_GRID[0], H_GRID[-1]))
        log(f"{name}: grid best H={H} gap={gap} {variant} K={K} VAL {best['val_auc']} OOS {best['oos_auc']} boundary={R['grid_boundary']}")
        # ---- TabPFN 5시드 on chosen cell ----
        parts = []
        for sd, m in (("up", up_raw), ("dn", dn_raw)):
            idx = np.flatnonzero(first_fire(m, gap)); idx = idx[(idx < n - H) & (ts[idx] >= np.datetime64(START)) & np.isfinite(atr_pct[idx])]
            mfe, mae = mfe_mae(h, l, c, idx, H, sd == "up"); a = atr_pct[idx]
            df = ind.iloc[idx][["timestamp"] + feat_cols].copy(); df["pos"] = idx; df["side"] = sd; df["is_bottom"] = 1 if sd == "up" else 0; df["peak"] = mfe / a; df["mae_m"] = mae / a; parts.append(df)
        F = pd.concat(parts).sort_values("timestamp").reset_index(drop=True).dropna(subset=feat_cols)
        F["hit"] = ((F["peak"] >= K) & ((F["mae_m"] < 2.0 * K) if variant == "mae_cap" else True)).astype(int)
        tsF = pd.to_datetime(F["timestamp"]); tr = (tsF < VAL_START).to_numpy(); va = ((tsF >= VAL_START) & (tsF < OOS_START)).to_numpy(); oo = ((tsF >= OOS_START) & (tsF < HOLDOUT_START)).to_numpy()
        cols = feat_cols + ["is_bottom"]
        from tabpfn import TabPFNClassifier
        per = {"VAL": [], "OOS": []}; aucs = {"VAL": [], "OOS": []}
        for sd_ in SEEDS:
            clf = TabPFNClassifier(device="cuda", random_state=int(sd_)).fit(F.loc[tr, cols], F.loc[tr, "hit"].to_numpy())
            for w, m in (("VAL", va), ("OOS", oo)):
                p = clf.predict_proba(F.loc[m, cols])[:, 1]; per[w].append(p); aucs[w].append(round(float(roc_auc_score(F.loc[m, "hit"], p)), 4))
        tab = {}
        for w, m in (("VAL", va), ("OOS", oo)):
            p = np.mean(per[w], axis=0); y = F.loc[m, "hit"].to_numpy(); pred = (p >= 0.5).astype(int)
            tab[w] = {"n": int(m.sum()), "hit_rate": round(float(y.mean()), 3), "auc": round(float(roc_auc_score(y, p)), 4), "auc_per_seed": aucs[w], "auc_seed_sd": round(float(np.std(aucs[w])), 4),
                      "bal_acc": round(float(balanced_accuracy_score(y, pred)), 4), "naive_acc": round(float(max(y.mean(), 1 - y.mean())), 4)}
            # 경제성: 상위 30% 선별의 신호 vs 반대 순손익 (F0 두 측면 라벨)
            sub = F.loc[m].assign(p=p); k = max(int(m.sum() * 0.3), 10); top = sub.nlargest(k, "p")
            e_all = econ.reindex(pd.MultiIndex.from_arrays([sub["timestamp"].to_numpy(), sub["is_bottom"].to_numpy()], names=["timestamp", "is_downside"]))
            e_top = econ.reindex(pd.MultiIndex.from_arrays([top["timestamp"].to_numpy(), top["is_bottom"].to_numpy()], names=["timestamp", "is_downside"]))
            tab[w]["econ_all_sig_bp"] = round(float(e_all["net_bp"].mean()), 2); tab[w]["econ_all_opp_bp"] = round(float(e_all["net_bp_flip"].mean()), 2)
            tab[w]["econ_top30_sig_bp"] = round(float(e_top["net_bp"].mean()), 2); tab[w]["econ_top30_opp_bp"] = round(float(e_top["net_bp_flip"].mean()), 2)
        R["tabpfn"] = tab
        # ---- permutation importance (VAL, seed0) ----
        clf = TabPFNClassifier(device="cuda", random_state=SEEDS[0]).fit(F.loc[tr, cols], F.loc[tr, "hit"].to_numpy()); X = F.loc[va, cols].to_numpy(); y = F.loc[va, "hit"].to_numpy()
        base_auc = roc_auc_score(y, clf.predict_proba(X)[:, 1]); imp = []
        for j, fcol in enumerate(cols):
            drops = []
            for _ in range(3):
                Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); drops.append(base_auc - roc_auc_score(y, clf.predict_proba(Xp)[:, 1]))
            imp.append((fcol, round(float(np.mean(drops)), 4)))
        imp.sort(key=lambda x: -x[1]); R["perm_importance_top8"] = imp[:8]
        # ---- 무작위 봉 기준선: 항상-on 모멘텀 방향(직전 15분 이동 부호), 같은 H·K 터치 라벨 ----
        allidx = np.flatnonzero(np.isfinite(mv3) & np.isfinite(atr_pct) & (np.arange(n) < n - H) & (ts >= np.datetime64(START)))
        samp = rng.choice(allidx, size=min(30000, len(allidx)), replace=False); upm = mv3[samp] > 0
        mfe_u, _ = mfe_mae(h, l, c, samp[upm], H, True); mfe_d, _ = mfe_mae(h, l, c, samp[~upm], H, False)
        base_hit = float(np.r_[mfe_u / atr_pct[samp[upm]], mfe_d / atr_pct[samp[~upm]]] .__ge__(K).mean())
        R["random_bar_momentum_baseline_hit"] = round(base_hit, 3); R["trigger_hit_rate_all"] = round(float(F["hit"].mean()), 3); R["lift_vs_baseline"] = round(float(F["hit"].mean() / max(base_hit, 1e-9)), 3)
        # ---- 판정(사전): TabPFN VAL·OOS AUC ≥ 0.58 둘 다, 시드 sd < 0.02, 상위30% 신호 순손익 > 반대 (VAL·OOS) → CHIP_CANDIDATE
        ok = all(tab[w]["auc"] >= 0.58 for w in tab) and all(tab[w]["auc_seed_sd"] < 0.02 for w in tab) and all(tab[w]["econ_top30_sig_bp"] > tab[w]["econ_top30_opp_bp"] for w in tab)
        R["verdict"] = "CHIP_CANDIDATE" if ok else "NOT_YET"; R["label"] = {"H": H, "gap": gap, "variant": variant, "K": K}
        # ---- 동결 컨텍스트 (TRAIN만) ----
        ctx = F.loc[tr, ["pos", "timestamp", "side", "hit", "peak", "is_bottom"] + feat_cols].rename(columns={"peak": "move_atr_mult"}); ctx.to_csv(CTX / f"{name}_train_context_frozen_20260904.csv", index=False)
        R["context_rows"] = int(len(ctx))
        report["signals"][name] = R
        log(f"{name}: TabPFN VAL {tab['VAL']['auc']} (sd {tab['VAL']['auc_seed_sd']}) OOS {tab['OOS']['auc']} · hit {R['trigger_hit_rate_all']} vs base {base_hit:.3f} lift {R['lift_vs_baseline']} · top30 econ VAL {tab['VAL']['econ_top30_sig_bp']}/{tab['VAL']['econ_top30_opp_bp']} OOS {tab['OOS']['econ_top30_sig_bp']}/{tab['OOS']['econ_top30_opp_bp']} · imp {imp[:3]} => {R['verdict']} ({time.time()-t0:.0f}s)")
    (OUT / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str)); (CTX / "manifest.json").write_text(json.dumps({k: v.get("label") for k, v in report["signals"].items()}, indent=1))
    print("HOMER_TREND_DONE")


if __name__ == "__main__":
    main()
