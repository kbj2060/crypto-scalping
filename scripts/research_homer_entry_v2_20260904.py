#!/usr/bin/env python3
"""호메로스 진입 모델 v2 -- 경제라벨(F0) + 레짐 분류기 + 증거신호 재료, 층별 게이트 내장 (2026-09-04).

사전등록: docs/experiments/homer_entry_v2_prereg_20260904.md (결과 보기 전 작성). 요지:
  모집단  매 5분봉 × 양방향 (트리거 게이트 없음 -> 앵커 선택 문제 없음)
  라벨    open[i+1] 진입 -> sim_exit(SL 5.0 / ARM 1.5 / Trail 0.1 ATR, 200봉, 비관 순서) -> 10bp 차감 > 0
  팔      F0 Tier0 23 | F1 +레짐 one-hot 6(OOF) | F2 +재료 8×(pct,age,dir)+정렬합(OOF) | F3 F1+F2
  학습기  TabPFN 5시드 18k 컨텍스트 앙상블 (서버 GPU) / HGB 프록시 (로컬 스모크·게이트)
  선정    VAL 상위 5% 분위(팔 간 호출빈도 일치) -> 순차 포트폴리오(동시보유 5) -> OOS 팔당 1회
  대조군  방향뒤집기 · 랜덤 부분표집 귀무(B=200) · 일군집 부트스트랩(팔·Δ vs F0) · 5시드 부호 · DSR/PBO
  게이트  L4/L1/L2/L2P/L3/T1/T2 -- scripts/gate_eth_entry_layers_20260903.py --pipeline <OUT>
  HOLDOUT ≥2026-04-01 은 로드하지 않는다(이 계보에서 소진). 심판은 전진 섀도우.

사용:
  python scripts/research_homer_entry_v2_20260904.py --stage build            # 로컬 CPU, 프레임·라벨·게이트 입력
  python scripts/research_homer_entry_v2_20260904.py --stage eval --learner hgb     # 로컬 프록시
  python scripts/research_homer_entry_v2_20260904.py --stage eval --learner tabpfn  # 서버 GPU
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "tmp/homer_entry_v2_20260904"
OOFD_MAT = ROOT / "tmp/eth_entry_oof_metalabel_20260903"
OOFD_REG = ROOT / "tmp/eth_entry_oof_regime_20260903"
POP_CFG = ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json"

TRAIN_START = pd.Timestamp("2024-05-01", tz="UTC")     # 레짐·재료 OOF 워밍업 제외 (모든 팔 동일)
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")           # HOLDOUT 경계 -- 이 이후 행은 만들지 않는다
LABEL_CELL = (5.0, 1.5, 0.1)
FORWARD_BARS = 200
COST_BP = 10.0
SEEDS = [20260829, 141592, 271828, 577215, 20260902]
CONTEXT_N = 18000
CHUNK = 10000
TOP_FRAC = 0.05
MAX_CONC = 5
B_NULL, B_BOOT = 200, 1000
SIGNALS = ["taker_delta_z_climax", "short_term_return_z", "liquidity_sweep", "orthogonal_combo",
           "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]
REG = {0: "bull", 1: "bear", 2: "chop"}
HGB_HP = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
              l2_regularization=1.0, early_stopping=False)


def log(m): print(f"[hev2] {m}", flush=True)


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------------- 라벨 (경제라벨 모델 원문)
def sim_exit(entry, atr, sign, H, L, C, sl, arm, trail):
    """비관 기준 트레일링 -- research_eth_v_rebound_ensemble_portfolio_sim_20260902.py 원문 그대로.
    봉마다 (1)불리한 쪽 스톱 판정 -> (2)유리한 쪽 best -> (3)무장 -> (4)트레일. (수익률, 청산봉오프셋)."""
    n = len(entry)
    stop = entry - sign * sl * atr
    armed = np.zeros(n, bool); best = entry.copy()
    done = np.zeros(n, bool); out = np.zeros(n); ex = np.full(n, H.shape[1] - 1)
    fav = np.where(sign[:, None] > 0, H, L)
    adv = np.where(sign[:, None] > 0, L, H)
    for t in range(H.shape[1]):
        if done.all():
            break
        a_ = adv[:, t]
        live = ~done
        hit = live & np.where(sign > 0, a_ <= stop, a_ >= stop)
        out = np.where(hit, sign * (stop - entry) / entry, out)
        ex = np.where(hit, t, ex); done = done | hit
        f_ = fav[:, t]
        live = ~done
        imp = live & (sign * (f_ - best) > 0)
        best = np.where(imp, f_, best)
        newly = live & ~armed & (sign * (best - entry) >= arm * atr)
        armed = armed | newly
        ns = best - sign * trail * atr
        u = live & armed & (sign * (ns - stop) > 0)
        stop = np.where(u, ns, stop)
    out = np.where(done, out, sign * (C[:, -1] - entry) / entry)
    return out, ex


def trail_single(side, e, a, H, L, C):
    """게이트 L2P용 단일 경로 백테스트 함수. a = ATR/entry(비율) -> 절대 ATR로 환산. raw move 반환."""
    out, _ = sim_exit(np.array([float(e)]), np.array([float(a) * float(e)]), np.array([float(side)]),
                      np.asarray(H, float)[None], np.asarray(L, float)[None], np.asarray(C, float)[None], *LABEL_CELL)
    return float(out[0])


# ----------------------------------------------------------------------------- 재료 (Phase 0 빌더 + 방향)
def build_material(cfg, kl):
    """OOF 메타라벨(`<sig>_oof.csv`) -> 봉별 `<sig>_pct`/`_age`/`_dir`/출처. 직전 발동이 H봉 안이면 그 pct·방향,
    age=경과/H; 아니면 pct 0 · age 1 · dir 0. 출처(foldk/final)는 L3 OOF 점검용."""
    n = len(kl); pos = {t: i for i, t in enumerate(pd.DatetimeIndex(kl["timestamp"]))}
    out = {}; stats = {}
    for name in SIGNALS:
        H = int(cfg[name]["horizon"])
        d = pd.read_csv(OOFD_MAT / f"{name}_oof.csv", parse_dates=["timestamp"])
        d = d[np.isfinite(d["pct_oof"])].copy()
        d["i"] = d["timestamp"].map(pos); d = d.dropna(subset=["i"]); d["i"] = d["i"].astype(int)
        d = d.sort_values(["i", "proba_oof"]).drop_duplicates("i", keep="last")
        src = d["oof_source"].fillna("").astype(str).str.replace(r"\(.*", "", regex=True).to_numpy()
        fire_i = d["i"].to_numpy(); pct_v = d["pct_oof"].to_numpy(float)
        dir_v = np.where(d["side"].astype(str).to_numpy() == "bottom", 1.0, -1.0)
        last = np.full(n, -10**9, dtype=np.int64); last[fire_i] = fire_i
        last = np.maximum.accumulate(last)
        el = np.arange(n) - last; active = el < H
        idx_of = np.full(n, -1, dtype=np.int64); idx_of[fire_i] = np.arange(len(fire_i))
        idx_last = np.where(last >= 0, idx_of[np.clip(last, 0, n - 1)], -1)
        ok = active & (idx_last >= 0); j = np.clip(idx_last, 0, len(pct_v) - 1)
        out[f"{name}_pct"] = np.where(ok, pct_v[j], 0.0)
        out[f"{name}_age"] = np.where(active, el / H, 1.0)
        out[f"{name}_dir"] = np.where(ok, dir_v[j], 0.0)
        out[f"{name}_oof_src"] = np.where(ok, src[j], "")
        stats[name] = {"H": H, "n_fires_oof": int(len(d)), "bar_coverage": round(float(active.mean()), 4)}
    M = pd.DataFrame(out); M.insert(0, "timestamp", kl["timestamp"].to_numpy())
    return M, stats


def load_regime(kind):
    r = pd.read_parquet(OOFD_REG / f"regime_oof_{kind}.parquet")
    r["timestamp"] = pd.to_datetime(r["timestamp"])
    df = pd.DataFrame({"timestamp": r["timestamp"].to_numpy()})
    code = r["regime_oof"].to_numpy(int)
    for k, nm in REG.items():
        df[f"reg_{kind}_{nm}"] = (code == k).astype(np.int8)
    df[f"regime_{kind}_oof_src"] = r["oof_source"].fillna("").astype(str).str.replace(r"\(.*", "", regex=True).to_numpy()
    return df


# ----------------------------------------------------------------------------- build
def stage_build():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    _s1 = _load("s1_hev2", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
    _s1.VAL_END = OOS_END                                   # long frame을 OOS 끝까지(HOLDOUT 제외)
    F0 = list(_s1.FEATURE_COLUMNS)
    log(f"프레임 빌드 (feasibility 원문) ... F0={len(F0)}")
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", atr_mult=1.5, t_sustain=0.2, full_bars=12)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick", atr_mult=1.5, t_sustain=0.2, full_bars=12)
    long = _s1.long_frame_for(sig, feat, sb, st).drop(columns=["status", "label", "split"], errors="ignore")
    long = long.loc[long["timestamp"] >= TRAIN_START].reset_index(drop=True)
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"
    log(f"  long rows {len(long):,}  {long['timestamp'].min()} ~ {long['timestamp'].max()}  ({time.time()-t0:.0f}s)")

    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    kl = kl.reset_index(drop=True)
    o, h, l, c = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close")); nk = len(kl)
    pos_of = pd.Series(np.arange(nk), index=pd.DatetimeIndex(kl["timestamp"]))
    long["ts"] = long["timestamp"].dt.tz_localize(None)
    long["pos"] = long["ts"].map(pos_of).to_numpy()
    long = long.loc[np.isfinite(long["pos"]) & (long["pos"] + FORWARD_BARS + 1 < nk)].reset_index(drop=True)
    long["pos"] = long["pos"].astype(int)

    # 경제라벨
    log("경제라벨 sim_exit ...")
    sl0, arm0, tr0 = LABEL_CELL
    idx_all = long["pos"].to_numpy(); sgn_all = np.where(long["is_downside"].to_numpy() == 1, 1.0, -1.0)
    atr_all = long["atr"].to_numpy(float)
    net = np.full(len(long), np.nan); exo = np.zeros(len(long), int)
    for s_ in range(0, len(long), 40000):
        e_ = min(s_ + 40000, len(long)); idx = idx_all[s_:e_]
        H = np.stack([h[j + 1:j + 1 + FORWARD_BARS] for j in idx]); L = np.stack([l[j + 1:j + 1 + FORWARD_BARS] for j in idx])
        C = np.stack([c[j + 1:j + 1 + FORWARD_BARS] for j in idx])
        pn, ex = sim_exit(o[idx + 1], atr_all[s_:e_], sgn_all[s_:e_], H, L, C, sl0, arm0, tr0)
        net[s_:e_] = pn * 1e4 - COST_BP; exo[s_:e_] = ex
    long["net_bp"] = net; long["exit_off"] = exo; long["y"] = (net > 0).astype(int)
    long["entry"] = o[idx_all + 1]; long["atr_pct"] = long["atr"] / long["entry"]
    # 방향뒤집기 대조군용: 같은 봉 반대쪽 행의 net_bp
    key = long["pos"].astype(np.int64) * 2 + long["is_downside"].astype(np.int64)
    opp = long["pos"].astype(np.int64) * 2 + (1 - long["is_downside"].astype(np.int64))
    long["net_bp_flip"] = pd.Series(net, index=key.to_numpy()).reindex(opp.to_numpy()).to_numpy()
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN", np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    log(f"  라벨률 {long['y'].mean():.4f}  split {long['split'].value_counts().to_dict()}  ({time.time()-t0:.0f}s)")

    # 레짐 OOF + 재료 OOF + 자기봉 수익률(조인시점 게이트용)
    reg = load_regime("eth").merge(load_regime("btc"), on="timestamp", how="outer")
    long = long.merge(reg, left_on="ts", right_on="timestamp", how="left", suffixes=("", "_r")).drop(columns=["timestamp_r"], errors="ignore")
    cfg = json.loads(POP_CFG.read_text())["cfg"]
    M, mstats = build_material(cfg, kl)
    long = long.merge(M, left_on="ts", right_on="timestamp", how="left", suffixes=("", "_m")).drop(columns=["timestamp_m"], errors="ignore")
    lr = pd.DataFrame({"ts": kl["timestamp"], "log_return": np.log(kl["close"] / kl["close"].shift(1))})
    long = long.merge(lr, on="ts", how="left")
    for k in ("eth", "btc"):
        for nm in REG.values():
            long[f"reg_{k}_{nm}"] = long[f"reg_{k}_{nm}"].fillna(0).astype(np.int8)
        long[f"regime_{k}_oof_src"] = long[f"regime_{k}_oof_src"].fillna("")
    side_sign = np.where(long["is_downside"].to_numpy() == 1, 1.0, -1.0)
    long["mat_signed_sum"] = sum(long[f"{s}_dir"] * long[f"{s}_pct"] for s in SIGNALS)
    long["mat_aligned_sum"] = side_sign * long["mat_signed_sum"].to_numpy()
    F1 = F0 + [f"reg_{k}_{nm}" for k in ("eth", "btc") for nm in REG.values()]
    MAT = [f"{s}_{x}" for x in ("pct", "age", "dir") for s in SIGNALS] + ["mat_aligned_sum"]
    F2 = F0 + MAT; F3 = F1 + MAT
    arms = {"F0": F0, "F1": F1, "F2": F2, "F3": F3}
    n_bad = int((long.loc[long["split"] == "TRAIN", "regime_eth_oof_src"].str.startswith("final")).sum())
    log(f"  레짐 TRAIN 'final' 출처 행 {n_bad} (0이어야)  재료 커버리지 {[(k, v['bar_coverage']) for k, v in mstats.items()]}")

    long = long.drop(columns=["timestamp"]).rename(columns={"ts": "timestamp"})
    long["signal"] = "every_bar"; long["sd"] = side_sign.astype(int)
    long["fi"] = long["pos"] + 1; long["ei"] = long["pos"] + 1 + FORWARD_BARS; long["btf"] = 1
    long["lim"] = long["entry"]; long["y_frac"] = long["net_bp"] / 1e4; long["atr_abs"] = long["atr"]
    rng = np.random.default_rng(20260904); long["l2_sample"] = (rng.random(len(long)) < 0.05).astype(int)
    long.to_parquet(OUT / "frame.parquet", index=False)
    long.to_parquet(OUT / "bar_features.parquet", index=False)
    fill_cols = ["timestamp", "signal", "side", "sd", "fi", "ei", "btf", "lim", "atr_pct", "atr_abs", "y_frac", "split", "l2_sample"] + F3
    long[fill_cols].rename(columns={"y_frac": "y"}).to_csv(OUT / "fills.csv", index=False)
    (OUT / "model_card.json").write_text(json.dumps({"feature_cols": F3, "arms": arms, "label_cell": LABEL_CELL,
                                                     "forward_bars": FORWARD_BARS, "cost_bp": COST_BP, "material_stats": mstats},
                                                    indent=2, ensure_ascii=False))
    gate_cfg = {
        "pipeline": "homer_entry_v2_20260904 -- 매 봉 × 양방향, 경제라벨(sim_exit 5.0/1.5/0.1, 200봉, 10bp), 다음 봉 시가 진입",
        "splits": {"VAL": "2025-09-01", "OOS": "2026-01-01", "HOLDOUT": "2026-04-01"},
        "known_ts": {"assumption": "모집단은 전체 봉(트리거 없음): 결정은 봉 τ 마감 후, 진입 open[τ+1] (fi=pos+1, btf=1). "
                                   "재료(F2/F3)가 쓰는 증거신호 발동은 라이브 compute_signals() raw 단일봉(L1로 확인)"},
        "label": {"fills": "tmp/homer_entry_v2_20260904/fills.csv", "ts_col": "timestamp", "y_col": "y", "entry_col": "lim",
                  "side_col": "sd", "atr_col": "atr_abs", "atr_is_absolute": True, "fill_idx_col": "fi", "exit_idx_col": "ei",
                  "bars_to_fill_col": "btf", "signal_col": "signal", "row_filter": "l2_sample == 1",
                  "exit": {"sl_atr": LABEL_CELL[0], "arm_atr": LABEL_CELL[1], "trail_atr": LABEL_CELL[2], "trail_anchor": "entry"},
                  "cost_roundtrip": COST_BP / 1e4, "notional": 1.0, "tol_mean_bp": 2.0, "tol_winrate_pp": 2.0},
        "trigger": {"module": "gate_eth_entry_triggers_v1_adapter_20260903", "fn": "build_fires", "warmup_bars": 4000, "sample_n": 120},
        "features": {"table": "tmp/homer_entry_v2_20260904/bar_features.parquet", "ts_col": "timestamp", "y_col": "y",
                     "cols_from_model_card": "tmp/homer_entry_v2_20260904/model_card.json",
                     "stacked": [{"col": f"{s}_pct", "source_col": f"{s}_oof_src"} for s in SIGNALS]
                                + [{"col": "reg_eth_chop", "source_col": "regime_eth_oof_src"},
                                   {"col": "reg_btc_chop", "source_col": "regime_btc_oof_src"}]},
        "scoring_parity": {"backtest": {"module": "scripts/research_homer_entry_v2_20260904.py", "fn": "trail_single", "style": "from_fill_bar"},
                           "live": {"adapter": "eth_v_rebound_econ_shadow"}, "n_paths": 300, "horizon_bars": 24},
        "controls": {"report": "tmp/homer_entry_v2_20260904/controls_hgb.json"},
        "selection": {"keep_frac": TOP_FRAC, "labels": ["y", "@recon"]},
        "seed": 20260904,
    }
    (OUT / "gate_config.json").write_text(json.dumps(gate_cfg, indent=2, ensure_ascii=False))
    log(f"build 완료 -> {OUT}  ({time.time()-t0:.0f}s)")


# ----------------------------------------------------------------------------- 평가 도구
def portfolio(cand, max_conc):
    """진입봉 순서대로 슬롯 제약 하에 체결 (경제라벨 모델 원문)."""
    cand = cand.sort_values(["entry_bar", "p"], ascending=[True, False])
    eb = cand["entry_bar"].to_numpy(); xb = cand["exit_bar"].to_numpy(); pn = cand["pnl_bp"].to_numpy()
    open_until, taken = [], []
    for k in range(len(cand)):
        open_until = [u for u in open_until if u > eb[k]]
        if len(open_until) < max_conc:
            open_until.append(xb[k]); taken.append(k)
    if not taken:
        return None
    t = cand.iloc[taken]; p = t["pnl_bp"].to_numpy()
    eq = np.cumsum(p); dd = eq - np.maximum.accumulate(eq)
    losses = (p <= 0).astype(int); mcl = cur = 0
    for x in losses:
        cur = cur + 1 if x else 0; mcl = max(mcl, cur)
    w = p > 0
    return {"n": int(len(p)), "exp_bp": float(p.mean()), "total_bp": float(p.sum()), "win_rate": float(w.mean()),
            "payoff": float(p[w].mean() / -p[~w].mean()) if w.any() and (~w).any() else None,
            "max_dd_bp": float(dd.min()), "max_consec_loss": int(mcl), "trades": t}


def resolve_both_sides(sel):
    """같은 봉 양방향 동시 호출 -> 확률 높은 쪽만, 동률 스킵 (라이브 규격 T5)."""
    g = sel.sort_values("p", ascending=False).drop_duplicates("pos", keep="first")
    dup = sel.groupby("pos")["p"].agg(["max", "count", "nunique"])
    tie = dup.index[(dup["count"] > 1) & (dup["nunique"] == 1)]
    return g.loc[~g["pos"].isin(tie)]


def calls_frame(s, p, cut):
    sel = s.assign(p=p).loc[lambda d: d["p"] >= cut]
    sel = resolve_both_sides(sel)
    return pd.DataFrame({"timestamp": sel["timestamp"].to_numpy(), "pos": sel["pos"].to_numpy(), "p": sel["p"].to_numpy(),
                         "entry_bar": sel["pos"].to_numpy() + 1, "exit_bar": sel["pos"].to_numpy() + 1 + sel["exit_off"].to_numpy(),
                         "pnl_bp": sel["net_bp"].to_numpy(), "pnl_flip_bp": sel["net_bp_flip"].to_numpy(),
                         "side": sel["side"].to_numpy()})


def day_boot(pnl, ts, B, rng):
    d = pd.Series(np.asarray(pnl, float), index=pd.DatetimeIndex(pd.to_datetime(np.asarray(ts))).normalize())
    days = d.index.unique().to_numpy(); per = d.groupby(level=0)
    sums = per.sum().reindex(days).to_numpy(); cnts = per.count().reindex(days).to_numpy()
    out = np.empty(B)
    for b in range(B):
        k = rng.integers(0, len(days), len(days)); out[b] = sums[k].sum() / max(cnts[k].sum(), 1)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def delta_day_boot(pnl_a, ts_a, pnl_b, ts_b, B, rng):
    """팔 A − F0(B)의 일 단위 페어드 부트스트랩: 일별 평균 bp 차이(어느 쪽이든 거래 없는 날은 0으로)."""
    da = pd.Series(np.asarray(pnl_a, float), index=pd.DatetimeIndex(pd.to_datetime(np.asarray(ts_a))).normalize()).groupby(level=0).mean()
    db = pd.Series(np.asarray(pnl_b, float), index=pd.DatetimeIndex(pd.to_datetime(np.asarray(ts_b))).normalize()).groupby(level=0).mean()
    days = da.index.union(db.index); diff = (da.reindex(days).fillna(0) - db.reindex(days).fillna(0)).to_numpy()
    out = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)])
    return float(diff.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def stats_of(r):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "trades"}


# ----------------------------------------------------------------------------- eval
def stage_eval(learner):
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    D = pd.read_parquet(OUT / "frame.parquet"); card = json.loads((OUT / "model_card.json").read_text()); arms = card["arms"]
    tr = D.loc[D["split"] == "TRAIN"].reset_index(drop=True)
    S = {w: D.loc[D["split"] == w].reset_index(drop=True) for w in ("VAL", "OOS")}
    log(f"학습기 {learner} · TRAIN {len(tr):,} VAL {len(S['VAL']):,} OOS {len(S['OOS']):,} · 팔 {list(arms)}")
    if learner == "tabpfn":
        from tabpfn import TabPFNClassifier
        import torch
        log(f"cuda {torch.cuda.is_available()} · mem_free {torch.cuda.mem_get_info()[0]/1e9:.2f}GB")
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
    probs = {}                                                   # arm -> split -> (n_seeds, n)
    for arm, cols in arms.items():
        probs[arm] = {w: [] for w in S}
        for sd in SEEDS:
            if learner == "tabpfn":
                rng = np.random.default_rng(sd)
                ctx = tr.iloc[np.sort(rng.choice(len(tr), size=min(CONTEXT_N, len(tr)), replace=False))]
                clf = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
                clf.fit(ctx[cols], ctx["y"].to_numpy())
            else:
                clf = HistGradientBoostingClassifier(**HGB_HP, random_state=sd).fit(tr[cols], tr["y"].to_numpy())
            for w, s in S.items():
                probs[arm][w].append(np.concatenate([clf.predict_proba(s[cols].iloc[k:k + CHUNK])[:, 1] for k in range(0, len(s), CHUNK)]))
            log(f"  {arm} seed {sd} 완료 ({time.time()-t0:.0f}s)")
        for w in S:
            probs[arm][w] = np.vstack(probs[arm][w])

    rng = np.random.default_rng(20260904)
    report = {"learner": learner, "seeds": SEEDS, "arms": {}, "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False, "holdout_touched": False,
              "selection": f"VAL top {TOP_FRAC:.0%} of ensemble p per arm -> OOS once", "max_concurrent": MAX_CONC, "cost_bp": COST_BP}
    controls = {"base": {}, "random_subsample": {}, "extra_passed": []}
    taken = {}
    daily = {}
    for arm in arms:
        P = {w: probs[arm][w].mean(axis=0) for w in S}
        cut = float(np.quantile(P["VAL"], 1 - TOP_FRAC))
        R = {"cut": round(cut, 4), "auc": {w: round(float(roc_auc_score(S[w]["y"], P[w])), 4) for w in S}, "windows": {}}
        taken[arm] = {}
        for w in S:
            cand = calls_frame(S[w], P[w], cut)
            r = portfolio(cand, MAX_CONC)
            if r is None:
                R["windows"][w] = {"error": "no trades"}; continue
            t = r["trades"]; taken[arm][w] = t
            flip = portfolio(cand.assign(pnl_bp=cand["pnl_flip_bp"]), MAX_CONC)
            # 랜덤 부분표집 귀무: 모델 확률 무시, 같은 크기 n의 무작위 호출 (양방향 규칙 동일)
            n_calls = len(cand); nulls = []
            for _ in range(B_NULL):
                rp = rng.random(len(S[w])); rcut = np.quantile(rp, 1 - n_calls / len(S[w]))
                rr = portfolio(calls_frame(S[w], rp, rcut), MAX_CONC)
                nulls.append(rr["exp_bp"] if rr else np.nan)
            nulls = np.asarray(nulls, float)
            lo, hi = day_boot(t["pnl_bp"], t["timestamp"], B_BOOT, rng)
            mo = pd.Series(t["pnl_bp"].to_numpy(), index=pd.to_datetime(t["timestamp"].to_numpy())).groupby(lambda x: x.to_period("M")).mean()
            per_seed = []
            for k in range(len(SEEDS)):
                pv = probs[arm]["VAL"][k]; ck = float(np.quantile(pv, 1 - TOP_FRAC))
                rk = portfolio(calls_frame(S[w], probs[arm][w][k], ck), MAX_CONC)
                per_seed.append(round(rk["exp_bp"], 3) if rk else None)
            R["windows"][w] = {**stats_of(r), "n_calls": int(n_calls), "flip": stats_of(flip) if flip else None,
                               "null_mean_bp": round(float(np.nanmean(nulls)), 3), "null_p95_bp": round(float(np.nanpercentile(nulls, 95)), 3),
                               "null_percentile": round(float((nulls < r["exp_bp"]).mean() * 100), 1),
                               "day_cluster_ci95": [round(lo, 3), round(hi, 3)], "monthly_exp_bp": {str(k): round(float(v), 2) for k, v in mo.items()},
                               "per_seed_exp_bp": per_seed, "seeds_positive": int(sum(1 for v in per_seed if v is not None and v > 0)),
                               "side_share_long": round(float((t["side"] == "bottom").mean()), 3)}
            controls["base"][f"{arm}/{w}"] = r["exp_bp"]; controls["random_subsample"][f"{arm}/{w}"] = [float(x) for x in nulls if np.isfinite(x)]
            if flip and r["exp_bp"] > flip["exp_bp"]:
                controls["extra_passed"].append(f"flip/{arm}/{w}")
            if w == "OOS":
                daily[arm] = pd.Series(t["pnl_bp"].to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(t["timestamp"].to_numpy())).normalize()).groupby(level=0).sum()
        report["arms"][arm] = R
        log(f"  {arm} cut {cut:.4f} " + " ".join(f"{w}: exp {R['windows'][w].get('exp_bp')} n {R['windows'][w].get('n')} "
                                                 f"flip {R['windows'][w].get('flip', {}) and R['windows'][w]['flip'].get('exp_bp')} "
                                                 f"null%{R['windows'][w].get('null_percentile')} seeds+{R['windows'][w].get('seeds_positive')}" for w in S))

    # Δ vs F0 (일 단위 페어드) + 사전등록 판정
    for arm in arms:
        if arm == "F0":
            continue
        R = report["arms"][arm]; R["delta_vs_F0"] = {}
        for w in S:
            if w in taken[arm] and w in taken["F0"]:
                m, lo, hi = delta_day_boot(taken[arm][w]["pnl_bp"], taken[arm][w]["timestamp"], taken["F0"][w]["pnl_bp"], taken["F0"][w]["timestamp"], B_BOOT, rng)
                R["delta_vs_F0"][w] = {"mean_bp": round(m, 3), "ci95": [round(lo, 3), round(hi, 3)]}
        Wv, Wo = R["windows"].get("VAL", {}), R["windows"].get("OOS", {})
        dv, do = R["delta_vs_F0"].get("VAL", {}), R["delta_vs_F0"].get("OOS", {})
        ok = (Wv.get("exp_bp", -1) > 0 and Wo.get("exp_bp", -1) > 0
              and Wv.get("flip") and Wv["exp_bp"] > Wv["flip"]["exp_bp"] and Wo.get("flip") and Wo["exp_bp"] > Wo["flip"]["exp_bp"]
              and Wv.get("null_percentile", 0) >= 95 and Wo.get("null_percentile", 0) >= 95
              and dv.get("ci95", [-1])[0] > 0 and do.get("mean_bp", -1) > 0 and Wo.get("seeds_positive", 0) >= 4)
        R["prereg_pass"] = bool(ok)
    # DSR / PBO (팔 4개 = trial)
    try:
        from core.selection_stats import deflated_sharpe_ratio, pbo_cscv
        days = sorted(set().union(*[d.index for d in daily.values()]))
        M = np.column_stack([daily[a].reindex(days).fillna(0.0).to_numpy() for a in arms if a in daily])
        best_val = max(arms, key=lambda a: report["arms"][a]["windows"].get("VAL", {}).get("exp_bp", -1e9))
        sr = lambda v: float(np.mean(v) / np.std(v, ddof=1)) if np.std(v, ddof=1) > 0 else 0.0
        dsr = deflated_sharpe_ratio(daily[best_val].reindex(days).fillna(0.0).to_numpy(), np.array([sr(M[:, j]) for j in range(M.shape[1])]))
        pbo = pbo_cscv(M, n_splits=10) if M.shape[0] >= 30 else {"error": "too few days"}
        report["multiple_testing"] = {"best_by_VAL": best_val, "dsr": dsr, "pbo": pbo}
        controls["dsr"] = dsr.get("deflated_sharpe_ratio"); controls["pbo"] = pbo.get("pbo")
    except Exception as ex:                                      # noqa: BLE001
        report["multiple_testing"] = {"error": f"{type(ex).__name__}: {ex}"}
    (OUT / f"report_{learner}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    (OUT / f"controls_{learner}.json").write_text(json.dumps(controls, default=float))
    # 게이트 T1이 읽는 controls 선언을 이 학습기의 산출로 갱신 (DSR/PBO 포함)
    gc_path = OUT / "gate_config.json"; gc = json.loads(gc_path.read_text())
    gc["controls"] = {"report": f"tmp/homer_entry_v2_20260904/controls_{learner}.json",
                      "dsr": controls.get("dsr"), "pbo": controls.get("pbo")}
    gc_path.write_text(json.dumps(gc, indent=2, ensure_ascii=False))
    for arm in arms:
        for w in S:
            if w in taken[arm]:
                taken[arm][w].to_csv(OUT / f"trades_{learner}_{arm}_{w}.csv", index=False)
    log(f"eval 완료 -> report_{learner}.json ({time.time()-t0:.0f}s)")
    for arm, R in report["arms"].items():
        log(f"  {arm}: " + " | ".join(f"{w} exp {R['windows'][w].get('exp_bp')}bp n {R['windows'][w].get('n')} null%{R['windows'][w].get('null_percentile')}" for w in S)
            + (f" | ΔF0 {R.get('delta_vs_F0')} | prereg {R.get('prereg_pass')}" if arm != "F0" else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "eval"], required=True)
    ap.add_argument("--learner", choices=["hgb", "tabpfn"], default="hgb")
    a = ap.parse_args()
    stage_build() if a.stage == "build" else stage_eval(a.learner)
