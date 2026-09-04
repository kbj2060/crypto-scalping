#!/usr/bin/env python3
"""ETH exit-timing **Phase 0 킬게이트** -- 모집단 = raw 인과 증거신호 발동 (2026-09-04).

WHY
---
②(청산/레짐/리스크)의 첫 작업. 09-03 설계(`docs/eth_rl_exit_timing_agent_design_20260903.md` §3)의
킬게이트를 그대로 쓰되, 모집단만 바꾼다. 원안은 h48qual/Omega 원시 진입후보였는데 사용자 지적대로
Omega 라이브는 시드 검증(3시드 6창 중 4창 부호반전)·DSR 0.88~0.92·PBO 0.44로 **엣지가 입증된 적이
없는 스택**이라 그 포지션에 맞춘 청산 모델은 배포 대상이 없고 h48qual 브라켓 기하/방향 편향이 피쳐에
박힌다. 원안이 Omega를 고른 유일한 이유("증거신호 진입은 앵커 미래참조로 막힘")는 09-02 raw 인과
모집단 + 09-03 L1 게이트 통과로 사라졌다.

  질문: "대시보드 증거신호가 발동한 자리에서 (재량으로) 들어간 포지션을, 재료 텐서/레짐이
         '끝까지 보유'보다 더 잘 닫게 해주는가?"

모집단 · 포지션 구조 (사전등록)
-------------------------------
- 발동: `tmp/eth_causal_population_metalabel_20260902/<sig>_causal_fires.csv` 8종 × 양측 (cluster_dedup 없음,
  L1 인과성 2026-09-03 확인). 발동 봉 T는 T 마감에 알려진다(known_ts = T).
- 진입: **다음 봉 시가** open[i+1] (봉 경계 진입 → 체결봉 크레딧(B형) 문제 없음). 방향 bottom=+1 / top=-1.
- 구조: 신호별 horizon H(config, 8~72봉) 타임아웃 + **하드 SL 3.0×atr_pct(발동행)** (intrabar 고저 판정 =
  라이브 컨벤션) · TP 없음 · 트레일링 없음 -- 청산 모델이 그 자리를 대신하는지를 묻는다.
- 창: 재료 OOF 워밍업(<2024-05-01) 제외 · TRAIN [2024-05-01, 2025-09-01) · VAL [2025-09-01, 2026-01-01) ·
  OOS [2026-01-01, 2026-04-01) · **HOLDOUT(≥2026-04-01) 미접촉**. 경계를 넘는 포지션은 버린다.
- 체크포인트: 실제 보유기간의 내부 10분위 9개(스모크 규약). 상태 = 직전 마감봉까지의 종가 기반 pos_*
  (CLAUDE.md 파리티: 가격변동 원시값). 결정 시각 = 방금 마감한 봉 ts[fi+t-1] -- **시장맥락·레짐·재료도 전부
  그 봉에서** 조인한다. ⚠️09-01 스모크는 맥락을 ts[fi+t](다음 봉)에서 조인했는데 그 행의 log_return/ret_1은
  다음 봉 종가를 담는다 → 한 봉 미래참조. 이 스크립트는 CSV 규약을 실측 검증한 뒤 마감봉에 조인하고,
  스모크 규약(다음 봉)으로 A팔을 한 번 더 학습해 그 한 봉의 크기를 진단으로 남긴다.
- 라벨(오라클): "지금 나가면(다음 봉 시가 체결) 실제 종단 결과보다 나은가" = exit_move_t > terminal_move.
  종단: SL 체결가(-3 ATR) 또는 타임아웃 종가. **학습·AUC는 timeout 포지션만**(SL은 라벨이 정의상 ~1).

세 arm
-----
  A  pos 10 + 시장맥락 19 + 수익률 5 + atr192 + 레짐 one-hot 3(**OOF 레짐** S12_K3 롤링) + 신호 one-hot 8
  B  A + 재료 `<sig>_pct` 8 + `<sig>_age` 8   (`tmp/eth_entry_oof_metalabel_20260903/` OOF본에서 직접 구성 --
     텐서 파일은 로컬·서버 어디에도 없고 빌더는 BTC 레짐 parquet 부재로 실패. 같은 코드로 재구성)
  C  B − 레짐 one-hot, **예측 레짐(OOF)별 모델 3개로 하드 라우팅**
  ※ 설계의 "재료·레짐 ≤2024-12 롤백 재훈련"은 OOF본(확장창 fold 2024-05/09, 2025-01/05, final<2025-09)이
    "평가창 직전까지만 학습" 성질을 이미 가지므로 불필요. 배포 GBM3 레짐 확률(TRAIN in-sample)은 쓰지 않는다.

킬 기준 (사전등록, 설계 §3-4)
---------------------------
  VAL timeout 체크포인트에서 ΔAUC = AUC(B or C) − AUC(A), 5시드 평균확률 기준,
  **후보(발동) 단위 군집 부트스트랩 95% CI 하한 > 0** 이고 **5시드 각각 ΔAUC > 0 (5/5)**.
  못 넘으면 "재료는 매매용 기여 없음(세 번째 0)" 선언, RL 착수 중단.
  보조(보고·경고): 일(day) 단위 군집 CI(09-03 '독립 42~45일' 교훈) · OOS ΔAUC 부호 · anti-stable
  (TRAIN-VAL 피쳐AUC 상관 음수면 중단) · 순열중요도 top15 재료 진입 여부 · 정책평가에서 무작위 exit
  대조와 트레일링(SL3/ARM1/Trail0.1) 비교(설계 실패조건 ③).

⚠️이 결과는 research/dev score다 -- 저장 원장 아님(발동→시뮬 직접), 그러나 timeout 선택은 결과 기준
선택이므로 절대 AUC는 편향될 수 있고(A vs B 상대비교는 공정) 승격 근거가 아니다.

사용: python scripts/research_eth_exit_timing_phase0_killgate_20260904.py [--seeds 5] [--boot 1000] [--quick]
산출: tmp/eth_exit_timing_phase0_20260904/{positions.csv, checkpoints.parquet, report.json, gate_config.json}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

from gate_eth_entry_layers_20260903 import trail  # noqa: E402  -- 저장소 표준 트레일링(비교군)

POP = ROOT / "tmp/eth_causal_population_metalabel_20260902"
OOFD = ROOT / "tmp/eth_entry_oof_metalabel_20260903"
OOF_REGIME = ROOT / "tmp/eth_entry_oof_regime_20260903/regime_oof_eth.parquet"
KL5 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
TF_PATHS = [ROOT / f"data/splits/year_oos/training_features_{y}.csv" for y in ("2024", "2025", "2026_rebuilt")]
OUT = ROOT / "tmp/eth_exit_timing_phase0_20260904"

WARMUP_END, VAL_START = pd.Timestamp("2024-05-01"), pd.Timestamp("2025-09-01")
OOS_START, HOLDOUT_START = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")
SL_ATR = 3.0
COST, COST_MAKER = 0.0010, 0.00078            # 표준 10bp / 09-04 실측 진입peg+청산taker 7.8bp(민감도)
DECILES = [k / 10.0 for k in range(1, 10)]
HGB_PARAMS = dict(max_depth=8, learning_rate=0.05, max_iter=300, l2_regularization=1.0, early_stopping=False)
INTERNAL_HOLDOUT_FRAC, EMBARGO_BARS = 0.20, 288
TRAIL_CMP = dict(sl=3.0, arm=1.0, tr=0.1)
RAW_CTX = ["log_return", "realized_vol_ratio", "garman_klass_vol", "atr_pct_rank_288", "bb_width_z",
           "hour_sin", "hour_cos", "session_europe", "session_us",
           "net_taker_ratio", "taker_acceleration", "cvd_slope_48", "price_cvd_divergence",
           "oi_change_rate", "funding_oi_divergence", "last_funding_rate", "funding_z_score",
           "btc_corr_60", "chop_index"]
RET_H = [1, 3, 6, 12, 24]
POS_COLS = ["pos_side", "pos_hold_bars", "pos_hold_frac", "pos_bars_left", "pos_unrealized", "pos_mfe",
            "pos_mae", "pos_giveback", "pos_dist_to_sl", "pos_sl"]
REGIME_COLS = ["regime_bull", "regime_bear", "regime_chop"]
REGIME_NAME = {0: "bull", 1: "bear", 2: "chop"}
FRAC = [1.0]


def log(m): print(f"[phase0 {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------- 데이터
def load_klines():
    kl = pd.read_csv(KL5, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"])
    kl = kl.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    pc = kl["close"].shift(1)
    tr = pd.concat([kl["high"] - kl["low"], (kl["high"] - pc).abs(), (kl["low"] - pc).abs()], axis=1).max(axis=1)
    kl["atr_pct_192"] = (tr.rolling(192, min_periods=192).mean() / kl["close"]).to_numpy()
    return kl


def load_context():
    use = ["timestamp", "close"] + RAW_CTX
    fr = pd.concat([pd.read_csv(p, usecols=use, parse_dates=["timestamp"]) for p in TF_PATHS], ignore_index=True)
    fr = fr.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    for k in RET_H:
        fr[f"ret_{k}"] = fr["close"].pct_change(k)
    return fr


def check_context_convention(ctx, kl):
    """훈련피쳐 CSV의 행 τ가 **봉 τ 자신의 종가**를 담는가(=τ 마감에야 알 수 있음) 검증.
    log_return(CSV) vs log(close_τ/close_{τ-1})(klines)를 대조한다."""
    m = ctx[["timestamp", "log_return"]].merge(kl[["timestamp", "close"]], on="timestamp", how="inner")
    m = m.sort_values("timestamp").reset_index(drop=True)
    own = np.log(m["close"] / m["close"].shift(1))
    prev = own.shift(1)
    s = m.sample(min(20000, len(m)), random_state=0).index
    r_own = float(np.corrcoef(m.loc[s, "log_return"].fillna(0), own.loc[s].fillna(0))[0, 1])
    r_prev = float(np.corrcoef(m.loc[s, "log_return"].fillna(0), prev.loc[s].fillna(0))[0, 1])
    conv = "row_uses_own_bar_close" if r_own > r_prev else "row_shifted_uses_prev_bar"
    log(f"맥락 CSV 규약 검증: corr(log_return, 자기봉) {r_own:.4f} vs corr(직전봉) {r_prev:.4f} → {conv}")
    return {"corr_own_bar": r_own, "corr_prev_bar": r_prev, "convention": conv}


def load_fires(cfg, kl_pos):
    rows = []
    for name, cc in cfg.items():
        d = pd.read_csv(POP / f"{name}_causal_fires.csv", usecols=["timestamp", "side", "atr_pct"], parse_dates=["timestamp"])
        d["signal"] = name; d["H"] = int(cc["horizon"]); rows.append(d)
    F = pd.concat(rows, ignore_index=True).drop_duplicates(["signal", "timestamp", "side"])
    F = F[(F.timestamp >= WARMUP_END) & (F.timestamp < HOLDOUT_START)].copy()
    assert F.timestamp.max() < HOLDOUT_START, "HOLDOUT 접촉"
    if FRAC[0] < 1.0:                                              # --frac: 배선 점검용 발동 서브샘플
        F = F.sample(frac=FRAC[0], random_state=0)
    F["i"] = F["timestamp"].map(kl_pos)
    n_un = int(F["i"].isna().sum())
    F = F.dropna(subset=["i"]).reset_index(drop=True); F["i"] = F["i"].astype(int)
    F["sd"] = np.where(F["side"] == "bottom", 1, -1)
    F["split"] = np.where(F.timestamp < VAL_START, "TRAIN", np.where(F.timestamp < OOS_START, "VAL", "OOS"))
    return F, n_un


def build_material(cfg, kl):
    """OOF 메타라벨 산출(`<sig>_oof.csv`)에서 봉별 `<sig>_pct`/`<sig>_age`/출처를 만든다 -- 텐서 빌더와 같은 규칙:
    직전 발동이 H봉 안이면 그 pct와 age=경과/H, 아니면 pct 0 · age 1. 출처(fold/final)는 L3 OOF 점검용."""
    n = len(kl); pos = {t: i for i, t in enumerate(pd.DatetimeIndex(kl["timestamp"]))}
    out = {}; stats = {}
    for name, cc in cfg.items():
        H = int(cc["horizon"])
        d = pd.read_csv(OOFD / f"{name}_oof.csv", parse_dates=["timestamp"])
        d = d[np.isfinite(d["pct_oof"])].copy()
        d["i"] = d["timestamp"].map(pos); d = d.dropna(subset=["i"]); d["i"] = d["i"].astype(int)
        d = d.sort_values(["i", "proba_oof"]).drop_duplicates("i", keep="last")
        src = d["oof_source"].fillna("").astype(str).str.replace(r"\(.*", "", regex=True).to_numpy()
        fire_i = d["i"].to_numpy(); pct_v = d["pct_oof"].to_numpy(float)
        last = np.full(n, -10**9, dtype=np.int64); last[fire_i] = fire_i
        last = np.maximum.accumulate(last)                          # 각 봉에서 직전(포함) 발동 인덱스
        el = np.arange(n) - last; active = el < H
        idx_of = np.full(n, -1, dtype=np.int64); idx_of[fire_i] = np.arange(len(fire_i))
        idx_last = np.where(last >= 0, idx_of[np.clip(last, 0, n - 1)], -1)
        pct = np.where(active & (idx_last >= 0), pct_v[np.clip(idx_last, 0, len(pct_v) - 1)], 0.0)
        age = np.where(active, el / H, 1.0)
        s_arr = np.where(active & (idx_last >= 0), src[np.clip(idx_last, 0, len(src) - 1)], "")
        out[f"{name}_pct"] = pct; out[f"{name}_age"] = age; out[f"{name}_oof_src"] = s_arr
        stats[name] = {"H": H, "n_fires_oof": int(len(d)), "bar_coverage": round(float(active.mean()), 4)}
    M = pd.DataFrame(out); M.insert(0, "timestamp", kl["timestamp"].to_numpy())
    return M, stats


# ----------------------------------------------------------------------------- 포지션·체크포인트
def simulate_positions(F, kl):
    o, h, l, c = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close")); n = len(kl)
    ts = kl["timestamp"].to_numpy()
    rec = []
    for r in F.itertuples(index=False):
        fi, H, sd, a = int(r.i) + 1, int(r.H), int(r.sd), float(r.atr_pct)
        if fi + H > n or not np.isfinite(a) or a <= 0:
            continue
        e = o[fi]; sl = SL_ATR * a
        stop = e * (1 - sl) if sd > 0 else e * (1 + sl)
        hh, ll, cc = h[fi:fi + H], l[fi:fi + H], c[fi:fi + H]
        hit = np.flatnonzero(ll <= stop) if sd > 0 else np.flatnonzero(hh >= stop)
        if len(hit):
            bars, reason, term = int(hit[0]) + 1, "sl", -sl
        else:
            bars, reason, term = H, "timeout", sd * (cc[-1] / e - 1.0)
        end_ts = pd.Timestamp(ts[fi + H - 1])
        # 경계를 넘는 포지션은 버린다 (분할 순수성)
        if (r.split == "TRAIN" and end_ts >= VAL_START) or (r.split == "VAL" and end_ts >= OOS_START) or \
           (r.split == "OOS" and end_ts >= HOLDOUT_START):
            continue
        tr_mv = trail(sd, e, a, hh, ll, cc, TRAIL_CMP["sl"], TRAIL_CMP["arm"], TRAIL_CMP["tr"])
        rec.append((r.timestamp, r.signal, r.side, sd, int(r.i), fi, fi + bars, e, a, sl, H, bars, reason, term, tr_mv, r.split))
    P = pd.DataFrame(rec, columns=["timestamp", "signal", "side", "sd", "i", "fi", "ei", "lim", "atr_pct", "sl_move", "H",
                                   "bars_held", "reason", "terminal_move", "trail_move", "split"])
    P["btf"] = 1
    P["y"] = P["terminal_move"] - COST                      # 종단 net move (게이트 L4용 라벨 열)
    P["pid"] = np.arange(len(P))
    return P


def build_checkpoints(P, kl):
    o, c = kl["open"].to_numpy(float), kl["close"].to_numpy(float); ts = kl["timestamp"].to_numpy()
    rows = []
    for r in P.itertuples(index=False):
        fi, B, H, sd, e, sl = int(r.fi), int(r.bars_held), int(r.H), int(r.sd), float(r.lim), float(r.sl_move)
        cps = sorted({int(round(B * f)) for f in DECILES}); cps = [t for t in cps if 1 <= t < B]
        if not cps:
            continue
        mv = sd * (c[fi:fi + cps[-1]] / e - 1.0)             # mv[k] = 봉 fi+k 마감 기준 (k=0: 진입봉)
        rmax = np.maximum.accumulate(np.concatenate([[0.0], mv])); rmin = np.minimum.accumulate(np.concatenate([[0.0], mv]))
        for t in cps:
            u = float(mv[t - 1]); mfe = float(rmax[t]); mae = float(rmin[t])
            gb = (mfe - u) / max(abs(mfe), 1e-8) if mfe > 0 else 0.0
            ex_mv = sd * (o[fi + t] / e - 1.0)                # 지금 나가면: 다음 봉 시가 체결
            rows.append((r.pid, ts[fi + t - 1], ts[fi + t], t, float(sd), float(t), t / H, float(H - t), u, mfe, mae,
                         float(np.clip(gb, 0, 10)), u + sl, sl, ex_mv, int(ex_mv > r.terminal_move)))
    C = pd.DataFrame(rows, columns=["pid", "ts_dec", "ts_next", "t", *POS_COLS, "exit_move", "label"])
    return C


def internal_split(P_train):
    d = P_train.sort_values("timestamp").reset_index(drop=True); k = int(len(d) * (1 - INTERNAL_HOLDOUT_FRAC))
    hs = d.loc[k, "timestamp"]; es = hs - pd.Timedelta(minutes=5 * EMBARGO_BARS)
    return set(d[d.timestamp < es].pid), set(d[d.timestamp >= hs].pid)


def fit(X, y, seed):
    m = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=int(seed))
    m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return m


def fit_routed(X, y, reg, seed):
    models = {}
    for g in (0, 1, 2):
        mk = reg == g
        if mk.sum() < 500 or len(np.unique(y[mk])) < 2:
            continue
        models[g] = fit(X[mk], y[mk], seed)
    return models


def predict_routed(models, X, reg, fallback):
    p = fallback.copy()
    for g, m in models.items():
        mk = reg == g
        if mk.any():
            p[mk] = m.predict_proba(X[mk])[:, 1]
    return p


def cluster_boot(y, pa, pb, groups, B, seed):
    rng = np.random.default_rng(seed)
    idx_by = {k: np.asarray(v) for k, v in pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).agg(list).items()}
    keys = list(idx_by.keys()); out = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        ix = np.concatenate([idx_by[keys[k]] for k in pick])
        if len(np.unique(y[ix])) < 2:
            continue
        out.append(roc_auc_score(y[ix], pb[ix]) - roc_auc_score(y[ix], pa[ix]))
    out = np.asarray(out)
    return {"ci95": [round(float(np.percentile(out, 2.5)), 5), round(float(np.percentile(out, 97.5)), 5)],
            "mean": round(float(out.mean()), 5), "n_clusters": len(keys), "B": int(len(out))}


def single_auc(X, y):
    out = {}
    for j, col in enumerate(X.columns):
        v = X[col].to_numpy(float); m = np.isfinite(v)
        if m.sum() < 200 or len(np.unique(y[m])) < 2 or np.nanstd(v[m]) == 0:
            continue
        out[col] = roc_auc_score(y[m], v[m])
    return out


def policy_eval(Pw, Cw, prob_by_arm, thr_by_arm, rng):
    """포지션별 net bp: 기준(끝까지) · 모델 첫 트리거 exit · 무작위 exit(같은 트리거율) · 트레일링."""
    res = {"10bp": {}, "7.8bp": {}}
    base = Pw["terminal_move"].to_numpy(); trail_mv = Pw["trail_move"].to_numpy()
    pid_to_row = {p: k for k, p in enumerate(Pw["pid"].to_numpy())}
    Cw = Cw.copy()
    for arm, prob in prob_by_arm.items():                    # 확률은 호출자의 Cw 행 순서와 정렬돼 있다 -- 정렬 전에 붙인다
        Cw[f"_p_{arm}"] = np.asarray(prob)
    Cw = Cw.sort_values(["pid", "t"])
    ex_lists = Cw.groupby("pid", sort=False)["exit_move"].agg(list)          # pid -> [exit_move per checkpoint]
    pids = ex_lists.index.to_numpy(); ex_arr = [np.asarray(v) for v in ex_lists.to_numpy()]
    pos_of = np.array([pid_to_row[p] for p in pids])
    def summ(mv):
        out = {}
        for cost, tag in ((COST, "10bp"), (COST_MAKER, "7.8bp")):
            net = (mv - cost) * 1e4
            out[tag] = {"n": int(len(net)), "mean_bp": round(float(net.mean()), 3), "median_bp": round(float(np.median(net)), 3),
                        "winrate": round(float((net > 0).mean()), 4)}
        return out
    for tag, s in summ(base).items(): res[tag]["hold_to_terminal"] = s
    for tag, s in summ(trail_mv).items(): res[tag]["trailing_SL3_ARM1_T0.1"] = s
    for arm in prob_by_arm:
        Cw["_trig"] = Cw[f"_p_{arm}"].to_numpy() >= thr_by_arm[arm]
        tr_lists = Cw.groupby("pid", sort=False)["_trig"].agg(list)
        mv = base.copy(); n_trig = 0
        for k, (ems, tg) in enumerate(zip(ex_arr, tr_lists.to_numpy())):
            hit = np.flatnonzero(np.asarray(tg))
            if len(hit):
                mv[pos_of[k]] = ems[hit[0]]; n_trig += 1
        for tag, s in summ(mv).items():
            s["trigger_rate"] = round(n_trig / len(Pw), 4); res[tag][f"model_{arm}"] = s
        # 무작위 exit 대조: 같은 트리거율로 무작위 체크포인트에서 나감 (5회 평균)
        p_trig = n_trig / len(Pw); acc = []
        for _ in range(5):
            mvr = base.copy()
            for k, ems in enumerate(ex_arr):
                if rng.random() < p_trig:
                    mvr[pos_of[k]] = ems[rng.integers(0, len(ems))]
            acc.append(mvr)
        for tag, s in summ(np.mean(acc, axis=0)).items():
            res[tag][f"random_exit_matched_{arm}"] = s
    return res


# ----------------------------------------------------------------------------- 메인
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--quick", action="store_true", help="시드 1·부트 100 (배선 점검용)")
    ap.add_argument("--frac", type=float, default=1.0, help="발동 서브샘플 비율 (배선 점검용, 결과는 보고에 쓰지 말 것)")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.boot = 1, 100
    FRAC[0] = a.frac
    global OUT
    if a.out:
        OUT = ROOT / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    SEEDS = sorted(int(x) for x in np.random.default_rng(20260904).integers(1, 10**6, a.seeds))
    log(f"시드(무작위 추출) {SEEDS} · 부트스트랩 {a.boot}")
    cfg = json.loads((POP / "config.json").read_text())["cfg"]; SIGS = list(cfg)

    kl = load_klines(); kl_pos = pd.Series(np.arange(len(kl)), index=kl["timestamp"].to_numpy())
    ctx = load_context(); conv = check_context_convention(ctx, kl)
    F, n_un = load_fires(cfg, kl_pos)
    log(f"발동 {len(F):,} (klines 미매칭 {n_un}) · split {F.split.value_counts().to_dict()}")
    P = simulate_positions(F, kl)
    log(f"포지션 {len(P):,} · reason {P.reason.value_counts().to_dict()} · split {P.split.value_counts().to_dict()}")
    C = build_checkpoints(P, kl)
    log(f"체크포인트 {len(C):,} · 라벨 기저율 {C.label.mean():.4f}")

    # ── 조인: 전부 결정 시각(방금 마감한 봉) ts_dec 기준
    M, mstats = build_material(cfg, kl)
    R = pd.read_parquet(OOF_REGIME)[["timestamp", "regime_oof"]]
    C = C.merge(ctx.drop(columns=["close"]).rename(columns={"timestamp": "ts_dec"}), on="ts_dec", how="left")
    C = C.merge(kl[["timestamp", "atr_pct_192"]].rename(columns={"timestamp": "ts_dec"}), on="ts_dec", how="left")
    C = C.merge(M.rename(columns={"timestamp": "ts_dec"}), on="ts_dec", how="left")
    C = C.merge(R.rename(columns={"timestamp": "ts_dec"}), on="ts_dec", how="left")
    # 스모크 규약 진단용: 맥락을 **다음 봉**(ts_next)에서 조인한 사본 열 (미래참조 크기 측정)
    nxt = ctx[["timestamp"] + RAW_CTX + [f"ret_{k}" for k in RET_H]].rename(columns={"timestamp": "ts_next"})
    nxt = nxt.rename(columns={c: f"NEXT_{c}" for c in nxt.columns if c != "ts_next"})
    C = C.merge(nxt, on="ts_next", how="left")
    for g, nm in REGIME_NAME.items():
        C[f"regime_{nm}"] = (C["regime_oof"] == g).astype(float)
    for s in SIGS:
        C[f"sig_{s}"] = 0.0
    pid_sig = P.set_index("pid")["signal"]; C["signal"] = C["pid"].map(pid_sig)
    for s in SIGS:
        C.loc[C["signal"] == s, f"sig_{s}"] = 1.0
    C["split"] = C["pid"].map(P.set_index("pid")["split"]); C["reason"] = C["pid"].map(P.set_index("pid")["reason"])
    C["day"] = pd.DatetimeIndex(C["ts_dec"]).floor("D")
    miss_ctx = float(C["log_return"].isna().mean()); miss_reg = float((C["regime_oof"].fillna(-1) < 0).mean())
    log(f"조인 결측: 맥락 {miss_ctx:.4f} · 레짐 {miss_reg:.4f} · 재료 커버리지 " +
        " ".join(f"{s[:6]}={mstats[s]['bar_coverage']:.2f}" for s in SIGS))

    CTX_COLS = RAW_CTX + [f"ret_{k}" for k in RET_H] + ["atr_pct_192"]
    SIG_COLS = [f"sig_{s}" for s in SIGS]
    MAT_COLS = [f"{s}_pct" for s in SIGS] + [f"{s}_age" for s in SIGS]
    FEAT = {"A": POS_COLS + CTX_COLS + REGIME_COLS + SIG_COLS,
            "B": POS_COLS + CTX_COLS + REGIME_COLS + SIG_COLS + MAT_COLS,
            "C": POS_COLS + CTX_COLS + SIG_COLS + MAT_COLS,
            "A_smokejoin": POS_COLS + [f"NEXT_{c}" for c in RAW_CTX + [f"ret_{k}" for k in RET_H]] + ["atr_pct_192"] + REGIME_COLS + SIG_COLS}

    # ── 학습 표본: timeout 포지션만 (사전등록)
    T = C[(C.split == "TRAIN") & (C.reason == "timeout")].copy()
    V = C[(C.split == "VAL") & (C.reason == "timeout")].copy()
    O = C[(C.split == "OOS") & (C.reason == "timeout")].copy()
    fit_pids, hold_pids = internal_split(P[(P.split == "TRAIN") & (P.reason == "timeout")])
    Tf, Th = T[T.pid.isin(fit_pids)], T[T.pid.isin(hold_pids)]
    log(f"timeout 체크포인트 TRAIN {len(T):,}(fit {len(Tf):,}/hold {len(Th):,}) · VAL {len(V):,} · OOS {len(O):,} · "
        f"포지션 TRAIN {T.pid.nunique():,} VAL {V.pid.nunique():,} OOS {O.pid.nunique():,}")

    def X(df, cols): return df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    yT, yV, yO = T["label"].to_numpy(), V["label"].to_numpy(), O["label"].to_numpy()
    regT, regV, regO = (d["regime_oof"].fillna(-1).to_numpy(int) for d in (T, V, O))

    probs = {"VAL": {k: [] for k in FEAT}, "OOS": {k: [] for k in FEAT}, "HOLD": {k: [] for k in FEAT}}
    per_seed_auc = []; m0 = {}; perm_top15 = []
    for si, seed in enumerate(SEEDS):
        row = {"seed": seed}
        for arm in ("A", "B", "C") + (("A_smokejoin",) if si == 0 else ()):
            t0 = time.time(); cols = FEAT[arm]
            if arm == "C":
                models = fit_routed(X(T, cols).to_numpy(), yT, regT, seed)
                fbA = probs["VAL"]["A"][-1]; fbO = probs["OOS"]["A"][-1]; fbH = probs["HOLD"]["A"][-1]
                pV = predict_routed(models, X(V, cols).to_numpy(), regV, fbA)
                pO = predict_routed(models, X(O, cols).to_numpy(), regO, fbO)
                pH = predict_routed(models, X(Th, cols).to_numpy(), Th["regime_oof"].fillna(-1).to_numpy(int), fbH)
                row["C_models"] = {REGIME_NAME[g]: int((regT == g).sum()) for g in models}
                if si == 0:
                    m0["C"] = models
            else:
                m = fit(X(T, cols), yT, seed)
                pV = m.predict_proba(X(V, cols))[:, 1]; pO = m.predict_proba(X(O, cols))[:, 1]; pH = m.predict_proba(X(Th, cols))[:, 1]
                if si == 0:
                    m0[arm] = m
                if arm == "B" and si == 0:
                    mh = fit(X(Tf, cols), Tf["label"].to_numpy(), seed)   # 순열중요도는 내부 holdout에서
                    pi = permutation_importance(mh, X(Th, cols), Th["label"].to_numpy(), n_repeats=3,
                                                random_state=seed, scoring="roc_auc", n_jobs=4)
                    imp = sorted(zip(cols, pi.importances_mean.tolist()), key=lambda z: -z[1])
                    perm_top15 = [{"feature": f, "auc_drop": round(v, 5)} for f, v in imp[:15]]
            probs["VAL"][arm].append(pV); probs["OOS"][arm].append(pO); probs["HOLD"][arm].append(pH)
            row[f"{arm}_val_auc"] = round(float(roc_auc_score(yV, pV)), 5); row[f"{arm}_oos_auc"] = round(float(roc_auc_score(yO, pO)), 5)
            log(f"  seed {seed} arm {arm:11s} VAL AUC {row[f'{arm}_val_auc']:.4f} OOS {row[f'{arm}_oos_auc']:.4f} ({time.time()-t0:.0f}s)")
        per_seed_auc.append(row)

    # ── 시드 평균 확률 → AUC · ΔAUC · 군집 부트스트랩
    avg = {w: {k: np.mean(v, axis=0) for k, v in probs[w].items() if v} for w in probs}
    auc = {w: {k: round(float(roc_auc_score(y, p)), 5) for k, p in avg[w].items()}
           for w, y in (("VAL", yV), ("OOS", yO), ("HOLD", Th["label"].to_numpy()))}
    log(f"시드평균 AUC VAL {auc['VAL']} · OOS {auc['OOS']}")
    kill = {}
    for arm in ("B", "C"):
        d_seed = [r[f"{arm}_val_auc"] - r["A_val_auc"] for r in per_seed_auc]
        fire_ci = cluster_boot(yV, avg["VAL"]["A"], avg["VAL"][arm], V["pid"].to_numpy(), a.boot, 1)
        day_ci = cluster_boot(yV, avg["VAL"]["A"], avg["VAL"][arm], V["day"].to_numpy(), a.boot, 2)
        oos_d = auc["OOS"][arm] - auc["OOS"]["A"]
        passed = fire_ci["ci95"][0] > 0 and all(x > 0 for x in d_seed) and len(d_seed) == a.seeds
        kill[arm] = {"delta_auc_val_seedavg": round(auc["VAL"][arm] - auc["VAL"]["A"], 5), "delta_auc_per_seed": [round(x, 5) for x in d_seed],
                     "seeds_positive": f"{sum(x > 0 for x in d_seed)}/{len(d_seed)}", "fire_cluster_ci95": fire_ci,
                     "day_cluster_ci95": day_ci, "delta_auc_oos_seedavg": round(oos_d, 5),
                     "PASS_preregistered": bool(passed), "day_ci_also_positive": bool(day_ci["ci95"][0] > 0),
                     "oos_sign_agrees": bool(np.sign(oos_d) == np.sign(auc["VAL"][arm] - auc["VAL"]["A"]))}
        log(f"  {arm}−A: ΔAUC {kill[arm]['delta_auc_val_seedavg']:+.4f} · 시드 {kill[arm]['seeds_positive']} · 발동군집 CI {fire_ci['ci95']} · "
            f"일군집 CI {day_ci['ci95']} · OOS Δ {oos_d:+.4f} → {'PASS' if passed else 'FAIL'}")

    # ── anti-stable: TRAIN vs VAL 단일피쳐 AUC 상관 (B 피쳐)
    sT, sV = single_auc(X(T, FEAT["B"]), yT), single_auc(X(V, FEAT["B"]), yV)
    common = [c for c in sT if c in sV]
    rho = float(spearmanr([sT[c] for c in common], [sV[c] for c in common]).correlation)
    top20 = sorted(common, key=lambda c: -abs(sT[c] - 0.5))[:20]
    keep_sign = int(sum(np.sign(sT[c] - 0.5) == np.sign(sV[c] - 0.5) for c in top20))
    mat_in_top15 = [d["feature"] for d in perm_top15 if d["feature"] in MAT_COLS]
    log(f"anti-stable: TRAIN-VAL 피쳐AUC Spearman ρ {rho:+.3f} · TRAIN top20 부호유지 {keep_sign}/20 · 순열중요도 top15 재료 {len(mat_in_top15)}개 {mat_in_top15}")

    # ── 정책평가 (VAL/OOS · 전체 포지션 & timeout만) -- 시드0 모델 재사용. 모델은 timeout으로 학습됐지만
    #    정책은 모든 포지션(sl 포함)에 적용된다: 결정 시점엔 어느 쪽으로 끝날지 모르기 때문이다.
    thr = {}
    for arm in ("A", "B", "C"):
        ph = probs["HOLD"][arm][0]; br = float(Th["label"].mean()); thr[arm] = float(np.quantile(ph, 1 - br))
    rng = np.random.default_rng(7); pol = {}
    for w, Pw_all in (("VAL", P[P.split == "VAL"]), ("OOS", P[P.split == "OOS"])):
        for scope, Pw in (("all", Pw_all), ("timeout_only", Pw_all[Pw_all.reason == "timeout"])):
            Cw = C[C.pid.isin(set(Pw.pid))].copy()
            pb = {}
            pb["A"] = m0["A"].predict_proba(X(Cw, FEAT["A"]))[:, 1]
            pb["B"] = m0["B"].predict_proba(X(Cw, FEAT["B"]))[:, 1]
            pb["C"] = predict_routed(m0["C"], X(Cw, FEAT["C"]).to_numpy(), Cw["regime_oof"].fillna(-1).to_numpy(int), pb["A"])
            pol[f"{w}/{scope}"] = policy_eval(Pw.reset_index(drop=True), Cw, pb, thr, rng)
            r10 = pol[f"{w}/{scope}"]["10bp"]
            log(f"  정책 {w}/{scope}: 보유 {r10['hold_to_terminal']['mean_bp']:+.1f} · 트레일 {r10['trailing_SL3_ARM1_T0.1']['mean_bp']:+.1f} · "
                + " · ".join(f"{arm} {r10[f'model_{arm}']['mean_bp']:+.1f}(무작위 {r10[f'random_exit_matched_{arm}']['mean_bp']:+.1f})" for arm in ("A", "B", "C")))

    verdict = "PASS" if any(kill[k]["PASS_preregistered"] for k in kill) else "FAIL -- 재료는 매매용 기여 없음(세 번째 0), RL 착수 중단"
    rep = {"script": Path(__file__).name, "run_utc": pd.Timestamp.utcnow().isoformat(), "seeds": SEEDS, "n_boot": a.boot,
           "population": "raw causal evidence-signal fires (8 signals × both sides), entry next-bar open, horizon per signal, hard SL 3.0 ATR, no TP/trail",
           "splits": {"TRAIN": [str(WARMUP_END.date()), str(VAL_START.date())], "VAL": [str(VAL_START.date()), str(OOS_START.date())],
                      "OOS": [str(OOS_START.date()), str(HOLDOUT_START.date())], "HOLDOUT": "untouched"},
           "fresh_forward_bar_by_bar": False, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False,
           "future_rows_used_for_entry": False, "research_grade_only": True, "risk_sizing_source": "none (price-move units, no notional)",
           "context_convention_check": conv, "join_convention": "all features at ts_dec = bar just closed; smoke joined at next bar (A_smokejoin diagnostic)",
           "counts": {"fires": int(len(F)), "positions": int(len(P)), "positions_by_reason": P.reason.value_counts().to_dict(),
                      "positions_by_split": P.split.value_counts().to_dict(), "checkpoints": int(len(C)),
                      "timeout_ckpt_train_val_oos": [int(len(T)), int(len(V)), int(len(O))],
                      "timeout_positions_train_val_oos": [int(T.pid.nunique()), int(V.pid.nunique()), int(O.pid.nunique())]},
           "material_coverage": mstats, "features": {k: len(v) for k, v in FEAT.items()},
           "per_seed_auc": per_seed_auc, "seedavg_auc": auc, "kill_gate": kill, "verdict": verdict,
           "diagnostics": {"anti_stable_spearman_train_val_feature_auc": round(rho, 4), "top20_sign_kept": keep_sign,
                           "perm_importance_top15_B": perm_top15, "material_in_top15": mat_in_top15,
                           "one_bar_context_leak_effect_seed0": {"A_val_auc": per_seed_auc[0]["A_val_auc"], "A_smokejoin_val_auc": per_seed_auc[0].get("A_smokejoin_val_auc")}},
           "policy_eval": pol, "thresholds": thr}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    P.to_csv(OUT / "positions.csv", index=False)
    keep_cols = ["pid", "ts_dec", "ts_next", "t", "split", "reason", "signal", "label", "exit_move", "regime_oof", "day"] + \
                sorted(set(FEAT["B"]) | {f"{s}_oof_src" for s in SIGS})
    C[keep_cols].to_parquet(OUT / "checkpoints.parquet", index=False)
    # 층 게이트 계약 (L4/L1/L3 -- L2는 봉 경계 진입이라 체결봉 크레딧 부류 해당 없음, L2P는 라이브 채점기 없음)
    gate_cfg = {
        "pipeline": "eth_exit_timing_phase0_20260904 -- exit 킬게이트 모집단(raw 인과 발동, 다음봉 시가 진입, H 타임아웃 + SL3ATR)",
        "splits": {"VAL": str(VAL_START.date()), "OOS": str(OOS_START.date()), "HOLDOUT": str(HOLDOUT_START.date())},
        "known_ts": {"assumption": "raw triggers from live compute_signals() (no cluster_dedup; L1 PASS 2026-09-03), known at trigger bar close; entry = next bar open (btf=1)"},
        "label": {"fills": str((OUT / "positions.csv").relative_to(ROOT)), "ts_col": "timestamp", "y_col": "y", "entry_col": "lim",
                  "side_col": "sd", "atr_col": "atr_pct", "fill_idx_col": "fi", "exit_idx_col": "ei", "bars_to_fill_col": "btf", "signal_col": "signal",
                  "exit": {"sl_atr": SL_ATR, "arm_atr": 1e9, "trail_atr": 0.0}, "cost_roundtrip": COST, "notional": 1.0,
                  "tol_mean_bp": 2.0, "tol_winrate_pp": 2.0},
        "trigger": {"module": "gate_eth_entry_triggers_v1_adapter_20260903", "fn": "build_fires", "warmup_bars": 4000, "sample_n": 120},
        "features": {"table": str((OUT / "checkpoints.parquet").relative_to(ROOT)), "ts_col": "ts_dec", "y_col": "label",
                     "cols": FEAT["B"], "stacked": [{"col": f"{s}_pct", "source_col": f"{s}_oof_src"} for s in SIGS]},
        "seed": SEEDS[0]}
    (OUT / "gate_config.json").write_text(json.dumps(gate_cfg, ensure_ascii=False, indent=2))
    log(f"→ {verdict}")
    log(f"산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
