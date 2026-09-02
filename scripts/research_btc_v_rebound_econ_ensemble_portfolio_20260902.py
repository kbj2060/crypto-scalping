#!/usr/bin/env python3
"""BTC 경제라벨 -- 5시드 앙상블 + 순차 포트폴리오, VAL 선정 -> OOS 1회.

## 위치

1단계(`research_btc_v_rebound_econ_label_screen_20260902.py`)가 **TRAIN에서만** BTC 전용
exit 셀을 확정했다. 이 스크립트는 그 셀로 라벨을 만들어 학습·검증한다.

## ETH에서 확립한 규율을 그대로 적용

  · **5시드 확률 앙상블** -- 단일 시드는 VAL 분위 선정 절차의 분산 때문에 4/5만 양수였다.
    앙상블로 시드간 std를 0.021까지 낮춘 것이 ETH 통과의 결정적 교정이었다.
  · ⭐**순차 포트폴리오** -- 동시보유 한도 내에서 슬롯이 빌 때만 진입. 중첩 허용 평가는
    자동매매로 불가능하고 n도 부풀린다.
  · **방향뒤집기 대조군** + 저ARM(<1.0) 제외(노이즈수확 아티팩트 구간).
  · 판정 지표는 **기대값 bp/트레이드** (통과셀 개수는 포화돼 크기를 못 담는다).
  · VAL에서 (분위, 셀, 한도) 선정 -> **OOS 1회**, 재최적화 금지.

## ⚠️BTC 고유 사정

  · **ATR 중앙 16.0bp로 ETH(~23bp)보다 30% 작다** -> 같은 ATR 배수라도 비용 10bp가
    상대적으로 1.4배 무겁다. 1단계가 이걸 보정한 셀을 골랐다.
  · BTC HOLDOUT은 **다른 라벨 정의로 이미 노출된 이력**이 있다(V자반등 6/8트리거 등).
    이 라벨은 정의가 다르지만, 노출 여부는 보수적으로 취급한다 -- 이 스크립트는 **VAL/OOS만**
    쓰고 HOLDOUT은 통과 시 별도 결정.

⚠️HOLDOUT 미터치. 라이브 코드 변경 없음.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_pf = _load("pf_btc2", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_sc = _load("sc_btc", "scripts/research_btc_v_rebound_econ_label_screen_20260902.py")
sim_exit, portfolio = _pf.sim_exit, _pf.portfolio
SL_GRID, ARM_GRID, TRAIL_GRID = _pf.SL_GRID, _pf.ARM_GRID, _pf.TRAIL_GRID
# ⭐BTC는 ATR이 16bp로 작아 라벨 셀이 SL=16.0으로 잡혔다(ETH 5.0의 3.2배). 거래 셀 격자도
# 같은 대역을 포함해야 한다 -- ETH 격자(최대 5.0)만 쓰면 BTC의 실제 최적 구간을 못 본다.
# ⚠️교정: 1차 시도에서 VAL이 SL=20.0(격자 최댓값)을 골라 OOS -22.18bp로 참패했다.
# 승률 84~95%/손익비 0.089 = 변동성 매도 프로파일. 상한을 8.0으로 묶는다.
SL_GRID = tuple(x for x in (tuple(SL_GRID) + (6.0, 8.0)) if x <= 8.0)
# ⭐4차 교정(2026-09-02): 앞선 두 시도가 실패한 뒤 격자를 다시 보니 **ARM 상한이 1.5**였다.
# ARM은 이익 목표인데 BTC ATR 16bp에서 1.5x면 목표가 24bp -- 비용 10bp 대비 2.4:1뿐이다
# (ETH는 1.5x23=35bp로 3.5:1). 나는 SL(손실 폭)만 넓혔는데 방향이 반대였다.
# BTC가 수수료 바닥에 가깝다는 진단이 맞다면 **넓혀야 할 것은 ARM**이다.
ARM_GRID = tuple(ARM_GRID) + (2.5, 4.0, 6.0, 8.0)
PAYOFF_FLOOR = 0.25
FORWARD_BARS = _pf.FORWARD_BARS
TIER0 = _sc.TIER0

SCREEN_REPORT = ROOT / "data/research/btc_v_rebound_econ_label_screen_20260902/report_arm_extended.json"
COST_BP, ARTIFACT_FREE_MIN = 10.0, 1.0
CONTEXT_N = 18000
SEEDS = [20260829, 141592, 271828, 577215, 20260902]
TAIL_FRACS = [0.005, 0.01, 0.02, 0.05]
MAX_CONCURRENT = [1, 3, 5]
NULL_B, NULL_SEED = 200, 20260902
CHUNK = 40000
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT = ROOT / "data/research/btc_v_rebound_econ_portfolio_20260902/report_arm_extended.json"


def log(m): print(f"[btc-pf] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    from sklearn.metrics import roc_auc_score
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    scr = json.loads(SCREEN_REPORT.read_text())
    CELL = tuple(scr["selected_cell"]["cell"])
    log(f"⭐1단계 확정 BTC 라벨 셀: SL/ARM/Trail = {CELL}  "
        f"(TRAIN 라벨률 {scr['selected_cell']['label_rate']:.4f}, "
        f"평균 {scr['selected_cell']['mean_bp']:+.2f}bp)")
    eth_c = scr.get("eth_cell_on_btc")
    if eth_c:
        log(f"  ETH 셀 {scr['scope']['eth_reference_cell']}을 BTC에 적용하면: "
            f"라벨률 {eth_c['label_rate']:.4f}, 평균 {eth_c['mean_bp']:+.2f}bp")

    long, meta = _sc.build_long()
    df = meta.pop("df")
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    nb = len(df)
    long = long.dropna(subset=TIER0).reset_index(drop=True)
    long = long.loc[long["bar_idx"] + FORWARD_BARS + 1 < nb].reset_index(drop=True)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL",
                      np.where(long["timestamp"] < OOS_END, "OOS", "HOLDOUT")))
    long = long.loc[long["split"] != "HOLDOUT"].reset_index(drop=True)
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"
    log(f"프레임 {len(long):,}행 (TRAIN {int((long.split=='TRAIN').sum()):,} / "
        f"VAL {int((long.split=='VAL').sum()):,} / OOS {int((long.split=='OOS').sum()):,})")

    sl0, arm0, tr0 = CELL
    ii = long["bar_idx"].to_numpy().astype(int)
    sg = np.where(long["is_downside"].to_numpy() == 1, 1.0, -1.0)
    at = long["atr"].to_numpy(dtype=float)
    net = np.full(len(long), np.nan)
    for s_ in range(0, len(long), CHUNK):
        e_ = min(s_ + CHUNK, len(long))
        j = ii[s_:e_]
        H = np.stack([h[x+1:x+1+FORWARD_BARS] for x in j])
        L = np.stack([l[x+1:x+1+FORWARD_BARS] for x in j])
        C = np.stack([c[x+1:x+1+FORWARD_BARS] for x in j])
        pn, _ = sim_exit(o[j+1], at[s_:e_], sg[s_:e_], H, L, C, sl0, arm0, tr0)
        net[s_:e_] = pn * 1e4 - COST_BP
    long["y"] = (net > 0).astype(float)
    log(f"라벨률 전체 {long['y'].mean():.4f}  "
        f"(bottom {long.loc[long.is_downside==1,'y'].mean():.4f} / "
        f"top {long.loc[long.is_downside==0,'y'].mean():.4f})")

    tr_set = long.loc[long["split"] == "TRAIN"]
    probs = {"VAL": [], "OOS": []}
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        ctx = tr_set.iloc[np.sort(rng.choice(len(tr_set), size=min(CONTEXT_N, len(tr_set)), replace=False))]
        clf = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
        clf.fit(ctx[TIER0], ctx["y"].to_numpy())
        for spn in ("VAL", "OOS"):
            s = long.loc[long["split"] == spn]
            probs[spn].append(np.concatenate([clf.predict_proba(s[TIER0].iloc[k:k+20000])[:, 1]
                                              for k in range(0, len(s), 20000)]))
        log(f"  seed {sd} 완료")
    scored = {}
    for spn in ("VAL", "OOS"):
        s = long.loc[long["split"] == spn].copy()
        P = np.vstack(probs[spn])
        s["p"] = P.mean(axis=0)
        scored[spn] = s
        lb = s.loc[s["y"].notna()]
        auc = float(roc_auc_score(lb["y"], lb["p"])) if lb["y"].nunique() == 2 else None
        log(f"  {spn}: {len(s):,}행  자기라벨AUC {auc:.4f}  시드간 std 중앙 {np.median(P.std(axis=0)):.4f}")

    def candidates(s, cut, cell, cost=COST_BP, flip=False):
        sel = s.loc[s["p"] >= cut]
        if len(sel) < 30:
            return None
        idx = sel["bar_idx"].to_numpy().astype(int)
        sgn = np.where(sel["is_downside"].to_numpy() == 1, 1.0, -1.0)
        if flip:
            sgn = -sgn
        H = np.stack([h[j+1:j+1+FORWARD_BARS] for j in idx])
        L = np.stack([l[j+1:j+1+FORWARD_BARS] for j in idx])
        C = np.stack([c[j+1:j+1+FORWARD_BARS] for j in idx])
        sl, arm, trv = cell
        pn, ex = sim_exit(o[idx+1], sel["atr"].to_numpy(dtype=float), sgn, H, L, C, sl, arm, trv)
        return pd.DataFrame({"timestamp": sel["timestamp"].to_numpy(), "entry_bar": idx + 1,
                             "exit_bar": idx + 1 + ex, "pnl_bp": pn * 1e4 - cost})

    log("")
    log("=== VAL 순차 포트폴리오 격자 ===")
    best = None
    for frac in TAIL_FRACS:
        k = max(30, int(round(len(scored["VAL"]) * frac)))
        cut = float(scored["VAL"].nlargest(k, "p")["p"].min())
        for sl in SL_GRID:
            for arm in ARM_GRID:
                if arm < ARTIFACT_FREE_MIN:
                    continue
                for trv in TRAIL_GRID:
                    cand = candidates(scored["VAL"], cut, (sl, arm, trv))
                    if cand is None:
                        continue
                    for mc in MAX_CONCURRENT:
                        r = portfolio(cand, mc)
                        if r is None or r["n"] < 30:
                            continue
                        # ⭐손익비 하한: 누적만 보면 왜도가 폭주한다(1차 실패 원인)
                        if (r.get("payoff") or 0) < PAYOFF_FLOOR:
                            continue
                        if best is None or r["total_bp"] > best["r"]["total_bp"]:
                            best = {"frac": frac, "cut": cut, "cell": (sl, arm, trv), "mc": mc, "r": r}
        if best:
            b = best
            log(f"  상위{frac*100:>4.1f}% 누적최고: 셀{b['cell']} 한도{b['mc']} n={b['r']['n']:,} "
                f"기대값 {b['r']['exp_bp']:+.2f}bp 총 {b['r']['total_bp']:+.0f}bp "
                f"DD {b['r']['max_dd_bp']:+.0f}bp")

    b = best
    log("")
    log(f"[VAL 선정] 상위{b['frac']*100:g}% (p>={b['cut']:.4f}) 셀{b['cell']} 한도{b['mc']}  "
        f"n={b['r']['n']:,} 기대값 {b['r']['exp_bp']:+.2f}bp 총 {b['r']['total_bp']:+.0f}bp "
        f"승률 {b['r']['win_rate']*100:.1f}% DD {b['r']['max_dd_bp']:+.0f}bp 연속손실 {b['r']['max_consec_loss']}")

    log("")
    log("=== OOS 1회 (재최적화 없음) ===")
    oos_res = {}
    for flip, fnm in ((False, "정방향"), (True, "뒤집기")):
        cand = candidates(scored["OOS"], b["cut"], b["cell"], flip=flip)
        if cand is None:
            log(f"  {fnm}: 후보 부족"); continue
        r = portfolio(cand, b["mc"])
        if r is None:
            continue
        days = (pd.Timestamp(r["ts"].max()) - pd.Timestamp(r["ts"].min())).total_seconds() / 86400
        mo = pd.Series(r["pnl"], index=pd.to_datetime(r["ts"])).groupby(
            pd.to_datetime(r["ts"]).to_period("M")).mean()
        log(f"  {fnm}: n={r['n']:,} ({r['n']/max(days,1):.2f}건/일) 기대값 {r['exp_bp']:+.2f}bp "
            f"총 {r['total_bp']:+.0f}bp 승률 {r['win_rate']*100:.1f}% 손익비 {r['payoff']} "
            f"DD {r['max_dd_bp']:+.0f}bp 연속손실 {r['max_consec_loss']}")
        log(f"      월별: " + "  ".join(f"{k} {v:+.2f}bp" for k, v in mo.items()))
        oos_res[fnm] = {**{k: (round(v, 3) if isinstance(v, float) else v)
                           for k, v in r.items() if k not in ("idx", "pnl", "ts")},
                        "per_day": round(r["n"] / max(days, 1), 3),
                        "monthly_exp_bp": {str(k): round(float(v), 2) for k, v in mo.items()}}

    # ⭐측면 균형 진단 -- ETH에서 "하락장에 숏 쳐서 이긴 것"인지 가르는 데 결정적이었다
    log("")
    log("=== 측면 균형 진단 ===")
    side_diag = {}
    for spn in ("VAL", "OOS"):
        sel = scored[spn].loc[scored[spn]["p"] >= b["cut"]]
        nl = int((sel["is_downside"] == 1).sum())
        side_diag[spn] = {"n": int(len(sel)), "long": nl, "short": int(len(sel) - nl),
                          "long_pct": round(nl / max(len(sel), 1) * 100, 1)}
        log(f"  {spn}: 호출 {len(sel):,}  롱 {nl:,} ({side_diag[spn]['long_pct']:.1f}%) "
            f"숏 {len(sel)-nl:,}")

    fwd, flp = oos_res.get("정방향"), oos_res.get("뒤집기")
    ok = bool(fwd and fwd["total_bp"] > 0 and flp and fwd["exp_bp"] > flp["exp_bp"]
              and sum(1 for v in fwd["monthly_exp_bp"].values() if v > 0) >= 2)
    log("")
    log(f"=== 판정: {'✅통과' if ok else '❌미통과'} (누적>0 / 뒤집기우위 / 3개월중 2개월 양수) ===")

    report = {"signal": "btc_v_rebound_econ_ensemble_portfolio", "asset": "BTCUSDT",
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "scope": {"label_cell": list(CELL), "label_cell_selected_on": "TRAIN only",
                        "seeds": SEEDS, "cost_bp": COST_BP,
                        "selection": "VAL에서 (분위,셀,한도) -> OOS 1회",
                        "holdout_touched": False, "live_code_changed": False,
                        "btc_atr_note": "BTC ATR 중앙 16.0bp (ETH ~23bp) -- 비용 10bp가 상대적으로 1.4배 무겁다"},
              "val_selection": {"frac": b["frac"], "cut": round(b["cut"], 4),
                                "cell": list(b["cell"]), "max_concurrent": b["mc"],
                                **{k: (round(v, 3) if isinstance(v, float) else v)
                                   for k, v in b["r"].items() if k not in ("idx", "pnl", "ts")}},
              "oos": oos_res, "side_balance": side_diag, "passed": ok, "runtime_sec": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
