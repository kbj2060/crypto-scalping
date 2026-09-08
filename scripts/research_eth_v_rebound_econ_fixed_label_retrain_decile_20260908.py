#!/usr/bin/env python3
"""V자반등 경제라벨을 **2026-09-07 수정 회계로 다시 만들고 재학습** -- 십분위 단조성 재검정.

## 왜 (사용자 승인, 2026-09-08)

배포 후보의 진입 품질 측정에서 확률 십분위별 MFE200에 단조성이 0이었다
(`eth_v_rebound_econ_entry_quality_mfe_20260908.md`). 다만 그 모델의 학습 라벨
"이 진입이 비용 후 이익인가"는 **legacy `sim_exit`**(걸 수 없는 스톱을 그 가격에 체결시키는
결함판)으로 만들어졌다. 즉 모델은 **회계 산물을 목표로** 학습됐다. 목표가 오염된 상태의
"실력 없음"은 구조의 한계가 아니라 목표의 문제일 수 있다.

⇒ 라벨만 `infeasible="exit"`(수정본)으로 바꿔 **같은 절차로 재학습**하고, 같은 십분위 테스트를
다시 돌린다. 여기서도 단조성이 없으면 이 피쳐/구조로 진입을 고르는 축은 종결이다.

## 바꾸는 것 / 안 바꾸는 것

  바꾸는 것: 라벨 산출의 청산 회계 (legacy -> fixed). **그것뿐이다.**
  그대로: 피쳐 23종 · 컨텍스트 18,000행/시드 · 시드 [141592, 271828, 577215] ·
          셀 (5.0, 1.5, 0.1) · 비용 10bp · 진입 다음 봉 시가 · 임계값 규칙 "VAL 상위 5%" ·
          같은 봉 양측면 dedup · 동시보유 5.

**파리티 자기검사(진단)**: (a) 재구성한 **legacy 라벨**이 동결 아티팩트
(`tabpfn_train_context_frozen_econ_5seed_20260902.csv`)의 `label`과 같은 값인지 --
겹치는 (timestamp, is_downside) 행에서 일치율을 잰다. 여기가 1.0이 아니면 라벨 재현이
깨진 것이므로 결과 해석 전에 원인을 밝혀야 한다. (b) 컨텍스트 샘플링 절차가 같은지는
타임스탬프 교집합 비율로 함께 보고한다(동결본이 다른 RNG 호출 순서를 썼을 수 있어
낮게 나와도 그 자체로는 오류가 아니다 -- 라벨 일치율이 본 검사다).

## 산출

  · 창별 확률 십분위 x {순손익 bp(수정회계, 배포셀), MFE200, MAE200} -- **신 모델과 구 모델을
    같은 표에** 놓는다(구 모델 확률은 직전 실행 캐시 재사용).
  · 십분위 순위 상관(Spearman) + (최상위 - 최하위) 십분위 차의 일군집 부트스트랩 CI.
  · 새 임계값(VAL 상위 5%)로 뽑은 진입의 건당/포트폴리오 성과 + 무작위 진입 대조군.
  · 라벨률과 창별 AUC(수정 라벨 기준).

⚠️읽기 전용. 라이브 코드 변경 없음. HO(2026-04~)는 소진된 창이라 서술용.
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
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_eq = _load("eq_retrain", "scripts/research_eth_v_rebound_econ_entry_quality_mfe_20260908.py")
_hc = _eq._hc
_s1 = _hc._s1
sim_stream = _hc.sim_stream
portfolio = _hc.portfolio
day_boot_ci = _hc.day_boot_ci
fixed_window_mfe = _eq.fixed_window_mfe

FEATURES, CTX_CSV = _hc.FEATURES, _hc.CTX_CSV
CELL, COST_BP, CAP = _hc.CELL, _hc.COST_BP, _hc.DEPLOYED_CAP
TRAIN_END, VAL_END, OOS_END = _hc.TRAIN_END, _hc.VAL_END, _hc.OOS_END
MAX_CONCURRENT = _hc.MAX_CONCURRENT

SEEDS = [141592, 271828, 577215]          # 라이브와 동일(동결 5시드 정렬 후 앞 3개)
CONTEXT_N = 18000
TAIL_FRAC = 0.05                          # 규격의 선정 규칙: VAL 상위 5%
PRED_CHUNK = 20000
FIXED_WIN = 200
BOOT = 2000
RNG_SEED = 20260908
OLD_SCORED = ROOT / "data/research/eth_v_rebound_econ_scored_20260908"
NEW_SCORED = ROOT / "data/research/eth_v_rebound_econ_fixedlabel_scored_20260908"
OUT = ROOT / "data/research/eth_v_rebound_econ_fixedlabel_retrain_20260908/report.json"


def log(m: str) -> None:
    print(f"[relabel] {m}", flush=True)


def decile_table(df, pcol, ycol_net, mfe, mae, days, rng):
    """확률 십분위별 순손익/MFE200/MAE200 + 순위상관 + (상위-하위) 차 CI."""
    d = pd.qcut(df[pcol], 10, labels=False, duplicates="drop")
    rows = []
    for k, v in df.assign(_d=d).groupby("_d"):
        idx = v.index.to_numpy()
        rows.append({"dec": int(k), "n": int(len(v)),
                     "p_lo": float(v[pcol].min()), "p_hi": float(v[pcol].max()),
                     "net_bp": float(ycol_net[idx].mean()),
                     "mfe200": float(mfe[idx].mean()), "mae200": float(mae[idx].mean())})
    rho = float(pd.Series([r["dec"] for r in rows]).corr(
        pd.Series([r["net_bp"] for r in rows]), method="spearman"))
    rho_mfe = float(pd.Series([r["dec"] for r in rows]).corr(
        pd.Series([r["mfe200"] for r in rows]), method="spearman"))
    top = df.index.to_numpy()[(d == d.max()).to_numpy()]
    bot = df.index.to_numpy()[(d == 0).to_numpy()]
    gap = float(ycol_net[top].mean() - ycol_net[bot].mean())
    # 상위/하위 십분위를 함께 담은 부호 벡터로 일군집 부트스트랩(같은 날 군집 보존)
    both = np.concatenate([top, bot])
    sgnv = np.concatenate([np.ones(len(top)), -np.ones(len(bot))])
    w = np.concatenate([np.full(len(top), 1.0 / len(top)), np.full(len(bot), 1.0 / len(bot))])
    vals = ycol_net[both] * sgnv * w * len(both)
    lo, hi = day_boot_ci(vals, days[both], rng, b=BOOT)
    return {"rows": rows, "spearman_net": rho, "spearman_mfe200": rho_mfe,
            "top_minus_bottom_bp": gap, "gap_ci95": [lo, hi]}


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)
    from tabpfn import TabPFNClassifier
    import torch
    log(f"cuda: {torch.cuda.is_available()}")
    NEW_SCORED.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    log("프레임 재구성 ...")
    _s1.VAL_END = pd.Timestamp("2030-01-01", tz="UTC")
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    long = _s1.long_frame_for(sig, feat, sb, st)
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    nk = len(kl)
    long["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1) for t in long["timestamp"]]
    # 라벨 창이 데이터 안에 온전히 들어오는 행만 -- 원본 백테스트와 같은 필터
    long = long.loc[(long["pos"] >= 0) & (long["pos"] + CAP + 1 < nk)].reset_index(drop=True)
    long["sgn"] = np.where(long["is_downside"].to_numpy() == 1, 1.0, -1.0)
    long["window"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                      np.where(long["timestamp"] < VAL_END, "VAL",
                       np.where(long["timestamp"] < OOS_END, "OOS", "HO")))
    log(f"  long {len(long):,}행 " + str(long["window"].value_counts().to_dict()))

    idx = long["pos"].to_numpy().astype(int)
    sgn = long["sgn"].to_numpy()
    atr = long["atr"].to_numpy(float)
    ent = o[idx + 1]
    days_all = long["timestamp"].dt.floor("D").to_numpy()

    log("라벨 산출 (fixed / legacy 둘 다) ...")
    r_fix, ex_fix, _ = sim_stream(idx + 1, ent, atr, sgn, h, l, c, *CELL,
                                  max_bars=CAP, infeasible="exit")
    r_leg, _, _ = sim_stream(idx + 1, ent, atr, sgn, h, l, c, *CELL,
                             max_bars=CAP, infeasible="ignore")
    net_fix = r_fix * 1e4 - COST_BP
    net_leg = r_leg * 1e4 - COST_BP
    long["y_fix"] = (net_fix > 0).astype(float)
    long["y_leg"] = (net_leg > 0).astype(float)
    tr_mask = long["window"] == "TRAIN"
    log(f"  라벨률 TRAIN  수정 {long.loc[tr_mask,'y_fix'].mean():.4f}  "
        f"legacy {long.loc[tr_mask,'y_leg'].mean():.4f} (배포 기록 0.75854)")

    # ---- 절차 파리티: legacy 라벨 + 같은 샘플링이 동결 아티팩트를 재현하는가 ----
    frozen = pd.read_csv(CTX_CSV)
    tr = long.loc[tr_mask].reset_index(drop=True)
    # (a) 라벨 재현: 동결본의 label이 재구성 legacy 라벨과 같은가
    key = pd.DataFrame({"ts": tr["timestamp"].dt.tz_localize(None),
                        "isd": tr["is_downside"].to_numpy().astype(int),
                        "y_leg": tr["y_leg"].to_numpy()})
    fz = frozen.copy()
    fz["ts"] = pd.to_datetime(fz["timestamp"]).dt.tz_localize(None)
    fz["isd"] = fz["is_downside"].astype(int)
    mg = fz.merge(key, on=["ts", "isd"], how="inner")
    lab_match = float((mg["label"].to_numpy() == mg["y_leg"].to_numpy()).mean()) if len(mg) else float("nan")
    log(f"  ⭐legacy 라벨 재현: 동결본과 겹치는 {len(mg):,}행에서 일치율 {lab_match:.6f}")
    parity = {"label_reproduction": {"matched_rows": int(len(mg)), "agreement": lab_match}}
    for sd in SEEDS:
        g = frozen.loc[frozen["seed"] == sd]
        r2 = np.random.default_rng(sd)
        pick = tr.iloc[np.sort(r2.choice(len(tr), size=min(CONTEXT_N, len(tr)), replace=False))]
        ts_a = set(pd.to_datetime(g["timestamp"]).dt.tz_localize(None))
        ts_b = set(pd.to_datetime(pick["timestamp"]).dt.tz_localize(None))
        inter = len(ts_a & ts_b)
        parity[str(sd)] = {"frozen_n": int(len(g)), "rebuilt_n": int(len(pick)),
                           "timestamp_overlap": inter,
                           "overlap_ratio": round(inter / max(1, len(g)), 4)}
        log(f"  파리티 시드 {sd}: 동결 {len(g):,}행 vs 재구성 {len(pick):,}행, "
            f"타임스탬프 교집합 {inter:,} ({inter/max(1,len(g))*100:.1f}%)")

    # ---- 재학습 (수정 라벨) ----
    log("재학습 (수정 라벨, 3시드) ...")
    models = []
    for sd in SEEDS:
        r2 = np.random.default_rng(sd)
        ctx = tr.iloc[np.sort(r2.choice(len(tr), size=min(CONTEXT_N, len(tr)), replace=False))]
        m = TabPFNClassifier(device="cuda", random_state=int(sd), ignore_pretraining_limits=True)
        m.fit(ctx[FEATURES], ctx["y_fix"].to_numpy())
        models.append(m)
        log(f"  시드 {sd} 적합 (라벨률 {ctx['y_fix'].mean():.4f})")

    log(f"  라벨 비교: 수정 라벨이 legacy와 다른 행 "
        f"{float((long['y_fix'] != long['y_leg']).mean())*100:.2f}%")
    report = {"question": "수정회계 라벨로 재학습하면 진입 선별 실력이 생기는가",
              "changed": "라벨 청산 회계 legacy -> fixed(infeasible='exit')만",
              "kept": {"features": len(FEATURES), "context_n": CONTEXT_N, "seeds": SEEDS,
                       "cell": list(CELL), "cost_bp": COST_BP, "cap_bars": CAP,
                       "selection_rule": "VAL 상위 5%", "max_concurrent": MAX_CONCURRENT},
              "label_rate_train": {"fixed": float(long.loc[tr_mask, "y_fix"].mean()),
                                   "legacy": float(long.loc[tr_mask, "y_leg"].mean())},
              "procedure_parity_vs_frozen": parity, "windows": {}}

    from sklearn.metrics import roc_auc_score
    cut_new = None
    for win in ("VAL", "OOS", "HO"):
        s = long.loc[long["window"] == win].copy()
        if s.empty:
            continue
        cache = NEW_SCORED / f"{win}.parquet"
        if cache.exists():
            sc = pd.read_parquet(cache)
            s = s.merge(sc[["timestamp", "sgn", "p_new"]], on=["timestamp", "sgn"], how="inner")
            log(f"[{win}] 캐시 재사용")
        else:
            log(f"[{win}] 채점 {len(s):,}행 ...")
            P = []
            for m in models:
                P.append(np.concatenate([m.predict_proba(s[FEATURES].iloc[k:k + PRED_CHUNK])[:, 1]
                                         for k in range(0, len(s), PRED_CHUNK)]))
                log(f"  시드 완료 ({time.time()-t0:.0f}s)")
            s["p_new"] = np.vstack(P).mean(axis=0)
            s[["timestamp", "sgn", "pos", "atr", "p_new"]].to_parquet(cache, index=False)

        old = OLD_SCORED / f"{win}.parquet"
        if old.exists():
            so = pd.read_parquet(old)[["timestamp", "sgn", "p"]].rename(columns={"p": "p_old"})
            s = s.merge(so, on=["timestamp", "sgn"], how="left")
        s = s.reset_index(drop=True)

        i2 = s["pos"].to_numpy().astype(int)
        g2 = s["sgn"].to_numpy()
        e2 = o[i2 + 1]
        r2_, ex2, _ = sim_stream(i2 + 1, e2, s["atr"].to_numpy(float), g2, h, l, c,
                                 *CELL, max_bars=CAP, infeasible="exit")
        net = r2_ * 1e4 - COST_BP
        y = (net > 0).astype(int)
        mfe, mae = fixed_window_mfe(i2 + 1, e2, g2, h, l, FIXED_WIN)
        days = s["timestamp"].dt.floor("D").to_numpy()

        rec = {"n_rows": int(len(s)), "label_rate_fixed": float(y.mean()),
               "auc_new": float(roc_auc_score(y, s["p_new"])),
               "deciles_new": decile_table(s, "p_new", net, mfe, mae, days, rng)}
        if "p_old" in s and s["p_old"].notna().all():
            rec["auc_old"] = float(roc_auc_score(y, s["p_old"]))
            rec["deciles_old"] = decile_table(s, "p_old", net, mfe, mae, days, rng)

        # ---- 새 임계값(VAL 상위 5%)로 진입 선정 ----
        if win == "VAL":
            k = max(30, int(round(len(s) * TAIL_FRAC)))
            cut_new = float(s.nlargest(k, "p_new")["p_new"].min())
            report["new_threshold"] = cut_new
            log(f"  [VAL] 새 임계값(상위 5%) = {cut_new:.4f}")
        passed = s.loc[s["p_new"] >= cut_new]
        keep = []
        for _, gg in passed.groupby("timestamp"):
            if len(gg) == 1:
                keep.append(gg.index[0]); continue
            top = gg["p_new"].max()
            tied = gg.loc[gg["p_new"] >= top - 1e-12]
            if len(tied) == 1:
                keep.append(tied.index[0])
        sel = passed.loc[keep].sort_values("pos")
        si = sel.index.to_numpy()
        if len(si) >= 30:
            lo, hi = day_boot_ci(net[si], days[si], rng, b=BOOT)
            cand = pd.DataFrame({"timestamp": sel["timestamp"].to_numpy(),
                                 "entry_bar": sel["pos"].to_numpy().astype(int) + 1,
                                 "exit_bar": sel["pos"].to_numpy().astype(int) + 1 + ex2[si],
                                 "pnl_bp": net[si]})
            pf = portfolio(cand, MAX_CONCURRENT)
            rec["selected_new"] = {"n": int(len(si)), "exp_bp": float(net[si].mean()),
                                   "ci95": [lo, hi], "win_rate": float((net[si] > 0).mean()),
                                   "mfe200_mean": float(mfe[si].mean()),
                                   "portfolio": {k2: (round(v, 3) if isinstance(v, float) else v)
                                                 for k2, v in pf.items()
                                                 if k2 not in ("idx", "pnl", "ts")}}
            rb = rng.choice(len(s), size=min(len(si) * 3, len(s)), replace=False)
            rec["random_baseline"] = {"n": int(len(rb)), "exp_bp": float(net[rb].mean()),
                                      "mfe200_mean": float(mfe[rb].mean())}

        report["windows"][win] = rec
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))

        log("")
        log(f"=== {win}  {len(s):,}행  라벨률(수정) {y.mean():.4f}  "
            f"AUC 신 {rec['auc_new']:.4f}" + (f" / 구 {rec['auc_old']:.4f}" if "auc_old" in rec else ""))
        for tag in ("deciles_new", "deciles_old"):
            if tag not in rec:
                continue
            d = rec[tag]
            log(f"  --- {'신(수정라벨 재학습)' if tag.endswith('new') else '구(배포, legacy라벨)'} "
                f"십분위: 순위상관 net {d['spearman_net']:+.3f} · MFE200 {d['spearman_mfe200']:+.3f} · "
                f"상위-하위 {d['top_minus_bottom_bp']:+.2f}bp "
                f"[{d['gap_ci95'][0]:+.2f},{d['gap_ci95'][1]:+.2f}]")
            log(f"      {'dec':>3s} {'p범위':>15s} {'순손익':>9s} {'MFE200':>9s} {'MAE200':>9s}")
            for r_ in d["rows"]:
                log(f"      {r_['dec']:>3d} {r_['p_lo']:.3f}~{r_['p_hi']:.3f} "
                    f"{r_['net_bp']:>+8.2f}bp {r_['mfe200']:>+8.1f} {r_['mae200']:>+8.1f}")
        if "selected_new" in rec:
            sn = rec["selected_new"]
            log(f"  선정({cut_new:.4f} 이상) n={sn['n']:,}  건당 {sn['exp_bp']:+.2f}bp "
                f"[{sn['ci95'][0]:+.2f},{sn['ci95'][1]:+.2f}]  승률 {sn['win_rate']*100:.1f}%  "
                f"| PF n={sn['portfolio']['n']:,} {sn['portfolio']['exp_bp']:+.2f}bp "
                f"총 {sn['portfolio']['total_bp']:+.0f}bp  | 무작위 {rec['random_baseline']['exp_bp']:+.2f}bp")

    report["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"저장: {OUT.relative_to(ROOT)} ({report['elapsed_sec']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
