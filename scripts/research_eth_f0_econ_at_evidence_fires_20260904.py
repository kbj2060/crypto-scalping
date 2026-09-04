#!/usr/bin/env python3
"""F0 매봉 경제모델을 증거신호 8종 발동 시점에 그대로 적용 -- "왜 V자반등 경제라벨만 살아남았나" 진단 (2026-09-04).

질문. V자반등 경제라벨 모델(F0: 매 봉 × 양방향, **트리거 없음**)은 살아남았는데, 같은 Tier0·같은 라벨족으로
8종 증거신호 **발동 모집단**을 재라벨링한 것은 11축 전부 실패했다(호메로스 §5.20·§5.22). 차이가 "라벨"이
아니라 "모집단(트리거 유무)"에 있다면 다음이 성립해야 한다.

  Q1 발동 (봉,측면)은 F0 순위에서 특별하지 않다 -- F0 컷(VAL 상위5%) 통과율이 기저(5%)와 비슷하거나 낮다.
  Q2 발동에 F0 게이트를 씌우면(발동 ∧ p_F0 ≥ 컷) 경제성이 F0 전체 수준으로 회복되는가.
       회복  -> 발동 자체가 문제가 아니라, 발동 모집단 안의 순위 정보를 표본 부족으로 학습 못 한 것.
       미회복 -> 발동 시점은 F0 엣지가 없는 시점(구조적 반선택). 라벨·모델·피쳐를 바꿔도 안 되는 이유.
  Q3 F0 호출은 발동과 시간적으로 어떤 관계인가 -- 발동 후 k봉 안에 몰리는가(농축비), 몰린다면 발동 봉 대비
       얼마나 더 진행한 지점인가(ATR 단위 초과진행; §5.20 "트리거 후 2~4 ATR 더 진행 후 반전"의 인과판).
  Q4 방향 IC -- §5.22의 IC 표는 롱·숏을 **합쳐** 계산해 대칭 피쳐(rsi 등)의 방향 IC가 상쇄된다. 측면별로 나눠
       매봉 모집단과 발동 모집단에서 Tier0의 방향 정보량을 다시 잰다.

사전 규칙(결과 보기 전 고정). F0 = v2 F0 그대로(TabPFN 5시드 18k, SEEDS·컨텍스트 추출 동일) · 컷 = VAL 상위 5%
(v2 동일) · 발동 = raw 인과 단일봉 발동(OOF csv) 및 GAP=12 첫발동 · 방향 = 발동 측면. 격자 없음, 재최적화 없음,
HOLDOUT 미접촉. OOS는 고정 규칙의 1회 조회. 다중성: 8신호×2발동정의 표는 기술통계, **Q2 판정은 8종 합집합
첫발동(pooled/first)** 하나 -- VAL·OOS 기대값>0 ∧ 둘 다 뒤집기 우위 ∧ OOS 일군집 CI 하한>0 이면 "회복".

사용:  --stage score  (서버 GPU) p_F0 산출 -> tmp/eth_f0_at_fires_20260904/p_f0.parquet (+ v2 F0 재현 대조)
       --stage diag   (CPU, torch 불필요) 진단 -> report.json
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


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


V2M = _load("hev2_mod", "scripts/research_homer_entry_v2_20260904.py")
V2, OOFD, SIGNALS, SEEDS = V2M.OUT, V2M.OOFD_MAT, V2M.SIGNALS, V2M.SEEDS
CONTEXT_N, CHUNK, TOP_FRAC, MAX_CONC, B_BOOT = V2M.CONTEXT_N, V2M.CHUNK, V2M.TOP_FRAC, V2M.MAX_CONC, V2M.B_BOOT
portfolio, calls_frame, day_boot, stats_of = V2M.portfolio, V2M.calls_frame, V2M.day_boot, V2M.stats_of
OUT = ROOT / "tmp/eth_f0_at_fires_20260904"
KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
GAP = 12                      # §5.22 재라벨링 실험의 인과 첫발동 규약 그대로
LAGS = (0, 3, 12, 48)
WINDOWS = ("VAL", "OOS")


def log(m): print(f"[f0fires] {m}", flush=True)


# ----------------------------------------------------------------------------- score
def stage_score():
    from tabpfn import TabPFNClassifier
    import torch
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(V2 / "frame.parquet"); cols = json.loads((V2 / "model_card.json").read_text())["arms"]["F0"]
    tr = D.loc[D["split"] == "TRAIN"].reset_index(drop=True)
    S = {w: D.loc[D["split"] == w].reset_index(drop=True) for w in WINDOWS}
    log(f"TRAIN {len(tr):,} VAL {len(S['VAL']):,} OOS {len(S['OOS']):,} · F0 {len(cols)} · cuda {torch.cuda.is_available()} "
        f"free {torch.cuda.mem_get_info()[0]/1e9:.2f}GB")
    P = {w: [] for w in S}
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        ctx = tr.iloc[np.sort(rng.choice(len(tr), size=min(CONTEXT_N, len(tr)), replace=False))]
        clf = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
        clf.fit(ctx[cols], ctx["y"].to_numpy())
        for w, s in S.items():
            P[w].append(np.concatenate([clf.predict_proba(s[cols].iloc[k:k + CHUNK])[:, 1] for k in range(0, len(s), CHUNK)]))
        log(f"  seed {sd} 완료 ({time.time()-t0:.0f}s)")
    frames = []
    for w, s in S.items():
        M = np.vstack(P[w]); f = s[["timestamp", "pos", "side", "is_downside", "split"]].copy()
        for k in range(len(SEEDS)):
            f[f"p_seed{k}"] = M[k]
        f["p_f0"] = M.mean(axis=0); frames.append(f)
    pf = pd.concat(frames, ignore_index=True); pf.to_parquet(OUT / "p_f0.parquet", index=False)
    # v2 F0 재현 대조 (같은 컷·같은 포트폴리오)
    cut = float(np.quantile(pf.loc[pf["split"] == "VAL", "p_f0"], 1 - TOP_FRAC))
    ref = json.loads((V2 / "report_tabpfn.json").read_text())["arms"]["F0"]
    par = {"cut": cut, "v2_cut": ref["cut"]}
    for w, s in S.items():
        r = portfolio(calls_frame(s, pf.loc[pf["split"] == w, "p_f0"].to_numpy(), cut), MAX_CONC)
        par[w] = {"exp_bp": round(r["exp_bp"], 3), "n": r["n"], "v2_exp_bp": ref["windows"][w]["exp_bp"], "v2_n": ref["windows"][w]["n"]}
    (OUT / "score_parity.json").write_text(json.dumps(par, indent=2))
    log(f"parity {par}  ({time.time()-t0:.0f}s)")


# ----------------------------------------------------------------------------- diag helpers
def load_fires():
    out = {}
    for s in SIGNALS:
        d = pd.read_csv(OOFD / f"{s}_oof.csv", parse_dates=["timestamp"])[["pos", "timestamp", "side", "split"]]
        d = d.drop_duplicates(["pos", "side"]).sort_values("pos").reset_index(drop=True)
        d["is_downside"] = (d["side"] == "bottom").astype(np.int8)
        ff = np.zeros(len(d), bool)
        for sd_ in (0, 1):
            idx = np.flatnonzero(d["is_downside"].to_numpy() == sd_); pos = d["pos"].to_numpy()[idx]
            keep = np.zeros(len(pos), bool); last = -10**9
            for j, p_ in enumerate(pos):
                if p_ - last > GAP:
                    keep[j] = True
                last = p_
            ff[idx] = keep
        d["first_fire"] = ff; out[s] = d
    return out


def fire_rows(Dw, F, first):
    f = F.loc[F["first_fire"]] if first else F
    return Dw.merge(f[["pos", "is_downside", "timestamp"]].rename(columns={"timestamp": "ts_fire"}), on=["pos", "is_downside"], how="inner")


def block(rows, cut, rng, Dw_p):
    """발동 (봉,측면) 행 집합에 대한 Q1/Q2 통계."""
    if len(rows) == 0:
        return {"n": 0}
    g = rows.loc[rows["p_f0"] >= cut]; u = rows.loc[rows["p_f0"] < cut]
    pct = np.searchsorted(np.sort(Dw_p), rows["p_f0"].to_numpy()) / len(Dw_p)
    o = {"n": int(len(rows)), "share_above_cut": round(float((rows["p_f0"] >= cut).mean()), 4),
         "p_median_pctile": round(float(np.median(pct)), 3), "net_all_bp": round(float(rows["net_bp"].mean()), 3),
         "wr_all": round(float((rows["net_bp"] > 0).mean()), 3), "flip_all_bp": round(float(rows["net_bp_flip"].mean()), 3),
         "n_gated": int(len(g)), "net_gated_bp": round(float(g["net_bp"].mean()), 3) if len(g) else None,
         "flip_gated_bp": round(float(g["net_bp_flip"].mean()), 3) if len(g) else None,
         "net_ungated_bp": round(float(u["net_bp"].mean()), 3) if len(u) else None}
    if len(g) >= 20:
        lo, hi = day_boot(g["net_bp"], g["timestamp"], B_BOOT, rng); o["gated_day_ci95"] = [round(lo, 2), round(hi, 2)]
        o["gated_days"] = int(pd.DatetimeIndex(g["timestamp"]).normalize().nunique())
        r = portfolio(calls_frame(rows, rows["p_f0"].to_numpy(), cut), MAX_CONC)
        o["gated_portfolio"] = stats_of(r) if r else None
    r0 = portfolio(calls_frame(rows, rows["p_f0"].to_numpy(), -1.0), MAX_CONC)
    o["ungated_portfolio"] = stats_of(r0) if r0 else None
    return o


def lag_arrays(F, npos):
    """측면별 '직전 raw 발동 이후 경과 봉' 배열 (npos 길이). 발동 없음 = 10**9."""
    out = {}
    for sd_ in (0, 1):
        pos = F.loc[F["is_downside"] == sd_, "pos"].to_numpy(); pos = pos[(pos >= 0) & (pos < npos)]
        last = np.full(npos, -10**9, dtype=np.int64); last[pos] = pos; last = np.maximum.accumulate(last)
        out[sd_] = np.arange(npos) - last
    return out


# ----------------------------------------------------------------------------- diag
def stage_diag():
    from scipy.stats import spearmanr
    t0 = time.time(); rng = np.random.default_rng(20260904)
    D = pd.read_parquet(V2 / "frame.parquet"); cols = json.loads((V2 / "model_card.json").read_text())["arms"]["F0"]
    pf = pd.read_parquet(OUT / "p_f0.parquet")
    D = D.merge(pf[["pos", "is_downside", "p_f0"]], on=["pos", "is_downside"], how="left")
    cut = float(np.quantile(D.loc[D["split"] == "VAL", "p_f0"], 1 - TOP_FRAC))
    fires = load_fires(); npos = int(D["pos"].max()) + 1
    # 발동 파일 timestamp ↔ 프레임 timestamp 정합 (인덱스 계약)
    chk = D[["pos", "timestamp"]].drop_duplicates("pos").merge(pd.concat(fires.values())[["pos", "timestamp"]].drop_duplicates("pos"),
                                                              on="pos", suffixes=("", "_f"))
    ts_ok = float((chk["timestamp"] == chk["timestamp_f"]).mean())
    assert ts_ok > 0.999, f"발동 pos↔timestamp 불일치 {ts_ok}"
    kl = pd.read_csv(KL, usecols=["timestamp", "close"], parse_dates=["timestamp"]).drop_duplicates("timestamp")
    bar = D[["pos", "timestamp", "atr"]].drop_duplicates("pos").merge(kl, on="timestamp", how="left").set_index("pos")
    close_of = bar["close"].reindex(range(npos)).to_numpy(float); atr_of = bar["atr"].reindex(range(npos)).to_numpy(float)
    log(f"cut {cut:.4f} · 발동 ts 정합 {ts_ok:.4f} · 봉 {npos:,} ({time.time()-t0:.0f}s)")

    rep = {"cut": cut, "gap": GAP, "top_frac": TOP_FRAC, "holdout_touched": False, "windows": {}}
    for w in WINDOWS:
        Dw = D.loc[D["split"] == w].reset_index(drop=True); Dw_p = Dw["p_f0"].to_numpy()
        R = {"n_rows": int(len(Dw)), "base_share_above_cut": round(float((Dw_p >= cut).mean()), 4)}
        f0 = portfolio(calls_frame(Dw, Dw_p, cut), MAX_CONC); R["f0_overall"] = stats_of(f0)
        R["f0_trade_sd_bp"] = round(float(f0["trades"]["pnl_bp"].std()), 2)
        # Q1/Q2 신호별 + 합집합
        R["signals"] = {}
        for s, F in fires.items():
            R["signals"][s] = {fd: block(fire_rows(Dw, F, fd == "first"), cut, rng, Dw_p) for fd in ("raw", "first")}
            opp = F.assign(is_downside=1 - F["is_downside"])         # 발동 봉의 반대 측면
            R["signals"][s]["opposite_side_raw"] = block(fire_rows(Dw, opp, False), cut, rng, Dw_p)
        allF = pd.concat(fires.values(), ignore_index=True)
        pooled = {}
        for fd in ("raw", "first"):
            f = allF.loc[allF["first_fire"]] if fd == "first" else allF
            f = f.drop_duplicates(["pos", "is_downside"])
            pooled[fd] = block(fire_rows(Dw, f.assign(first_fire=True), True), cut, rng, Dw_p)
        R["pooled"] = pooled
        # Q3 F0 호출 vs 발동 시차 · 초과진행
        picks = calls_frame(Dw, Dw_p, cut); ppos = picks["pos"].to_numpy(); pside = (picks["side"].to_numpy() == "bottom").astype(int)
        allpos = Dw["pos"].to_numpy(); allside = Dw["is_downside"].to_numpy().astype(int)
        lagrep = {}; lag_frame = pd.DataFrame({"pos": ppos, "side": picks["side"].to_numpy(), "pnl_bp": picks["pnl_bp"].to_numpy(),
                                                "timestamp": picks["timestamp"].to_numpy()})
        min_same = np.full(len(ppos), 10**9); min_same_all = np.full(len(allpos), 10**9)
        for s, F in fires.items():
            L = lag_arrays(F, npos)
            lag_p = np.where(pside == 1, L[1][ppos], L[0][ppos]); lag_a = np.where(allside == 1, L[1][allpos], L[0][allpos])
            lag_frame[f"lag_{s}"] = lag_p
            min_same = np.minimum(min_same, lag_p); min_same_all = np.minimum(min_same_all, lag_a)
            ent = {}
            for k in LAGS:
                pr, br = float((lag_p <= k).mean()), float((lag_a <= k).mean())
                ent[f"<= {k}"] = {"pick_rate": round(pr, 4), "base_rate": round(br, 4), "enrichment": round(pr / br, 2) if br > 0 else None}
            # 발동 후 k봉 안 호출의 성과 vs 그 외
            near = lag_p <= 12
            ent["pnl_near12_bp"] = round(float(picks["pnl_bp"].to_numpy()[near].mean()), 2) if near.any() else None
            ent["n_near12"] = int(near.sum())
            # 초과진행: 같은 측면 첫발동(GAP) 이후 48봉 안 호출에서 (발동봉 종가 → 호출봉 종가) 방향 반대 진행, ATR(발동봉) 단위
            Lf = lag_arrays(F.loc[F["first_fire"]], npos)
            lagf = np.where(pside == 1, Lf[1][ppos], Lf[0][ppos]); m = lagf <= 48
            if m.any():
                fp = ppos[m] - lagf[m]; sgn = np.where(pside[m] == 1, 1.0, -1.0)
                exc = sgn * (close_of[fp] - close_of[ppos[m]]) / atr_of[fp]          # 롱: 발동 후 더 떨어졌으면 +
                # 무조건부 참조: 같은 창의 모든 (봉,측면) 행
                lagfa = np.where(allside == 1, Lf[1][allpos], Lf[0][allpos]); ma = lagfa <= 48
                fpa = allpos[ma] - lagfa[ma]; sga = np.where(allside[ma] == 1, 1.0, -1.0)
                exca = sga * (close_of[fpa] - close_of[allpos[ma]]) / atr_of[fpa]
                ent["excursion_since_first_fire_atr"] = {"n_picks": int(m.sum()), "pick_q25_50_75": [round(float(x), 2) for x in np.nanpercentile(exc, [25, 50, 75])],
                                                         "pick_lag_median": float(np.median(lagf[m])),
                                                         "all_rows_q25_50_75": [round(float(x), 2) for x in np.nanpercentile(exca, [25, 50, 75])],
                                                         "pick_share_beyond_1atr": round(float((exc >= 1.0).mean()), 3),
                                                         "all_share_beyond_1atr": round(float((exca >= 1.0).mean()), 3)}
            lagrep[s] = ent
        lag_frame["lag_any_signal_same_side"] = min_same
        byb = {}
        edges = [(-1, 0, "lag0"), (0, 3, "lag1_3"), (3, 12, "lag4_12"), (12, 48, "lag13_48"), (48, 10**10, "none_gt48")]
        for lo_, hi_, nm in edges:
            m = (min_same > lo_) & (min_same <= hi_)
            byb[nm] = {"n": int(m.sum()), "share": round(float(m.mean()), 3), "exp_bp": round(float(picks["pnl_bp"].to_numpy()[m].mean()), 2) if m.any() else None,
                       "base_share": round(float(((min_same_all > lo_) & (min_same_all <= hi_)).mean()), 3)}
        R["picks_vs_fires"] = {"per_signal": lagrep, "any_signal_same_side_lag_buckets": byb}
        lag_frame.to_csv(OUT / f"picks_lag_{w}.csv", index=False)
        rep["windows"][w] = R
        log(f"{w}: F0 {f0['exp_bp']:.2f}bp n{f0['n']} · pooled first n{pooled['first']['n']} above {pooled['first']['share_above_cut']} "
            f"gated {pooled['first'].get('net_gated_bp')} (n{pooled['first'].get('n_gated')}) · ({time.time()-t0:.0f}s)")

    # Q4 측면별 IC (TRAIN)
    T = D.loc[D["split"] == "TRAIN"]; feats = [c for c in cols if c != "is_downside"]
    allF = pd.concat(fires.values(), ignore_index=True).drop_duplicates(["pos", "is_downside"])
    TF = T.merge(allF[["pos", "is_downside"]], on=["pos", "is_downside"], how="inner")
    piv = T.pivot_table(index="pos", columns="is_downside", values="net_bp"); piv = piv.dropna()
    Tl = T.loc[T["is_downside"] == 1].set_index("pos").reindex(piv.index)
    ddir = (piv[1] - piv[0]).to_numpy()
    firebars = np.isin(piv.index.to_numpy(), allF["pos"].unique())
    sub = np.zeros(len(piv), bool); sub[rng.choice(len(piv), size=min(120000, len(piv)), replace=False)] = True
    ic = {}
    for c in feats:
        v = pd.to_numeric(Tl[c], errors="coerce").to_numpy(float); ok = np.isfinite(v)
        r_all = spearmanr(v[ok & sub], ddir[ok & sub]).correlation
        r_fire = spearmanr(v[ok & firebars], ddir[ok & firebars]).correlation if (ok & firebars).sum() > 200 else np.nan
        row = {"dir_ic_every_bar": round(float(r_all), 4), "dir_ic_fire_bars": round(float(r_fire), 4) if np.isfinite(r_fire) else None}
        for sd_, nm in ((1, "long"), (0, "short")):
            a = T.loc[T["is_downside"] == sd_]; b = TF.loc[TF["is_downside"] == sd_]
            ia = a.index[rng.choice(len(a), size=min(120000, len(a)), replace=False)]
            va = pd.to_numeric(a.loc[ia, c], errors="coerce").to_numpy(float); na = a.loc[ia, "net_bp"].to_numpy(float); oa = np.isfinite(va)
            vb = pd.to_numeric(b[c], errors="coerce").to_numpy(float); nb = b["net_bp"].to_numpy(float); ob = np.isfinite(vb)
            row[f"ic_{nm}_every_bar"] = round(float(spearmanr(va[oa], na[oa]).correlation), 4)
            row[f"ic_{nm}_fires"] = round(float(spearmanr(vb[ob], nb[ob]).correlation), 4) if ob.sum() > 200 else None
        ic[c] = row
    top_dir = sorted(feats, key=lambda c: -abs(ic[c]["dir_ic_every_bar"]))[:8]
    rep["direction_ic_train"] = {"n_bars_every": int(len(piv)), "n_fire_bars": int(firebars.sum()), "n_fire_rows": int(len(TF)),
                                 "top8_by_every_bar_dir_ic": {c: ic[c] for c in top_dir}, "all": ic}
    # 검정력: F0 꼬리 효과를 ±2bp로 재는 데 필요한 n
    sd_tr = rep["windows"]["OOS"]["f0_trade_sd_bp"]
    rep["power"] = {"f0_trade_sd_bp": sd_tr, "n_for_se_2bp": int((sd_tr / 2.0) ** 2),
                    "pooled_first_gated_n": {w: rep["windows"][w]["pooled"]["first"].get("n_gated") for w in WINDOWS},
                    "per_signal_first_n": {w: {s: rep["windows"][w]["signals"][s]["first"]["n"] for s in SIGNALS} for w in WINDOWS}}
    # 사전등록 판정 (pooled/first 하나)
    pv, po = rep["windows"]["VAL"]["pooled"]["first"], rep["windows"]["OOS"]["pooled"]["first"]
    gv, go = pv.get("gated_portfolio") or {}, po.get("gated_portfolio") or {}
    rescued = bool(gv.get("exp_bp", -1) > 0 and go.get("exp_bp", -1) > 0 and (pv.get("flip_gated_bp") is not None and pv["net_gated_bp"] > pv["flip_gated_bp"])
                   and (po.get("flip_gated_bp") is not None and po["net_gated_bp"] > po["flip_gated_bp"]) and (po.get("gated_day_ci95", [-1])[0] > 0))
    rep["prereg"] = {"rule": "pooled/first: VAL&OOS gated portfolio exp>0, both flip-inferior, OOS gated day-CI lower>0", "rescued": rescued}
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    # 요약 표
    print(f"\n{'window':>5s} {'signal':>26s} {'def':>6s} {'n':>6s} {'above%':>7s} {'pctl':>5s} {'net_all':>8s} {'gated':>8s} {'n_g':>5s} {'flip_g':>8s} {'ungated':>8s} {'gport':>7s} {'CI':>16s}")
    for w in WINDOWS:
        R = rep["windows"][w]
        print(f"{w:>5s} {'F0 overall':>26s} {'':>6s} {R['f0_overall']['n']:6d} {R['base_share_above_cut']*100:6.1f}% {'':>5s} {'':>8s} {R['f0_overall']['exp_bp']:8.2f}")
        for s in SIGNALS + ["pooled"]:
            for fd in ("raw", "first"):
                b = R["pooled"][fd] if s == "pooled" else R["signals"][s][fd]
                if not b.get("n"):
                    continue
                gp = b.get("gated_portfolio") or {}
                print(f"{w:>5s} {s:>26s} {fd:>6s} {b['n']:6d} {b['share_above_cut']*100:6.1f}% {b['p_median_pctile']:5.2f} {b['net_all_bp']:8.2f} "
                      f"{(b['net_gated_bp'] if b['net_gated_bp'] is not None else float('nan')):8.2f} {b['n_gated']:5d} "
                      f"{(b['flip_gated_bp'] if b['flip_gated_bp'] is not None else float('nan')):8.2f} "
                      f"{(b['net_ungated_bp'] if b['net_ungated_bp'] is not None else float('nan')):8.2f} "
                      f"{(gp.get('exp_bp') if gp else float('nan')):7.2f} {str(b.get('gated_day_ci95', '')):>16s}")
    print("\n[picks vs fires: any-signal same-side lag buckets]")
    for w in WINDOWS:
        print(w, json.dumps(rep["windows"][w]["picks_vs_fires"]["any_signal_same_side_lag_buckets"]))
    print("\n[direction IC top8 every-bar vs fire bars]")
    for c, r in rep["direction_ic_train"]["top8_by_every_bar_dir_ic"].items():
        print(f"  {c:>24s} dir_every {r['dir_ic_every_bar']:+.4f} dir_fire {r['dir_ic_fire_bars']} | long every {r['ic_long_every_bar']:+.4f} fires {r['ic_long_fires']} | short every {r['ic_short_every_bar']:+.4f} fires {r['ic_short_fires']}")
    print("\n[prereg]", rep["prereg"], "\n[power]", json.dumps(rep["power"]))
    log(f"diag 완료 -> {OUT/'report.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["score", "diag"], required=True)
    a = ap.parse_args(); stage_score() if a.stage == "score" else stage_diag()
