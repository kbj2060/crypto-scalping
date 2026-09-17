#!/usr/bin/env python3
"""Zeus L4 — **실제 부모 후보 집합**에서 사이징 경주 (2026-09-17, dev/CPU)

## 왜 다시
09-16 전체판·09-17 더블배리어판은 **무작위 진입**이었다. 우리 부모는 무작위가 아니다 --
후보 봉의 ATR 중앙값이 전체 봉의 **64.8%**(조용한 봉에서 발화). 그래서 역변동성 사이징이
쓸 변동성 범위가 좁다. **실제 후보에서 다시 재야 한다.**

## ⭐새 축 — p 기반 켈리
더블 배리어는 페이오프가 고정(+TP/−SL)이라 건당 분산이 **오직 p 의 함수**다.
그리고 부모의 **방향 확신도**가 p 를 안다(stageN: 10분위 Δp **+4.52pp** CI[+1.81,+7.24] ·
단조 0.974). 그러니 크기는 변동성이 아니라 **p** 로 정해야 한다:

    실효배당 b = (TP − 비용)/(SL + 비용) = 1.4748
    f* = (p(1+b) − 1)/b          ← 손익분기 p*=40.408% 에서 정확히 0

p 는 **방향 확신도를 TRAIN 쪽 분위로 보정**해 만든다(TEST 미사용). 이게 L4 의 최소판이다.

## 팔 (사전 지정 · ①고정을 1번 팔로)
  ①고정 · 역ATR 1/atr14 · 1/atr288 · 동일위험 1/safeMAE · **⭐켈리(p)** · **⭐켈리(p) 상한 0.5×**
판정: 위험 분포(SD·하위1%·50배 청산율) + 로그성장 + 하루 순bp. 평균 명목 정규화.
⚠️[[eth_kelly_is_leverage_not_split_and_ignores_path_20260915]] — f* 는 배수지 진입비율이 아니다.
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402
import research_omega461_parent_quality_calibration_and_bias_20260917 as F  # noqa: E402
import research_omega461_side_skill_decomposition_20260917 as K  # noqa: E402

OUT = E.OUT
TPB, SLB, COST = K.BASE_TP * 1e4, K.BASE_SL * 1e4, 1.02
B_EFF = (TPB - COST) / (SLB + COST)
BE = (SLB + COST) / (TPB + SLB)
LIQ_X = 50.0


HERE = Path(__file__).resolve().parents[1]   # 워크트리. 일부 스크립트는 여기에만 있다.


def _mod(rel, name):
    path = HERE / rel
    if not path.exists():
        path = ROOT / rel
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); argv, sys.argv = sys.argv, ["x"]
    try: sp.loader.exec_module(m)
    finally: sys.argv = argv
    return m


def log(*a): print(*a, flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    assert abs((BE * (1 + B_EFF) - 1) / B_EFF) < 1e-12, "손익분기에서 f*≠0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} · 더블배리어 TP{K.BASE_TP*100:g}%/SL{K.BASE_SL*100:g}% · "
        f"손익분기 p*={BE*100:.3f}% · 실효배당 b={B_EFF:.4f}")
    MQ = _mod("scripts/live_eth_mae_quantile_model_20260913.py", "MQ")
    SV = _mod("scripts/live_eth_sizing_vol_model_20260912.py", "SV")
    for mod, fn in ((MQ, "eth_mae_quantile_model.joblib"), (SV, "eth_sizing_vol_model.joblib"),
                    (MQ.svm, "eth_sizing_vol_model.joblib")):
        mod.ARTIFACT = ROOT / "data" / "live" / fn
        if hasattr(mod, "_CACHE"): mod._CACHE.clear()
    art = MQ.load_model(); assert art and art.get("models"), "MAE 분위 모델 로드 실패"

    df, base_cols = E.load()
    bun = torch.load(E.BUNDLE, map_location="cpu", weights_only=False)
    experts = {}
    for en in ("bull", "bear", "chop"):
        pay = dict(bun["models"][en])
        m = tabm.ThreeHeadTabM(int(pay["n_features"]), cfg=tabm.ThreeHeadConfig(**dict(pay["config"]))).to(device)
        m.load_state_dict(pay["state_dict"]); m.eval()
        experts[en] = (m, dict(pay["scaler"]))

    rows_all = {}
    for name, t0, t1, v0, v1 in K.FOLDS:
        if not (v1 < K.DEPLOYED_SEEN[0] or v0 > K.DEPLOYED_SEEN[1]):
            continue
        te = df[(df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")].reset_index(drop=True)
        tr = df[(df.timestamp >= t0) & (df.timestamp <= t1 + " 23:59:59")].reset_index(drop=True)
        xr = tabm._base_input(te, base_cols); ev = tabm._route_probs(te).argmax(1)
        xt = tabm._base_input(tr, base_cols); evt = tabm._route_probs(tr).argmax(1)
        D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
        Dt = np.zeros((len(tr), 3))
        for ei, en in enumerate(("bull", "bear", "chop")):
            m, sc = experts[en]
            s_ = ev == ei
            if s_.any(): D[s_], Q[s_] = E.heads(m, tabm._standardize_apply(xr[s_], sc), device)
            st = evt == ei
            if st.any(): Dt[st], _ = E.heads(m, tabm._standardize_apply(xt[st], sc), device)
        _, side = F.gate(D, Q, E.Q_THRESH)
        idx = np.where(side != 0)[0]
        if len(idx) < 100: continue
        conf = D.max(1)[idx]
        rows_all[name] = (te, side, idx, conf, Dt, tr)
        log(f"  {name}: 후보 {len(idx):,} · 확신도 중앙 {np.median(conf):.3f}")

    # TRAIN 확신도 분위 <-> 실현 p 는 TRAIN 에서 배리어를 돌려야 알 수 있다 -> 그것도 계산
    pnl_all, w_all, day_all, mae_all = [], {}, [], []
    kel, kel_cap = [], []
    atr14s, atr288s, safes = [], [], []
    for name, (te, side, idx, conf, Dt, tr) in rows_all.items():
        hi = pd.to_numeric(te["high"]).to_numpy(float); lo = pd.to_numeric(te["low"]).to_numpy(float)
        cl = pd.to_numeric(te["close"]).to_numpy(float)
        r, h, _rs, _rn, mae = K._first_touch_open(idx, side, hi, lo, cl, K.BASE_TP, K.BASE_SL, K.MAXBARS)
        pnl_all.append(r * 1e4 - COST); mae_all.append(np.maximum(-mae * 1e4, 0.0))
        day_all.append(te.timestamp.dt.floor("D").to_numpy()[idx])
        # ── TRAIN 에서 «확신도 분위별 실현 p» 를 만든다 (TEST 미사용, 인과적) ──
        trhi = pd.to_numeric(tr["high"]).to_numpy(float); trlo = pd.to_numeric(tr["low"]).to_numpy(float)
        trcl = pd.to_numeric(tr["close"]).to_numpy(float)
        dat = Dt.argmax(1)
        cand_tr = np.where((dat != 0) & (np.arange(len(tr)) < len(tr) - K.MAXBARS - 1))[0]
        cand_tr = cand_tr[::7]                      # 부표집(속도). 비율 추정엔 충분하다.
        assert len(cand_tr) > 2000, f"{name} TRAIN 후보 부족 {len(cand_tr)}"
        side_tr = np.zeros(len(tr))
        side_tr[cand_tr] = np.where(dat[cand_tr] == 1, 1.0, -1.0)   # ⭐모델이 고른 측면
        _r, _h, _rs2, rn_tr, _m2 = K._first_touch_open(cand_tr, side_tr, trhi, trlo, trcl,
                                                       K.BASE_TP, K.BASE_SL, K.MAXBARS)
        hit_tr = (rn_tr == 1).astype(float)
        conf_tr = Dt.max(1)[cand_tr]
        qs = np.quantile(conf_tr, np.linspace(0, 1, 11)[1:-1])      # TRAIN 분위 경계
        b_tr = np.digitize(conf_tr, qs)
        p_bin = np.array([hit_tr[b_tr == k].mean() if (b_tr == k).sum() > 50 else np.nan
                          for k in range(10)])
        p_bin = pd.Series(p_bin).ffill().bfill().to_numpy()         # 빈 분위는 이웃으로
        mono = float(np.corrcoef(np.arange(10), p_bin)[0, 1])
        log(f"    {name} TRAIN 보정: 후보 {len(cand_tr):,} · p {p_bin.min()*100:.2f}~{p_bin.max()*100:.2f}% "
            f"· 분위-p 상관 {mono:+.3f}")
        p_hat = np.clip(p_bin[np.clip(np.digitize(conf, qs), 0, 9)], 0.30, 0.65)
        assert p_hat.std() > 1e-6, "p 추정이 상수다 -- 켈리 팔이 고정과 같아진다"
        f = np.clip((p_hat * (1 + B_EFF) - 1) / B_EFF, 0.0, None)
        kel.append(f); kel_cap.append(np.minimum(f, 0.5 * np.quantile(f, 0.99)))
        tr_ = np.maximum(hi - lo, np.maximum(np.abs(hi - np.roll(cl, 1)), np.abs(lo - np.roll(cl, 1))))
        a14 = pd.Series(tr_).rolling(14, min_periods=14).mean().to_numpy() / np.maximum(cl, 1e-12)
        a288 = pd.Series(tr_).rolling(288, min_periods=144).mean().to_numpy() / np.maximum(cl, 1e-12)
        atr14s.append(a14[idx]); atr288s.append(a288[idx])
        base = MQ.svm.build_features(te.timestamp, cl, pd.to_numeric(te["quote_volume"]).to_numpy(float),
                                     pd.to_numeric(te["trades"]).to_numpy(float), hi, lo)
        rr = base.iloc[idx].copy().reset_index(drop=True)
        rr["log_h"] = np.log(np.maximum(h, 1) * 5.0); rr["side"] = side[idx]
        safes.append(MQ.safe_mae(art["models"], rr[MQ.FEATURES], art["mult"]))

    pnl = np.concatenate(pnl_all); mae = np.concatenate(mae_all); days = np.concatenate(day_all)
    FD = _mod("scripts/research_sizing_horserace_fulldata_20260916.py", "FD")
    arms = {"①고정": np.ones(len(pnl)),
            "역ATR 1/atr14": 1.0 / np.maximum(np.concatenate(atr14s), 1e-9),
            "역ATR 1/atr288": 1.0 / np.maximum(np.concatenate(atr288s), 1e-9),
            "동일위험 1/safeMAE": 1.0 / np.maximum(np.concatenate(safes), 1e-9),
            "⭐켈리(p)": np.concatenate(kel) + 1e-9,
            "⭐켈리(p) 상한0.5×": np.concatenate(kel_cap) + 1e-9}
    arms = {k: v for k, v in arms.items() if np.isfinite(v).all() and v.sum() > 0}
    log(f"\n후보 {len(pnl):,}건 · 고유일 {len(np.unique(days)):,} · 평균 {pnl.mean():+.2f}bp")
    log(f"{'규칙':<22}{'가중CV':>8}{'평균bp':>9}{'SD':>9}{'vs고정':>8}{'하위1%':>10}{'vs고정':>8}{'50배청산율':>11}")
    res = []
    fx = None
    for k, w in arms.items():
        s = FD.summarize(pnl, w, days, mae)
        if fx is None: fx = s
        res.append({"arm": k, **s, "d_sd": (s["sd"]/fx["sd"]-1)*100, "d_p1": (s["p1"]/fx["p1"]-1)*100})
        log(f"{k:<22}{s['cv']:>8.2f}{s['mean']:>+9.2f}{s['sd']:>9.1f}{(s['sd']/fx['sd']-1)*100:>+7.1f}%"
            f"{s['p1']:>10.1f}{(s['p1']/fx['p1']-1)*100:>+7.1f}%{s['liq']*100:>10.2f}%")
    (OUT / "stageR_l4_sizing_candidates.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"\n저장: {OUT}/stageR_l4_sizing_candidates.json")
    log("  (SD·하위1%·청산율은 작을수록 좋다. 평균은 사이징이 만들지 않는다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
