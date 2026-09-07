#!/usr/bin/env python3
"""현행 모델들을 **MASHT 피쳐로 갈아끼워** 짝비교 (2026-09-07, 서버 GPU).

사용자: *"TabPFN/MASHT 모델을 써서 기존에 쓰던 증거신호와 v자 급등락과
       v자반등자동매매 모델들을 TabPFN/MASHT로 바꿔서 테스트 진행해줘"*

## 무엇이 바뀌는가
현행 8종 칩·V자 급등락은 **이미 TabPFN 을 쓴다**. 바뀌는 것은 **피쳐 표현**이다:
  현행(A)  발동 봉에서 손으로 만든 23~26개 스칼라(delta_z·atr·rsi·bb_pctb·…)
  MASHT(B) 발동 봉을 **포함해 끝나는 48봉 × 8채널** 창 → MultiRocket 2016 + Hydra 768 = 2784열
  합침(C)  A + B
같은 모집단·같은 라벨·같은 분할에서 A/B/C 를 나란히 놓는다.

## 채널 (앵커 러너와 동일 정의를 import 로 재사용)
logret · path_atr · dem14 · p_fast · delta_z · ret3_z · kalman_dev_z · hl_range
방향 채널은 `side` 로 정렬한다(bottom=-1/top=+1) -- 라벨이 측면별로 정의돼 있기 때문.

## 규약
- Rocket 변환은 **TRAIN 에서만 fit**, 평가창은 transform 만 (배포 형태).
- 분할: TRAIN ≤2025-08-31 · VAL 2025-09~12 · OOS 2026-01~03 · HOLDOUT_SPENT 2026-04~
  ⚠️HOLDOUT 구간은 이 저장소에서 이미 소진됐다 -- **research/dev score** 로만 읽는다.
- 판정은 절대 AUC 가 아니라 **A 대비 짝비교 증분**(같은 날 표집 CI). 원시 차이로 판단 금지.
- 🔴**누수 가드**: 1차 실행에서 `orthogonal_combo`/`smt_divergence` 의 A팔 AUC 가 **1.0000** 이었다.
  두 CSV 에 `move_atr_mult`(= 라벨 `hit` 의 정의 그 자체, K×ATR 도달배수)가 피쳐로 들어 있었다.
  이 저장소에서 재발한 함정이다(BTC/ETH 튜닝 파리티 감사에서 같은 컬럼이 같은 방식으로 샜다).
  1차 수정은 "TRAIN 단일피쳐 AUC>0.95 자동 제외" 가드였는데 **두 번 더 실패했다**:
    (a) `Xa` 를 가드 **앞에서** 만들어 버려 "제외했다"고 출력하면서 그대로 넣었다
        (헤더 피쳐수 23 vs 열수 A 24 불일치가 증거).
    (b) 임계를 통과하는 미래파생이 남는다 -- `mae_atr_mult`(실현 최대역행폭)는 0.8502 라 살아남는다.
  ⇒ 최종 해법은 임계가 아니다. **A팔은 배포가 실제로 쓰는 피쳐 목록만 쓴다**
     (`live_evidence_signal_metalabel_20260829.py::METALABEL_SIGNALS[*]["feature_columns"]`,
      기본 `FEATURE_COLUMNS` 23개 · V자는 `live_eth_sweep_v_rebound_signal_20260829.FEATURES`).
     그 목록에는 미래파생이 하나도 없음을 교차확인했다. 이러면 비교가 정확히
     **"배포 피쳐 vs MASHT"** 가 되고, 추측이 사라진다. 임계 가드는 이중 안전장치로 남긴다.
- 현행 배포 수치(각 신호의 공표 AUC)와 내 A팔 수치는 프로토콜이 달라 다를 수 있다.
  비교는 **내 하네스 안의 A vs B** 로만 한다.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/eth_masht_swap_20260907"
K_WIN = 48
SEED = 20260907
N_EST = 4
BOOT = 1200
L = "data/labels"

TASKS = [
    ("taker_delta_climax", f"{L}/eth_5m_taker_delta_climax_metalabel_v5_gap12_20260830/eth_5m_taker_delta_climax_metalabel_v5_gap12_features.csv", "hit"),
    ("short_term_return_z", f"{L}/eth_5m_short_term_return_z_metalabel_20260829/eth_5m_short_term_return_z_metalabel_features.csv", "hit"),
    ("liquidity_sweep", f"{L}/eth_5m_liquidity_sweep_topdown_metalabel_20260830/eth_5m_liquidity_sweep_topdown_metalabel_features_H30_GAP12_K4.0.csv", "hit"),
    ("orthogonal_combo", f"{L}/eth_5m_orthogonal_combo_metalabel_20260830/eth_5m_orthogonal_combo_metalabel_features.csv", "hit"),
    ("smt_divergence", f"{L}/eth_5m_smt_divergence_metalabel_20260831/eth_5m_smt_divergence_metalabel_features.csv", "hit"),
    ("fib_extension", f"{L}/eth_5m_fib_extension_exhaustion_metalabel_20260831/eth_5m_fib_extension_exhaustion_metalabel_FINAL_features.csv", "hit"),
    ("dalton_rule2", f"{L}/eth_5m_dalton_rule2_balance_edge_metalabel_20260830/eth_5m_dalton_rule2_balance_edge_metalabel_features.csv", "hit"),
    ("v_rebound(V자급등락)", f"{L}/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv", "label"),
]
SPLITS = [("TRAIN", None, "2025-09-01"), ("VAL", "2025-09-01", "2026-01-01"),
          ("OOS", "2026-01-01", "2026-04-01"), ("HOLDOUT_SPENT", "2026-04-01", None)]
DROP = {"pos", "timestamp", "side", "hit", "label", "pred_dir_ret", "is_bottom"}


def split_of(ts: pd.Series) -> np.ndarray:
    out = np.array(["?"] * len(ts), dtype=object)
    for name, lo, hi in SPLITS:
        m = np.ones(len(ts), bool)
        if lo: m &= (ts >= lo).to_numpy()
        if hi: m &= (ts < hi).to_numpy()
        out[m] = name
    return out


def day_ci(y, p, d, rng, B=BOOT):
    u = np.unique(d)
    if len(u) < 5: return (np.nan, np.nan)
    idx = {x: np.flatnonzero(d == x) for x in u}
    o = [roc_auc_score(y[i], p[i]) for i in
         (np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)]) for _ in range(B))
         if len(np.unique(y[i])) > 1]
    return (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))) if len(o) > B // 3 else (np.nan, np.nan)


def diff_ci(y, p1, p2, d, rng, B=BOOT):
    u = np.unique(d); idx = {x: np.flatnonzero(d == x) for x in u}; o = []
    for _ in range(B):
        i = np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)])
        if len(np.unique(y[i])) > 1:
            o.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))) if len(o) > B // 3 else (np.nan, np.nan)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    import importlib.util, torch
    from aeon.transformations.collection.convolution_based import MultiRocket, HydraTransformer
    from tabpfn import TabPFNClassifier

    ML = None
    try:
        import importlib.util as _iu
        _s = _iu.spec_from_file_location("mlfeat", ROOT / "scripts/live_evidence_signal_metalabel_20260829.py")
        ML = _iu.module_from_spec(_s); _s.loader.exec_module(ML)
    except Exception as e:
        print(f"⚠️배포 피쳐 목록 로드 실패({type(e).__name__}) -- 숫자컬럼 전체로 폴백", flush=True)
    VR = None
    try:
        import importlib.util as _iu2
        _s2 = _iu2.spec_from_file_location("vrfeat", ROOT / "scripts/live_eth_sweep_v_rebound_signal_20260829.py")
        VR = _iu2.module_from_spec(_s2); _s2.loader.exec_module(VR)
    except Exception:
        pass

    def deployed_features(task_name: str, cols) -> list[str] | None:
        """배포가 실제로 모델에 넣는 피쳐 목록. 없으면 None(폴백)."""
        key = {"taker_delta_climax": "taker_delta_z_climax", "short_term_return_z": "short_term_return_z",
               "liquidity_sweep": "liquidity_sweep", "orthogonal_combo": "orthogonal_combo",
               "smt_divergence": "smt_divergence", "fib_extension": "fib_extension_exhaustion"}.get(task_name)
        if key and ML is not None and key in ML.METALABEL_SIGNALS:
            fc = ML.METALABEL_SIGNALS[key].get("feature_columns", ML.FEATURE_COLUMNS)
        elif task_name.startswith("v_rebound") and VR is not None:
            fc = list(getattr(VR, "FEATURES", []))
        elif ML is not None:
            fc = ML.FEATURE_COLUMNS          # dalton 등 -- 공통 23개
        else:
            return None
        keep = [c for c in fc if c in cols]
        return keep if len(keep) >= 10 else None

    def _load(n, p):
        s = importlib.util.spec_from_file_location(n, ROOT / p)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
    B = _load("ab", "scripts/build_eth_anchor_label_dataset_20260907.py")
    OC = _load("oc", "scripts/build_eth_anchor_oscillator_cores_20260907.py")
    WB = _load("wb", "scripts/build_eth_anchor_window_tensor_20260907.py")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    eth = B._load_kl(B.ETH_KL); fund = B._load_funding(); btc = B._load_kl(B.BTC_KL)
    tmax = min(eth["timestamp"].max(), btc["timestamp"].max(), fund["calc_time"].max())
    eth = eth[eth["timestamp"] <= tmax].reset_index(drop=True)
    sig = B.compute_signals(eth, btc_df=btc[btc["timestamp"] <= tmax],
                            funding_df=fund[fund["calc_time"] <= tmax])
    C = OC.compute_cores(sig, B)
    close = sig["close"].to_numpy(float); high = sig["high"].to_numpy(float)
    low = sig["low"].to_numpy(float); atr = sig["atr_pct"].to_numpy(float)
    BAR = {"logret": np.concatenate([[np.nan], np.diff(np.log(close))]),
           "dem14": sig["dem"].to_numpy(float), "p_fast": C["p_fast"].to_numpy(float),
           "delta_z": C["delta_z"].to_numpy(float), "ret3_z": C["ret3_z"].to_numpy(float),
           "kalman_dev_z": sig["kalman_dev_z"].to_numpy(float),
           "hl_range": (high - low) / np.maximum(close, 1e-12)}
    TS = pd.to_datetime(sig["timestamp"].to_numpy())
    POS = pd.Series(np.arange(len(TS)), index=TS)
    print(f"[준비] 신호 프레임 {len(sig):,} (상한 {tmax}) · device {dev} · 채널 {WB.CHANNELS}", flush=True)

    def windows(tss: pd.Series, sides: np.ndarray):
        idx = POS.reindex(tss).to_numpy()
        ok = ~pd.isna(idx)
        idx = np.where(ok, idx, 0).astype(int)
        cs = np.where(sides == "top", 1.0, -1.0)
        n = len(tss); X = np.full((n, len(WB.CHANNELS), K_WIN), np.nan, np.float32)
        tsv = TS.to_numpy(); offs = np.arange(K_WIN - 1, -1, -1) * np.timedelta64(5, "m")
        good = np.zeros(n, bool)
        for i in range(n):
            if not ok[i] or idx[i] - K_WIN + 1 < 0: continue
            w = slice(idx[i] - K_WIN + 1, idx[i] + 1)
            if not np.array_equal(tsv[w], tss.to_numpy()[i] - offs): continue
            cl = close[w]; a = atr[idx[i]]
            for c, ch in enumerate(WB.CHANNELS):
                v = (cl - cl[-1]) / max(a * cl[-1], 1e-12) if ch == "path_atr" else BAR[ch][w]
                if ch in WB.DIRECTIONAL: v = v * cs[i]
                elif ch in WB.CENTERED: v = 0.5 + (v - 0.5) * cs[i]
                X[i, c, :] = v
            good[i] = np.isfinite(X[i]).all()
        return X, good

    rows, incs = [], []
    for name, path, lab in TASKS:
        f = ROOT / path
        if not f.exists():
            print(f"\n[{name}] 파일 없음 -- 건너뜀"); continue
        D = pd.read_csv(f)
        tc = next(c for c in D.columns if "time" in c.lower())
        D[tc] = pd.to_datetime(D[tc], utc=True).dt.tz_localize(None)   # ⚠️v_rebound 는 tz-aware
        D = D.rename(columns={tc: "timestamp"}).sort_values("timestamp").reset_index(drop=True)
        y_all = D[lab].to_numpy()
        sides = D["side"].to_numpy() if "side" in D else np.array(["bottom"] * len(D))
        dep = deployed_features(name, set(D.columns))
        if dep:
            feats, feat_src = dep, f"배포목록 {len(dep)}"
        else:
            feats = [c for c in D.columns if c not in DROP and D[c].dtype.kind in "fib"]
            feat_src = f"폴백(숫자전체) {len(feats)}"
        Xw, good = windows(D["timestamp"], sides)
        sp = split_of(D["timestamp"]); day = D["timestamp"].dt.floor("D").to_numpy()
        ok = good & np.isfinite(y_all)
        tr = ok & (sp == "TRAIN")
        # 🔴누수 가드: TRAIN 단일피쳐 AUC 가 0.95 를 넘으면 라벨 파생으로 보고 제외한다.
        leaked = []
        if tr.sum() >= 50 and len(np.unique(y_all[tr])) > 1:
            yt = y_all[tr].astype(int)
            for c in list(feats):
                v = D[c].to_numpy(float)[tr]
                m = np.isfinite(v)
                if m.sum() < 50 or len(np.unique(yt[m])) < 2:
                    continue
                a = roc_auc_score(yt[m], v[m])
                if max(a, 1 - a) > 0.95:
                    leaked.append((c, round(a, 4))); feats.remove(c)
        if leaked:
            print(f"   🔴누수 제외 {leaked}", flush=True)
        # ⚠️Xa 는 **가드 이후에** 만든다. 1차 수정판은 가드 앞에서 만들어 버려
        #   "제외했다"고 출력하면서 실제로는 그대로 넣었다(헤더 피쳐수 23 vs 열수 A 24 불일치).
        Xa = D[feats].to_numpy(np.float64)
        print(f"\n[{name}] {len(D):,}행 · 창유효 {good.sum():,} · 현행피쳐 {len(feats)} "
              f"· TRAIN {tr.sum():,} · 양성률 {y_all[ok].mean():.3f} · A팔={feat_src}", flush=True)
        if tr.sum() < 200:
            print("   TRAIN 부족 -- 건너뜀"); continue
        t0 = time.time()
        mr = MultiRocket(n_kernels=252, random_state=SEED, n_jobs=10).fit(Xw[tr].astype(np.float32))
        hy = HydraTransformer(n_kernels=8, n_groups=16, random_state=SEED, n_jobs=10).fit(Xw[tr].astype(np.float32))
        Z = np.full((len(D), 2784), np.nan, np.float32)
        sel = np.flatnonzero(ok)
        Z[sel] = np.concatenate([np.asarray(mr.transform(Xw[sel].astype(np.float32))),
                                 np.asarray(hy.transform(Xw[sel].astype(np.float32)))], axis=1)
        ARMS = {"A_현행": Xa, "B_MASHT": Z, "C_합침": np.concatenate([Xa, Z], axis=1)}
        print(f"   변환 {time.time()-t0:.0f}s · 열수 A {Xa.shape[1]} / B {Z.shape[1]} / C {ARMS['C_합침'].shape[1]}", flush=True)
        keep = {}
        for arm, Xm in ARMS.items():
            t1 = time.time()
            clf = TabPFNClassifier(device=dev, n_estimators=N_EST, random_state=SEED,
                                   ignore_pretraining_limits=True, memory_saving_mode=True)
            clf.fit(np.nan_to_num(Xm[tr]).astype(np.float32), y_all[tr].astype(int))
            rec = {"task": name, "arm": arm, "n_feat": Xm.shape[1], "n_train": int(tr.sum())}
            keep[arm] = {}
            for w, _, _ in SPLITS[1:]:
                te = ok & (sp == w)
                if te.sum() < 40 or len(np.unique(y_all[te])) < 2: continue
                p = clf.predict_proba(np.nan_to_num(Xm[te]).astype(np.float32))[:, 1]
                a = roc_auc_score(y_all[te].astype(int), p)
                lo, hi = day_ci(y_all[te].astype(int), p, day[te], rng)
                rec[f"{w}_auc"], rec[f"{w}_lo"], rec[f"{w}_n"] = a, lo, int(te.sum())
                keep[arm][w] = (y_all[te].astype(int), p, day[te])
            rec["mean3"] = np.nanmean([rec.get(f"{w}_auc", np.nan) for w, _, _ in SPLITS[1:]])
            if arm == "A_현행" and rec["mean3"] > 0.99:
                rec["LEAK_FAIL"] = True
                print(f"   🔴🔴A팔 mean3 {rec['mean3']:.4f} > 0.99 -- 잔여 누수. 이 과제 비교 무효.",
                      flush=True)
            rows.append(rec)
            print(f"   {arm:<9} " + " · ".join(
                f"{w[:3]} {rec.get(f'{w}_auc', float('nan')):.4f}[{rec.get(f'{w}_lo', float('nan')):.3f}]"
                for w, _, _ in SPLITS[1:]) + f"  mean3 {rec['mean3']:.4f} ({time.time()-t1:.0f}s)", flush=True)
        for arm in ("B_MASHT", "C_합침"):
            r = {"task": name, "arm": arm}
            for w, _, _ in SPLITS[1:]:
                if w in keep.get(arm, {}) and w in keep.get("A_현행", {}):
                    y, pb, d = keep[arm][w]; _, pa, _ = keep["A_현행"][w]
                    r[f"{w}_d"] = roc_auc_score(y, pb) - roc_auc_score(y, pa)
                    r[f"{w}_lo"], r[f"{w}_hi"] = diff_ci(y, pb, pa, d, rng)
            r["n_win_gt0"] = sum(1 for w, _, _ in SPLITS[1:]
                                 if np.isfinite(r.get(f"{w}_lo", np.nan)) and r[f"{w}_lo"] > 0)
            r["n_win_lt0"] = sum(1 for w, _, _ in SPLITS[1:]
                                 if np.isfinite(r.get(f"{w}_hi", np.nan)) and r[f"{w}_hi"] < 0)
            incs.append(r)
            print(f"   Δ{arm[0]}-A   " + " · ".join(
                f"{w[:3]} {r.get(f'{w}_d', float('nan')):+.4f}[{r.get(f'{w}_lo', float('nan')):+.3f},{r.get(f'{w}_hi', float('nan')):+.3f}]"
                for w, _, _ in SPLITS[1:]) + f"  창>0 {r['n_win_gt0']}/3 · 창<0 {r['n_win_lt0']}/3", flush=True)

    A = pd.DataFrame(rows); I = pd.DataFrame(incs)
    A.to_csv(OUT / "swap_arms.csv", index=False); I.to_csv(OUT / "swap_increments.csv", index=False)
    print("\n" + "=" * 104, flush=True)
    print("요약 — MASHT 가 현행 피쳐를 이기는가 (짝비교 CI 하한>0 인 창 수)", flush=True)
    print("=" * 104, flush=True)
    for arm in ("B_MASHT", "C_합침"):
        s = I[I.arm == arm]
        print(f"\n  [{arm}]  두 창 이상 우세 {int((s.n_win_gt0>=2).sum())}/{len(s)} 과제 · "
              f"두 창 이상 열세 {int((s.n_win_lt0>=2).sum())}/{len(s)}")
        for _, r in s.iterrows():
            v = "✅우세" if r.n_win_gt0 >= 2 else ("🔴열세" if r.n_win_lt0 >= 2 else "동등")
            print(f"    {r.task:<22}{v}  " + " · ".join(
                f"{w[:3]} {r.get(f'{w}_d', float('nan')):+.4f}" for w, _, _ in SPLITS[1:]))
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
