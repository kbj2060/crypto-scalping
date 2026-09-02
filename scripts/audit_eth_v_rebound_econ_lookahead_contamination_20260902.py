#!/usr/bin/env python3
"""E0 경제라벨 자동매매 후보의 **룩어헤드/오염 심층 감사** -- 섀도우 운용 전 관문.

## 왜 지금

이 파이프라인은 2026-09-02에 새로 만든 것이라 기존 Tier0 감사
(`reference_tier0_23_feature_lookahead_and_contamination_audit_20260830`)가 커버하지 않는
경로가 있다. 특히 **피쳐가 항상 전체 CSV로 계산된다** -- `build_all_bar_frame()`은
`_tier0.SOURCE`를 통째로 읽고 VAL_END와 무관하다. 롤링만 쓰면 무해하지만 전역 통계가
하나라도 섞이면 HOLDOUT 데이터가 TRAIN 피쳐를 오염시킨다.

## 검사 항목

  ⭐**T1 절단 재계산 (결정적)** -- 데이터를 시점 T에서 자르고 피쳐를 **처음부터 다시 계산**해
     전체 계산본과 비교한다. 후방 전용 피쳐라면 t<=T의 값이 **비트 단위로 같아야** 한다.
     하나라도 다르면 그 피쳐는 미래 데이터를 쓴다. `compute_indicators`/`add_creative_*`/
     `add_broad_*`/`add_causal_columns` 내부를 읽지 않고도 전 경로를 한 번에 검증한다.

  **T2 라벨 창의 split 경계 침범** -- E0 라벨은 진입 후 FORWARD_BARS(200봉=16.7h)를 본다.
     TRAIN 끝자락 봉의 라벨은 VAL 구간 가격으로 만들어진다(purge 미적용). 몇 행이며
     제거하면 결과가 바뀌는지.

  ⭐**T3 TabPFN 배치 의존성** -- TabPFN은 in-context learner라 predict 시 배치 내부 통계로
     정규화할 수 있다. 그렇다면 **같은 행도 어떤 행들과 함께 예측하느냐에 따라 확률이 달라져**
     transductive 누출이 된다. 청크 크기를 바꿔가며 동일 행의 확률을 비교한다.
     (이 저장소에서 한 번도 확인한 적 없음)

  **T4 피쳐 목록 위생** -- `triggers`(문자열, local_extreme 포함 = 전방 창 사용) 같은 컬럼이
     FEATURES에 섞이지 않았는지. label/net_bp 등 라벨 파생 컬럼도 마찬가지.

  **T5 동일 봉 양방향 동시 진입** -- (bar, long)과 (bar, short)이 둘 다 임계값을 넘으면
     포트폴리오가 헤지 포지션을 여는가. 라이브에서 이상 동작이 된다.

  **T6 단일피쳐 방향 AUC** -- 각 피쳐 하나로 다음 봉 방향을 맞히는 정도. 0.55 초과면 누출 의심
     (154피쳐 감사와 같은 기준). 과거 방향 대조군도 같이.

⚠️읽기 전용. 라이브 코드/아티팩트 변경 없음.

Run on the server via handoff.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
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


_pf = _load("pf_audit", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
_feas, _bt = _s1._feas, _pf._bt
_tier0 = _feas._tier0
TIER0, FORWARD_BARS = _pf.TIER0, _pf.FORWARD_BARS
sim_exit = _pf.sim_exit
LABEL_CELL, CONTEXT_N, SEEDS, CHUNK = _pf.LABEL_CELL, _pf.CONTEXT_N, _pf.SEEDS, _pf.CHUNK

ETH_CSV = _feas.ETH_CSV
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
CUT = 0.8158
TMPDIR = ROOT / "tmp/vreb_lookahead_audit_20260902"
OUT = ROOT / "data/research/eth_v_rebound_econ_lookahead_audit_20260902/report.json"
TOL = 1e-9


def log(m): print(f"[audit] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    report = {"signal": "v_rebound_econ_lookahead_contamination_audit", "asset": "ETHUSDT",
              "scope": {"target": "E0 경제라벨 자동매매 후보", "read_only": True,
                        "features": TIER0, "n_features": len(TIER0)}, "tests": {}}

    # =========================================================
    # T4 피쳐 목록 위생 (가장 싸므로 먼저)
    # =========================================================
    log("=== T4 피쳐 목록 위생 ===")
    banned = [c for c in TIER0 if c in ("triggers", "label", "y", "net_bp", "status",
                                        "status_b", "status_t", "held_up", "is_candidate",
                                        "pos", "split", "p", "proba")]
    dup = [c for c in set(TIER0) if TIER0.count(c) > 1]
    log(f"  피쳐 {len(TIER0)}개, 금지 컬럼 혼입 {len(banned)}건, 중복 {len(dup)}건")
    log(f"  {'✅' if not banned and not dup else '❌'} {TIER0}")
    report["tests"]["T4_feature_hygiene"] = {"n_features": len(TIER0), "banned_present": banned,
                                             "duplicates": dup, "passed": not banned and not dup}

    # =========================================================
    # ⭐T1 절단 재계산
    # =========================================================
    log("")
    log("=== ⭐T1 절단 재계산 (결정적 룩어헤드 검사) ===")
    feat_full = _feas.build_all_bar_frame()
    log(f"  전체 계산본 {len(feat_full):,}행 ({feat_full['timestamp'].min()} ~ {feat_full['timestamp'].max()})")
    raw = pd.read_csv(ETH_CSV)
    TMPDIR.mkdir(parents=True, exist_ok=True)
    n = len(raw)
    cut_points = [int(n * f) for f in (0.45, 0.65, 0.85, 0.97)]
    t1 = {}
    orig_source, orig_eth = _tier0.SOURCE, _feas.ETH_CSV
    try:
        for T in cut_points:
            tmp = TMPDIR / f"eth_trunc_{T}.csv"
            raw.iloc[:T].to_csv(tmp, index=False)
            _tier0.SOURCE = tmp
            _feas.ETH_CSV = tmp
            ft = _feas.build_all_bar_frame()
            m = ft.merge(feat_full, on="timestamp", how="inner", suffixes=("_t", "_f"))
            m = m.tail(2000)      # 절단 지점 직전 2000봉만 비교(그 앞은 자명하게 동일)
            worst = {}
            for c in TIER0:
                if c == "is_downside" or f"{c}_t" not in m.columns:
                    continue
                a = pd.to_numeric(m[f"{c}_t"], errors="coerce").to_numpy(dtype=float)
                b = pd.to_numeric(m[f"{c}_f"], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() == 0:
                    continue
                d = float(np.max(np.abs(a[ok] - b[ok])))
                nanmis = int((np.isfinite(a) != np.isfinite(b)).sum())
                if d > TOL or nanmis:
                    worst[c] = {"max_abs_diff": d, "nan_mismatch": nanmis}
            ts_cut = str(ft["timestamp"].max())
            t1[str(T)] = {"cut_ts": ts_cut, "compared_rows": int(len(m)),
                          "features_differing": worst}
            log(f"  T={T:,} ({ts_cut}) 비교 {len(m):,}행 -> "
                f"{'✅ 차이 없음' if not worst else '❌ 차이 발생: ' + str(list(worst))}")
            for c, v in worst.items():
                log(f"      {c:24s} max|diff| {v['max_abs_diff']:.6g}  NaN불일치 {v['nan_mismatch']}")
    finally:
        _tier0.SOURCE, _feas.ETH_CSV = orig_source, orig_eth
        shutil.rmtree(TMPDIR, ignore_errors=True)
    t1_pass = all(not v["features_differing"] for v in t1.values())
    log(f"  ⇒ {'✅전 절단점에서 동일 -- 피쳐는 후방 전용' if t1_pass else '❌미래 데이터 사용 피쳐 존재'}")
    report["tests"]["T1_truncation"] = {"cut_points": t1, "passed": t1_pass}

    # =========================================================
    # 공통 프레임 (T2/T3/T5/T6)
    # =========================================================
    log("")
    log("프레임 재구성 (T2/T3/T5/T6용) ...")
    _s1.VAL_END = OOS_END
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    long = _s1.long_frame_for(sig, feat, sb, st)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    nk = len(kl)
    long["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1) for t in long["timestamp"]]
    long = long.loc[(long["pos"] >= 0) & (long["pos"] + FORWARD_BARS + 1 < nk)].reset_index(drop=True)

    sl0, arm0, tr0 = LABEL_CELL
    ii = long["pos"].to_numpy().astype(int)
    sg = np.where(long["is_downside"].to_numpy() == 1, 1.0, -1.0)
    at = long["atr"].to_numpy(dtype=float)
    net = np.full(len(long), np.nan)
    exit_off = np.full(len(long), -1)
    for s_ in range(0, len(long), CHUNK):
        e_ = min(s_ + CHUNK, len(long))
        j = ii[s_:e_]
        H = np.stack([h[x+1:x+1+FORWARD_BARS] for x in j])
        L = np.stack([l[x+1:x+1+FORWARD_BARS] for x in j])
        C = np.stack([c[x+1:x+1+FORWARD_BARS] for x in j])
        pn, ex = sim_exit(o[j+1], at[s_:e_], sg[s_:e_], H, L, C, sl0, arm0, tr0)
        net[s_:e_] = pn * 1e4 - 10.0
        exit_off[s_:e_] = ex
    long["y"] = (net > 0).astype(float)
    long["exit_pos"] = ii + 1 + exit_off

    # =========================================================
    # T2 라벨 창의 split 경계 침범
    # =========================================================
    log("")
    log("=== T2 라벨 창의 split 경계 침범 (purge 미적용) ===")
    tr_mask = long["split"] == "TRAIN"
    train_end_pos = int(np.searchsorted(kl["timestamp"].to_numpy(),
                                        np.datetime64(TRAIN_END.tz_localize(None))))
    crosses = tr_mask & (long["exit_pos"] >= train_end_pos)
    n_cross = int(crosses.sum())
    log(f"  TRAIN 행 {int(tr_mask.sum()):,} 중 라벨 창이 TRAIN_END를 넘는 행: **{n_cross:,}** "
        f"({n_cross/max(int(tr_mask.sum()),1)*100:.3f}%)")
    log(f"  (최대 침범 거리 {int((long.loc[crosses,'exit_pos'] - train_end_pos).max()) if n_cross else 0}봉)")
    report["tests"]["T2_boundary_leak"] = {
        "train_rows": int(tr_mask.sum()), "crossing_rows": n_cross,
        "pct": round(n_cross / max(int(tr_mask.sum()), 1) * 100, 4),
        "note": "purge 미적용. 비중이 작으면 실질 영향 미미하나 명시 필요."}

    # =========================================================
    # ⭐T3 TabPFN 배치 의존성
    # =========================================================
    log("")
    log("=== ⭐T3 TabPFN 배치 의존성 (transductive 누출) ===")
    from tabpfn import TabPFNClassifier
    tr_set = long.loc[long["split"] == "TRAIN"]
    rng = np.random.default_rng(SEEDS[0])
    ctx = tr_set.iloc[np.sort(rng.choice(len(tr_set), size=min(CONTEXT_N, len(tr_set)), replace=False))]
    clf = TabPFNClassifier(device="cuda", random_state=SEEDS[0], ignore_pretraining_limits=True)
    clf.fit(ctx[TIER0], ctx["y"].to_numpy())
    oos = long.loc[long["split"] == "OOS"].reset_index(drop=True)
    probe = oos.iloc[:2000]
    ref = clf.predict_proba(probe[TIER0])[:, 1]
    t3 = {}
    for cs in (2000, 500, 100, 20):
        pr = np.concatenate([clf.predict_proba(probe[TIER0].iloc[k:k+cs])[:, 1]
                             for k in range(0, len(probe), cs)])
        d = float(np.max(np.abs(pr - ref)))
        t3[f"chunk_{cs}"] = round(d, 8)
        log(f"  청크 {cs:>5}: 기준(2000행 일괄) 대비 max|Δp| = {d:.3e}")
    # 순서 뒤집기
    rev = clf.predict_proba(probe[TIER0].iloc[::-1])[:, 1][::-1]
    d_rev = float(np.max(np.abs(rev - ref)))
    t3["reversed_order"] = round(d_rev, 8)
    log(f"  순서 역전  : max|Δp| = {d_rev:.3e}")
    t3_pass = max(t3.values()) < 1e-4
    log(f"  ⇒ {'✅배치 무관 -- transductive 누출 없음' if t3_pass else '⚠️배치에 따라 확률이 달라짐 -- 라이브/백테스트 불일치 위험'}")
    report["tests"]["T3_tabpfn_batch_dependence"] = {**t3, "passed": t3_pass,
                                                      "threshold": 1e-4}

    # =========================================================
    # T5 동일 봉 양방향 동시 진입
    # =========================================================
    log("")
    log("=== T5 동일 봉 양방향 동시 호출 ===")
    P = []
    for sd in SEEDS:
        r2 = np.random.default_rng(sd)
        cx = tr_set.iloc[np.sort(r2.choice(len(tr_set), size=min(CONTEXT_N, len(tr_set)), replace=False))]
        m2 = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
        m2.fit(cx[TIER0], cx["y"].to_numpy())
        P.append(np.concatenate([m2.predict_proba(oos[TIER0].iloc[k:k+20000])[:, 1]
                                 for k in range(0, len(oos), 20000)]))
    oos["p"] = np.vstack(P).mean(axis=0)
    sel = oos.loc[oos["p"] >= CUT]
    both = sel.groupby("timestamp").size()
    n_both = int((both >= 2).sum())
    log(f"  호출 {len(sel):,}건 / 고유 봉 {len(both):,}  양방향 동시 호출 봉 **{n_both:,}** "
        f"({n_both/max(len(both),1)*100:.2f}%)")
    log(f"  {'✅없음' if n_both == 0 else '⚠️존재 -- 라이브에서 헤지 포지션이 열린다. 배선 시 한쪽만 취하는 규칙 필요'}")
    report["tests"]["T5_both_sides"] = {"calls": int(len(sel)), "unique_bars": int(len(both)),
                                        "both_sides_bars": n_both,
                                        "pct": round(n_both / max(len(both), 1) * 100, 3)}

    # =========================================================
    # T6 단일피쳐 방향 AUC
    # =========================================================
    log("")
    log("=== T6 단일피쳐 방향 AUC (0.55 초과면 누출 의심) ===")
    cl = c
    fwd1 = np.full(nk, np.nan); fwd1[:-1] = cl[1:] / cl[:-1] - 1
    bwd1 = np.full(nk, np.nan); bwd1[1:] = cl[1:] / cl[:-1] - 1
    sub = long.loc[long["split"] != "TRAIN"]
    pi = sub["pos"].to_numpy().astype(int)
    yf = (fwd1[pi] > 0).astype(float); yb = (bwd1[pi] > 0).astype(float)

    def rank_auc(x, y):
        m_ = np.isfinite(x) & np.isfinite(y)
        x, y = x[m_], y[m_]
        if len(x) < 1000 or len(np.unique(y)) < 2:
            return np.nan
        r = pd.Series(x).rank().to_numpy()
        n1, n0 = float((y == 1).sum()), float((y == 0).sum())
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    t6, suspect = {}, []
    for cname in TIER0:
        x = pd.to_numeric(sub[cname], errors="coerce").to_numpy(dtype=float)
        af, ab = rank_auc(x, yf), rank_auc(x, yb)
        t6[cname] = {"auc_next": round(af, 4) if af == af else None,
                     "auc_prev": round(ab, 4) if ab == ab else None}
        if af == af and abs(af - 0.5) >= 0.05:
            suspect.append(cname)
    for cname, v in sorted(t6.items(), key=lambda kv: -abs((kv[1]["auc_next"] or 0.5) - 0.5))[:8]:
        log(f"  {cname:24s} 다음봉 {v['auc_next']}  직전봉 {v['auc_prev']}")
    log(f"  ⇒ {'✅0.55 초과 없음' if not suspect else '❌의심: ' + str(suspect)}")
    report["tests"]["T6_single_feature_auc"] = {"per_feature": t6, "suspect": suspect,
                                                "passed": not suspect}

    # =========================================================
    # 종합
    # =========================================================
    checks = {"T1 절단재계산": t1_pass,
              "T3 배치무관": t3_pass,
              "T4 피쳐위생": report["tests"]["T4_feature_hygiene"]["passed"],
              "T6 단일피쳐AUC": not suspect}
    log("")
    log("=== 종합 ===")
    for k, v in checks.items():
        log(f"  {'✅' if v else '❌'} {k}")
    log(f"  ℹ️ T2 경계침범 {n_cross:,}행({report['tests']['T2_boundary_leak']['pct']}%) -- 아래 해석 참고")
    log(f"  ℹ️ T5 양방향 동시호출 {n_both:,}봉({report['tests']['T5_both_sides']['pct']}%)")
    report["overall_passed"] = all(checks.values())
    log("")
    log(f"⇒ {'✅룩어헤드/오염 없음' if report['overall_passed'] else '❌문제 발견 -- 아래 항목 확인'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
