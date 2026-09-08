#!/usr/bin/env python3
"""V자반등 경제라벨 자동매매(섀도우)의 **최대 보유시간 제거** 효과 측정.

## 질문 (2026-09-08 사용자)

대시보드에 붙어 있는 V자반등 자동매매(`live_eth_v_rebound_econ_shadow_runner_20260902.py`)는
`MAX_HOLD_BARS = 200`(5분봉 200개 = 1,000분 = 16.7시간) 시간청산을 가지고 있다. 이걸 풀면
성과가 얼마나 달라지는가.

## 설계 (진입 고정, 청산만 바꾼다)

시간청산은 **청산 축**이므로 진입 선택은 건드리지 않는다. 배포 서빙 규격 그대로 후보를 만들고
(동결 컨텍스트 3시드 TabPFN, p>=0.8221, 같은 봉 양측면이면 확률 높은 쪽만 -- 라이브 코드에서
상수/피쳐/임계값을 그대로 import), **같은 진입 집합**에 대해 보유 상한만 바꿔 재시뮬레이션한다.
따라서 창별 비교는 완전 짝지음(paired)이고 CI는 일(day) 군집 부트스트랩으로 낸다.

  · 상한 격자: 200(배포) / 400 / 800 / 1600 / 3200 / 무제한(데이터 끝까지, 미완결은 censored)
  · 회계: `infeasible="exit"`(2026-09-07 수정본, 판정) + legacy(결함 원문, 규격서 수치와 대조용)
  · 비용 10.0bp(테이커), 진입 = 다음 봉 시가, 셀 (5.0, 1.5, 0.1) -- 전부 배포값
  · 건당(무제약) 통계와 **동시보유 5 순차 포트폴리오** 통계를 함께 낸다. 상한을 풀면 슬롯이
    오래 잠기므로 "건당 기대값"만 보면 반쪽이다.
  · 무작위 진입 기준선(같은 개수·같은 창) -- 상한 효과가 이 모델 특유인지 청산 구조 자체의
    성질인지 가른다.

## 창

VAL 2025-09-01~2026-01-01 · OOS 2026-01-01~2026-04-01 이 판정 창이다.
HO(2026-04-01~데이터 끝)는 **이미 1회 노출로 소진된 HOLDOUT 구간**이라 서술용으로만 찍는다 --
여기 숫자로 상한을 고르면 안 된다.

⚠️읽기 전용. 라이브 코드 변경 없음. GPU(서버) 필요.

Run:
  ~/miniconda3/envs/quant_ai/bin/python scripts/research_eth_v_rebound_econ_hold_cap_sweep_20260908.py
"""
from __future__ import annotations

import argparse
import importlib
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


_pf = _load("pf_holdcap", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
sim_exit_ref = _pf.sim_exit                     # 파리티 자기검사용 기준 구현
portfolio = _pf.portfolio

# 서빙 상수는 **라이브 코드에서 그대로** 가져온다(재선언 금지 -- 파리티가 조용히 깨진다).
_SIG = importlib.import_module("live_eth_v_rebound_econ_autotrade_signal_20260902")
_RUN = importlib.import_module("live_eth_v_rebound_econ_shadow_runner_20260902")
FEATURES = _SIG.FEATURES
CTX_CSV = _SIG.CTX_CSV
CUT = _SIG.PROBA_THRESHOLD
BRACKET = _SIG.BRACKET
MAX_CONCURRENT = _SIG.MAX_CONCURRENT
ENSEMBLE_SEEDS = _SIG.ENSEMBLE_SEEDS
DEPLOYED_CAP = _RUN.MAX_HOLD_BARS
COST_BP = _RUN.COST_BP

CELL = (BRACKET["sl_atr"], BRACKET["arm_atr"], BRACKET["trail_atr"])
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
CAPS = [200, 400, 800, 1600, 3200, None]
PRED_CHUNK = 20000
BOOT = 2000
RNG_SEED = 20260908
OUT = ROOT / "data/research/eth_v_rebound_econ_hold_cap_20260908/report.json"


def log(m: str) -> None:
    print(f"[holdcap] {m}", flush=True)


# ---------------------------------------------------------------- 청산 시뮬
def sim_stream(pos_entry, entry, atr, sign, h, l, c, sl, arm, trail,
               max_bars=None, infeasible="exit"):
    """`_pf.sim_exit`과 **같은 규약**의 스트리밍 구현. 상한 없는 보유를 다루기 위해 필요하다.

    창을 미리 쌓지 않고 살아있는 트레이드만 봉을 따라 전진시킨다. 컬럼 t는 전역 인덱스
    `pos_entry + t`이고 t=0이 진입봉(시가 진입)이다. 반환: (수익률, 청산 오프셋, 사유).
    사유: stop / stop_infeasible / time(상한 도달) / censored(데이터 끝).
    """
    n = len(entry)
    nk = len(c)
    stop = entry - sign * sl * atr
    best = entry.copy()
    armed = np.zeros(n, bool)
    out = np.zeros(n)
    ex = np.zeros(n, np.int64)
    why = np.empty(n, dtype=object)
    act = np.arange(n)
    t = 0
    while act.size:
        if max_bars is not None and t >= max_bars:
            gi = pos_entry[act] + (t - 1)
            out[act] = sign[act] * (c[gi] - entry[act]) / entry[act]
            ex[act] = t - 1
            why[act] = "time"
            break
        gi = pos_entry[act] + t
        oob = gi >= nk
        if oob.any():
            a = act[oob]
            gp = pos_entry[a] + (t - 1)
            out[a] = sign[a] * (c[gp] - entry[a]) / entry[a]
            ex[a] = t - 1
            why[a] = "censored"
            act = act[~oob]
            gi = gi[~oob]
            if act.size == 0:
                break
        s_ = sign[act]
        e_ = entry[act]
        hi, lo, cl = h[gi], l[gi], c[gi]
        adv = np.where(s_ > 0, lo, hi)
        fav = np.where(s_ > 0, hi, lo)
        st_ = stop[act]
        hit = np.where(s_ > 0, adv <= st_, adv >= st_)
        if hit.any():
            a = act[hit]
            out[a] = s_[hit] * (st_[hit] - e_[hit]) / e_[hit]
            ex[a] = t
            why[a] = "stop"
        keep = ~hit
        act = act[keep]
        if act.size == 0:
            break
        s_, e_, cl, fav = s_[keep], e_[keep], cl[keep], fav[keep]
        imp = s_ * (fav - best[act]) > 0
        best[act] = np.where(imp, fav, best[act])
        armed[act] |= ~armed[act] & (s_ * (best[act] - e_) >= arm * atr[act])
        ns = best[act] - s_ * trail * atr[act]
        u = armed[act] & (s_ * (ns - stop[act]) > 0)
        bad = u & (s_ * (ns - cl) > 0) if infeasible != "ignore" else np.zeros(act.size, bool)
        if infeasible == "exit" and bad.any():
            a = act[bad]
            out[a] = s_[bad] * (cl[bad] - e_[bad]) / e_[bad]
            ex[a] = t
            why[a] = "stop_infeasible"
        u = u & ~bad
        stop[act] = np.where(u, ns, stop[act])
        if infeasible == "exit":
            act = act[~bad]
        t += 1
    return out, ex, why


def parity_check(sel, o, h, l, c, infeasible):
    """스트리밍 구현이 기준 구현(`_pf.sim_exit`)과 상한 200에서 비트 수준으로 같은지."""
    ok = sel.loc[sel["pos"] + 1 + DEPLOYED_CAP <= len(c)]   # 창이 데이터 안에 다 들어오는 건만
    idx = ok["pos"].to_numpy().astype(int)
    sgn = ok["sgn"].to_numpy()
    atr = ok["atr"].to_numpy(float)
    ent = o[idx + 1]
    H = np.stack([h[j + 1:j + 1 + DEPLOYED_CAP] for j in idx])
    L = np.stack([l[j + 1:j + 1 + DEPLOYED_CAP] for j in idx])
    C = np.stack([c[j + 1:j + 1 + DEPLOYED_CAP] for j in idx])
    r0, e0 = sim_exit_ref(ent, atr, sgn, H, L, C, *CELL, infeasible=infeasible)
    r1, e1, _ = sim_stream(idx + 1, ent, atr, sgn, h, l, c, *CELL,
                           max_bars=DEPLOYED_CAP, infeasible=infeasible)
    dr = float(np.abs(r0 - r1).max())
    de = int(np.abs(e0 - e1).max())
    log(f"  파리티({infeasible}, n={len(idx):,}): |Δ수익|max={dr:.3e} |Δ청산봉|max={de}")
    if dr > 1e-12 or de != 0:
        raise SystemExit("파리티 실패 -- 스트리밍 구현이 기준 sim_exit과 다르다")


# ---------------------------------------------------------------- 통계
def day_boot_ci(vals, days, rng, b=BOOT):
    """일 군집 부트스트랩 평균 CI95."""
    uniq, inv = np.unique(days, return_inverse=True)
    buckets = [vals[inv == k] for k in range(len(uniq))]
    means = np.empty(b)
    for i in range(b):
        pick = rng.integers(0, len(buckets), len(buckets))
        means[i] = np.concatenate([buckets[j] for j in pick]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def trade_stats(pnl, ex, why, days, rng, base=None):
    w = pnl > 0
    lo, hi = day_boot_ci(pnl, days, rng)
    d = {"n": int(len(pnl)), "exp_bp": float(pnl.mean()), "ci95": [lo, hi],
         "total_bp": float(pnl.sum()), "win_rate": float(w.mean()),
         "med_bars": float(np.median(ex + 1)), "mean_bars": float((ex + 1).mean()),
         "p95_bars": float(np.percentile(ex + 1, 95)),
         "reason": {k: int(v) for k, v in zip(*np.unique(why, return_counts=True))}}
    if base is not None:
        dif = pnl - base
        dlo, dhi = day_boot_ci(dif, days, rng)
        d["delta_vs_cap200"] = {"mean_bp": float(dif.mean()), "ci95": [dlo, dhi],
                                "n_changed": int((np.abs(dif) > 1e-9).sum())}
    return d


# ---------------------------------------------------------------- 메인
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="VAL,OOS,HO")
    ap.add_argument("--random-baseline", type=int, default=1)
    args = ap.parse_args()
    want = [w.strip() for w in args.windows.split(",") if w.strip()]
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)

    from tabpfn import TabPFNClassifier
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    log("프레임 재구성 (배포 백테스트와 같은 경로) ...")
    _s1.VAL_END = pd.Timestamp("2030-01-01", tz="UTC")      # 창 절단 해제
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
    data_end = kl["timestamp"].iloc[-1]
    log(f"  klines {nk:,}봉  마지막 {data_end}")

    long["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1) for t in long["timestamp"]]
    long = long.loc[(long["pos"] >= 0) & (long["pos"] + 2 < nk)].reset_index(drop=True)
    long["sgn"] = np.where(long["is_downside"].to_numpy() == 1, 1.0, -1.0)
    long["window"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                      np.where(long["timestamp"] < VAL_END, "VAL",
                       np.where(long["timestamp"] < OOS_END, "OOS", "HO")))
    log(f"  long {len(long):,}행  " + str(long["window"].value_counts().to_dict()))

    log(f"동결 컨텍스트 TabPFN {ENSEMBLE_SEEDS}시드 적합 ...")
    ctx = pd.read_csv(CTX_CSV)
    keep = sorted(ctx["seed"].unique())[:ENSEMBLE_SEEDS]
    models = []
    for sd in keep:
        g = ctx.loc[ctx["seed"] == sd]
        m = TabPFNClassifier(device="cuda", random_state=int(sd), ignore_pretraining_limits=True)
        m.fit(g[FEATURES], g["label"].to_numpy())
        models.append(m)
    log(f"  시드 {keep} (라이브와 동일: 정렬 후 앞 {ENSEMBLE_SEEDS}개)")

    report = {"question": "MAX_HOLD_BARS 제거 시 성능 차이",
              "deployed": {"cap_bars": DEPLOYED_CAP, "cell": list(CELL), "cut": CUT,
                           "seeds": [int(x) for x in keep], "max_concurrent": MAX_CONCURRENT,
                           "cost_bp": COST_BP},
              "caps": [x if x is not None else "none" for x in CAPS],
              "data_end": str(data_end), "windows": {}}
    OUT.parent.mkdir(parents=True, exist_ok=True)

    for win in want:
        s = long.loc[long["window"] == win].copy()
        if s.empty:
            log(f"[{win}] 행 없음 -- 건너뜀")
            continue
        log("")
        log(f"=== {win} ({s['timestamp'].min()} ~ {s['timestamp'].max()}, {len(s):,}행) 채점 ===")
        P = []
        for m in models:
            P.append(np.concatenate([m.predict_proba(s[FEATURES].iloc[k:k + PRED_CHUNK])[:, 1]
                                     for k in range(0, len(s), PRED_CHUNK)]))
            log(f"  시드 채점 완료 ({time.time()-t0:.0f}s)")
        s["p"] = np.vstack(P).mean(axis=0)

        # 라이브 규칙: 같은 봉에서 롱·숏 둘 다 통과하면 확률 높은 쪽만. 동률이면 둘 다 버린다.
        passed = s.loc[s["p"] >= CUT]
        keep_idx = []
        for _, g in passed.groupby("timestamp"):
            if len(g) == 1:
                keep_idx.append(g.index[0])
                continue
            top = g["p"].max()
            tied = g.loc[g["p"] >= top - 1e-12]
            if len(tied) == 1:
                keep_idx.append(tied.index[0])
        sel = passed.loc[keep_idx].sort_values("pos").reset_index(drop=True)
        log(f"  후보 {len(sel):,}건 (전체 {len(s):,}행의 {len(sel)/len(s)*100:.2f}%)")
        if len(sel) < 30:
            continue

        wrec = {"span": [str(s["timestamp"].min()), str(s["timestamp"].max())],
                "rows_scored": int(len(s)), "entries": int(len(sel)), "acct": {}}
        for acct in ("exit", "ignore"):
            parity_check(sel, o, h, l, c, acct)
            idx = sel["pos"].to_numpy().astype(int)
            sgn = sel["sgn"].to_numpy()
            atr = sel["atr"].to_numpy(float)
            ent = o[idx + 1]
            days = sel["timestamp"].dt.floor("D").to_numpy()
            rows = {}
            base_pnl = None
            for cap in CAPS:
                r, ex, why = sim_stream(idx + 1, ent, atr, sgn, h, l, c, *CELL,
                                        max_bars=cap, infeasible=acct)
                pnl = r * 1e4 - COST_BP
                key = str(cap) if cap is not None else "none"
                if base_pnl is None:
                    base_pnl = pnl.copy()
                st_ = trade_stats(pnl, ex, why, days, rng,
                                  base=None if cap == CAPS[0] else base_pnl)
                cand = pd.DataFrame({"timestamp": sel["timestamp"].to_numpy(),
                                     "entry_bar": idx + 1, "exit_bar": idx + 1 + ex,
                                     "pnl_bp": pnl})
                pf = portfolio(cand, MAX_CONCURRENT)
                if pf is None:
                    continue
                st_["portfolio"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in pf.items() if k not in ("idx", "pnl", "ts")}
                ndays = max(1.0, (sel["timestamp"].max() - sel["timestamp"].min()).total_seconds() / 86400)
                st_["portfolio"]["trades_per_day"] = round(pf["n"] / ndays, 2)
                rows[key] = st_
            wrec["acct"][acct] = rows

            tag = "수정본" if acct == "exit" else "legacy(결함)"
            log(f"  --- {win} / {tag} 회계 ---")
            log(f"  {'상한':>6s} {'건당bp':>9s} {'CI95':>18s} {'Δ vs200':>9s} "
                f"{'중앙보유':>7s} {'시간청산':>7s} | {'PF n':>5s} {'PF bp':>8s} {'PF총bp':>9s} {'PF DD':>9s}")
            for key, d in rows.items():
                p_ = d["portfolio"]
                dl = d.get("delta_vs_cap200", {}).get("mean_bp")
                tm = d["reason"].get("time", 0) + d["reason"].get("censored", 0)
                log(f"  {key:>6s} {d['exp_bp']:>+8.2f}bp [{d['ci95'][0]:>+7.2f},{d['ci95'][1]:>+7.2f}] "
                    f"{('' if dl is None else f'{dl:+8.2f}'):>9s} {d['med_bars']:>6.0f}봉 "
                    f"{tm/d['n']*100:>6.1f}% | {p_['n']:>5d} {p_['exp_bp']:>+7.2f}bp "
                    f"{p_['total_bp']:>+8.0f}bp {p_['max_dd_bp']:>+8.0f}bp")

        if args.random_baseline:
            # 같은 창·같은 건수의 무작위 진입: 상한 효과가 청산 구조 자체의 성질인지 확인.
            rb = rng.choice(len(s), size=min(len(sel) * 3, len(s)), replace=False)
            rs = s.iloc[np.sort(rb)]
            idx = rs["pos"].to_numpy().astype(int)
            sgn = np.where(rng.random(len(idx)) < 0.5, 1.0, -1.0)
            atr = rs["atr"].to_numpy(float)
            ent = o[idx + 1]
            days = rs["timestamp"].dt.floor("D").to_numpy()
            base_pnl = None
            rrows = {}
            for cap in CAPS:
                r, ex, why = sim_stream(idx + 1, ent, atr, sgn, h, l, c, *CELL,
                                        max_bars=cap, infeasible="exit")
                pnl = r * 1e4 - COST_BP
                if base_pnl is None:
                    base_pnl = pnl.copy()
                rrows[str(cap) if cap is not None else "none"] = trade_stats(
                    pnl, ex, why, days, rng, base=None if cap == CAPS[0] else base_pnl)
            wrec["random_entry"] = rrows
            log(f"  --- {win} / 무작위 진입 {len(idx):,}건 (수정본 회계) ---")
            for key, d in rrows.items():
                dl = d.get("delta_vs_cap200", {}).get("mean_bp")
                log(f"  {key:>6s} {d['exp_bp']:>+8.2f}bp "
                    f"{('' if dl is None else f'Δ{dl:+7.2f}'):>10s} 중앙 {d['med_bars']:>4.0f}봉")

        report["windows"][win] = wrec
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    report["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"저장: {OUT.relative_to(ROOT)}  ({report['elapsed_sec']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
