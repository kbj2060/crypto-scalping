#!/usr/bin/env python3
"""V자반등 경제라벨 자동매매의 **진입 품질** -- MFE/MAE 해부.

## 질문 (2026-09-08 사용자)

"진입하고 나서 최대 미실현이익이 항상 플러스인지, 계속 마이너스인 경우는 얼마나 되는지."

## 측정 (진입 집합은 배포 서빙 규격 그대로)

동결 컨텍스트 3시드 TabPFN · p>=0.8221 · 같은 봉 양측면 dedup · 다음 봉 시가 진입
(`live_eth_v_rebound_econ_autotrade_signal_20260902.py`에서 상수·피쳐를 그대로 import).

  · **MFE**(최대 미실현이익) = max_t sign*(유리쪽 극값[t] - 진입가)/진입가.
    유리쪽은 롱이면 고가, 숏이면 저가 -- 라이브 배리어 판정과 같은 **봉 고가/저가** 규약.
  · **MAE**(최대 미실현손실) = 같은 식의 min, 불리쪽 극값 기준.
  · 두 가지 창을 함께 낸다:
      (a) `exit` -- 실제 청산봉까지(배포 규칙: SL5.0/ARM1.5/Trail0.1, 상한 200봉, 09-07 수정회계).
          "그 트레이드를 들고 있던 동안 실제로 볼 수 있었던 최대 평가이익".
      (b) `200` -- 청산과 무관하게 진입 후 200봉 고정. "청산 규칙을 빼면 기회가 있었는가".
  · 비용 10bp가 기준선이다. MFE < 10bp면 **어떤 청산 규칙으로도 이익 실현이 불가능**했던 진입이다.
  · 피크 반납 = MFE(exit) - 총수익(비용 전). 진입이 나빴는지 청산이 반납했는지를 가른다.

## 대조

  · **무작위 진입** 같은 창·같은 봉 모집단에서 3배수 추출(방향도 무작위).
    모델 진입의 MFE 분포가 무작위보다 나은지 -- 진입 품질의 유일한 절대 기준.
  · **확률 십분위** 전 모집단(선정 전)에서 p 십분위별 MFE/MAE/순손익.
    모델 점수가 진입 품질을 실제로 랭킹하는지 본다(임계값 위만 보면 알 수 없다).

⚠️읽기 전용. 라이브 코드 변경 없음. GPU 필요.
채점 결과는 `data/research/eth_v_rebound_econ_scored_20260908/`에 저장해 재실행 시 재사용한다.
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


_hc = _load("holdcap_eq", "scripts/research_eth_v_rebound_econ_hold_cap_sweep_20260908.py")
_s1 = _hc._s1
sim_stream = _hc.sim_stream
FEATURES, CTX_CSV, CUT = _hc.FEATURES, _hc.CTX_CSV, _hc.CUT
CELL, COST_BP, DEPLOYED_CAP = _hc.CELL, _hc.COST_BP, _hc.DEPLOYED_CAP
ENSEMBLE_SEEDS, MAX_CONCURRENT = _hc.ENSEMBLE_SEEDS, _hc.MAX_CONCURRENT
TRAIN_END, VAL_END, OOS_END = _hc.TRAIN_END, _hc.VAL_END, _hc.OOS_END

FIXED_WIN = 200
PRED_CHUNK = 20000
MFE_CHUNK = 8000
RNG_SEED = 20260908
SCORED = ROOT / "data/research/eth_v_rebound_econ_scored_20260908"
OUT = ROOT / "data/research/eth_v_rebound_econ_entry_quality_20260908/report.json"
BUCKETS = [(-1e18, 0.0), (0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 50.0), (50.0, 1e18)]
BUCKET_NAMES = ["<=0(계속 마이너스)", "0~5bp", "5~10bp(비용미만)", "10~20bp", "20~50bp", ">50bp"]


def log(m: str) -> None:
    print(f"[eq] {m}", flush=True)


def sim_track(pos_entry, entry, atr, sign, h, l, c, sl, arm, trail, max_bars):
    """`sim_stream`과 **같은 청산 규약**에 MFE/MAE/피크시점 추적을 붙인 판.

    `sim_stream`의 `best`는 트레일 무장용이라 진입가로 하한이 걸려 있다(=max(0,MFE)).
    여기서는 진입가보다 한 번도 위로 못 간 진입을 세야 하므로 **별도 추적**한다.
    반환 (수익률, 청산오프셋, 사유, mfe, mae, t_mfe) -- 앞 셋은 `sim_stream`과 일치해야 한다.
    """
    n = len(entry)
    nk = len(c)
    stop = entry - sign * sl * atr
    best = entry.copy()
    armed = np.zeros(n, bool)
    out = np.zeros(n)
    ex = np.zeros(n, np.int64)
    why = np.empty(n, dtype=object)
    mfe = np.full(n, -np.inf)
    mae = np.full(n, np.inf)
    t_mfe = np.zeros(n, np.int64)
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
        # 이 봉을 실제로 들고 있었으므로 청산 여부와 무관하게 극값을 반영한다.
        fex = s_ * (fav - e_) / e_ * 1e4
        aex = s_ * (adv - e_) / e_ * 1e4
        upd = fex > mfe[act]
        t_mfe[act] = np.where(upd, t, t_mfe[act])
        mfe[act] = np.where(upd, fex, mfe[act])
        mae[act] = np.minimum(mae[act], aex)
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
        bad = u & (s_ * (ns - cl) > 0)
        if bad.any():
            a = act[bad]
            out[a] = s_[bad] * (cl[bad] - e_[bad]) / e_[bad]
            ex[a] = t
            why[a] = "stop_infeasible"
        u = u & ~bad
        stop[act] = np.where(u, ns, stop[act])
        act = act[~bad]
        t += 1
    return out, ex, why, mfe, mae, t_mfe


def fixed_window_mfe(pos_entry, entry, sign, h, l, nbars):
    """청산과 무관한 고정 창 MFE/MAE(bp). 데이터 끝은 있는 만큼만 본다."""
    n = len(entry)
    mfe = np.full(n, -np.inf)
    mae = np.full(n, np.inf)
    nk = len(h)
    for s0 in range(0, n, MFE_CHUNK):
        e0 = min(s0 + MFE_CHUNK, n)
        idx = pos_entry[s0:e0]
        sg = sign[s0:e0]
        en = entry[s0:e0]
        room = np.minimum(nbars, nk - idx)
        m = room.max()
        cols = idx[:, None] + np.arange(m)[None, :]
        valid = np.arange(m)[None, :] < room[:, None]
        cols = np.clip(cols, 0, nk - 1)
        H, L = h[cols], l[cols]
        fav = np.where(sg[:, None] > 0, H, L)
        adv = np.where(sg[:, None] > 0, L, H)
        f = sg[:, None] * (fav - en[:, None]) / en[:, None] * 1e4
        a = sg[:, None] * (adv - en[:, None]) / en[:, None] * 1e4
        mfe[s0:e0] = np.where(valid, f, -np.inf).max(axis=1)
        mae[s0:e0] = np.where(valid, a, np.inf).min(axis=1)
    return mfe, mae


def bucket_share(v):
    return {nm: float(((v > lo) & (v <= hi)).mean()) if i else float((v <= hi).mean())
            for i, (nm, (lo, hi)) in enumerate(zip(BUCKET_NAMES, BUCKETS))}


def q(v, ps=(5, 25, 50, 75, 95)):
    return {f"p{p}": float(np.percentile(v, p)) for p in ps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="VAL,OOS,HO")
    args = ap.parse_args()
    want = [w.strip() for w in args.windows.split(",") if w.strip()]
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)
    SCORED.mkdir(parents=True, exist_ok=True)
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
    long = long.loc[(long["pos"] >= 0) & (long["pos"] + 2 < nk)].reset_index(drop=True)
    long["sgn"] = np.where(long["is_downside"].to_numpy() == 1, 1.0, -1.0)
    long["window"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                      np.where(long["timestamp"] < VAL_END, "VAL",
                       np.where(long["timestamp"] < OOS_END, "OOS", "HO")))

    models = None
    report = {"question": "진입 품질 -- MFE가 항상 플러스인가",
              "spec": {"cut": CUT, "cell": list(CELL), "cap_bars": DEPLOYED_CAP,
                       "cost_bp": COST_BP, "seeds": ENSEMBLE_SEEDS,
                       "mfe_convention": "봉 고가/저가(라이브 배리어와 동일), 진입가=다음 봉 시가"},
              "windows": {}}

    for win in want:
        s = long.loc[long["window"] == win].copy()
        if s.empty:
            continue
        cache = SCORED / f"{win}.parquet"
        if cache.exists():
            sc = pd.read_parquet(cache)
            n_before = len(s)
            s = s.merge(sc[["timestamp", "sgn", "p"]], on=["timestamp", "sgn"], how="inner")
            log(f"[{win}] 캐시 재사용 {cache.name} ({len(s):,}/{n_before:,}행)")
        else:
            if models is None:
                from tabpfn import TabPFNClassifier
                ctx = pd.read_csv(CTX_CSV)
                keep = sorted(ctx["seed"].unique())[:ENSEMBLE_SEEDS]
                models = []
                for sd in keep:
                    g = ctx.loc[ctx["seed"] == sd]
                    m = TabPFNClassifier(device="cuda", random_state=int(sd),
                                         ignore_pretraining_limits=True)
                    m.fit(g[FEATURES], g["label"].to_numpy())
                    models.append(m)
                log(f"  모델 적합 완료 시드 {keep}")
            log(f"[{win}] 채점 {len(s):,}행 ...")
            P = []
            for m in models:
                P.append(np.concatenate([m.predict_proba(s[FEATURES].iloc[k:k + PRED_CHUNK])[:, 1]
                                         for k in range(0, len(s), PRED_CHUNK)]))
                log(f"  시드 완료 ({time.time()-t0:.0f}s)")
            s["p"] = np.vstack(P).mean(axis=0)
            s[["timestamp", "pos", "sgn", "atr", "p"]].to_parquet(cache, index=False)

        idx_all = s["pos"].to_numpy().astype(int)
        sgn_all = s["sgn"].to_numpy()
        ent_all = o[idx_all + 1]
        log(f"[{win}] 모집단 고정창({FIXED_WIN}봉) MFE/MAE ...")
        mfe_pop, mae_pop = fixed_window_mfe(idx_all + 1, ent_all, sgn_all, h, l, FIXED_WIN)
        s["mfe200"] = mfe_pop
        s["mae200"] = mae_pop

        # ---- 배포 진입 집합 ----
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
        idx = sel["pos"].to_numpy().astype(int)
        sgn = sel["sgn"].to_numpy()
        atr = sel["atr"].to_numpy(float)
        ent = o[idx + 1]

        r, ex, why, mfe_x, mae_x, t_mfe = sim_track(idx + 1, ent, atr, sgn, h, l, c,
                                                    *CELL, max_bars=DEPLOYED_CAP)
        r0, e0, w0 = sim_stream(idx + 1, ent, atr, sgn, h, l, c, *CELL,
                                max_bars=DEPLOYED_CAP, infeasible="exit")
        dr = float(np.abs(r - r0).max())
        log(f"  파리티(sim_track vs sim_stream): |Δ수익|max={dr:.3e} "
            f"|Δ청산봉|max={int(np.abs(ex-e0).max())}")
        if dr > 1e-12 or int(np.abs(ex - e0).max()) != 0:
            raise SystemExit("파리티 실패")

        gross = r * 1e4
        pnl = gross - COST_BP
        rec = {"entries": int(len(sel)),
               "span": [str(sel["timestamp"].min()), str(sel["timestamp"].max())],
               "mfe_exit": {"mean": float(mfe_x.mean()), **q(mfe_x),
                            "share_le0": float((mfe_x <= 0).mean()),
                            "share_lt_cost": float((mfe_x < COST_BP).mean()),
                            "buckets": bucket_share(mfe_x)},
               "mfe_200": {"mean": float(sel['mfe200'].mean()), **q(sel["mfe200"].to_numpy()),
                           "share_le0": float((sel["mfe200"] <= 0).mean()),
                           "share_lt_cost": float((sel["mfe200"] < COST_BP).mean()),
                           "buckets": bucket_share(sel["mfe200"].to_numpy())},
               "mae_exit": {"mean": float(mae_x.mean()), **q(mae_x)},
               "mae_200": {"mean": float(sel["mae200"].mean()), **q(sel["mae200"].to_numpy())},
               "t_mfe_bars": {"mean": float(t_mfe.mean() + 1), **q(t_mfe + 1)},
               "giveback_bp": {"mean": float((mfe_x - gross).mean()),
                               **q(mfe_x - gross),
                               "share_gave_back_all": float((gross <= 0).mean())},
               "pnl_bp": {"mean": float(pnl.mean()), "win_rate": float((pnl > 0).mean())},
               "by_mfe_bucket": {}}

        for nm, (lo, hi) in zip(BUCKET_NAMES, BUCKETS):
            m = (mfe_x > lo) & (mfe_x <= hi) if lo > -1e17 else (mfe_x <= hi)
            if m.sum() == 0:
                continue
            rec["by_mfe_bucket"][nm] = {"n": int(m.sum()), "share": float(m.mean()),
                                        "pnl_bp": float(pnl[m].mean()),
                                        "mae_bp": float(mae_x[m].mean()),
                                        "bars": float((ex[m] + 1).mean())}

        # ---- 무작위 진입 대조 ----
        rb = rng.choice(len(s), size=min(len(sel) * 3, len(s)), replace=False)
        rs = s.iloc[np.sort(rb)]
        ridx = rs["pos"].to_numpy().astype(int)
        rsgn = np.where(rng.random(len(ridx)) < 0.5, 1.0, -1.0)
        rent = o[ridx + 1]
        _, rex, _, rmfe, rmae, _ = sim_track(ridx + 1, rent, rs["atr"].to_numpy(float), rsgn,
                                             h, l, c, *CELL, max_bars=DEPLOYED_CAP)
        rmfe200, rmae200 = fixed_window_mfe(ridx + 1, rent, rsgn, h, l, FIXED_WIN)
        rec["random_entry"] = {"n": int(len(ridx)),
                               "mfe_exit_mean": float(rmfe.mean()),
                               "mfe_exit_share_le0": float((rmfe <= 0).mean()),
                               "mfe_exit_share_lt_cost": float((rmfe < COST_BP).mean()),
                               "mfe_200_mean": float(rmfe200.mean()),
                               "mfe_200_share_lt_cost": float((rmfe200 < COST_BP).mean()),
                               "mae_exit_mean": float(rmae.mean())}

        # ---- 확률 십분위 (선정 전 전 모집단) ----
        dec = pd.qcut(s["p"], 10, labels=False, duplicates="drop")
        g = s.assign(dec=dec).groupby("dec")
        rec["proba_deciles"] = [
            {"dec": int(k), "n": int(len(v)), "p_lo": float(v["p"].min()),
             "p_hi": float(v["p"].max()), "mfe200": float(v["mfe200"].mean()),
             "mae200": float(v["mae200"].mean()),
             "share_mfe200_lt_cost": float((v["mfe200"] < COST_BP).mean())}
            for k, v in g]

        report["windows"][win] = rec
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))

        log("")
        log(f"=== {win}  진입 {len(sel):,}건 ===")
        log(f"  MFE(청산까지)  평균 {mfe_x.mean():+7.2f}bp  중앙 {np.median(mfe_x):+7.2f}bp  "
            f"<=0 {(mfe_x<=0).mean()*100:5.1f}%  <비용10bp {(mfe_x<COST_BP).mean()*100:5.1f}%")
        log(f"  MFE(200봉고정) 평균 {sel['mfe200'].mean():+7.2f}bp  중앙 {sel['mfe200'].median():+7.2f}bp  "
            f"<=0 {(sel['mfe200']<=0).mean()*100:5.1f}%  <비용10bp {(sel['mfe200']<COST_BP).mean()*100:5.1f}%")
        log(f"  MAE(청산까지)  평균 {mae_x.mean():+7.2f}bp  중앙 {np.median(mae_x):+7.2f}bp")
        log(f"  피크 도달 {np.median(t_mfe+1):.0f}봉(중앙)  피크반납 평균 {(mfe_x-gross).mean():+.2f}bp  "
            f"총수익<=0 {(gross<=0).mean()*100:.1f}%")
        log(f"  무작위진입: MFE(청산) 평균 {rmfe.mean():+7.2f}bp  <=0 {(rmfe<=0).mean()*100:5.1f}%  "
            f"<비용 {(rmfe<COST_BP).mean()*100:5.1f}%  | MFE200 평균 {rmfe200.mean():+7.2f}bp")
        log("  MFE(청산) 구간별:")
        for nm, d in rec["by_mfe_bucket"].items():
            log(f"    {nm:>18s} {d['share']*100:5.1f}%  n={d['n']:>5,}  "
                f"순손익 {d['pnl_bp']:+8.2f}bp  MAE {d['mae_bp']:+8.2f}bp  보유 {d['bars']:5.1f}봉")
        log("  확률 십분위(전 모집단): dec  p범위        MFE200   MAE200  MFE<비용")
        for d in rec["proba_deciles"]:
            log(f"    {d['dec']:>2d}  {d['p_lo']:.3f}~{d['p_hi']:.3f}  {d['mfe200']:+8.2f}  "
                f"{d['mae200']:+8.2f}  {d['share_mfe200_lt_cost']*100:5.1f}%")

    report["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"저장: {OUT.relative_to(ROOT)} ({report['elapsed_sec']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
