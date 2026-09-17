#!/usr/bin/env python3
"""Zeus Baseline v1 — **새 더블 배리어로 품질 머리를 재학습**하고 배포 품질머리와 맞붙인다.

## 왜 이게 마지막 구멍인가
배리어 설계(§docs/experiments/omega461_exit_barrier_design_20260917.md)의 모든 숫자는
**배포 품질머리** 기준이었다. 청산을 TP1.5%/SL1% 더블 배리어로 바꿨으면 품질 머리가 배우는
대상도 바뀌어야 한다 -- 그걸 안 하면 「청산 규칙과 품질 머리가 서로 다른 것을 보는」 상태다.

## 설계
 · 방향 머리 타깃: **zigzag (그대로)** -- 두 부모 공통이고 이번 결정에서 안 건드린다.
 · 품질 머리 타깃: **새 더블 배리어 action** (TP1.5%/SL1%, 시간청산 없음, 미해소=CASH)
 · 인과 워크포워드 5폴드 + CAND. 임계값 q 는 **TRAIN 꼬리에서만** 잡는다(TEST 미사용).
 · 평가는 라이브와 같은 hard routing + 같은 더블 배리어로 실제 체결 시뮬.
 · 판정은 **하루 순bp**(건당 × 건/일 − 비용) -- 건당만 보면 「폭 넓히기」가 항상 이긴다.

## 대조군 (사전 지정)
 A. 배포 품질머리 (고정 아티팩트, 재학습 없음)         <- 현재 baseline
 B. 새 라벨 품질머리 (이 스크립트)
 C. 품질머리 없음(방향만, q 게이트 해제)                <- 품질 머리가 «일을 하는가»
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402
import research_omega461_parent_quality_calibration_and_bias_20260917 as F  # noqa: E402
import research_omega461_parent_debias_retrain_20260917 as H  # noqa: E402
import research_omega461_side_skill_decomposition_20260917 as K  # noqa: E402

SEEDS = H.SEEDS5
TARGET = H.TARGET                      # 통과율 3.5% (TRAIN 꼬리에서 맞춘다)
LABEL = (ROOT / "tmp/omega461_longwindow_20260917/zeus_double_barrier_labels_20260917"
         / "zeus_db_tp15_sl10_usdc3x_3.06bp.parquet")
OUTJ = E.OUT / "stageM_zeus_quality_head.json"
CACHE = E.OUT / "stageM_probs.npz"
COST = {"usdc": 1.02, "peg": 5.52}


def log(*a, **k): print(*a, flush=True)


def fit(xtr, ydir, yqual, w, xv, ydv, yqv, wv, *, seed, ei, device):
    """E.fit_expert 와 같되 **방향/품질 타깃을 따로** 받는다(기존은 같은 y 를 둘 다에 썼다)."""
    import random
    torch.manual_seed(seed + ei); np.random.seed(seed + ei); random.seed(seed + ei)
    m = tabm.ThreeHeadTabM(xtr.shape[1], cfg=tabm.CFG).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=float(tabm.CFG.lr),
                            weight_decay=float(tabm.CFG.weight_decay))
    dl = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ydir),
                                  torch.from_numpy(yqual), torch.from_numpy(w)),
                    batch_size=int(tabm.CFG.batch_size), shuffle=True)
    k = int(tabm.CFG.k)
    best, state, stale, bep = float("inf"), None, 0, 0
    ce = torch.nn.functional.cross_entropy
    for ep in range(E.EPOCHS):
        m.train()
        for xb, db, qb, wb in dl:
            xb, db, qb, wb = xb.to(device), db.to(device), qb.to(device), wb.to(device)
            o = m(xb)
            td = db[:, None].expand(-1, k).reshape(-1)
            tq = qb[:, None].expand(-1, k).reshape(-1)
            ld = ce(o["direction"].reshape(-1, 3), td, reduction="none").reshape(-1, k)
            lq = ce(o["quality"].reshape(-1, 3), tq, reduction="none").reshape(-1, k)
            den = torch.clamp(wb.sum(), min=1.0)
            loss = (ld.mean(1) * wb).sum() / den + float(tabm.CFG.quality_loss_weight) * (lq.mean(1) * wb).sum() / den
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()
        m.eval()
        with torch.no_grad():
            o = m(torch.from_numpy(xv).to(device))
            td = torch.from_numpy(ydv).to(device)[:, None].expand(-1, k).reshape(-1)
            ld = ce(o["direction"].reshape(-1, 3), td, reduction="none").reshape(-1, k)
            wvt = torch.from_numpy(wv).to(device)
            vl = float((ld.mean(1) * wvt).sum() / torch.clamp(wvt.sum(), min=1.0))
        if vl < best - 1e-5:
            best, bep, stale = vl, ep + 1, 0
            state = {a: v.detach().clone() for a, v in m.state_dict().items()}
        else:
            stale += 1
            if stale >= E.PATIENCE:
                break
    m.load_state_dict(state); m.eval()
    return m, {"inner_val_loss": best, "best_epoch": bep}


def realize(te, side, idx):
    """더블 배리어 TP1.5%/SL1% 로 실제 체결시켜 (건당bp, 보유봉, 날짜) 를 낸다."""
    hi = pd.to_numeric(te["high"]).to_numpy(np.float64)
    lo = pd.to_numeric(te["low"]).to_numpy(np.float64)
    cl = pd.to_numeric(te["close"]).to_numpy(np.float64)
    r, h, res, _rn, _mae = K._first_touch_open(idx, side, hi, lo, cl, K.BASE_TP, K.BASE_SL, K.MAXBARS)
    return r * 1e4, h.astype(float), te.timestamp.dt.floor("D").to_numpy()[idx], res


def econ(pnl, hold, days, tag):
    lo_, hi_, nd = E.block_ci(pnl, days)
    per_day = 288.0 / max(hold.mean(), 1e-9)
    return {"tag": tag, "n": int(len(pnl)), "indep_days": nd,
            "gross_bp": float(pnl.mean()), "ci95": [lo_, hi_],
            "median_bp": float(np.median(pnl)), "win_rate": float((pnl > 0).mean()),
            "median_hold": float(np.median(hold)), "mean_hold": float(hold.mean()),
            "trades_per_day": per_day,
            "net_day_usdc": (float(pnl.mean()) - COST["usdc"]) * per_day,
            "net_day_peg": (float(pnl.mean()) - COST["peg"]) * per_day}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} · seeds={SEEDS} · 배리어 TP{K.BASE_TP*100:g}%/SL{K.BASE_SL*100:g}% 더블")
    df, base_cols = E.load()
    lab = pd.read_parquet(LABEL, columns=["timestamp", "tb_action"])
    lab["timestamp"] = pd.to_datetime(lab["timestamp"]).dt.tz_localize(None)
    n0 = len(df)
    df = df.merge(lab.rename(columns={"tb_action": "y_qual"}), on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)
    log(f"품질 라벨 조인 {n0:,} -> {len(df):,}행 · "
        f"품질 CASH/LONG/SHORT {np.bincount(df.y_qual.to_numpy(), minlength=3)/len(df)}")
    agree = float((df.y_qual.to_numpy() == pd.to_numeric(df.zigzag_action).to_numpy()).mean())
    log(f"⭐방향(zigzag) vs 품질(더블배리어) 타깃 일치율 {agree:.4f}")

    b = torch.load(E.BUNDLE, map_location="cpu", weights_only=False)
    dep = {}
    for ename in ("bull", "bear", "chop"):
        pay = dict(b["models"][ename])
        m = tabm.ThreeHeadTabM(int(pay["n_features"]),
                               cfg=tabm.ThreeHeadConfig(**dict(pay["config"]))).to(device)
        m.load_state_dict(pay["state_dict"]); m.eval()
        dep[ename] = (m, dict(pay["scaler"]))

    cache = dict(np.load(CACHE, allow_pickle=True)) if CACHE.exists() else {}
    rows, pooled = [], {"A": [], "B": [], "C": []}
    for name, t0, t1, v0, v1 in K.FOLDS:
        tm = (df.timestamp >= t0) & (df.timestamp <= t1 + " 23:59:59")
        vm = (df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")
        tr, te = df[tm].reset_index(drop=True), df[vm].reset_index(drop=True)
        assert tr.timestamp.max() < te.timestamp.min(), f"{name} TRAIN 이 TEST 를 침범"
        ydir_t = pd.to_numeric(tr["zigzag_action"]).to_numpy(np.int64)
        yq_t = tr["y_qual"].to_numpy(np.int64)
        rt = tabm._route_probs(tr)
        ev = tabm._route_probs(te).argmax(1)
        xv_raw = tabm._base_input(te, base_cols)
        n = len(tr); split = max(int(n * 0.85), min(n - 1, 512))
        log(f"\n{'='*92}\n=== {name} TRAIN {t0}~{t1} {n:,} · TEST {v0}~{v1} {len(te):,}\n{'='*92}")

        # ── A. 배포 품질머리 ──
        D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
        for ei, ename in enumerate(("bull", "bear", "chop")):
            m, sc = dep[ename]
            sel = ev == ei
            if sel.any():
                D[sel], Q[sel] = E.heads(m, tabm._standardize_apply(xv_raw[sel], sc), device)
        seen = not (v1 < K.DEPLOYED_SEEN[0] or v0 > K.DEPLOYED_SEEN[1])

        # ── B. 새 라벨 품질머리 ──
        ck = f"{name}|B"
        if ck in cache:
            z = dict(cache[ck].item()); Ds, Qs, tDm, tQm = list(z["Ds"]), list(z["Qs"]), z["tDm"], z["tQm"]
        else:
            xs, scaler = tabm._standardize_fit(tabm._base_input(tr, base_cols))
            xv_std = tabm._standardize_apply(xv_raw, scaler)
            tail_expert = rt[split:].argmax(1)
            Ds, Qs, tDs, tQs = [], [], [], []
            for seed in SEEDS:
                Db = np.zeros((len(te), 3)); Qb = np.zeros((len(te), 3))
                tD = np.zeros((n - split, 3)); tQ = np.zeros((n - split, 3))
                for ei in range(3):
                    w = H.w_old(ydir_t, rt[:, ei].astype(np.float32))
                    mm, _ = fit(xs[:split], ydir_t[:split], yq_t[:split], w[:split],
                                xs[split:], ydir_t[split:], yq_t[split:], w[split:],
                                seed=seed, ei=ei, device=device)
                    sel = ev == ei
                    if sel.any():
                        Db[sel], Qb[sel] = E.heads(mm, xv_std[sel], device)
                    ts_ = tail_expert == ei
                    if ts_.any():
                        tD[ts_], tQ[ts_] = E.heads(mm, xs[split:][ts_], device)
                Ds.append(Db); Qs.append(Qb); tDs.append(tD); tQs.append(tQ)
                log(f"  B/seed {seed} 완료")
            tDm, tQm = np.mean(tDs, 0), np.mean(tQs, 0)
            cache[ck] = np.array({"Ds": Ds, "Qs": Qs, "tDm": tDm, "tQm": tQm}, dtype=object)
            np.savez(CACHE, **cache)
        Dm, Qm = np.mean(Ds, 0), np.mean(Qs, 0)
        ql, qsh = H.thresholds(tDm, tQm, symmetric=True)

        arms = {"B": H.side_from(Dm, Qm, ql, qsh),
                "C": np.where(Dm.argmax(1) == 1, 1.0, np.where(Dm.argmax(1) == 2, -1.0, 0.0))}
        if not seen:
            arms["A"] = F.gate(D, Q, E.Q_THRESH)[1]
        for tag, side in arms.items():
            idx = np.where(side != 0)[0]
            if len(idx) < 100:
                log(f"  {tag}: 후보 {len(idx)} -- 건너뜀"); continue
            pnl, hold, days, res = realize(te, side, idx)
            r = econ(pnl, hold, days, tag); r["fold"] = name
            rows.append(r); pooled[tag].append((pnl, hold, days))
            log(f"  {tag} {name}: 후보 {len(idx):,} · 건당 {r['gross_bp']:+7.2f} "
                f"CI[{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}] · 중앙보유 {r['median_hold']:.0f}봉 "
                f"· {r['trades_per_day']:.2f}건/일 · 순/일 USDC {r['net_day_usdc']:.1f}")
        json.dump(rows, open(OUTJ, "w"), indent=2, default=float)

    log(f"\n{'='*92}\n=== 풀링 ===")
    log(f"{'팔':<4}{'폴드':>5}{'건수':>8}{'건당bp':>9}{'CI':>20}{'중앙보유':>9}{'건/일':>7}"
        f"{'순/일USDC':>10}{'순/일peg':>9}")
    summ = {}
    names = {"A": "배포 품질머리", "B": "새 라벨 품질머리", "C": "품질머리 없음"}
    for tag in ("A", "B", "C"):
        if not pooled[tag]:
            continue
        pnl = np.concatenate([x[0] for x in pooled[tag]])
        hold = np.concatenate([x[1] for x in pooled[tag]])
        days = np.concatenate([x[2] for x in pooled[tag]])
        r = econ(pnl, hold, days, tag); r["folds"] = len(pooled[tag]); summ[tag] = r
        log(f"{tag:<4}{r['folds']:>5}{r['n']:>8,}{r['gross_bp']:>+9.2f}"
            f"  [{r['ci95'][0]:+7.2f},{r['ci95'][1]:+7.2f}]{r['median_hold']:>9.0f}"
            f"{r['trades_per_day']:>7.2f}{r['net_day_usdc']:>10.1f}{r['net_day_peg']:>9.1f}"
            f"   {names[tag]}")
    if "A" in summ and "B" in summ:
        log(f"\n⭐B − A (순bp/일 USDC) = {summ['B']['net_day_usdc'] - summ['A']['net_day_usdc']:+.1f}"
            f"   ⚠️A 는 배포 학습구간 밖 폴드만이라 폴드 집합이 다르다 -- 폴드별 표를 같이 본다")
    if "C" in summ and "B" in summ:
        log(f"⭐B − C (순bp/일 USDC) = {summ['B']['net_day_usdc'] - summ['C']['net_day_usdc']:+.1f}"
            f"   (품질 머리가 실제로 일을 하는가)")
    json.dump({"folds": rows, "pooled": summ, "label": str(LABEL),
               "tp": K.BASE_TP, "sl": K.BASE_SL, "seeds": SEEDS}, open(OUTJ, "w"), indent=2, default=float)
    log(f"저장: {OUTJ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
