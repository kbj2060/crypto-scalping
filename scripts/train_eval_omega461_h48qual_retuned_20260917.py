#!/usr/bin/env python3
"""L — 비용 가정을 정정한 h48 라벨로 **품질 머리를 재학습**하고 발화율/경제를 잰다.

배포 h48qual 은 같은 VAL 에서 **83건(0.2%·하루 0.7건)** 밖에 안 쏜다. 원인은 품질 라벨의
`fee_cost = 42bp`(테이커 7bp × 2다리 × 3배)인데 우리 실측 USDC 메이커 왕복은 **1.02bp** 다.
K단계에서 배리어를 그대로 두고 비용만 바꾸니 라벨 활성률이 **59.8% → 79.6%** 로 올랐다
(원래 값으로 기존 ETH 라벨을 **99.93% 재현**하는 관문 통과 후).

여기서는 **방향 머리는 zigzag, 품질 머리는 h48** 로 **타깃을 분리**해 학습한다
(= 배포 h48qual 의 `quality_mode=quality_label_action`). 지금까지 내 스크립트는 두 머리에
같은 타깃을 줬다(zig075 의 `same_as_direction`). 그 차이가 이 판의 핵심이다.

나머지는 H 최선 구성 고정: base(2025-01~2026-02) · 무작위 5시드 앙상블 · old 가중 ·
hard routing. 게이트는 **배포값 q=0.50 원본**과 **TRAIN 꼬리 대칭 보정** 둘 다 낸다.
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
import research_omega461_parent_pnl_concentration_20260917 as G  # noqa: E402
import research_omega461_parent_debias_retrain_20260917 as H  # noqa: E402

RET = ROOT / "tmp/omega461_longwindow_20260917/h48_retune"
VARIANTS = ["orig_taker3x_42bp", "usdc3x_3.1bp", "usdc1x_1.0bp"]
Q_DEPLOYED = 0.50


def log(*a, **k): print(*a, flush=True)


def fit_two_targets(xtr, ydir, yqual, wtr, xv, ydv, yqv, wv, *, seed, ei, device):
    """방향/품질 타깃이 **다르다**. 배포 h48qual 의 quality_label_action 규약."""
    torch.manual_seed(seed + ei); np.random.seed(seed + ei)
    m = tabm.ThreeHeadTabM(xtr.shape[1], cfg=tabm.CFG).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=float(tabm.CFG.lr),
                            weight_decay=float(tabm.CFG.weight_decay))
    dl = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ydir),
                                  torch.from_numpy(yqual), torch.from_numpy(wtr)),
                    batch_size=int(tabm.CFG.batch_size), shuffle=True)
    k = int(tabm.CFG.k)
    best, state, stale, bep = float("inf"), None, 0, 0
    for ep in range(E.EPOCHS):
        m.train()
        for xb, yd, yq, wb in dl:
            xb, yd, yq, wb = xb.to(device), yd.to(device), yq.to(device), wb.to(device)
            o = m(xb); den = torch.clamp(wb.sum(), min=1.0)
            ld = torch.nn.functional.cross_entropy(
                o["direction"].reshape(-1, 3), yd[:, None].expand(-1, k).reshape(-1),
                reduction="none").reshape(-1, k)
            lq = torch.nn.functional.cross_entropy(
                o["quality"].reshape(-1, 3), yq[:, None].expand(-1, k).reshape(-1),
                reduction="none").reshape(-1, k)
            loss = (ld.mean(1) * wb).sum() / den + float(tabm.CFG.quality_loss_weight) * (lq.mean(1) * wb).sum() / den
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()
        m.eval()
        with torch.no_grad():
            o = m(torch.from_numpy(xv).to(device))
            t = torch.from_numpy(ydv).to(device)[:, None].expand(-1, k).reshape(-1)
            ld = torch.nn.functional.cross_entropy(o["direction"].reshape(-1, 3), t,
                                                   reduction="none").reshape(-1, k)
            wvt = torch.from_numpy(wv).to(device)
            vl = float((ld.mean(1) * wvt).sum() / torch.clamp(wvt.sum(), min=1.0))
        if vl < best - 1e-5:
            best, bep, stale = vl, ep + 1, 0
            state = {a: v.detach().clone() for a, v in m.state_dict().items()}
        else:
            stale += 1
            if stale >= E.PATIENCE: break
    m.load_state_dict(state); m.eval()
    return m, bep


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, base_cols = E.load()
    vm = (df.timestamp >= E.VAL[0]) & (df.timestamp <= E.VAL[1] + " 23:59:59")
    val = df[vm].reset_index(drop=True)
    yv = pd.to_numeric(val["zigzag_action"]).to_numpy(np.int64)
    xv_raw = tabm._base_input(val, base_cols)
    ev_expert = tabm._route_probs(val).argmax(1)
    vdays = val.timestamp.dt.floor("D").to_numpy()
    fwd1 = val["fwd_1h_bp"].to_numpy(np.float64)
    t0, t1 = E.ARMS["base"]
    results = []

    for var in VARIANTS:
        q = pd.read_parquet(RET / f"h48cons_{var}.parquet")
        q["timestamp"] = pd.to_datetime(q["timestamp"])
        d = df.merge(q[["timestamp", "tb_action"]], on="timestamp", how="left")
        d["tb_action"] = d["tb_action"].fillna(0).astype(np.int64)   # 결측=CASH (배포 padding 규약)
        tr = d[(d.timestamp >= t0) & (d.timestamp <= t1 + " 23:59:59")].reset_index(drop=True)
        ydir = pd.to_numeric(tr["zigzag_action"]).to_numpy(np.int64)
        yqual = tr["tb_action"].to_numpy(np.int64)
        rt = tabm._route_probs(tr)
        n = len(tr); split = max(int(n * 0.85), min(n - 1, 512))
        xs, scaler = tabm._standardize_fit(tabm._base_input(tr, base_cols))
        xv_std = tabm._standardize_apply(xv_raw, scaler)
        tail_expert = rt[split:].argmax(1)
        agree = float((ydir == yqual).mean())
        log(f"\n{'='*88}\n=== {var} · TRAIN {n:,}행 "
            f"· 방향(zigzag) {(np.bincount(ydir,minlength=3)/n).round(3)} "
            f"· 품질(h48) {(np.bincount(yqual,minlength=3)/n).round(3)} "
            f"· 두 타깃 일치 {agree:.3f}\n{'='*88}")

        Ds, Qs, tDs, tQs = [], [], [], []
        for seed in H.SEEDS5:
            D = np.zeros((len(val), 3)); Q = np.zeros((len(val), 3))
            tD = np.zeros((n - split, 3)); tQ = np.zeros((n - split, 3))
            for ei in range(3):
                w = H.w_old(ydir, rt[:, ei].astype(np.float32))
                m, _ = fit_two_targets(xs[:split], ydir[:split], yqual[:split], w[:split],
                                       xs[split:], ydir[split:], yqual[split:], w[split:],
                                       seed=seed, ei=ei, device=device)
                sel = ev_expert == ei
                if sel.any():
                    D[sel], Q[sel] = E.heads(m, xv_std[sel], device)
                ts = tail_expert == ei
                if ts.any():
                    tD[ts], tQ[ts] = E.heads(m, xs[split:][ts], device)
            Ds.append(D); Qs.append(Q); tDs.append(tD); tQs.append(tQ)
            log(f"  seed {seed} 완료")
        Dm, Qm = np.mean(Ds, 0), np.mean(Qs, 0)
        tDm, tQm = np.mean(tDs, 0), np.mean(tQs, 0)

        for gate_name, (ql, qsh) in (("q=0.50(배포값)", (Q_DEPLOYED, Q_DEPLOYED)),
                                     ("대칭보정", H.thresholds(tDm, tQm, symmetric=True))):
            side = H.side_from(Dm, Qm, ql, qsh)
            if (side != 0).sum() < 30:
                log(f"  [{var}/{gate_name}] 통과 {int((side!=0).sum())}건 -- 너무 적어 건너뜀")
                continue
            r = F.bias_report(f"{var}/{gate_name}", side, fwd1, yv, vdays)
            r.update({k: v for k, v in G.day_stats(side, fwd1, vdays).items() if not k.startswith("_")})
            r.update({"variant": var, "gate": gate_name, "q_long": ql, "q_short": qsh,
                      "pass_rate": float((side != 0).mean()),
                      "per_day": float((side != 0).sum() / len(np.unique(vdays))),
                      "dir_qual_agree": agree})
            results.append(r)
            es = r["excess_vs_short"]
            log(f"  [{gate_name}] q={ql:.3f}/{qsh:.3f} · n {r['n']:,} ({r['per_day']:.1f}건/일 · "
                f"통과율 {r['pass_rate']:.3f}) · 롱 {r['long_share']*100:.1f}% · 총 {r['gross_bp']:+.2f} · "
                f"중앙 {r['median_bp']:+.2f} · 양수일 {r['pos_day_share']*100:.1f}% · "
                f"모델−숏 {es['bp']:+.2f} [{es['ci95'][0]:+.2f},{es['ci95'][1]:+.2f}]")
        (E.OUT / "stageL_h48qual_retrain.json").write_text(json.dumps(results, indent=2, default=float))

    log(f"\n{'='*88}\n=== 요약 (배포 h48qual: 83건 · 0.7건/일 · +2.12bp / 배포 zig075: 1,225건 · 10.0건/일 · +4.94bp) ===")
    log(f"{'판':<34}{'건':>7}{'건/일':>8}{'통과율':>8}{'롱%':>7}{'총bp':>8}{'중앙':>7}{'양수일':>8}{'모델−숏':>9}")
    for r in results:
        log(f"{r['name']:<34}{r['n']:>7,}{r['per_day']:>8.1f}{r['pass_rate']:>8.3f}"
            f"{r['long_share']*100:>6.1f}%{r['gross_bp']:>8.2f}{r['median_bp']:>7.2f}"
            f"{r['pos_day_share']*100:>7.1f}%{r['excess_vs_short']['bp']:>+9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
