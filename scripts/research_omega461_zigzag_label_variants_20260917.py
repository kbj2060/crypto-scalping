#!/usr/bin/env python3
"""I — 라벨 설계 5판. **버퍼 폭**(활성 구간을 넓힐까)과 **soft 타깃**(1 대신 0.19~1.0 연속)을 붙인다.

## 사실 확인 (제안된 전제와 다른 점)
① 라벨은 **봉 하나가 아니라 구간 전체**다. 2024 실측: LONG 918구간·길이 중앙 **34봉**,
   SHORT 925구간·29봉, CASH 1,844구간·**5봉**. 「1 부근을 모두 1로」는 이미 그렇고,
   남은 손잡이는 피벗 주변 CASH 버퍼(`transition_buffer`, 현행 2)를 줄이는 것이다.
② **soft 라벨이 이미 있는데 학습에 안 쓰인다.** `zigzag_soft_{cash,long,short}` 가 생성기
   산출물이고 LONG 봉의 soft_long 은 평균 0.7395·분위[.19,.54,.84,.99,1.0] 로 연속이다.
   쓰이는 곳은 `_quality_target_hard_rule` 의 `side_soft>=0.70` 이진 임계 한 줄뿐이고 그건
   h48qual 전용 -- **zig075 는 same_as_direction 이라 soft 를 아예 안 본다.** 현 부모의 방향
   손실은 순수 hard CE 다.

## 판
  L0 버퍼2/hard(현행) · L1 버퍼2/soft · L2 버퍼2/smoothing0.1 · L3 버퍼0/hard · L4 버퍼0/soft

나머지는 H 최선 구성으로 고정: base(2025-01~2026-02) · **무작위 5시드 앙상블** ·
old 가중 · **측면 대칭 게이트**. 라벨만 바뀐다.

⚠️**bacc 는 라벨 정의가 바뀌면 서로 비교가 안 된다.** 전 판 공통으로 **현행(버퍼2) 라벨
기준**으로 재고, 판정은 경제 수치(총bp·중앙 건당bp·양수일·모델−롱/숏)로 한다.
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
import build_wave3_action_labels_20260531 as W  # noqa: E402
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402
import research_omega461_parent_quality_calibration_and_bias_20260917 as F  # noqa: E402
import research_omega461_parent_pnl_concentration_20260917 as G  # noqa: E402
import research_omega461_parent_debias_retrain_20260917 as H  # noqa: E402

BASE = E.OUT.parent          # tmp/omega461_longwindow_20260917 (E.OUT 은 그 아래 stageE)
ARCH = ROOT / "data/eth_5m_2021_2023_archive.csv"
SOFT = ["zigzag_soft_cash", "zigzag_soft_long", "zigzag_soft_short"]
VARIANTS = [("L0_buf2_hard", 2, "hard"), ("L1_buf2_soft", 2, "soft"),
            ("L2_buf2_smooth", 2, "smooth"), ("L3_buf0_hard", 0, "hard"),
            ("L4_buf0_soft", 0, "soft")]
SMOOTH_EPS = 0.1


def log(*a, **k): print(*a, flush=True)


def label_dir(buf: int) -> Path:
    """버퍼 폭을 바꾼 라벨을 연도별로 생성(정본과 같은 연도별 분리)."""
    d = BASE / f"zigzag_labels_buf{buf}"
    if (d / "zigzag_action_labels_2026.csv").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    prm = json.loads((BASE / "zz_repro" / "zigzag_action_label_audit.json").read_text())["params"]
    kw = dict(min_reversal_pct=float(prm["zigzag_reversal_pct"]),
              min_wave_bars=int(prm["min_wave_bars"]), transition_buffer=int(buf),
              atr_window=int(prm["atr_window"]), atr_multiplier=float(prm["atr_multiplier"]),
              mae_penalty=float(prm["mae_penalty"]),
              softmax_temperature=float(prm["softmax_temperature"]),
              min_risk_floor=float(prm["min_risk_floor"]))
    src = {2024: ROOT / "data/splits/year_oos/training_features_2024.csv",
           2025: ROOT / "data/splits/year_oos/training_features_2025.csv",
           2026: ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"}
    for y, p in src.items():
        fr = W._read_frame(p, expected_year=y)
        lab = W.build_zigzag_action_labels(fr, **kw)
        s = W._summary(lab)
        log(f"  버퍼{buf} {y}: {s['rows']}행 · counts {s['counts']} · CASH {s['ratios']['0']:.4f}")
        lab.to_csv(d / f"zigzag_action_labels_{y}.csv", index=False)
    return d


def join(df, ld: Path):
    lab = pd.concat([pd.read_csv(ld / f"zigzag_action_labels_{y}.csv",
                                 usecols=["timestamp", "zigzag_action"] + SOFT)
                     for y in (2025, 2026)], ignore_index=True)
    lab["timestamp"] = pd.to_datetime(lab["timestamp"]).dt.tz_localize(None)
    lab = lab.drop_duplicates("timestamp").sort_values("timestamp")
    return df.drop(columns=["zigzag_action"]).merge(lab, on="timestamp", how="inner") \
             .sort_values("timestamp").reset_index(drop=True)


def fit_soft(xtr, ttr, wtr, xv, tv, wv, *, seed, ei, device):
    """타깃이 분포(B,3)인 경우의 학습. hard 는 원-핫으로 넘기면 같은 함수로 처리된다."""
    torch.manual_seed(seed + ei); np.random.seed(seed + ei)
    m = tabm.ThreeHeadTabM(xtr.shape[1], cfg=tabm.CFG).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=float(tabm.CFG.lr),
                            weight_decay=float(tabm.CFG.weight_decay))
    dl = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ttr),
                                  torch.from_numpy(wtr)),
                    batch_size=int(tabm.CFG.batch_size), shuffle=True)
    k = int(tabm.CFG.k)
    best, state, stale, bep = float("inf"), None, 0, 0
    for ep in range(E.EPOCHS):
        m.train()
        for xb, tb, wb in dl:
            xb, tb, wb = xb.to(device), tb.to(device), wb.to(device)
            o = m(xb); den = torch.clamp(wb.sum(), min=1.0)
            ld = -(tb.unsqueeze(1) * torch.log_softmax(o["direction"], -1)).sum(-1)   # (B,k)
            lq = -(tb.unsqueeze(1) * torch.log_softmax(o["quality"], -1)).sum(-1)
            loss = (ld.mean(1) * wb).sum() / den + float(tabm.CFG.quality_loss_weight) * (lq.mean(1) * wb).sum() / den
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()
        m.eval()
        with torch.no_grad():
            o = m(torch.from_numpy(xv).to(device))
            tvt = torch.from_numpy(tv).to(device); wvt = torch.from_numpy(wv).to(device)
            ld = -(tvt.unsqueeze(1) * torch.log_softmax(o["direction"], -1)).sum(-1)
            vl = float((ld.mean(1) * wvt).sum() / torch.clamp(wvt.sum(), min=1.0))
        if vl < best - 1e-5:
            best, bep, stale = vl, ep + 1, 0
            state = {a: v.detach().clone() for a, v in m.state_dict().items()}
        else:
            stale += 1
            if stale >= E.PATIENCE: break
    m.load_state_dict(state); m.eval()
    return m, bep


def targets(y, soft, mode):
    if mode == "soft":
        t = soft / np.clip(soft.sum(1, keepdims=True), 1e-9, None)
    else:
        t = np.eye(3, dtype=np.float64)[y]
        if mode == "smooth":
            t = t * (1 - SMOOTH_EPS) + SMOOTH_EPS / 3.0
    return t.astype(np.float32)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} · seeds={H.SEEDS5}")
    df0, base_cols = E.load()
    vm = (df0.timestamp >= E.VAL[0]) & (df0.timestamp <= E.VAL[1] + " 23:59:59")
    val_ref = df0[vm].reset_index(drop=True)
    yv_ref = pd.to_numeric(val_ref["zigzag_action"]).to_numpy(np.int64)   # ⚠️공통 기준(버퍼2)
    xv_raw = tabm._base_input(val_ref, base_cols)
    ev_expert = tabm._route_probs(val_ref).argmax(1)
    vdays = val_ref.timestamp.dt.floor("D").to_numpy()
    fwd1 = val_ref["fwd_1h_bp"].to_numpy(np.float64)
    t0, t1 = E.ARMS["base"]
    results = []

    for tag, buf, mode in VARIANTS:
        ld = label_dir(buf)
        df = join(df0, ld)
        tr = df[(df.timestamp >= t0) & (df.timestamp <= t1 + " 23:59:59")].reset_index(drop=True)
        assert len(tr) > 100_000, f"{tag} TRAIN 이 작다: {len(tr)}"
        yt = pd.to_numeric(tr["zigzag_action"]).to_numpy(np.int64)
        soft_t = tr[SOFT].to_numpy(np.float64)
        T = targets(yt, soft_t, mode)
        rt = tabm._route_probs(tr)
        n = len(tr); split = max(int(n * 0.85), min(n - 1, 512))
        xs, scaler = tabm._standardize_fit(tabm._base_input(tr, base_cols))
        xv_std = tabm._standardize_apply(xv_raw, scaler)
        tail_expert = rt[split:].argmax(1)
        log(f"\n{'='*88}\n=== {tag} (버퍼 {buf} · 타깃 {mode}) TRAIN {n:,}행 "
            f"· 라벨비중 {(np.bincount(yt,minlength=3)/n).round(4)} "
            f"· 타깃 활성평균 {T[:,1:].max(1).mean():.4f}\n{'='*88}")

        Ds, Qs, tDs, tQs = [], [], [], []
        for seed in H.SEEDS5:
            D = np.zeros((len(val_ref), 3)); Q = np.zeros((len(val_ref), 3))
            tD = np.zeros((n - split, 3)); tQ = np.zeros((n - split, 3))
            for ei in range(3):
                w = H.w_old(yt, rt[:, ei].astype(np.float32))
                m, bep = fit_soft(xs[:split], T[:split], w[:split],
                                  xs[split:], T[split:], w[split:],
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
        ql, qsh = H.thresholds(np.mean(tDs, 0), np.mean(tQs, 0), symmetric=True)
        side = H.side_from(Dm, Qm, ql, qsh)
        r = F.bias_report(tag, side, fwd1, yv_ref, vdays)
        r.update({k: v for k, v in G.day_stats(side, fwd1, vdays).items() if not k.startswith("_")})
        r.update({"variant": tag, "buffer": buf, "target": mode,
                  "bacc_vs_ref": E.bacc(yv_ref, Dm.argmax(1)),
                  "train_label_shares": (np.bincount(yt, minlength=3) / n).tolist(),
                  "q_long": ql, "q_short": qsh})
        results.append(r)
        el, es = r["excess_vs_long"], r["excess_vs_short"]
        log(f"  → n {r['n']:,} · 롱 {r['long_share']*100:.1f}% · 총 {r['gross_bp']:+.2f} · "
            f"중앙 {r['median_bp']:+.2f} · 양수일 {r['pos_day_share']*100:.1f}% · "
            f"최고1일제거 {r['drop_top1d_bp']:+.2f} · bacc(공통기준) {r['bacc_vs_ref']:.4f}")
        log(f"     모델−롱 {el['bp']:+.2f} [{el['ci95'][0]:+.2f},{el['ci95'][1]:+.2f}] · "
            f"모델−숏 {es['bp']:+.2f} [{es['ci95'][0]:+.2f},{es['ci95'][1]:+.2f}]"
            f"{'  🟢' if es['ci95'][0] > 0 else ''}")
        (E.OUT / "stageI_label_variants.json").write_text(json.dumps(results, indent=2, default=float))

    log(f"\n{'='*88}\n=== 요약 (base · 5시드 앙상블 · 대칭게이트 · 라벨만 교체) ===")
    log(f"{'판':<18}{'CASH%':>7}{'롱%':>7}{'총bp':>8}{'중앙':>7}{'양수일':>8}{'bacc':>8}{'모델−숏':>9}")
    for r in results:
        log(f"{r['variant']:<18}{r['train_label_shares'][0]*100:>6.1f}%{r['long_share']*100:>6.1f}%"
            f"{r['gross_bp']:>8.2f}{r['median_bp']:>7.2f}{r['pos_day_share']*100:>7.1f}%"
            f"{r['bacc_vs_ref']:>8.4f}{r['excess_vs_short']['bp']:>+9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
