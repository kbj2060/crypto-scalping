#!/usr/bin/env python3
"""E단계 — Omega4.6.1 **부모(zig075)** 를 되살린 장기 데이터로 학습하고 실제로 테스트한다.

## 왜 zig075 인가
두 부모(h48qual·zig075)의 **방향 라벨은 동일한 zigzag** 다. runtime_contract 로 확인:
  h48qual  direction=zigzag_action_labels_20260531 · quality_mode=quality_label_action
  zig075   direction=zigzag_action_labels_20260531 · quality_mode=same_as_direction
즉 둘의 차이는 **품질 머리의 타깃**뿐이다. zig075 는 추가 라벨 없이 완결되고, h48qual 은
`sltp_h48_conservative_padded` 를 2022~2023 으로 다시 만들어야 한다. 그래서 zig075.

## 이번 판이 이전과 다른 점
1. ⭐**펀딩이 진짜 값이다.** 옛 프레임은 2022~2024 펀딩이 전량 중앙값이라 파생 9열
   (`ou_halflife` 포함)이 같이 상수였다. `/fapi/v1/fundingRate` 로 5,343건을 복구해
   채움 324,577행 → **0행**, 연도간 상수열 11개 → 3개(2022 롱숏비, 바이낸스 이력 한계).
2. ⭐**라벨이 2021-12~2026-08 로 늘었다.** 역공학은 필요 없었다 — 생성기가
   `build_wave3_action_labels_20260531.py` 라는 이름으로 커밋돼 있었고(내부 DEFAULT_OUT 이
   zigzag_action_labels_20260531), 손대지 않고 돌리니 **2024·2025 가 행 단위 완전 재현**
   (일치율 1.000000, 210,481행). 2026 은 정본이 02-28 절단본이라 그 앞 13,601행이 완전 일치.
3. ⭐**추론을 라이브와 똑같이 hard routing 으로 한다**(`_route_expert` 가 전문가 **하나**를
   고른다). D단계 사다리는 확률 가중 혼합이었다 -- 판 사이 비교에는 무해했지만 절대 수치를
   라이브와 견주려면 안 된다.

## 설계
| 판 | TRAIN | 비고 |
|---|---|---|
| deep | 2022-01-01 ~ 2026-02-28 | 되살린 전 구간 |
| base | 2025-01-01 ~ 2026-02-28 | 배포 깊이 대조군, 끝 날짜 동일 |

VAL **2026-03-01~06-30** — 배포 부모의 VAL(2025-10~12)·OOS(2026-01~02) **둘 다 밖**이고
예약된 single-touch OOS(2026-07-01~09-30) **앞**이다. 전진 분할이라 겹침이 없다.

⚠️학습하는 건 direction/quality 두 머리다(exit 는 2025 전용 lifecycle 아티팩트를 요구).
경제 수치는 **총이익 방향 읽기**이지 라이브 경로(ATR 배리어·exit 머리·라우터·사이징)가 아니다.
승격 근거로 쓰지 않는다.
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402

BASE = ROOT / "tmp/omega461_longwindow_20260917"
PARQUET = BASE / "features_with_regime_2022_2026_realfunding.parquet"
LABELS = BASE / "zigzag_labels_full"
BUNDLE = (ROOT / "tmp/causal_regen_20260516"
          / "omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01"
            "_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629"
          / "true_3head_tabm_bundle.pt")
OUT = BASE / "stageE"
ARMS = {"deep": ("2022-01-01", "2026-02-28"), "base": ("2025-01-01", "2026-02-28")}
VAL = ("2026-03-01", "2026-06-30")
SEEDS = [170917, 430211, 885604]
EPOCHS, PATIENCE = 6, 2
Q_THRESH = 0.75              # 배포 zig075 의 quality_threshold
HORIZONS = {"1h": 12, "4h": 48}
COST_USDC_BP, COST_PEG_BP = 1.02, 5.52   # USDC 메이커(AS 0.51×2) · 배포 peg 실측


def log(*a): print(*a, flush=True)


def load():
    b = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    base_cols, input_cols = list(b["base_cols"]), list(b["models"]["bull"]["input_columns"])
    assert input_cols == base_cols + list(tabm.POS_COLS), "배포 번들 115열 계약 불일치"

    df = pd.read_parquet(PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    lab = pd.concat([pd.read_csv(p, usecols=["timestamp", "zigzag_action"])
                     for p in sorted(LABELS.glob("zigzag_action_labels_*.csv"))], ignore_index=True)
    lab["timestamp"] = pd.to_datetime(lab["timestamp"]).dt.tz_localize(None)
    lab = lab.drop_duplicates("timestamp").sort_values("timestamp")
    n0 = len(df)
    df = df.merge(lab, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    log(f"프레임 {n0:,} -> 라벨 조인 {len(df):,}행  [{df.timestamp.min()} ~ {df.timestamp.max()}]")
    assert not [c for c in base_cols if c not in df.columns], "base_cols 결손"

    y = pd.to_numeric(df["zigzag_action"], errors="raise").to_numpy(np.int64)
    sh = np.bincount(y, minlength=3) / len(y)
    log(f"라벨 비중 CASH/LONG/SHORT = {sh.round(4)}")
    assert sh.min() > 0.02, "라벨 퇴화"

    # 전방수익(bp) -- 경제 읽기용. 마지막 h봉은 NaN 이고 VAL 에서 제외된다.
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)
    for name, h in HORIZONS.items():
        fwd = np.full(len(c), np.nan)
        fwd[:-h] = (c[h:] / c[:-h] - 1.0) * 1e4
        df[f"fwd_{name}_bp"] = fwd
    return df, base_cols


def fit_expert(xtr, ytr, wtr, xv, yv, wv, *, seed, ei, device):
    torch.manual_seed(seed + ei); np.random.seed(seed + ei); random.seed(seed + ei)
    m = tabm.ThreeHeadTabM(xtr.shape[1], cfg=tabm.CFG).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=float(tabm.CFG.lr),
                            weight_decay=float(tabm.CFG.weight_decay))
    dl = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr), torch.from_numpy(wtr)),
                    batch_size=int(tabm.CFG.batch_size), shuffle=True)
    k = int(tabm.CFG.k)
    best, state, stale, bep, steps = float("inf"), None, 0, 0, 0
    for ep in range(EPOCHS):
        m.train()
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            o = m(xb); t = yb[:, None].expand(-1, k).reshape(-1)
            ld = torch.nn.functional.cross_entropy(o["direction"].reshape(-1, 3), t, reduction="none").reshape(-1, k)
            lq = torch.nn.functional.cross_entropy(o["quality"].reshape(-1, 3), t, reduction="none").reshape(-1, k)
            den = torch.clamp(wb.sum(), min=1.0)
            loss = (ld.mean(1) * wb).sum() / den + float(tabm.CFG.quality_loss_weight) * (lq.mean(1) * wb).sum() / den
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step(); steps += 1
        m.eval()
        with torch.no_grad():
            o = m(torch.from_numpy(xv).to(device))
            t = torch.from_numpy(yv).to(device)[:, None].expand(-1, k).reshape(-1)
            ld = torch.nn.functional.cross_entropy(o["direction"].reshape(-1, 3), t, reduction="none").reshape(-1, k)
            vl = float((ld.mean(1) * torch.from_numpy(wv).to(device)).sum()
                       / torch.clamp(torch.from_numpy(wv).to(device).sum(), min=1.0))
        if vl < best - 1e-5:
            best, bep, stale = vl, ep + 1, 0
            state = {a: v.detach().clone() for a, v in m.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE: break
    m.load_state_dict(state); m.eval()
    return m, {"inner_val_loss": best, "best_epoch": bep, "steps": steps}


def heads(model, x, device, chunk=65536):
    d, q = [], []
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            o = model(torch.from_numpy(x[i:i + chunk]).to(device))
            d.append(torch.softmax(o["direction"], -1).mean(1).cpu().numpy())
            q.append(torch.softmax(o["quality"], -1).mean(1).cpu().numpy())
    return np.concatenate(d), np.concatenate(q)


def bacc(y, p):
    return float(np.mean([float((p[y == c] == c).mean()) for c in (0, 1, 2) if (y == c).any()]))


def block_ci(vals, days, n=2000, seed=17):
    """날짜블록 부트스트랩 -- 5분봉은 독립이 아니다. 독립 단위는 «일»이다."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(days)
    by = {d: vals[days == d] for d in uniq}
    means = np.array([np.concatenate([by[d] for d in rng.choice(uniq, len(uniq), replace=True)]).mean()
                      for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(uniq)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")
    df, base_cols = load()

    vm = (df.timestamp >= VAL[0]) & (df.timestamp <= VAL[1] + " 23:59:59")
    val = df[vm].reset_index(drop=True)
    assert len(val) > 20_000, f"VAL 이 너무 작다: {len(val)}"
    yv = pd.to_numeric(val["zigzag_action"]).to_numpy(np.int64)
    xv_raw = tabm._base_input(val, base_cols)
    rv = tabm._route_probs(val)
    ev_expert = rv.argmax(1)                       # ⭐라이브와 같은 hard routing
    vdays = val.timestamp.dt.floor("D").to_numpy()
    log(f"VAL {VAL[0]}~{VAL[1]} {len(val):,}행 · 독립일 {len(np.unique(vdays))} · "
        f"라벨비중 {(np.bincount(yv, minlength=3)/len(yv)).round(4)} · "
        f"라우팅 {np.bincount(ev_expert, minlength=3)/len(val)}")

    rows = []
    for arm, (t0, t1) in ARMS.items():
        tm = (df.timestamp >= t0) & (df.timestamp <= t1 + " 23:59:59")
        tr = df[tm].reset_index(drop=True)
        assert tr.timestamp.max() < pd.Timestamp(VAL[0]), "TRAIN 이 VAL 을 침범했다"
        yt = pd.to_numeric(tr["zigzag_action"]).to_numpy(np.int64)
        rt = tabm._route_probs(tr)
        n = len(tr); split = max(int(n * 0.85), min(n - 1, 512))
        log(f"\n=== {arm}  TRAIN {t0}~{t1}  {n:,}행 (내부검증 {n-split:,}) · "
            f"라벨비중 {(np.bincount(yt, minlength=3)/n).round(4)}")
        xs, scaler = tabm._standardize_fit(tabm._base_input(tr, base_cols))
        xv_std = tabm._standardize_apply(xv_raw, scaler)

        for seed in SEEDS:
            D = np.zeros((len(val), 3)); Q = np.zeros((len(val), 3)); diag = {}
            for ei, ename in enumerate(("bull", "bear", "chop")):
                w = compute_sample_weight("balanced", y=yt).astype(np.float32) * rt[:, ei].astype(np.float32)
                assert w.sum() > 0, f"{arm}/{ename} 라우팅 가중치 전부 0"
                m, dg = fit_expert(xs[:split], yt[:split], w[:split], xs[split:], yt[split:], w[split:],
                                   seed=seed, ei=ei, device=device)
                sel = ev_expert == ei
                if sel.any():
                    d, q = heads(m, xv_std[sel], device)
                    D[sel], Q[sel] = d, q
                diag[ename] = dg
            dir_action = D.argmax(1)
            q_for = np.where(dir_action > 0, Q[np.arange(len(Q)), dir_action], Q[:, 0])
            final = np.where((dir_action != 0) & (q_for >= Q_THRESH), dir_action, 0)
            side = np.where(final == 1, 1.0, np.where(final == 2, -1.0, 0.0))

            sh = np.bincount(final, minlength=3) / len(final)
            assert sh.max() < 0.995, f"{arm}/seed{seed} 게이트 후 퇴화: {sh}"
            r = {"arm": arm, "seed": seed, "train_rows": n,
                 "val_bacc_pre_gate": bacc(yv, dir_action),
                 "val_bacc_post_gate": bacc(yv, final),
                 "gate_pass_rate": float((final != 0).mean()),
                 "side_acc_post_gate": float((final[(final != 0) & (yv != 0)] == yv[(final != 0) & (yv != 0)]).mean())
                                       if ((final != 0) & (yv != 0)).any() else float("nan"),
                 "experts": diag, "econ": {}}
            for hname in HORIZONS:
                f = val[f"fwd_{hname}_bp"].to_numpy(np.float64)
                ok = (side != 0) & np.isfinite(f)
                pnl = side[ok] * f[ok]
                lo, hi, nd = block_ci(pnl, vdays[ok]) if ok.sum() > 100 else (np.nan, np.nan, 0)
                r["econ"][hname] = {"n": int(ok.sum()), "indep_days": nd,
                                    "gross_bp": float(pnl.mean()) if ok.any() else float("nan"),
                                    "ci95": [lo, hi]}
            rows.append(r)
            e1 = r["econ"]["1h"]
            log(f"  seed {seed}: bacc {r['val_bacc_pre_gate']:.4f} → 게이트후 {r['val_bacc_post_gate']:.4f} · "
                f"통과율 {r['gate_pass_rate']:.3f} · 방향정확 {r['side_acc_post_gate']:.4f} · "
                f"1h 총 {e1['gross_bp']:+.2f}bp CI[{e1['ci95'][0]:+.2f},{e1['ci95'][1]:+.2f}] "
                f"· best_epoch {[v['best_epoch'] for v in diag.values()]}")

    (OUT / "stageE_results.json").write_text(json.dumps(rows, indent=2, default=float))
    log(f"\n=== 요약 (시드 {len(SEEDS)}개) · 비용선: USDC {COST_USDC_BP}bp · peg {COST_PEG_BP}bp ===")
    for arm in ARMS:
        s = [r for r in rows if r["arm"] == arm]
        for hname in HORIZONS:
            g = np.array([r["econ"][hname]["gross_bp"] for r in s])
            los = np.array([r["econ"][hname]["ci95"][0] for r in s])
            sa = np.array([r["side_acc_post_gate"] for r in s])
            pr = np.array([r["gate_pass_rate"] for r in s])
            verdict = ("🟢USDC 넘음" if los.min() > COST_USDC_BP else
                       "🔴CI 가 비용선 아래" if g.mean() > 0 else "🔴음수")
            log(f"{arm:>4s} {hname}: 총 {g.mean():+7.2f}bp [{g.min():+.2f},{g.max():+.2f}] · "
                f"CI하한 최소 {los.min():+7.2f} · 방향정확 {sa.mean():.4f} · 통과율 {pr.mean():.3f} "
                f"· {verdict}")
    for hname in HORIZONS:
        d = np.array([r["econ"][hname]["gross_bp"] for r in rows if r["arm"] == "deep"])
        b = np.array([r["econ"][hname]["gross_bp"] for r in rows if r["arm"] == "base"])
        spread = max(d.max() - d.min(), b.max() - b.min())
        delta = d.mean() - b.mean()
        log(f"Δ{hname}(deep - base) = {delta:+.2f}bp · 시드폭 {spread:.2f} -> "
            f"{'시드잡음보다 크다' if abs(delta) > spread else '🔴시드잡음 안'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
