#!/usr/bin/env python3
"""D단계 -- Omega4.6.1 부모(zig075) 아키텍처로 「학습창 깊이」만 바꾼 사다리 3판.

배포 부모의 실학습창은 2025-01~09 한 해다(A~C 기록 §2). 여기서 묻는 건 하나:
**같은 레시피에 데이터를 더 주면 방향이 좋아지는가.**

세 판 전부 TRAIN 끝을 2025-09-30 으로 맞춘다 -- 깊이만 변하고 「최신성」은 고정이다.
(두 판 제안에서 세 판으로 늘렸다. 원안은 2024판이 2025판보다 «깊고 동시에 오래된» 창이라
 깊이와 최신성이 섞였다. 끝을 맞추면 사다리가 순수해진다.)

  m09  2025-01-01~2025-09-30  <- **배포 부모의 TRAIN 창과 정확히 같다**
  m15  2024-07-01~2025-09-30
  m21  2024-01-01~2025-09-30

m09 가 배포창과 같으므로 사다리의 바닥은 재현 점검을 겸한다.

⚠️**라벨이 2026-02-28 16:00 에서 끝난다**(2024 105,380 · 2025 105,101 · 2026 16,897행,
2026 파일은 1~2월뿐). 원래 제안한 공통 VAL 2026-03-01~06-30 은 **라벨이 없어 불가능**하다.
그래서 VAL 은 2025-10-01~2026-02-28 -- 배포 부모의 VAL+OOS 와 같은 창이고 이미 여러 번
소진됐다. ⇒ **이 판정은 research/dev 점수다.** 그래도 세 판이 같은 창을 똑같이 보므로
«깊이가 이득인가»라는 상대 비교 자체는 성립한다. single-touch OOS(2026-07-01~09-30)는
건드리지 않았고 최종 판정은 거기서 한다.

라벨은 배포 부모와 같은 zigzag_action_labels_20260531 이고 2024 부터만 존재한다 --
그래서 사다리의 바닥이 2024 다(역공학은 2024 재현 관문에서 실패, 기록 §6).

**직접 훈련하는 건 direction/quality 두 머리뿐이다.** exit 머리는 2025 전용 lifecycle
아티팩트(TABM_2025)를 요구하고, 「언제 나갈까」는 깊이 질문이 아니다. 세 판 전부 같은
방식으로 끄므로 판 사이 비교는 그대로 성립한다. 정식 3-머리 학습은 아니다 -- 승격 근거로
쓰지 않는다.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402

PARQUET = ROOT / "tmp/omega461_longwindow_20260917/features_with_regime_2022_2026.parquet"
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
DEPLOYED_BUNDLE = (ROOT / "tmp/causal_regen_20260516"
                   / "omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01"
                     "_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629"
                   / "true_3head_tabm_bundle.pt")
OUT = ROOT / "tmp/omega461_longwindow_20260917/stageD"

ARMS = {
    "m09": ("2025-01-01", "2025-09-30"),
    "m15": ("2024-07-01", "2025-09-30"),
    "m21": ("2024-01-01", "2025-09-30"),
}
VAL = ("2025-10-01", "2026-02-28")
SEEDS = [170917, 430211, 885604]   # 무작위 추출, 고정간격 아님 (Seed-Diversity 계약)
EPOCHS, PATIENCE = 6, 2
TAG = ""     # 대조군 실행이 결과 파일을 덮지 않도록 호출측이 바꾼다


def log(*a):
    print(*a, flush=True)


def load() -> tuple[pd.DataFrame, list[str]]:
    bundle = torch.load(DEPLOYED_BUNDLE, map_location="cpu", weights_only=False)
    input_cols = list(bundle["models"]["bull"]["input_columns"])
    base_cols = list(bundle["base_cols"])
    assert input_cols == base_cols + list(tabm.POS_COLS), "배포 번들의 115열 계약과 어긋난다"

    df = pd.read_parquet(PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    lab = pd.concat([pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{y}.csv",
                                 usecols=["timestamp", "zigzag_action"])
                     for y in (2024, 2025, 2026)], ignore_index=True)
    lab["timestamp"] = pd.to_datetime(lab["timestamp"]).dt.tz_localize(None)
    lab = lab.drop_duplicates("timestamp").sort_values("timestamp")

    n_before = len(df)
    df = df.merge(lab, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    log(f"프레임 {n_before} -> 라벨 조인 {len(df)}행  "
        f"[{df.timestamp.min()} ~ {df.timestamp.max()}]")

    missing = [c for c in base_cols if c not in df.columns]
    assert not missing, f"base_cols 결손: {missing}"
    y = pd.to_numeric(df["zigzag_action"], errors="raise").to_numpy(np.int64)
    assert set(np.unique(y)) <= {0, 1, 2}, f"라벨 클래스 이상: {np.unique(y)}"
    shares = np.bincount(y, minlength=3) / len(y)
    log(f"라벨 비중 CASH/LONG/SHORT = {shares.round(4)}")
    assert shares.min() > 0.02, "라벨 퇴화 -- 여기서 멈춘다"
    return df, base_cols


def make_x(frame: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """tabm._base_input 과 같은 규약: base_cols 를 수치화/무한대 제거/0채움, pos_* 는 0."""
    return tabm._base_input(frame, base_cols)


def fit_expert(xtr, ytr, wtr, xin_val, yin_val, win_val, *, seed, expert_idx, device):
    torch.manual_seed(seed + expert_idx)
    np.random.seed(seed + expert_idx)
    random.seed(seed + expert_idx)
    model = tabm.ThreeHeadTabM(xtr.shape[1], cfg=tabm.CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tabm.CFG.lr),
                            weight_decay=float(tabm.CFG.weight_decay))
    dl = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr),
                                  torch.from_numpy(wtr)),
                    batch_size=int(tabm.CFG.batch_size), shuffle=True, drop_last=False)
    k = int(tabm.CFG.k)
    best, best_state, stale, best_epoch, steps = float("inf"), None, 0, 0, 0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            out = model(xb)
            tgt = yb[:, None].expand(-1, k).reshape(-1)
            ld = torch.nn.functional.cross_entropy(out["direction"].reshape(-1, 3), tgt,
                                                   reduction="none").reshape(-1, k)
            lq = torch.nn.functional.cross_entropy(out["quality"].reshape(-1, 3), tgt,
                                                   reduction="none").reshape(-1, k)
            den = torch.clamp(wb.sum(), min=1.0)
            loss = ((ld.mean(1) * wb).sum() / den
                    + float(tabm.CFG.quality_loss_weight) * (lq.mean(1) * wb).sum() / den)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            steps += 1
        model.eval()
        with torch.no_grad():
            o = model(torch.from_numpy(xin_val).to(device))
            vy = torch.from_numpy(yin_val).to(device)
            vw = torch.from_numpy(win_val).to(device)
            ld = torch.nn.functional.cross_entropy(
                o["direction"].reshape(-1, 3), vy[:, None].expand(-1, k).reshape(-1),
                reduction="none").reshape(-1, k)
            vloss = float((ld.mean(1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
        if vloss < best - 1e-5:
            best, best_epoch, stale = vloss, epoch + 1, 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, {"inner_val_loss": best, "best_epoch": best_epoch, "steps": steps}


def proba(model, x, device):
    with torch.no_grad():
        out = model(torch.from_numpy(x).to(device))["direction"]     # (n, k, 3)
        return torch.softmax(out, dim=-1).mean(dim=1).cpu().numpy()  # TabM: k 평균


def bacc(y, pred):
    accs = [float((pred[y == c] == c).mean()) for c in (0, 1, 2) if (y == c).any()]
    return float(np.mean(accs))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")
    df, base_cols = load()

    vm = (df.timestamp >= VAL[0]) & (df.timestamp <= VAL[1] + " 23:59:59")
    val = df[vm].reset_index(drop=True)
    assert len(val) > 20_000, f"VAL 이 너무 작다: {len(val)}"
    yv = pd.to_numeric(val["zigzag_action"]).to_numpy(np.int64)
    xv = tabm._standardize_apply  # noqa: F841  (스케일러는 판마다 다르므로 아래서 적용)
    xv_raw = make_x(val, base_cols)
    rv = tabm._route_probs(val)
    log(f"VAL {VAL[0]}~{VAL[1]} {len(val)}행 · 라벨비중 "
        f"{(np.bincount(yv, minlength=3)/len(yv)).round(4)}")

    rows = []
    for arm, (t0, t1) in ARMS.items():
        tm = (df.timestamp >= t0) & (df.timestamp <= t1 + " 23:59:59")
        tr = df[tm].reset_index(drop=True)
        assert tr.timestamp.max() < pd.Timestamp(VAL[0]), "TRAIN 이 VAL 을 침범했다"
        yt = pd.to_numeric(tr["zigzag_action"]).to_numpy(np.int64)
        xt_raw = make_x(tr, base_cols)
        rt = tabm._route_probs(tr)
        n = len(tr)
        split = max(int(n * 0.85), min(n - 1, 512))
        log(f"\n=== {arm}  TRAIN {t0}~{t1}  {n}행 (내부검증 {n-split}행) "
            f"· 라벨비중 {(np.bincount(yt, minlength=3)/n).round(4)}")
        xs, scaler = tabm._standardize_fit(xt_raw)   # 판마다 한 번 (전문가·시드 무관)
        xv_std = tabm._standardize_apply(xv_raw, scaler)

        for seed in SEEDS:
            pv = np.zeros((len(val), 3))
            diag = {}
            for ei, ename in enumerate(tabm.EXPERT_NAMES if hasattr(tabm, "EXPERT_NAMES")
                                       else ["bull", "bear", "chop"]):
                w = (compute_sample_weight("balanced", y=yt).astype(np.float32)
                     * rt[:, ei].astype(np.float32))
                assert w.sum() > 0, f"{arm}/{ename} 라우팅 가중치가 전부 0"
                m, d = fit_expert(xs[:split], yt[:split], w[:split],
                                  xs[split:], yt[split:], w[split:],
                                  seed=seed, expert_idx=ei, device=device)
                pv += proba(m, xv_std, device) * rv[:, ei:ei+1]
                diag[ename] = d
            pred = pv.argmax(1)
            sh = np.bincount(pred, minlength=3) / len(pred)
            assert sh.max() < 0.98, f"{arm}/seed{seed} 예측 퇴화: {sh}"
            r = {"arm": arm, "seed": seed, "train_rows": n,
                 "val_bacc": bacc(yv, pred),
                 "val_acc": float((pred == yv).mean()),
                 "val_trade_recall": float((pred[yv != 0] != 0).mean()),
                 "val_trade_precision": float((yv[pred != 0] != 0).mean()) if (pred != 0).any() else 0.0,
                 "val_side_acc": float((pred[(pred != 0) & (yv != 0)] == yv[(pred != 0) & (yv != 0)]).mean())
                                 if ((pred != 0) & (yv != 0)).any() else float("nan"),
                 "pred_share": sh.round(4).tolist(),
                 "experts": diag}
            rows.append(r)
            log(f"  seed {seed}: bacc {r['val_bacc']:.4f} · acc {r['val_acc']:.4f} · "
                f"방향정확 {r['val_side_acc']:.4f} · 진입비중 {1-sh[0]:.3f} · "
                f"best_epoch {[v['best_epoch'] for v in diag.values()]}")

    (OUT / f"stageD_results{TAG}.json").write_text(json.dumps(rows, indent=2, default=float))
    log("\n=== 판별 요약 (시드 3개 평균 ± 폭) ===")
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        b = np.array([r["val_bacc"] for r in sub])
        s = np.array([r["val_side_acc"] for r in sub])
        log(f"{arm} (n={sub[0]['train_rows']:>7,}) bacc {b.mean():.4f} [{b.min():.4f},{b.max():.4f}] "
            f"· 방향정확 {s.mean():.4f} [{s.min():.4f},{s.max():.4f}]")
    base = np.array([r["val_bacc"] for r in rows if r["arm"] == "m09"])
    for arm in [a for a in ("m15", "m21") if a in ARMS and len(base)]:
        a = np.array([r["val_bacc"] for r in rows if r["arm"] == arm])
        d = a.mean() - base.mean()
        spread = max(a.max() - a.min(), base.max() - base.min())
        log(f"Δbacc({arm} - m09) = {d:+.4f}  · 시드폭 {spread:.4f}  "
            f"-> {'시드잡음보다 크다' if abs(d) > spread else '🔴시드잡음 안에 있다'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
