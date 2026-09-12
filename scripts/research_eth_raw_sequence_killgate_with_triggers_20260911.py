#!/usr/bin/env python3
"""원시 시퀀스 딥러닝 킬게이트 + 이벤트 트리거 양성대조 (2026-09-11).

왜: 같은 날 research_eth_exit_timing_killgate_predictability_20260911.py 가 **표 형태 피쳐**로
청산 타이밍을 0/5 기각했다(VAL AUC .530~.540). 남은 정당한 반론은 하나다 —
"원시 봉 시퀀스를 시퀀스 모델에 넣으면 표 피쳐가 못 본 걸 볼 수도 있다."
이 스크립트가 그 반론을 닫는다.

⭐**양성 대조군이 핵심이다.** 딥러닝이 실패했을 때 «데이터에 정보가 없다»와 «내 모델이
망가졌다»는 겉보기가 같다. 그래서 **같은 모델·같은 입력·같은 학습 루프**로 이벤트 트리거
4종을 같이 푼다. 이 저장소의 선행 측정은 변동성확장 AUC .805 · 레짐 .699 · 극점 .722 로
**학습이 되는 축**이라고 말한다. 거기서 높은 AUC 가 나오는데 청산에서만 .54 면,
그건 모델 탓이 아니라 정보 탓이다. 트리거까지 .54 면 이 스크립트가 고장난 것이다.

라벨 정의는 이 저장소 것을 따르되 **재구성**이다 — 배포 성능 재현이 아니라 양성대조다:
  변동성확장  volexp = std(lr,12)/std(lr,288) 가 앞으로 H봉 안에 1.8 을 넘는가
              (`live_eth_breakout_detector_20260911.py` 의 전환 정의)
  레짐추세    앞으로 12봉 순이동이 atr*sqrt(12) 이상인가 (추세 vs 횡보)
  극점        앞으로 12봉 안에 ±6봉 국소극점이 생기는가
              (`live_eth_sweep_v_rebound_signal_20260829.py::LOCAL_EXTREME_W=6`)
  V자반등     앞으로 H봉 안에 1.0*atr 하락 후 1.5*atr 되돌림
  청산보유    ★킬게이트 — 남은 창의 최고 종가가 지금 종가보다 위인가

경계 계약(CLAUDE.md): **모든 라벨은 봉 t+1 부터 본다. 시퀀스는 봉 t 에서 끝난다.**
정규화는 창 자신의 통계로만 한다(인과적이고 스케일프리).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.live_evidence_signal_dashboard_20260823 import compute_signals, SIGNAL_ORDER  # noqa: E402

# 개발기 워크트리는 data/ 가 gitignore 라 본 체크아웃을 가리켜야 하고, 서버는 리포 안에 있다.
# 경로를 하드코딩하면 서버에서 조용히 깨진다.
DATA = Path(os.environ.get("ETH_DATA_DIR", str(ROOT / "data")))
if not (DATA / "eth_5m_1year.csv").exists():
    DATA = Path("/home/kbj20/crypto-scalping/data")   # 개발기 폴백
L = 96                 # 시퀀스 길이(8시간)
H = 24                 # 라벨 창(2시간)
CH = 6
SPLITS = {"IS": ("2024-01-01", "2025-08-31"), "VAL": ("2025-09-01", "2025-12-31"),
          "OOS": ("2026-01-01", "2026-12-31")}
MAX_TRAIN = 60_000     # CPU 학습 상한 — 부분표집은 시간축 균일
EPOCHS, BATCH = 12, 512
torch.manual_seed(0)
np.random.seed(0)


class TCN(nn.Module):
    """작은 확장 합성곱 — 수용영역 1+2*(1+2+4+8)=31... 4층 dilation 1/2/4/8, k=3 → 31봉.
    L=96 을 다 보도록 마지막에 전역 평균+최대 풀링을 얹는다. 파라미터 ~2만."""

    def __init__(self, ch: int = CH, w: int = 32):
        super().__init__()
        layers, c = [], ch
        for d in (1, 2, 4, 8):
            layers += [nn.Conv1d(c, w, 3, padding=d, dilation=d), nn.GELU(), nn.BatchNorm1d(w)]
            c = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * w, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):                       # x: (B, CH, L)
        h = self.body(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], -1)).squeeze(-1)


def build_channels(eth: pd.DataFrame) -> np.ndarray:
    """봉별 원시 채널 (n, CH). 정규화는 창 단위로 뒤에서 한다."""
    o, h, l, c = (eth[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v, tb = eth["volume"].to_numpy(float), eth["taker_buy_base"].to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    return np.column_stack([
        lr,                                     # 수익률
        (h - c) / c, (c - l) / c,               # 위/아래 꼬리
        (c - o) / c,                            # 몸통
        np.log1p(v) - np.log1p(pd.Series(v).rolling(288, min_periods=1).median().to_numpy()),
        tb / np.maximum(v, 1e-9) - 0.5,         # 테이커 매수 비중
    ]).astype(np.float32)


def norm_windows(w: np.ndarray) -> np.ndarray:
    """창 자신의 통계로 표준화 — 인과적이고 스케일프리(변동성 수준을 지운다)."""
    m = w.mean(-1, keepdims=True)
    s = w.std(-1, keepdims=True) + 1e-6
    return np.clip((w - m) / s, -8, 8)


def labels(eth: pd.DataFrame, sig: pd.DataFrame, n: int) -> dict[str, np.ndarray]:
    """모든 라벨은 봉 t+1 부터 본다. 반환 배열 길이 n, 못 만드는 구간은 NaN."""
    c = eth["close"].to_numpy(float)
    hi, lo = eth["high"].to_numpy(float), eth["low"].to_numpy(float)
    atr = sig["atr_pct"].to_numpy() * c
    N = len(c)
    fut_c = np.lib.stride_tricks.sliding_window_view(c, H)          # [i] = c[i:i+H]
    out: dict[str, np.ndarray] = {}

    # ★킬게이트 — 남은 창의 최고 종가가 지금보다 위인가 (보유 경과 1봉 / 12봉 두 지점)
    for age, rem in ((1, H - 1), (12, H - 12)):
        y = np.full(N, np.nan)
        w = np.lib.stride_tricks.sliding_window_view(c, rem)
        lim = N - age - rem
        t = np.arange(lim)
        y[t] = (w[t + age + 1].max(1) > c[t + age]).astype(float)
        out[f"★청산보유(경과{age}봉)"] = y

    # 변동성확장 — volexp 가 앞으로 H봉 안에 1.8 돌파
    lr = pd.Series(np.diff(np.log(c), prepend=np.log(c[0])))
    volexp = (lr.rolling(12).std() / lr.rolling(288).std()).to_numpy()
    ve = np.lib.stride_tricks.sliding_window_view(np.nan_to_num(volexp, nan=0.0), H)
    y = np.full(N, np.nan); y[:len(ve) - 1] = (ve[1:].max(1) >= 1.8).astype(float)
    out["변동성확장"] = y

    # 레짐추세 — 앞으로 12봉 순이동이 atr*sqrt(12) 이상
    k = 12
    y = np.full(N, np.nan)
    lim = N - k - 1
    t = np.arange(lim)
    y[t] = (np.abs(c[t + 1 + k] - c[t]) >= atr[t] * np.sqrt(k)).astype(float)
    out["레짐추세"] = y

    # 극점 — 앞으로 12봉 안에 ±6봉 국소극점 발생
    W = 6
    lowmin = pd.Series(lo).rolling(2 * W + 1, center=True).min().to_numpy()
    highmax = pd.Series(hi).rolling(2 * W + 1, center=True).max().to_numpy()
    isext = ((lo <= lowmin) | (hi >= highmax)).astype(float)
    ew = np.lib.stride_tricks.sliding_window_view(np.nan_to_num(isext), 12)
    y = np.full(N, np.nan); y[:len(ew) - 1] = (ew[1:].max(1) > 0).astype(float)
    out["극점"] = y

    # V자반등 — 앞으로 H봉 안에 1.0*atr 하락 후 1.5*atr 되돌림 (하락 이후 구간에서만 되돌림 인정)
    y = np.full(N, np.nan)
    fl = np.lib.stride_tricks.sliding_window_view(lo, H)
    fh = np.lib.stride_tricks.sliding_window_view(hi, H)
    lim = len(fl) - 1
    t = np.arange(lim)
    drop = fl[t + 1] <= (c[t] - atr[t])[:, None]
    first = np.where(drop.any(1), drop.argmax(1), H)
    run = np.maximum.accumulate(fh[t + 1][:, ::-1], axis=1)[:, ::-1]   # [j] = 이후 최고 고가
    j = np.clip(first, 0, H - 1)
    trough = np.minimum.accumulate(fl[t + 1], axis=1)[np.arange(lim), j]
    y[t] = (drop.any(1) & (run[np.arange(lim), j] >= trough + 1.5 * atr[t])).astype(float)
    out["V자반등"] = y
    return {k: v[:n] for k, v in out.items()}


def train_eval(Xw: np.ndarray, y: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    """같은 모델·같은 루프로 모든 타깃을 푼다. VAL AUC 로 조기종료, OOS 는 한 번만 본다."""
    tr = np.flatnonzero(masks["IS"])
    if len(tr) > MAX_TRAIN:                      # 시간축 균일 부분표집
        tr = tr[np.linspace(0, len(tr) - 1, MAX_TRAIN).astype(int)]
    va, oo = np.flatnonzero(masks["VAL"]), np.flatnonzero(masks["OOS"])
    pos = y[tr].mean()
    model = TCN()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6), dtype=torch.float32))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, EPOCHS * (len(tr) // BATCH + 1))

    def score(idx: np.ndarray) -> tuple[float, np.ndarray]:
        model.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, len(idx), 4096):
                xb = torch.from_numpy(norm_windows(Xw[idx[i:i + 4096]]))
                ps.append(torch.sigmoid(model(xb)).numpy())
        p = np.concatenate(ps)
        return (roc_auc_score(y[idx], p) if y[idx].std() else np.nan), p

    best, best_state, hist = -1.0, None, []
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            xb = torch.from_numpy(norm_windows(Xw[b]))
            opt.zero_grad()
            loss = lossf(model(xb), torch.from_numpy(y[b].astype(np.float32)))
            loss.backward()
            opt.step()
            sched.step()
        a, _ = score(va)
        hist.append(a)
        if a > best:
            best, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    val_auc, _ = score(va)
    oos_auc, _ = score(oo)
    is_auc, _ = score(tr[:20000])
    return dict(pos_rate=pos, n_train=len(tr), n_val=len(va), n_oos=len(oo),
                IS=is_auc, VAL=val_auc, OOS=oos_auc, best_epoch=int(np.argmax(hist)) + 1)


def baselines(Xtab: np.ndarray, names: list[str], y: np.ndarray, masks: dict) -> dict:
    """대조군 둘 — (1) 표 피쳐 GBM (2) 모델 없는 최고 단일피쳐. 딥러닝이 이걸 못 이기면 무의미."""
    tr, va, oo = (np.flatnonzero(masks[k]) for k in ("IS", "VAL", "OOS"))
    g = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, l2_regularization=1.0,
                                       random_state=0, early_stopping=True)
    g.fit(Xtab[tr], y[tr])
    gv = roc_auc_score(y[va], g.predict_proba(Xtab[va])[:, 1])
    go = roc_auc_score(y[oo], g.predict_proba(Xtab[oo])[:, 1])
    best, bname = 0.5, "-"
    for j, nm in enumerate(names):
        col = Xtab[va][:, j]
        if not np.isfinite(col).all() or np.std(col) == 0:
            continue
        a = roc_auc_score(y[va], col)
        a = max(a, 1 - a)
        if a > best:
            best, bname = a, nm
    return dict(gbm_VAL=gv, gbm_OOS=go, best_feat=bname, best_feat_VAL=best)


def main() -> int:
    t0 = time.time()
    eth = pd.read_csv(DATA / "eth_5m_1year.csv", parse_dates=["timestamp"])
    btc = pd.read_csv(DATA / "btc_5m_1year.csv", parse_dates=["timestamp"])
    eth = eth[eth.timestamp >= "2024-01-01"].reset_index(drop=True)
    sig = compute_signals(eth, btc_df=btc)

    base = build_channels(eth)
    # (i) 행이 창의 **끝** 봉이 되도록 맞춘다 — Xw[i] 는 봉 i 에서 끝나는 L봉 시퀀스
    view = np.lib.stride_tricks.sliding_window_view(base, L, axis=0)   # (N-L+1, CH, L)
    n = view.shape[0]
    off = L - 1                                                        # view[i] 의 끝 봉 = i+off
    ts = eth["timestamp"].to_numpy()[off:off + n]

    tabfeats = ["p_fast", "p_slow", "delta_z", "vol_z", "ret3_z", "atr_pct",
                "lower_wick_ratio", "upper_wick_ratio", "dem", "kalman_dev_z"]
    Xtab = np.column_stack([sig[tabfeats].to_numpy(float)[off:off + n]] +
                           [(sig[f"bottom_{s}"].fillna(False).to_numpy().astype(float)
                             - sig[f"top_{s}"].fillna(False).to_numpy().astype(float))[off:off + n]
                            for s, _ in SIGNAL_ORDER])
    tabnames = tabfeats + [f"fire_{s}" for s, _ in SIGNAL_ORDER]

    ys = labels(eth, sig, len(eth))
    ys = {k: v[off:off + n] for k, v in ys.items()}
    finite_tab = np.isfinite(Xtab).all(1)

    print(f"# 시퀀스 {n:,}개 · 길이 {L}봉(8h) · 채널 {CH} · 라벨창 {H}봉(2h)")
    print(f"# 모델 TCN(확장합성곱 4층, 파라미터 {sum(p.numel() for p in TCN().parameters()):,}) · CPU")
    print(f"# 학습 상한 {MAX_TRAIN:,} · 에폭 {EPOCHS} · VAL 로 조기종료, OOS 는 한 번만\n")
    hdr = (f"{'타깃':22s} {'양성률':>6s} {'학습n':>7s} {'검증n':>7s} "
           f"{'딥 IS':>6s} {'딥 VAL':>7s} {'딥 OOS':>7s} | {'GBM VAL':>8s} {'GBM OOS':>8s} | {'최고단일피쳐':>20s}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for name, y in ys.items():
        ok = np.isfinite(y) & finite_tab
        masks = {k: ok & (ts >= np.datetime64(a)) & (ts <= np.datetime64(b + "T23:59"))
                 for k, (a, b) in SPLITS.items()}
        if min(m.sum() for m in masks.values()) < 500:
            print(f"{name:22s} 표본 부족 — 건너뜀"); continue
        yy = np.nan_to_num(y)
        d = train_eval(view, yy, masks)
        b = baselines(Xtab, tabnames, yy, masks)
        rows.append(dict(target=name, **d, **b))
        print(f"{name:22s} {d['pos_rate']:6.3f} {d['n_train']:7,d} {d['n_val']:7,d} "
              f"{d['IS']:6.3f} {d['VAL']:7.3f} {d['OOS']:7.3f} | {b['gbm_VAL']:8.3f} {b['gbm_OOS']:8.3f} | "
              f"{b['best_feat'][:13]+' '+format(b['best_feat_VAL'],'.3f'):>20s}")
    df = pd.DataFrame(rows)
    out = ROOT / "tmp/evidence_entry_peak_exit_20260911"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "raw_seq_killgate.csv", index=False)

    print(f"\n=== 판정 ===")
    ctrl = df[~df.target.str.startswith("★")]
    gate = df[df.target.str.startswith("★")]
    print(f"양성대조(이벤트 트리거 4종) 딥 VAL AUC: {ctrl.VAL.min():.3f}~{ctrl.VAL.max():.3f}"
          f"  최고 {ctrl.loc[ctrl.VAL.idxmax(),'target']}")
    print(f"킬게이트(청산) 딥 VAL AUC: {gate.VAL.min():.3f}~{gate.VAL.max():.3f}"
          f"  OOS {gate.OOS.min():.3f}~{gate.OOS.max():.3f}")
    ok = ((gate.VAL >= 0.55) & (gate.OOS >= 0.55)).sum()
    print(f"킬게이트 통과(VAL·OOS 둘 다 >= 0.55): {ok}/{len(gate)}")
    print(f"\n소요 {time.time()-t0:.0f}s · 저장 {out/'raw_seq_killgate.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
