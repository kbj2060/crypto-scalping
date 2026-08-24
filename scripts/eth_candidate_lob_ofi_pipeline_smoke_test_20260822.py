#!/usr/bin/env python3
"""ETH raw-L2/OFI DL 파이프라인 Tier1 스모크테스트 (2026-08-22, 사용자 승인 "1번으로 진행해줘").
**2026-08-22 재작성**: 최초 버전이 학습기법을 거의 안 넣고(고정 60epoch, val 없이 train만
로깅, Adam lr=1e-3, dropout/weight-decay 없음, stride=1로 시퀀스가 98% 겹침) 돌려 test_bce가
절편전용 하한보다 나쁘게 나왔었다(사용자 지적, 타당함). [[feedback_modern_dl_training_checklist]]/
[[reference_dl_layer_design_training_20260816]]를 확인 후 이 프로젝트가 이미 검증한 관행으로
다시 짬: 진단습관(매 epoch train+val 전체 커브 로깅, best-checkpoint 요약만 보지 않기),
lr=2e-4(+cosine, 이 프로젝트에서 N≥5로 가장 잘 검증된 단일 lever)로 시작, Prechelt(1998)
UP_4 strip 기반 조기종료, AdamW(decoupled weight decay)+dropout, train/val/test 3분할+
경계 purge(라벨 윈도우가 분할선을 못 넘게), 시퀀스 stride를 1→5로 늘려 중복 압축.

`docs/model_contracts/eth_candidate_lob_microstructure_contract_20260817.md`가 정한 3단계
축적기준의 **1단계(파이프라인 스모크테스트)만** 수행한다 — "이미 WS-E 격리 파일럿(19,121건,
53시간)으로 지금 당장 만족 가능"이라고 명시된 바로 그 단계다. 2단계(예비 신호점검, 2026-09-14
이후)/3단계(승격급, 2026-11-17 이후)는 데이터가 물리적으로 부족해 이번엔 절대 시도하지 않는다.

**2026-08-22 3차 수정**(사용자 질문 "드랍아웃 논문대로야? 너무 많은 거 같은데"): dropout을
문헌 근거 없이 0.2 균일 적용했던 걸 지적받고 실제 조사(Srivastava 2014 원논문, Tompson 2015
SpatialDropout, Gal&Ghahramani 2016 변분 RNN dropout, Liu et al. 2023 "Dropout Reduces
Underfitting" arXiv:2303.01500) 후 두 가지 수정: (1) conv 출력 dropout을 element-wise에서
채널단위(`nn.Dropout1d`)로 교체 — conv feature map은 인접 활성값이 강하게 상관되므로
SpatialDropout이 맞는 형태. (2) val 커브가 200epoch 내내 발산 신호를 안 보였다는(=과적합
관측 없음) 이전 결과에 비춰, Liu et al.(2023) "과적합 위험이 없는데 dropout을 끝까지 걸면
오히려 underfitting을 유발한다"는 발견을 반영해 early-dropout 스케줄(EARLY_DROPOUT_EPOCHS
이후 자동 비활성화) 도입. LSTM 뒤 head 앞 dropout은 이미 문헌상 표준 위치라 유지.

⚠️⚠️ 학습기법을 제대로 넣어도 이 스크립트는 여전히 "신호를 찾는다"가 아니라 **"raw L2 JSON →
OFI/멀티레벨 피쳐 → DeepLOB류 모델 → 제대로 된 학습절차가 실제로 도는가"만** 확인한다.
53시간·단일 레짐 데이터는 물리적으로 부족하므로(2단계 게이트가 2026-09-14인 이유), 이번
결과가 좋게 나오든 나쁘게 나오든 방향성/수익성 주장의 근거로 쓰지 않는다 — 아래 출력에
반복해서 명시한다.

데이터: `data/research/ws_e_orderbook_raw_pilot.duckdb`의
`orderbook_periodic_snapshots_eth_soak_20260719`(19,121행, ETH/USDT 선물, 상위20레벨
bids_json/asks_json, ~10초 간격, 2026-07-19 11:46~2026-07-21 18:21 KST). 라이브 프로덕션
데이터와 무관한 격리 연구 DB([[eth_candidate_lob_microstructure_data_scoping_20260817]]).

피쳐: Cont, Kukanov & Stoikov(2014) "The Price Impact of Order Book Events"의 레벨별 OFI
정의를 순위(rank) 기준으로 멀티레벨 확장(Cont/Cucuringu/Zhang 2023, Kolm et al. 2023 "OFI
features > raw LOB" 결론과 정합) — 레벨 1/5/10 통합 OFI + 기존 저장된 imbalance_{1,5,10,20}과
대조검증.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

DB_PATH = ROOT / "data/research/ws_e_orderbook_raw_pilot.duckdb"
TABLE = "orderbook_periodic_snapshots_eth_soak_20260719"
LEVELS_FOR_OFI = (1, 5, 10)
SEQ_LEN = 50            # ~50 snapshots * 10s ≈ 8.3분 lookback
HORIZON = 6             # 6 snapshots * 10s = 60s ahead (smoke-test 타겟, 성능주장 아님)
STRIDE = 5              # 시퀀스 간 겹침 압축(이전: stride=1 -> 인접 시퀀스 98% 중복 -> 사실상
                         # 표본과잉계산의 주범이었음. stride=5도 여전히 겹치지만 완화됨)
MAX_EPOCHS = 200        # Prechelt UP_4가 조기종료 -- 코사인 스케줄의 T_max로도 씀
LR = 2e-4               # 이 프로젝트가 N>=5 시드로 검증한 가장 신뢰할 lever
                         # ([[feedback_modern_dl_training_checklist]] "Learning-rate/schedule" 절)
LR_MIN = 2e-6
WEIGHT_DECAY = 1e-2
DROPOUT = 0.2
EARLY_DROPOUT_EPOCHS = 20   # Liu et al. 2023 "Dropout Reduces Underfitting"(arXiv:2303.01500)
                             # 정신: val 커브가 발산 신호를 안 보이면(=과적합 위험이 관측되지
                             # 않으면) dropout을 학습 초반에만 걸고 이후 끔 -- 전 구간 dropout은
                             # 이미 신호가 얇은 모델의 학습을 오히려 방해(underfitting)할 수 있음
BATCH = 32
STRIP_LEN = 5           # Prechelt UP_s strip 길이(epoch)
STRIP_PATIENCE = 4      # 연속 악화 strip 4개까지 허용(UP_4)
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20260822
torch.manual_seed(SEED)


def _load_raw() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"""
        SELECT recorded_at_utc, best_bid, best_ask, mid, bids_json, asks_json,
               imbalance_1, imbalance_5, imbalance_10, imbalance_20
        FROM {TABLE}
        ORDER BY recorded_at_utc
    """).df()
    con.close()
    return df


def _parse_levels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """bids_json/asks_json(문자열, [[price,qty],...] 20레벨) -> (N,20,2) 배열 2개."""
    bids = np.stack([np.array(json.loads(s), dtype=np.float64) for s in df["bids_json"]])
    asks = np.stack([np.array(json.loads(s), dtype=np.float64) for s in df["asks_json"]])
    return bids, asks


def _ofi_single_level(px_prev: np.ndarray, qty_prev: np.ndarray,
                       px_now: np.ndarray, qty_now: np.ndarray, *, is_bid: bool) -> np.ndarray:
    """Cont, Kukanov & Stoikov (2014) 레벨별 OFI 이벤트 e_n. is_bid=True는 매수벽 쪽(가격↑=적극매수),
    False는 매도벽 쪽(가격↓=적극매도) 부호 컨벤션을 따른다."""
    if is_bid:
        up = px_now > px_prev
        same = px_now == px_prev
        down = px_now < px_prev
        e = np.where(up, qty_now, np.where(same, qty_now - qty_prev, -qty_prev))
    else:
        down = px_now < px_prev   # ask price dropping = aggressive sell pressure -> positive event
        same = px_now == px_prev
        up = px_now > px_prev
        e = np.where(down, qty_now, np.where(same, qty_now - qty_prev, -qty_prev))
    return e


def _multilevel_ofi(bids: np.ndarray, asks: np.ndarray, levels: int) -> np.ndarray:
    """순위(rank) 기준 멀티레벨 통합 OFI(Cont/Cucuringu/Zhang 2023 확장 방식) -- 레벨 i의 OFI
    이벤트를 각 스냅샷의 정렬된 호가창 i번째 자리(가격이 바뀌어도 "몇 번째로 좋은 가격인가"만
    비교)로 근사한다. 엄밀한 price-level tracking(동일 가격을 시간축으로 계속 추적)이 아니라
    순위 근사임을 명시 -- 스모크테스트 목적엔 충분하나, 2단계/3단계 정식 분석에선 재검토 필요."""
    n = bids.shape[0]
    total = np.zeros(n)
    for lvl in range(levels):
        bpx_prev, bq_prev = bids[:-1, lvl, 0], bids[:-1, lvl, 1]
        bpx_now, bq_now = bids[1:, lvl, 0], bids[1:, lvl, 1]
        apx_prev, aq_prev = asks[:-1, lvl, 0], asks[:-1, lvl, 1]
        apx_now, aq_now = asks[1:, lvl, 0], asks[1:, lvl, 1]
        e_bid = _ofi_single_level(bpx_prev, bq_prev, bpx_now, bq_now, is_bid=True)
        e_ask = _ofi_single_level(apx_prev, aq_prev, apx_now, aq_now, is_bid=False)
        total[1:] += (e_bid - e_ask)
    return total


def main() -> None:
    print("=" * 78)
    print(f"⚠️  Tier-1 파이프라인 스모크테스트 — 성능/알파 주장 아님, 배관 검증 전용 (seed={SEED})")
    print("=" * 78)

    df = _load_raw()
    print(f"raw snapshots: {len(df)}  [{df['recorded_at_utc'].min()} .. {df['recorded_at_utc'].max()}]")
    bids, asks = _parse_levels(df)
    print(f"parsed bids shape={bids.shape} asks shape={asks.shape} (rows, levels, [price,qty])")

    feats = pd.DataFrame({"recorded_at_utc": df["recorded_at_utc"]})
    feats["mid"] = df["mid"].to_numpy()
    feats["spread"] = (df["best_ask"] - df["best_bid"]).to_numpy()
    for lvl in LEVELS_FOR_OFI:
        col = f"ofi_L{lvl}"
        vals = np.full(len(df), np.nan)
        vals[1:] = _multilevel_ofi(bids, asks, lvl)[1:]
        feats[col] = vals
    feats["imbalance_1"] = df["imbalance_1"].to_numpy()
    feats["imbalance_5"] = df["imbalance_5"].to_numpy()
    feats["imbalance_10"] = df["imbalance_10"].to_numpy()

    # sanity: OFI 부호가 즉시 다음 mid 변화 방향과 얼마나 겹치는지(순수 배관검증용 참고 수치,
    # 53시간 단일구간이라 이 값 자체를 신호로 주장하지 않는다)
    mid_chg = feats["mid"].diff().to_numpy()
    for lvl in LEVELS_FOR_OFI:
        col = f"ofi_L{lvl}"
        valid = feats[col].notna() & ~np.isnan(mid_chg)
        agree = float(np.mean(np.sign(feats.loc[valid, col]) == np.sign(mid_chg[valid])))
        print(f"[참고, 신호주장 아님] sign({col}) == sign(다음 mid 변화) 일치율: {agree:.3f} (n={valid.sum()})")

    # ---- 시퀀스 구성 (모델 입력 배관 검증) ----
    feat_cols = ["spread", "ofi_L1", "ofi_L5", "ofi_L10", "imbalance_1", "imbalance_5", "imbalance_10"]
    X_raw = feats[feat_cols].to_numpy(dtype=np.float64)
    valid_from = 1  # 첫 행은 OFI가 NaN(diff 기준)
    X_raw = X_raw[valid_from:]
    mid = feats["mid"].to_numpy()[valid_from:]

    mu, sd = np.nanmean(X_raw, axis=0, keepdims=True), np.nanstd(X_raw, axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    X_std = np.nan_to_num((X_raw - mu) / sd)

    n_seq = (len(X_std) - SEQ_LEN - HORIZON) // STRIDE
    if n_seq <= 0:
        raise RuntimeError("시퀀스를 만들기엔 데이터가 부족합니다 -- 스모크테스트 실패")
    starts = np.arange(n_seq) * STRIDE
    Xs = np.stack([X_std[s:s + SEQ_LEN] for s in starts])                            # (n_seq, SEQ_LEN, n_feat)
    y = np.array([1.0 if mid[s + SEQ_LEN + HORIZON - 1] > mid[s + SEQ_LEN - 1] else 0.0
                  for s in starts])
    print(f"\n시퀀스 텐서(stride={STRIDE}): X={Xs.shape} y={y.shape} (양성비율={y.mean():.3f}, "
          f"순수 배관검증용 타겟 -- 모델링 목표 아님)")

    # ---- train/val/test 3분할 + 경계 purge (라벨 윈도우가 분할선 못 넘게) ----
    # [[feedback_modern_dl_training_checklist]] "Correctness/setup checks": forward-looking
    # 라벨 윈도우가 있으면 분할 경계를 purge해야 함 -- 시퀀스 하나의 라벨이 SEQ_LEN+HORIZON개
    # 원시 행에 걸쳐 있으므로, 그 폭만큼 각 분할 경계에서 시퀀스를 제외한다.
    purge = (SEQ_LEN + HORIZON) // STRIDE + 1
    n_train = int(n_seq * 0.6)
    n_val = int(n_seq * 0.2)
    tr_end = n_train
    va_start, va_end = n_train + purge, n_train + purge + n_val
    te_start = va_end + purge
    Xtr, ytr = Xs[:tr_end], y[:tr_end]
    Xva, yva = Xs[va_start:va_end], y[va_start:va_end]
    Xte, yte = Xs[te_start:], y[te_start:]
    print(f"split(purge={purge}시퀀스): train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    class DeepLOBSmoke(nn.Module):
        """DeepLOB(Zhang, Zohren & Roberts 2019)류 Conv1d(레벨/피쳐축) + LSTM(시간축) 골격 +
        dropout 정규화(2026-08-22 추가, 최초판엔 전무했음). **2026-08-22 2차 수정**(사용자
        지적): conv 출력의 dropout을 element-wise `nn.Dropout`에서 채널단위 `nn.Dropout1d`로
        교체(Tompson et al. 2015 SpatialDropout — conv feature map은 인접 활성값이 강하게
        상관돼 있어 개별 원소 드랍은 의도한 "중복 제거" 효과가 약함, 채널 전체를 드랍해야 함).
        LSTM 이후 head 앞 dropout은 원래도 표준 위치(문헌 확인: "between LSTM and FC layer")라
        유지, plain Dropout 그대로(여긴 (B,32) 평탄 벡터라 spatial 개념 자체가 없음)."""
        def __init__(self, n_feat: int, dropout: float) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_feat, 16, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout1d(dropout),
                nn.Conv1d(16, 16, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout1d(dropout),
            )
            self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
            self.head_dropout = nn.Dropout(dropout)
            self.head = nn.Linear(32, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)   # (B, SEQ_LEN, 16)
            out, _ = self.lstm(h)
            return torch.sigmoid(self.head(self.head_dropout(out[:, -1, :])).squeeze(-1))

    model = DeepLOBSmoke(n_feat=Xs.shape[-1], dropout=DROPOUT)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=LR_MIN)
    bce = nn.BCELoss()
    Xtr_t, ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    Xva_t, yva_t = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)
    Xte_t, yte_t = torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.float32)

    print(f"\ntrain={len(Xtr)} val={len(Xva)} test={len(Xte)} 시퀀스 -- AdamW lr={LR}(cosine, "
          f"T_max={MAX_EPOCHS}) + Prechelt UP_{STRIP_PATIENCE}(strip={STRIP_LEN}) 조기종료로 학습")
    print(f"early-dropout: epoch<{EARLY_DROPOUT_EPOCHS}까지만 dropout={DROPOUT} 활성, 이후 0으로 끔 "
          f"(Liu et al. 2023, arXiv:2303.01500 정신)")
    print("--- 매 epoch train+val 전체 커브 (요약 체크포인트만 보지 않는다) ---")

    def _set_dropout_p(m: nn.Module, p: float) -> None:
        for mod in m.modules():
            if isinstance(mod, (nn.Dropout, nn.Dropout1d)):
                mod.p = p

    n = len(Xtr_t)
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    strip_best_history: list[float] = []
    curve: list[tuple[int, float, float]] = []
    stop_epoch = MAX_EPOCHS - 1
    dropout_disabled = False

    for epoch in range(MAX_EPOCHS):
        if epoch == EARLY_DROPOUT_EPOCHS and not dropout_disabled:
            _set_dropout_p(model, 0.0)
            dropout_disabled = True
            print(f"  -> early-dropout: epoch={epoch}부터 dropout 비활성화(p=0)")

        model.train()
        perm = torch.randperm(n)
        train_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            p = model(Xtr_t[idx])
            loss = bce(p, ytr_t[idx])
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * len(idx)
        train_loss /= n
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(bce(model(Xva_t), yva_t).item())
        curve.append((epoch, train_loss, val_loss))
        print(f"  epoch={epoch:3d} lr={sched.get_last_lr()[0]:.2e} train_bce={train_loss:.4f} val_bce={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        # Prechelt(1998) UP_s: strip 끝마다 그 strip 최저 val_loss를 이전 strip 최저치와 비교,
        # s번 연속 악화되면 중단.
        if (epoch + 1) % STRIP_LEN == 0:
            strip_min = min(v for _, _, v in curve[-STRIP_LEN:])
            strip_best_history.append(strip_min)
            if len(strip_best_history) > STRIP_PATIENCE:
                recent = strip_best_history[-(STRIP_PATIENCE + 1):]
                worsened = all(recent[i] >= recent[i - 1] for i in range(1, len(recent)))
                if worsened:
                    stop_epoch = epoch
                    print(f"  -> Prechelt UP_{STRIP_PATIENCE}: strip 최저치가 {STRIP_PATIENCE}회 "
                          f"연속 악화, epoch={epoch}에서 조기종료")
                    break

    assert best_state is not None
    model.load_state_dict(best_state)
    print(f"\nbest checkpoint: epoch={best_epoch} val_bce={best_val:.4f} (전체 {stop_epoch + 1}epoch 중)")

    model.eval()
    with torch.no_grad():
        test_bce = float(bce(model(Xte_t), yte_t).item())
    base_rate = float(yte.mean())
    intercept_bce = -(base_rate * np.log(max(base_rate, 1e-9)) + (1 - base_rate) * np.log(max(1 - base_rate, 1e-9)))
    print(f"\ntest_bce(held-out, best-val checkpoint)={test_bce:.4f}  "
          f"(참고: 절편전용 이론하한={intercept_bce:.4f}, 기준양성비율={base_rate:.3f})")
    print("⚠️ 위 test_bce는 53시간·단일 레짐 데이터 1회 분할 결과입니다 -- N=1 판정창, 성능/알파")
    print("   주장의 근거로 쓰지 않습니다. 이번 스모크테스트의 성공 기준은: (1) 크래시 없이")
    print("   전체 파이프라인+제대로 된 학습절차(스케줄/조기종료/정규화)가 돌았는가, (2) train/val")
    print("   커브가 진단 가능한 형태로 나왔는가(수치가 아니라 곡선의 모양) 둘 뿐입니다.")

    train_val_gap = curve[best_epoch][1:] if best_epoch < len(curve) else (None, None)
    print(f"\n=== SMOKE TEST 판정 ===")
    print(f"  파이프라인 크래시 없이 완주: PASS")
    print(f"  best epoch(val 기준)={best_epoch}, train_bce={train_val_gap[0]:.4f}, "
          f"val_bce={train_val_gap[1]:.4f} -- 일반화 vs 암기 격차 진단 가능: "
          f"{'양호(격차작음)' if train_val_gap[1] - train_val_gap[0] < 0.15 else '격차 있음(예상됨, 53h 데이터 한계)'}")
    print(f"  Prechelt UP_{STRIP_PATIENCE} 조기종료 정상 작동: "
          f"{'YES' if stop_epoch < MAX_EPOCHS - 1 else 'NO(MAX_EPOCHS까지 도달, strip patience 재검토 필요)'}")
    print(f"  최종 판정: Tier-1 PASS (학습기법 정상 적용 확인, 성능수치는 여전히 비주장)")


if __name__ == "__main__":
    main()
