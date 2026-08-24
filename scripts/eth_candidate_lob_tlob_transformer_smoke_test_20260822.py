#!/usr/bin/env python3
"""ETH raw-L2/OFI LOB 전용 트랜스포머(OFI-TLOB-lite) Tier-1 스모크테스트 (2026-08-22).

사용자 지시("작은 Conv1d+LSTM 보다는 요즘 transformer가 대세... 트랜스포머 쪽으로 진행")로
`eth_candidate_lob_ofi_pipeline_smoke_test_20260822.py`(DeepLOB류 Conv1d+LSTM)를 대체하는
트랜스포머 버전. **아키텍처는 구현 전 아티팩트로 제시하고 사용자 컨펌을 받은 뒤 작성함**
([[feedback_dl_architecture_requires_user_confirmation]] 신규 원칙 최초 적용).

**2026-08-22 5차 수정**(사용자 요청 "트랜스포머도 이진으로 바꿔서 진행해보는건 어때? 3진은
너무 어려운거 같아"): DeepLOB와 공정 비교(둘 다 N=5 시드검증까지 마침, 근거는
[[eth_candidate_lob_ofi_pipeline_smoke_test_20260822]] "5차" 절)를 위해 컨펌된 3진(상승/보합/
하락) 타겟을 DeepLOB와 동일한 이진(단순 상승/하락, `mid[t+H] > mid[t]`) 타겟으로 교체. 출력
헤드를 3-class softmax에서 DeepLOB와 동일한 sigmoid+BCELoss로 변경 — 아키텍처(Dual Attention+
MLPLOB+AttentionDrop+Stochastic Depth)는 그대로, 타겟/헤드/손실함수만 DeepLOB와 맞춘 통제비교.

## 문헌 근거 (전부 실제 조사, arXiv 직접 확인)

- **TLOB** (Berti & Kasneci, 2025, arXiv:2502.15757) — 주 골격. Dual Attention(Temporal
  Self-Attention + Feature Self-Attention을 블록마다 병렬 적용) + Bilinear Normalization
  (입력층) + MLPLOB(표준 트랜스포머 FFN 대신 Feature-Mixing MLP + Temporal-Mixing MLP,
  GeLU+LayerNorm). FI-2010/NASDAQ뿐 아니라 **Bitcoin 데이터로도 검증됨**(SOTA 대비 F1 +3.7).
  원논문 자체 어블레이션: attention head 1/2/4 간 성능차 없어 head=1 채택, Adam(Lion 대비
  우세) lr=1e-4, 시퀀스길이 128(하이퍼파라미터탐색 결과), 블록 4개.
- **Bilinear Normalization** (Tran et al. 2020/2021, arXiv:2003.00598, arXiv:2109.00983) —
  LOB 4백만+ 이벤트로 검증된 정규화, 표준 z-score보다 비정상성(non-stationarity)에 강함.
  ⚠️ 원논문 수식 전체를 확보하지 못해(WebFetch로 메타데이터 수준만 확인), 아래 `BilinearNorm`은
  "시간축+피쳐축 양쪽을 정규화해 학습가능 게이트로 결합"이라는 핵심 아이디어를 따른 근사
  구현이다 — 원논문 수식의 1:1 재현이 아님을 명시한다.
- **AttentionDrop** (2025, arXiv:2504.12088) — 트랜스포머 어텐션 경로에 head-dropout보다
  세밀한 단위로 드랍을 거는 정규화. ⚠️ WebFetch로 메타데이터만 확인됐고 논문 고유의 신규
  메커니즘(어떤 세분화 단위인지) 전체는 확보 못함 — 대신 "softmax 이후 attention weight
  행렬에 드랍아웃을 건다"는 기술된 지점과 정확히 같은 곳에 이미 존재하는 PyTorch
  `nn.MultiheadAttention(dropout=...)` 내장 기능으로 구현한다(같은 지점을 타겟하는 것을
  근거로 채택, 논문의 정확한 신규 알고리즘 재현이라 주장하지 않음).
- **Stochastic Depth** (Huang et al. 2016) — 잔차블록을 학습 중 확률적으로 통째로 스킵
  (survival probability p_l, 깊이에 따라 선형감소, 테스트시엔 잔차분기를 p_l로 스케일).
  원논문은 54블록짜리 매우 깊은 ResNet에서 p_L=0.5(마지막블록)를 씀 — 이 프로젝트는 **블록
  2개뿐인 훨씬 얕은 모델**이라 그대로 가져오면 과함, p_L=0.8로 완화(아래 STOCH_DEPTH_MIN_SURVIVAL).
- **PatchTST(2023)/iTransformer(2024)** — 참고만: 패치토큰화(시퀀스50엔 과함, 생략),
  변수축 어텐션 역전(=TLOB의 Feature Self-Attention과 같은 개념, 이미 포함됨).
- ⚠️ **반증 근거, 정직하게 병기**: 이 축이 "설계도"로 인용해온 Bieganowski & Ślepaczuk(2026,
  arXiv:2602.00776)를 재확인하니 실제로는 **CatBoost(GBDT)를 쓰지 트랜스포머가 아니었다**.
  이 저장소 자체의 기존 발견(918-실험 벤치마크·DRW/G-Research Kaggle 크립토대회 전부 선형/GBDT
  우승, [[dl_crypto_trading_literature_survey_20260817]])과 같은 방향 — 트랜스포머가 문헌에서
  압도적으로 지지된다고 오해하면 안 된다. 사용자의 트랜스포머 전환 결정 자체를 반박하는 근거는
  아니고(TLOB류가 유효한 것도 사실), 균형있게 병기하는 것.

## 이 저장소용 각색 (원본 TLOB 대비 변경점, 컨펌받은 4가지 결정 반영)

| 항목 | TLOB 원본 | 이 구현 | 근거 |
|---|---|---|---|
| 입력표현 | 원시 10레벨×4(40차원) | 기존 검증된 OFI/imbalance 7피쳐 | Kolm 2023 "OFI>raw LOB", 파이프라인 재사용 |
| 시퀀스길이 | 128 | 50(유지) | 53시간 데이터 규모, 이전 스모크테스트와 비교가능성 |
| 블록 수 | 4 | **2**(컨펌됨) | 데이터 규모 대비 과다용량 방지(FastBiNLOB 스케일링법칙 논지) |
| 출력 | 3진(상승/보합/하락) | **이진**(5차 수정, DeepLOB와 통제비교 위해 재변경) | 최초 컨펌은 3진(TLOB 원본대로)이었으나, 사용자 요청으로 DeepLOB와 동일한 이진 타겟/BCELoss로 교체 — 아키텍처는 불변, 타겟만 통제 |
| 정규화 | Bilinear Norm | Bilinear Norm(근사) | 동일 |
| 신규 정규화 | (없음) | AttentionDrop + Stochastic Depth | 이번 요청으로 추가 |

Tier-1 스모크테스트 범위는 불변 — 데이터(WS-E 격리 파일럿, 53시간)가 그대로이므로 트랜스포머로
바꿔도 09-14(2단계) 전까지는 신호/성능 주장을 하지 않는다. 아래 출력에 반복 명시한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import eth_candidate_lob_ofi_pipeline_smoke_test_20260822 as deeplob  # noqa: E402 -- OFI 피쳐엔지니어링 재사용

DB_PATH = deeplob.DB_PATH
SEQ_LEN = deeplob.SEQ_LEN
HORIZON = deeplob.HORIZON
STRIDE = deeplob.STRIDE

N_BLOCKS = 2                    # 컨펌: TLOB원본 4 -> 2로 축소(데이터규모)
MAX_EPOCHS = 200
LR = 1e-4                       # TLOB 원논문 자체 어블레이션 값(Adam, Lion 대비 우세)
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-2
ATTN_DROPOUT = 0.1              # AttentionDrop(2025) 정신 -- MHA 내장 dropout(softmax 이후 위치)
MLP_DROPOUT = 0.1
EARLY_DROPOUT_EPOCHS = 20       # Liu et al. 2023 정신 유지(attn/mlp dropout에만 적용)
STOCH_DEPTH_MIN_SURVIVAL = 0.8  # Huang 2016 p_L, 원논문 0.5(54블록)보다 완화(우리는 2블록)
BATCH = 32
STRIP_LEN = 5
STRIP_PATIENCE = 4
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20260822
torch.manual_seed(SEED)


class BilinearNorm(nn.Module):
    """Tran et al. 2021 Bilinear Normalization 정신을 따른 근사 구현(원논문 수식 1:1 재현 아님,
    WebFetch로 확보한 건 "시간축+피쳐축 양쪽 정규화를 결합"이라는 핵심 아이디어까지). 학습가능
    게이트로 두 정규화를 섞고, 채널별 scale/shift를 둔다."""
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))   # sigmoid(0)=0.5 시작
        self.scale = nn.Parameter(torch.ones(n_feat))
        self.shift = nn.Parameter(torch.zeros(n_feat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, T, F)
        mu_t, sd_t = x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True) + 1e-6
        x_temporal = (x - mu_t) / sd_t
        mu_f, sd_f = x.mean(dim=2, keepdim=True), x.std(dim=2, keepdim=True) + 1e-6
        x_feature = (x - mu_f) / sd_f
        g = torch.sigmoid(self.gate)
        return (g * x_temporal + (1 - g) * x_feature) * self.scale + self.shift


class TLOBBlockBody(nn.Module):
    """Dual Attention(Temporal+Feature) + MLPLOB. Stochastic Depth 래퍼가 바깥에서 잔차를
    관리하므로 여기엔 블록 전체를 감싸는 잔차는 없다(내부 서브컴포넌트 잔차는 유지, TLOB
    원논문의 MLP-mixer식 구조 그대로)."""
    def __init__(self, d_model: int, seq_len: int, attn_dropout: float, mlp_dropout: float) -> None:
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads=1, dropout=attn_dropout, batch_first=True)
        self.feature_attn = nn.MultiheadAttention(seq_len, num_heads=1, dropout=attn_dropout, batch_first=True)
        self.combine = nn.Linear(d_model * 2, d_model)
        self.attn_norm = nn.LayerNorm(d_model)

        self.feat_mix = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(mlp_dropout), nn.Linear(d_model, d_model))
        self.feat_norm = nn.LayerNorm(d_model)
        self.temp_mix = nn.Sequential(
            nn.Linear(seq_len, seq_len), nn.GELU(), nn.Dropout(mlp_dropout), nn.Linear(seq_len, seq_len))
        self.temp_norm = nn.LayerNorm(seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, T, F)
        t_out, _ = self.temporal_attn(x, x, x)
        xt = x.transpose(1, 2)                             # (B, F, T)
        f_out_t, _ = self.feature_attn(xt, xt, xt)
        f_out = f_out_t.transpose(1, 2)                    # back to (B, T, F)
        combined = self.attn_norm(self.combine(torch.cat([t_out, f_out], dim=-1)))

        h = self.feat_norm(combined + self.feat_mix(combined))
        ht = h.transpose(1, 2)
        ht = self.temp_norm(ht + self.temp_mix(ht))
        return ht.transpose(1, 2)


class StochasticTLOBBlock(nn.Module):
    """Huang et al. 2016 Stochastic Depth -- 학습 중 베르누이(survival_prob)로 블록 전체를
    스킵(=항등함수)하거나 x + body(x)를 계산. 평가 시엔 x + survival_prob * body(x)로 결정적
    스케일링(원논문 test-time 규칙)."""
    def __init__(self, body: TLOBBlockBody, survival_prob: float) -> None:
        super().__init__()
        self.body = body
        self.survival_prob = survival_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if torch.rand(()).item() < self.survival_prob:
                return x + self.body(x)
            return x
        return x + self.survival_prob * self.body(x)


class OFITLOBLite(nn.Module):
    """TLOB(Berti & Kasneci 2025)를 이 저장소의 OFI 7피쳐+N_BLOCKS=2로 각색한 버전. 출력은
    DeepLOB(`eth_candidate_lob_ofi_pipeline_smoke_test_20260822.py`)와 동일한 이진
    sigmoid(스칼라) — 공정 비교를 위해 5차 수정에서 3-class softmax에서 교체."""
    def __init__(self, n_feat: int, seq_len: int, n_blocks: int, attn_dropout: float,
                 mlp_dropout: float, min_survival: float) -> None:
        super().__init__()
        self.norm = BilinearNorm(n_feat)
        survivals = [1.0 - (l / max(n_blocks - 1, 1)) * (1.0 - min_survival) for l in range(n_blocks)]
        self.blocks = nn.ModuleList([
            StochasticTLOBBlock(TLOBBlockBody(n_feat, seq_len, attn_dropout, mlp_dropout), p)
            for p in survivals
        ])
        self.reduce = nn.Sequential(nn.Linear(n_feat, max(n_feat // 2, 2)), nn.GELU())
        self.head = nn.Linear(seq_len * max(n_feat // 2, 2), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.reduce(x)                 # (B, T, F//2)
        return torch.sigmoid(self.head(x.flatten(1)).squeeze(-1))   # (B,) DeepLOB와 동일 형태


def _set_dropout_p(m: nn.Module, p: float) -> None:
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = p
        if isinstance(mod, nn.MultiheadAttention):
            mod.dropout = p


def main() -> None:
    print("=" * 78)
    print(f"⚠️  OFI-TLOB-lite Tier-1 스모크테스트 — 성능/알파 주장 아님, 배관 검증 전용 (seed={SEED})")
    print("=" * 78)

    df = deeplob._load_raw()
    print(f"raw snapshots: {len(df)}  [{df['recorded_at_utc'].min()} .. {df['recorded_at_utc'].max()}]")
    bids, asks = deeplob._parse_levels(df)

    import pandas as pd
    feats = pd.DataFrame({"recorded_at_utc": df["recorded_at_utc"]})
    feats["mid"] = df["mid"].to_numpy()
    feats["spread"] = (df["best_ask"] - df["best_bid"]).to_numpy()
    for lvl in deeplob.LEVELS_FOR_OFI:
        vals = np.full(len(df), np.nan)
        vals[1:] = deeplob._multilevel_ofi(bids, asks, lvl)[1:]
        feats[f"ofi_L{lvl}"] = vals
    feats["imbalance_1"] = df["imbalance_1"].to_numpy()
    feats["imbalance_5"] = df["imbalance_5"].to_numpy()
    feats["imbalance_10"] = df["imbalance_10"].to_numpy()

    feat_cols = ["spread", "ofi_L1", "ofi_L5", "ofi_L10", "imbalance_1", "imbalance_5", "imbalance_10"]
    X_raw = feats[feat_cols].to_numpy(dtype=np.float64)[1:]
    mid = feats["mid"].to_numpy()[1:]
    mu, sd = np.nanmean(X_raw, axis=0, keepdims=True), np.nanstd(X_raw, axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    X_std = np.nan_to_num((X_raw - mu) / sd)

    n_seq = (len(X_std) - SEQ_LEN - HORIZON) // STRIDE
    starts = np.arange(n_seq) * STRIDE
    Xs = np.stack([X_std[s:s + SEQ_LEN] for s in starts])
    fwd_ret = np.array([mid[s + SEQ_LEN + HORIZON - 1] / mid[s + SEQ_LEN - 1] - 1.0 for s in starts])
    print(f"\n시퀀스 텐서(stride={STRIDE}): X={Xs.shape}, fwd_ret 준비완료 -- 순수 배관검증용 타겟")

    purge = (SEQ_LEN + HORIZON) // STRIDE + 1
    n_train = int(n_seq * 0.6)
    n_val = int(n_seq * 0.2)
    tr_end = n_train
    va_start, va_end = n_train + purge, n_train + purge + n_val
    te_start = va_end + purge

    # 이진 타겟(5차 수정, DeepLOB와 동일 정의): 단순 상승/하락, 임계값 없음
    y = (fwd_ret > 0).astype(np.float32)
    print(f"이진 타겟(DeepLOB와 동일 정의, fwd_ret>0): 양성비율={y.mean():.3f}")

    Xtr, ytr = Xs[:tr_end], y[:tr_end]
    Xva, yva = Xs[va_start:va_end], y[va_start:va_end]
    Xte, yte = Xs[te_start:], y[te_start:]
    print(f"split(purge={purge}시퀀스): train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    model = OFITLOBLite(n_feat=Xs.shape[-1], seq_len=SEQ_LEN, n_blocks=N_BLOCKS,
                         attn_dropout=ATTN_DROPOUT, mlp_dropout=MLP_DROPOUT,
                         min_survival=STOCH_DEPTH_MIN_SURVIVAL)
    n_params = sum(p.numel() for p in model.parameters())
    survivals = [b.survival_prob for b in model.blocks]
    print(f"\n모델 파라미터 수: {n_params:,}  블록별 survival_prob(Stochastic Depth): {survivals}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=LR_MIN)
    bce = nn.BCELoss()   # 5차: CrossEntropyLoss -> DeepLOB와 동일한 BCELoss
    Xtr_t, ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    Xva_t, yva_t = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)
    Xte_t, yte_t = torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.float32)

    print(f"\ntrain={len(Xtr)} val={len(Xva)} test={len(Xte)} 시퀀스 -- AdamW lr={LR}(cosine, "
          f"T_max={MAX_EPOCHS}) + Prechelt UP_{STRIP_PATIENCE}(strip={STRIP_LEN})")
    print(f"early-dropout: epoch<{EARLY_DROPOUT_EPOCHS}까지만 attn/mlp dropout 활성, 이후 0")
    print("--- 매 epoch train+val 전체 커브 ---")

    n = len(Xtr_t)
    best_val, best_state, best_epoch = float("inf"), None, -1
    strip_best_history: list[float] = []
    curve: list[tuple[int, float, float]] = []
    stop_epoch = MAX_EPOCHS - 1
    dropout_disabled = False

    for epoch in range(MAX_EPOCHS):
        if epoch == EARLY_DROPOUT_EPOCHS and not dropout_disabled:
            _set_dropout_p(model, 0.0)
            dropout_disabled = True
            print(f"  -> early-dropout: epoch={epoch}부터 attn/mlp dropout 비활성화")

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
            val_p = model(Xva_t)
            val_loss = float(bce(val_p, yva_t).item())
            val_acc = float(((val_p > 0.5).float() == yva_t).float().mean().item())
        curve.append((epoch, train_loss, val_loss))
        print(f"  epoch={epoch:3d} lr={sched.get_last_lr()[0]:.2e} train_bce={train_loss:.4f} "
              f"val_bce={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if (epoch + 1) % STRIP_LEN == 0:
            strip_min = min(v for _, _, v in curve[-STRIP_LEN:])
            strip_best_history.append(strip_min)
            if len(strip_best_history) > STRIP_PATIENCE:
                recent = strip_best_history[-(STRIP_PATIENCE + 1):]
                if all(recent[i] >= recent[i - 1] for i in range(1, len(recent))):
                    stop_epoch = epoch
                    print(f"  -> Prechelt UP_{STRIP_PATIENCE}: epoch={epoch}에서 조기종료")
                    break

    assert best_state is not None
    model.load_state_dict(best_state)
    print(f"\nbest checkpoint: epoch={best_epoch} val_bce={best_val:.4f} (전체 {stop_epoch + 1}epoch 중)")

    model.eval()
    with torch.no_grad():
        test_p = model(Xte_t)
        test_bce = float(bce(test_p, yte_t).item())
        test_acc = float(((test_p > 0.5).float() == yte_t).float().mean().item())
    base_rate = float(yte.mean())
    intercept_bce = -(base_rate * np.log(max(base_rate, 1e-9)) + (1 - base_rate) * np.log(max(1 - base_rate, 1e-9)))
    majority_acc = max(base_rate, 1 - base_rate)
    print(f"\ntest_bce(held-out)={test_bce:.4f}  test_acc={test_acc:.3f}  "
          f"(참고: 절편전용 이론하한={intercept_bce:.4f}, 다수클래스 기준선={majority_acc:.3f}, "
          f"기준양성비율={base_rate:.3f})")
    print("⚠️ 위 수치는 53시간·단일 레짐 데이터 1회 분할 결과입니다 -- N=1 판정창, 성능/알파")
    print("   주장의 근거로 쓰지 않습니다. 트랜스포머로 바꿔도 이 사실은 변하지 않습니다 --")
    print("   Tier-2 게이트(2026-09-14) 전까지는 신호 유무를 판단하지 않습니다.")

    print(f"\n=== SMOKE TEST 판정 ===")
    print(f"  파이프라인 크래시 없이 완주: PASS")
    print(f"  best epoch(val 기준)={best_epoch}, train_bce={curve[best_epoch][1]:.4f}, "
          f"val_bce={curve[best_epoch][2]:.4f}")
    print(f"  Prechelt UP_{STRIP_PATIENCE} 조기종료: {'YES' if stop_epoch < MAX_EPOCHS - 1 else 'NO'}")
    print(f"  Stochastic Depth 블록별 survival_prob: {survivals}")
    print(f"  최종 판정: Tier-1 PASS (TLOB식 트랜스포머 배관 정상 확인, 성능수치는 여전히 비주장)")


if __name__ == "__main__":
    main()
