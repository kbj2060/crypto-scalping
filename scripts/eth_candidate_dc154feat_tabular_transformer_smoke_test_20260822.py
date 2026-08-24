#!/usr/bin/env python3
"""ETH DC 154피쳐 엔지니어링셋 + FT-Transformer식 tabular 트랜스포머 이진 방향 스모크테스트
(2026-08-22, 사용자 요청 "150여개 피쳐에 트랜스포머 적용, 이진으로, 방금 만든 아키텍처를 더
깊게").

## 이 실험이 필요한 배경 (닫힌 축 위에서 진행하는 이유)

[[eth_dc_engineered154_feature_set_20260820]]가 이 정확한 154피쳐셋(158캐노니컬 정리+RIT조합
30+금융ML문헌12)을 TabM으로 N=5 시드 재학습해 **chance 수준**(cond_acc 48.8~51.0%, 독립
재현배치도 48.4~49.9%)이라는 결론을 이미 냈다. 같은 세션 라벨퓨전 연구
([[eth_label_fusion_combined_model_research_20260821]])는 이 피쳐 기반 개별모델의 BCE가
절편전용 이론하한(0.6928)과 사실상 동일함도 확인했다 — 이건 "모델이 약해서"가 아니라
"추출 가능한 정보 자체가 없다"는 신호라 원칙적으로 어떤 아키텍처를 얹어도 하한 아래로는
못 내려간다. 사용자가 그럼에도 Optuna HP탐색+N시드검증까지 "승격 가능할 때까지" 계속
진행하라고 지시(2026-08-22, /loop) — 아래 구조는 그 지시를 안전하게 수행하기 위한 것이다.

## 아키텍처 — TLOB 블록의 tabular 각색 (FT-Transformer식)

라이브 LOB판(`eth_candidate_lob_tlob_transformer_smoke_test_20260822.py`)의 TLOB 블록은
Temporal Self-Attention(시간축)+Feature Self-Attention(피쳐축) 듀얼 구조였다. 154피쳐는 시간축이
없는 단일 bar 스냅샷이라 **Temporal Self-Attention을 제거**하고, FT-Transformer(Gorishniy et
al. 2021, arXiv:2106.11959)식으로 각 피쳐를 독립 토큰으로 임베딩(+CLS 토큰)한 뒤 그 154+1개
토큰 간 어텐션만 사용한다. 나머지(AttentionDrop, Stochastic Depth, MLPLOB식 dual-mixing)는
LOB판에서 그대로 재사용 — `StochasticTLOBBlock`은 아키텍처에 무관한 래퍼라 직접 import.

## 데이터/타겟/스플릿 — 리포지토리 유일 split 규약(2026-08-22 확정)

| 티어 | 규칙 | 구체값 | 용도 |
|---|---|---|---|
| TRAIN(fit) | 2024-01-01 ~ 최근분기 직전 분기말 | 2024-01-01 ~ 2026-03-31 | 학습. purge=48bar 후 VAL과 경계 |
| VAL(판정 티어) | 최근 완결 분기 | 2026-04-01 ~ 2026-06-30(Q2) | 조기종료+체크포인트+**보고 지표, Optuna 목적함수** |
| OOS(단일터치) | 다음 분기, 사용자 override로 조기실행 | 2026-07-01 ~ 데이터상 최근일 | 참고 지표. **기본 비활성 — `--eval-oos` 명시해야 계산** |

⚠️ **OOS는 기본적으로 계산하지 않는다(2026-08-22 2차 리팩터)**: Optuna로 수십~수백 trial을
돌리면 매 trial이 `main()`을 그대로 재실행할 경우 OOS를 그만큼 반복 조회하게 된다 — 이 저장소가
반복적으로 경고해온 "OOS 재사용" 문제를 대량으로 재현하는 것. 그래서 `train_and_eval()`은
`eval_oos=False`가 기본값이고, Optuna 목적함수는 VAL만 본다. OOS는 HP탐색이 끝난 뒤 **최종
확정 설정으로 N≥5 시드검증할 때 그 배치 전체에서 한 번**(시드마다 결정 자체는 다르지만, 이미
CONFIRMED된 이 저장소 관례상 "N=5 시드검증 배치"는 단일 조정된 조회로 취급) 계산한다.

## 성능 관련 — 2026-08-22 진단 및 수정

1차 버전은 CPU 전용(`torch` 기본 device)에 BATCH=256이었다 — 서버에 CUDA GPU가 실제로 있는데도
(`torch.cuda.is_available()==True`, 사용자 질문으로 확인) 안 쓰고 있었고, TRAIN이 LOB판(2,287
행)보다 100배+ 많은(~21만 행) 이 데이터에 LOB판 배치사이즈를 그대로 써서 에폭당 배치 수가
~1000개에 달해 단일 실행이 30분+ 걸렸다(사용자가 "cuda는 안 쓰고 있어?"로 지적, 로그가 안
보여 멈춘 줄 알았으나 `ps`로 482%CPU 확인해 실제로는 느리게 진행 중이었음도 함께 확인).
GPU로 옮기고 배치를 키우고, 데이터 로딩(피쳐엔지니어링 포함, 비용의 큰 부분)과 학습을
분리해 Optuna가 매 trial마다 데이터를 다시 안 만들게 리팩터했다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import eth_dc_engineered_features_canonicaldata_20260820 as feat154  # noqa: E402 -- 154피쳐 컬럼 로직 재사용

omega = feat154.omega

TRAIN_START, TRAIN_END = "2024-01-01", "2026-03-31"    # 앵커드walk-forward fit 구간
VAL_START, VAL_END = "2026-04-01", "2026-06-30"         # 앵커드walk-forward VAL(2026 Q2)
OOS_START = "2026-07-01"                                 # 단일터치 OOS 시작 -- 끝은 데이터상 최근일(override)
SCRATCH = ROOT / "tmp/dc154_ilias_split_20260822"
FORWARD_BARS = 48        # 4h, 이 154피쳐 축(스태킹/라벨퓨전 스크립트)이 이미 쓰는 관례

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_HP = {
    "d_token": 16,
    "n_blocks": 4,
    "lr": 1e-4,
    "lr_min": 1e-6,
    "weight_decay": 1e-2,
    "attn_dropout": 0.1,
    "mlp_dropout": 0.1,
    "min_survival": 0.8,
    "batch": 4096,             # 262k+ 행 규모에 맞춰 확대(구 256 -> 에폭당 배치 ~1000개였음)
    "max_epochs": 30,           # SDPA수정 후에도 에폭당 수십초 -- Optuna 다회실행 감안해 상한 축소(구 80)
    "early_dropout_epochs": 15,
    "strip_len": 5,
    "strip_patience": 4,
}


class FeatureTokenizer(nn.Module):
    """FT-Transformer(Gorishniy et al. 2021) 피쳐 토크나이저 -- 각 수치 피쳐를 독립 아핀변환으로
    d_token 임베딩, CLS 토큰 prepend."""
    def __init__(self, n_feat: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_feat, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_feat, d_token))
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, n_feat)
        tok = x.unsqueeze(-1) * self.weight + self.bias    # (B, n_feat, d_token)
        cls = self.cls.expand(x.shape[0], -1, -1)
        return torch.cat([cls, tok], dim=1)                 # (B, n_feat+1, d_token)


class TabularBlockBody(nn.Module):
    """TLOBBlockBody의 tabular 각색 -- Temporal Self-Attention 삭제(시간축 없음), 토큰(=피쳐)간
    어텐션 하나만 유지 + MLPLOB식 dual-mixing(채널믹싱+토큰믹싱)은 그대로.

    ⚠️ 2026-08-22 속도수정: `nn.MultiheadAttention`을 직접 QKV projection +
    `F.scaled_dot_product_attention`(SDPA, PyTorch 2.0+ fused 커널)으로 교체했다 --
    마이크로벤치마크로 확인: raw SDPA는 58회(=1에폭 배치수) 호출에 0.7초인데, 원래 코드의
    실측 에폭당 소요는 100초+였다(100배+ 괴리, 순수 연산량으로는 설명 불가). `nn.
    MultiheadAttention`은 `dropout>0`일 때 fused 경로를 못 타고 훨씬 느린 레퍼런스 구현으로
    빠지는 게 알려진 문제라 이게 원인으로 특정됨(먼저 시도한 `.item()` 동기화 제거는 효과
    없었음 -- 그 가설은 틀렸다). num_heads=1(원 TLOB 어블레이션 근거 그대로)이라 head
    reshape 불필요."""
    def __init__(self, d_token: int, n_tokens: int, attn_dropout: float, mlp_dropout: float) -> None:
        super().__init__()
        self.qkv = nn.Linear(d_token, d_token * 3)
        self.attn_dropout = attn_dropout   # _set_dropout_p가 갱신하는 대상(SDPA는 함수인자라 모듈속성으로 들고 있어야 함)
        self.proj = nn.Linear(d_token, d_token)
        self.attn_norm = nn.LayerNorm(d_token)
        self.chan_mix = nn.Sequential(
            nn.Linear(d_token, d_token), nn.GELU(), nn.Dropout(mlp_dropout), nn.Linear(d_token, d_token))
        self.chan_norm = nn.LayerNorm(d_token)
        self.tok_mix = nn.Sequential(
            nn.Linear(n_tokens, n_tokens), nn.GELU(), nn.Dropout(mlp_dropout), nn.Linear(n_tokens, n_tokens))
        self.tok_norm = nn.LayerNorm(n_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, n_tokens, d_token)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        dropout_p = self.attn_dropout if self.training else 0.0
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        combined = self.attn_norm(self.proj(a))
        h = self.chan_norm(combined + self.chan_mix(combined))
        ht = h.transpose(1, 2)
        ht = self.tok_norm(ht + self.tok_mix(ht))
        return ht.transpose(1, 2)


class FastStochasticBlock(nn.Module):
    """`tlob.StochasticTLOBBlock`과 기대값은 동일(Huang 2016 Stochastic Depth)하지만 매
    블록·매 배치 `torch.rand(()).item()` 동기화를 없앤 버전. ⚠️ 2026-08-22: 이 수정 자체는
    속도개선 효과가 없었음이 실측으로 확인됐다(수정 전 508초/3에폭 -> 수정 후 537초/3에폭,
    개선 없음) -- 진짜 원인은 `nn.MultiheadAttention`의 fused 커널 미사용이었다(아래
    `TabularBlockBody` docstring 참고). 그래도 `torch.where`로 device 위에서 마스킹하는 게
    구조적으로 더 나은 패턴이라 되돌리지 않고 유지한다."""
    def __init__(self, body: nn.Module, survival_prob: float) -> None:
        super().__init__()
        self.body = body
        self.survival_prob = survival_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            keep = (torch.rand((), device=x.device) < self.survival_prob)
            return torch.where(keep, x + self.body(x), x)
        return x + self.survival_prob * self.body(x)


class DC154TabularTransformer(nn.Module):
    """154피쳐 -> FeatureTokenizer -> n_blocks x StochasticTLOBBlock(TabularBlockBody) -> CLS
    풀링 -> sigmoid. DeepLOB/TLOB-LOB와 동일한 이진 출력 형태."""
    def __init__(self, n_feat: int, d_token: int, n_blocks: int, attn_dropout: float,
                 mlp_dropout: float, min_survival: float) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_feat, d_token)
        n_tokens = n_feat + 1
        survivals = [1.0 - (l / max(n_blocks - 1, 1)) * (1.0 - min_survival) for l in range(n_blocks)]
        self.blocks = nn.ModuleList([
            FastStochasticBlock(TabularBlockBody(d_token, n_tokens, attn_dropout, mlp_dropout), p)
            for p in survivals
        ])
        self.head = nn.Linear(d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, n_feat) standardized
        h = self.tokenizer(x)
        for block in self.blocks:
            h = block(h)
        return torch.sigmoid(self.head(h[:, 0, :]).squeeze(-1))   # CLS 토큰 풀링


def _set_dropout_p(m: nn.Module, p: float) -> None:
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = p
        if isinstance(mod, TabularBlockBody):
            mod.attn_dropout = p


def load_data() -> dict:
    """비용이 큰 부분(CSV 로딩+154피쳐 엔지니어링+표준화) -- Optuna 등에서 한 번만 호출하고
    재사용할 것. 반환값은 전부 numpy 배열(디바이스 이동은 train_and_eval에서)."""
    print("=" * 78, flush=True)
    print("⚠️  DC154피쳐 tabular 트랜스포머 -- 데이터 로딩(이 부분이 비용의 큰 몫)", flush=True)
    print(f"device={DEVICE}", flush=True)
    print("=" * 78, flush=True)

    SCRATCH.mkdir(parents=True, exist_ok=True)

    raw = pd.concat(
        [pd.read_csv(ROOT / f"data/splits/year_oos/training_features_{y}.csv", low_memory=False)
         for y in ("2024", "2025", "2026_rebuilt")],
        ignore_index=True,
    )
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    if raw["timestamp"].duplicated().any():
        raise RuntimeError("2024+2025+2026 concat에 중복 timestamp 존재")

    regime3_dir = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
    regime3_concat = pd.concat(
        [pd.read_csv(regime3_dir / f"training_features_{y}_regime3_current_sensitive_hmm_wide24.csv")
         for y in ("2024", "2025", "2026_rebuilt")],
        ignore_index=True,
    )
    regime3_path = SCRATCH / "regime3_current_2024_2025_2026_concat.csv"
    regime3_concat.to_csv(regime3_path, index=False)
    raw, _ = omega._overlay_required(raw, regime3_path, omega.REGIME3_CURRENT_COLS, tag="regime3_current_2024_2025_2026")

    print("피쳐엔지니어링(조합30+금융ML12) 시작...", flush=True)
    full = feat154._attach_engineered_columns(raw)
    feat_cols = feat154.FINAL_FEATURE_LIST
    missing = [c for c in feat_cols if c not in full.columns]
    if missing:
        raise RuntimeError(f"154피쳐 중 프레임에 없는 컬럼: {missing}")
    for col in ("timestamp", "close"):
        if col not in full.columns:
            raise RuntimeError(f"{col} 컬럼 없음")
    print(f"full={full.shape}  feat_cols={len(feat_cols)}  "
          f"range=[{full['timestamp'].min()}..{full['timestamp'].max()}]", flush=True)

    full["fwd_ret"] = full["close"].shift(-FORWARD_BARS) / full["close"] - 1.0
    full["y_up"] = (full["fwd_ret"] > 0).astype(np.float32)
    full = full.dropna(subset=["fwd_ret"]).reset_index(drop=True)

    ts = full["timestamp"]
    train_mask = (ts >= TRAIN_START) & (ts <= TRAIN_END)
    val_mask = (ts >= VAL_START) & (ts <= VAL_END)
    oos_mask = ts >= OOS_START
    train_idx = np.flatnonzero(train_mask.to_numpy())
    val_idx = np.flatnonzero(val_mask.to_numpy())
    oos_idx = np.flatnonzero(oos_mask.to_numpy())
    purge = FORWARD_BARS
    train_idx = train_idx[:-purge] if len(train_idx) > purge else train_idx[:0]
    val_idx = val_idx[:-purge] if len(val_idx) > purge else val_idx[:0]
    oos_end_ts = full.loc[oos_idx, "timestamp"].max() if len(oos_idx) else None
    print(f"TRAIN({TRAIN_START}~{TRAIN_END}, purge={purge}bar)={len(train_idx)}  "
          f"VAL({VAL_START}~{VAL_END}, purge={purge}bar)={len(val_idx)}  "
          f"OOS({OOS_START}~{oos_end_ts}, 부분분기 override, 기본 미평가)={len(oos_idx)}", flush=True)

    mu = full.loc[train_idx, feat_cols].to_numpy(dtype=np.float64)
    mu, sd = np.nanmean(mu, axis=0, keepdims=True), np.nanstd(mu, axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0

    def _std(idx: np.ndarray) -> np.ndarray:
        raw_x = full.loc[idx, feat_cols].to_numpy(dtype=np.float64)
        return np.nan_to_num((raw_x - mu) / sd).astype(np.float32)

    Xtr = _std(train_idx)
    ytr = full.loc[train_idx, "y_up"].to_numpy(dtype=np.float32)
    Xva = _std(val_idx)
    yva = full.loc[val_idx, "y_up"].to_numpy(dtype=np.float32)
    Xoos = _std(oos_idx)
    yoos = full.loc[oos_idx, "y_up"].to_numpy(dtype=np.float32)
    print(f"양성비율: train={ytr.mean():.3f} val={yva.mean():.3f} oos={yoos.mean():.3f}", flush=True)

    return {
        "Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva, "Xoos": Xoos, "yoos": yoos,
        "feat_cols": feat_cols, "n_feat": len(feat_cols),
        "oos_window": f"{OOS_START}~{oos_end_ts}",
    }


def train_and_eval(data: dict, hp: dict, seed: int, *, eval_oos: bool = False, verbose: bool = True) -> dict:
    """`hp`는 DEFAULT_HP를 base로 부분 override(Optuna trial이 채울 부분만 넘기면 됨).
    `eval_oos=False`(기본)면 OOS는 아예 계산하지 않는다 -- Optuna 반복 조회로부터 OOS 보호."""
    cfg = {**DEFAULT_HP, **hp}
    torch.manual_seed(seed)

    Xtr_t = torch.tensor(data["Xtr"], device=DEVICE)
    ytr_t = torch.tensor(data["ytr"], device=DEVICE)
    Xva_t = torch.tensor(data["Xva"], device=DEVICE)
    yva_t = torch.tensor(data["yva"], device=DEVICE)

    model = DC154TabularTransformer(
        n_feat=data["n_feat"], d_token=cfg["d_token"], n_blocks=cfg["n_blocks"],
        attn_dropout=cfg["attn_dropout"], mlp_dropout=cfg["mlp_dropout"],
        min_survival=cfg["min_survival"],
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    survivals = [b.survival_prob for b in model.blocks]

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"], eta_min=cfg["lr_min"])
    bce = nn.BCELoss()

    if verbose:
        print(f"\n모델 파라미터 수: {n_params:,}  블록별 survival_prob: {survivals}  device={DEVICE}", flush=True)
        print(f"train={len(Xtr_t)} val={len(Xva_t)} 행 -- AdamW lr={cfg['lr']}(cosine, "
              f"T_max={cfg['max_epochs']}) batch={cfg['batch']} + "
              f"Prechelt UP_{cfg['strip_patience']}(strip={cfg['strip_len']})", flush=True)

    n = len(Xtr_t)
    best_val, best_state, best_epoch = float("inf"), None, -1
    strip_best_history: list[float] = []
    curve: list[tuple[int, float, float]] = []
    stop_epoch = cfg["max_epochs"] - 1
    dropout_disabled = False

    for epoch in range(cfg["max_epochs"]):
        if epoch == cfg["early_dropout_epochs"] and not dropout_disabled:
            _set_dropout_p(model, 0.0)
            dropout_disabled = True

        model.train()
        perm = torch.randperm(n, device=DEVICE)
        train_loss_sum = torch.zeros((), device=DEVICE)   # 배치마다 .item() 동기화(WSL2 GPU
                                                             # 패스스루에서 특히 느림) 대신 device
                                                             # 위에서 누적, 에폭당 1회만 동기화
        for i in range(0, n, cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            p = model(Xtr_t[idx])
            loss = bce(p, ytr_t[idx])
            loss.backward()
            opt.step()
            train_loss_sum += loss.detach() * len(idx)
        train_loss = float(train_loss_sum.item()) / n
        sched.step()

        model.eval()
        with torch.no_grad():
            val_p = model(Xva_t)
            val_loss = float(bce(val_p, yva_t).item())
            val_acc = float(((val_p > 0.5).float() == yva_t).float().mean().item())
        curve.append((epoch, train_loss, val_loss))
        if verbose:
            print(f"  epoch={epoch:3d} lr={sched.get_last_lr()[0]:.2e} train_bce={train_loss:.4f} "
                  f"val_bce={val_loss:.4f} val_acc={val_acc:.3f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if (epoch + 1) % cfg["strip_len"] == 0:
            strip_min = min(v for _, _, v in curve[-cfg["strip_len"]:])
            strip_best_history.append(strip_min)
            if len(strip_best_history) > cfg["strip_patience"]:
                recent = strip_best_history[-(cfg["strip_patience"] + 1):]
                if all(recent[i] >= recent[i - 1] for i in range(1, len(recent))):
                    stop_epoch = epoch
                    if verbose:
                        print(f"  -> Prechelt UP_{cfg['strip_patience']}: epoch={epoch}에서 조기종료", flush=True)
                    break

    assert best_state is not None
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_p_final = model(Xva_t)
        val_bce_final = float(bce(val_p_final, yva_t).item())
        val_acc_final = float(((val_p_final > 0.5).float() == yva_t).float().mean().item())
    base_rate = float(data["yva"].mean())
    intercept_bce = -(base_rate * np.log(max(base_rate, 1e-9)) + (1 - base_rate) * np.log(max(1 - base_rate, 1e-9)))

    result = {
        "val_bce": val_bce_final, "val_acc": val_acc_final,
        "val_intercept_bce": intercept_bce, "val_base_rate": base_rate,
        "best_epoch": best_epoch, "stop_epoch": stop_epoch, "n_params": n_params,
        "hp": cfg, "seed": seed,
    }
    if verbose:
        print(f"\nval_bce(2026 Q2, best checkpoint)={val_bce_final:.4f}  val_acc={val_acc_final:.3f}  "
              f"(참고: 절편전용 이론하한={intercept_bce:.4f}, 기준양성비율={base_rate:.3f})", flush=True)

    if eval_oos:
        Xoos_t = torch.tensor(data["Xoos"], device=DEVICE)
        yoos_t = torch.tensor(data["yoos"], device=DEVICE)
        with torch.no_grad():
            oos_p = model(Xoos_t)
            oos_bce = float(bce(oos_p, yoos_t).item())
            oos_acc = float(((oos_p > 0.5).float() == yoos_t).float().mean().item())
        oos_base_rate = float(data["yoos"].mean())
        oos_intercept_bce = -(oos_base_rate * np.log(max(oos_base_rate, 1e-9))
                               + (1 - oos_base_rate) * np.log(max(1 - oos_base_rate, 1e-9)))
        result.update({
            "oos_bce": oos_bce, "oos_acc": oos_acc,
            "oos_intercept_bce": oos_intercept_bce, "oos_base_rate": oos_base_rate,
        })
        if verbose:
            print(f"\n⚠️ OOS(단일터치, {data['oos_window']})={oos_bce:.4f}  oos_acc={oos_acc:.3f}  "
                  f"(참고: 절편전용 이론하한={oos_intercept_bce:.4f}, 기준양성비율={oos_base_rate:.3f})",
                  flush=True)

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int, nargs="?", default=20260822)
    ap.add_argument("--eval-oos", action="store_true", help="단일터치 OOS도 평가(기본 비활성)")
    ap.add_argument("--n-blocks", type=int, default=None)
    ap.add_argument("--d-token", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--attn-dropout", type=float, default=None)
    ap.add_argument("--mlp-dropout", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()

    hp_override = {}
    for cli_name, hp_key in [("n_blocks", "n_blocks"), ("d_token", "d_token"), ("lr", "lr"),
                              ("weight_decay", "weight_decay"), ("attn_dropout", "attn_dropout"),
                              ("mlp_dropout", "mlp_dropout"), ("batch", "batch")]:
        v = getattr(args, cli_name)
        if v is not None:
            hp_override[hp_key] = v

    data = load_data()
    result = train_and_eval(data, hp_override, args.seed, eval_oos=args.eval_oos, verbose=True)

    print(f"\n=== SMOKE TEST 판정 ===")
    print(f"  파이프라인 크래시 없이 완주: PASS")
    print(f"  best epoch(val 기준)={result['best_epoch']}, val_bce={result['val_bce']:.4f}")
    print(f"  Prechelt 조기종료: {'YES' if result['stop_epoch'] < result['hp']['max_epochs'] - 1 else 'NO'}")
    print(f"  최종 판정: PASS")


if __name__ == "__main__":
    main()
