"""zig075 quality 게이트를 TabPFN 메타라벨 필터로 대체할 수 있는가 — Stage 0 싼 falsification.

배경
----
Omega4.6.1의 direction_head 축은 닫혔다(h48qual 0/5 시드가 always_short에 패배, zig075는 N=5로
always_short과 통계적 구분 불가). 대체 모델 6종(TabM HP서치 0/40+, GBDT 0/48, 오토인코더 0/18,
TCN 0/75, CNN, one-vs-rest)도 전부 기각됐다. 그러나 TabPFN 은 그 목록에 없다 — 아직 안 해본 축이다.

이 저장소의 반복 교훈은 **"재분류는 실패하고 필터링만 작동한다"** 이므로, TabPFN 을
direction_head 대체(재분류)가 아니라 **quality 게이트 자리(필터)** 에 넣는다. zig075 의 현행
quality_head 는 `quality_mode=same_as_direction` — direction 과 **같은 라벨로 학습된 별도 헤드**다
(같은 출력이 아님: 실측 corr(dir_confidence, quality_for_action)=0.959, 평균절대차 0.024, 임계
0.75 통과율 8.78% vs 8.76%). 즉 실질적으로 **방향 확신도 필터**이고, 이는 h48qual 에서 규명된
근본 원인("게이트 ≈ direction_head confidence 필터")과 같은 구조다. 이 스크립트는 그 필터를
TabPFN 메타라벨 분류기로 갈아끼웠을 때 판별력이 실제로 올라가는지를 **재학습 없이**(부모 TabM 은
동결, 저장된 예측만 사용) 측정한다.

Stage 0 이 통과해야만 Stage 1(greedy_replay bar-by-bar 경제성)로 넘어간다.

사전등록 (실행 전 확정)
----------------------
모집단   : 부모 holdout 창의 후보 봉 = `dir_action != 0` (quality 게이트 이전, 메타라벨링의
           표준 primary-side 정의). train_predictions 는 쓰지 않는다 — 부모 학습 구간의 예측을
           side 로 쓰면 부모의 과적합 확신이 메타라벨로 새기 때문
           (scripts/diagnose_eth_h48qual_dirhead_metalabel_via_event_label_engine_20260815.py 의
           동일 판단을 따름).
             W1 = validation 2025-10-01..12-31  (후보 19,174)
             W2 = oos        2026-01-01..02-28  (후보 12,029)
헤드라인 : W1 학습 → W2 평가 (인과 전진). 역방향(W2→W1)은 참고로만 보고.
라벨     : 라이브 배리어 컨벤션 삼중배리어 이진 라벨.
           진입 = close[i] (봉 i 확정 후 결정 → 봉 i 종가 진입),
           배리어 탐색 = 봉 i+1 부터 (사건 라벨 경계 계약: 피쳐 창 끝 i < 탐색 시작 i+1),
           TP/SL = omega4_6_1_live.py::_ComponentConfig 라이브 ATR 공식
                   (atr_window=192, tp_mult=12, sl_mult=6, min_tp=.075, min_sl=.040,
                    max_tp=.22, max_sl=.12) — atr_pct 는 봉 i 까지만 사용,
           배리어 판정 = intrabar 고가/저가 (h48qual/zig075 의 문서화된 라이브 컨벤션),
           호라이즌 상한 H=2016봉(7일) 초과시 비용차감 부호로 결정.
           라벨 1 = 비용 후 이익.
피쳐     : 부모 base_cols 102개에서 **원시 수준(level) 13개 제외한 89개** 를 공통 기반으로 쓴다
           (open/high/low/close/volume/quote_volume/trades/taker_buy_base/taker_buy_quote/
            sum_open_interest_value/close_btc/volume_btc/quote_volume_btc 제외 — 학습창과 평가창의
            가격대가 달라 원시 수준은 분포 밖 값이 된다, 저장소의 알려진 오염 함정). 두 arm:
             A) headline  = 89 + 부모 헤드 출력 12개 + side = 102개.
                메타라벨링의 표준 구성이자 **incumbent 정보의 진부분집합이 아닌 상위집합** —
                quality_for_action 이 피쳐로 들어가므로 TabPFN 은 최소한 현행 게이트를 재현할 수
                있다. 못 이기면 "추가 정보 없음"이 곧바로 결론이 된다.
             B) ablation  = 89 (부모 출력 없음). 부모가 이미 인코딩하지 않은 무언가를 TabPFN 이
                따로 찾아내는지 분리해서 본다.
모델     : TabPFNClassifier, N=5 진짜 무작위 시드. 컨텍스트 상한 18,000행(저장소 관례) —
           초과시 균등솎기(2026-09-02 발견: 균등솎기 >= 무작위). 실제 사용 행수를 반드시 기록한다.
대조군   : (a) incumbent `quality_for_action` (현행 게이트 점수 그대로),
           (b) always_long / always_short 기저율,
           (c) 무작위 필터(커버리지 매칭, B=200),
           (d) 라벨 셔플 귀무(B=32, TabPFN 재학습),
           (e) **측면 매칭 귀무**(선택 집합의 롱/숏 구성을 보존한 채 각 측면 풀에서 무작위
               추출, 부트스트랩 2000회). 두 창 다 하락장이라 SHORT 기저율이 LONG 을 크게
               웃돌아(W2: 58.6% vs 25.0%), pooled 기저율을 귀무로 쓰면 "숏을 더 고르는"
               필터가 실력 없이도 이긴다.
킬 기준  : 아래 셋 중 하나라도 걸리면 이 축을 종결한다.
           K1. W2 에서 TabPFN AUC 평균이 incumbent AUC 를 초과하지 못함.
           K2. W2 TabPFN AUC 가 셔플 귀무 분포의 95분위를 넘지 못함.
           K3. incumbent 커버리지에 매칭한 정밀도가 무작위 필터 분포의 95분위를 넘지 못함.
           K4. 측면 매칭 귀무 대비 초과 정밀도의 부트스트랩 95% CI 가 5시드 중 4개 미만에서만
               0 을 배제 — 즉 "숏을 고른 것 말고 한 일이 없다".

준수
----
fresh_forward_bar_by_bar : 해당 없음(Stage 0 은 판별력 진단, 경제성 주장 아님 — Stage 1 이 담당)
trade_ledgers_used_as_input   : false
saved_parent_exit_timestamps_used : false
future_rows_used_for_entry    : false
라이브 파일(trading_bot.py, omega4_6_1_live.py, runtime_config.py, .env) 미변경.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

ZIG075_DIR = ROOT / (
    "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620"
    "_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629"
)
BUNDLE = ZIG075_DIR / "true_3head_tabm_bundle.pt"

FRAMES = {
    "W1": {
        "base": ROOT / "data/splits/year_oos/training_features_2025.csv",
        "overlay": ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
                        / "training_features_2025_regime3_current_sensitive_hmm_wide24.csv",
        "pred": ZIG075_DIR / "validation_predictions_q075.csv",
        "label": "validation 2025-10-01..12-31",
    },
    "W2": {
        "base": ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
        "overlay": ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
                        / "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv",
        "pred": ZIG075_DIR / "oos_predictions_q075.csv",
        "label": "oos 2026-01-01..02-28",
    },
}

# omega4_6_1_live.py::_ComponentConfig 기본값 (자산 공통, 모든 components_override 호출부가 그대로 둠)
ATR_WINDOW, TP_MULT, SL_MULT = 192, 12.0, 6.0
MIN_TP, MIN_SL, MAX_TP, MAX_SL = 0.075, 0.040, 0.22, 0.12
QUALITY_THRESHOLD = 0.75
HORIZON = 2016  # 7일 (5분봉)
CTX_CAP = 18000  # TabPFN 컨텍스트 상한, 저장소 관례
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594]  # 진짜 무작위 추출
N_SHUFFLE = 32
N_RANDOM = 200
RNG_MASTER = 20260909
SKIP_DONE = True  # 저장된 scores_*.npz 가 있는 arm 은 재실행하지 않는다(부분 재개)

# 학습창/평가창의 가격대가 달라 분포 밖이 되는 원시 수준 컬럼
LEVEL_COLS = [
    "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value",
    "close_btc", "volume_btc", "quote_volume_btc",
]

# 결정 시점에 실제로 가용한 부모 헤드 출력 (메타라벨링의 primary-model 신뢰도 피쳐)
PARENT_COLS = [
    "router_confidence", "router_margin",
    "dir_p_cash", "dir_p_long", "dir_p_short", "dir_confidence",
    "dir_side_edge", "dir_trade_prob",
    "quality_p_cash", "quality_p_long", "quality_p_short", "quality_for_action",
]

OUT = ROOT / "tmp/eth_zig075_quality_metalabel_tabpfn_20260909"


def load_predictions(path: Path) -> pd.DataFrame:
    """validation 은 컬럼명에 `_oof_`, oos 는 없다 — 두 네이밍을 모두 흡수."""
    header = list(pd.read_csv(path, nrows=0).columns)
    wanted = ["dir_action", "final_action", *PARENT_COLS]
    pick = {k: next(c for c in header if c.endswith(k)) for k in wanted}
    df = pd.read_csv(path, usecols=["timestamp", *pick.values()])
    df = df.rename(columns={v: k for k, v in pick.items()})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def load_frame(spec: dict, base_cols: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(spec["base"], low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = (frame.dropna(subset=["timestamp"]).sort_values("timestamp")
                  .drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    overlay = pd.read_csv(spec["overlay"], low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    missing = [c for c in base_cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{spec['base'].name}: base_cols 누락 {len(missing)}개 -> {missing[:10]}")
    return frame


def build_meta_labels(frame: pd.DataFrame, preds: pd.DataFrame, fee: float, slip: float) -> pd.DataFrame:
    """후보 봉마다 라이브 배리어 컨벤션으로 격리 트레이드를 해소해 이진 메타라벨을 만든다.

    진입은 봉 i 종가, 배리어 탐색은 봉 i+1 부터(사건 라벨 경계 계약). 배리어 판정은 intrabar
    고가/저가 — h48qual/zig075 의 라이브 컨벤션(`omega4_6_1_live.py::evaluate_exit` 의
    bar_high_move/bar_low_move).

    ATR 과 배리어 탐색은 **예측 창이 아니라 전체 연도 프레임** 위에서 계산한다. 예측 창으로
    먼저 자르면 (a) 창 시작 192봉의 ATR 이 잘린 윈도우로 계산되고 (b) 창 끝 7일치 후보의
    lookahead 가 잘려 인위적으로 `time` 해소가 된다 — 둘 다 창 경계 인공물이다.
    """
    merged = frame.merge(preds, on="timestamp", how="left", validate="one_to_one")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged["side_feat"] = np.where(merged["dir_action"] == 1, 1.0,
                                   np.where(merged["dir_action"] == 2, -1.0, 0.0))

    atr_pct = atr_eval._atr_pct(merged, ATR_WINDOW)
    tp_arr = np.clip(np.maximum(MIN_TP, atr_pct * TP_MULT), 0.0, MAX_TP)
    sl_arr = np.clip(np.maximum(MIN_SL, atr_pct * SL_MULT), 0.0, MAX_SL)

    high = pd.to_numeric(merged["high"], errors="raise").to_numpy(np.float64)
    low = pd.to_numeric(merged["low"], errors="raise").to_numpy(np.float64)
    close = pd.to_numeric(merged["close"], errors="raise").to_numpy(np.float64)
    dir_action = merged["dir_action"].to_numpy()  # 예측 창 밖은 NaN (left join)
    n = len(merged)

    cand_idx = np.flatnonzero(np.isfinite(dir_action) & (dir_action != 0))
    cand_idx = cand_idx[cand_idx < n - 2]  # 최소 한 봉의 탐색 여지 확보

    sides = np.where(dir_action[cand_idx] == 1, 1, -1).astype(np.int8)
    cost = 2.0 * (fee + slip)

    labels = np.empty(len(cand_idx), dtype=np.int8)
    net_move = np.empty(len(cand_idx), dtype=np.float64)
    reasons = np.empty(len(cand_idx), dtype=object)
    hold = np.empty(len(cand_idx), dtype=np.int32)
    truncated = np.zeros(len(cand_idx), dtype=bool)

    for k, i in enumerate(cand_idx):
        side = sides[k]
        entry = close[i] * (1 + slip) if side > 0 else close[i] * (1 - slip)
        j0, j1 = i + 1, min(i + 1 + HORIZON, n)
        truncated[k] = (j1 - j0) < HORIZON  # 프레임 끝이라 7일 lookahead 를 다 못 본 후보
        h, l = high[j0:j1], low[j0:j1]
        if side > 0:
            up = (h - entry) / entry
            dn = (l - entry) / entry
        else:
            up = (entry - l) / entry
            dn = (entry - h) / entry
        tp_hit = np.flatnonzero(up >= tp_arr[i])
        sl_hit = np.flatnonzero(dn <= -sl_arr[i])
        t_tp = tp_hit[0] if tp_hit.size else np.iinfo(np.int32).max
        t_sl = sl_hit[0] if sl_hit.size else np.iinfo(np.int32).max
        if t_tp == t_sl == np.iinfo(np.int32).max:
            # 시간 종료 — 마지막 봉 종가로 비용차감 부호 판정
            last = close[j1 - 1]
            move = (last - entry) / entry if side > 0 else (entry - last) / entry
            labels[k], net_move[k], reasons[k], hold[k] = int(move - cost > 0), move - cost, "time", j1 - 1 - i
        elif t_tp <= t_sl:
            # 같은 봉에 둘 다 닿으면 TP 우선 — 라이브 evaluate_exit 의 SL->TP 평가 순서와 반대이므로
            # 낙관 편향이 되지 않도록 동시 터치는 SL 로 처리한다(보수적 채택).
            if t_tp == t_sl:
                labels[k], net_move[k], reasons[k], hold[k] = 0, -sl_arr[i] - cost, "both->sl", int(t_sl) + 1
            else:
                labels[k], net_move[k], reasons[k], hold[k] = 1, tp_arr[i] - cost, "tp", int(t_tp) + 1
        else:
            labels[k], net_move[k], reasons[k], hold[k] = 0, -sl_arr[i] - cost, "sl", int(t_sl) + 1

    out = pd.DataFrame({
        "row": cand_idx,
        "timestamp": merged["timestamp"].to_numpy()[cand_idx],
        "side": sides,
        "quality_for_action": merged["quality_for_action"].to_numpy()[cand_idx],
        "tp": tp_arr[cand_idx],
        "sl": sl_arr[cand_idx],
        "label": labels,
        "net_move": net_move,
        "reason": reasons,
        "hold_bars": hold,
        "horizon_truncated": truncated,
    })
    return merged, out


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney U 기반 AUC (동점은 평균 순위로 처리)."""
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def stride_subsample(n: int, cap: int) -> np.ndarray:
    """균등솎기 — 2026-09-02 발견에서 무작위와 최소 동급으로 확인됨."""
    if n <= cap:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, cap).astype(np.int64))


def fit_tabpfn(Xtr, ytr, Xte, seed: int, device: str) -> np.ndarray:
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(device=device, random_state=seed)
    clf.fit(Xtr, ytr.astype(int))
    return clf.predict_proba(Xte)[:, 1]


def precision_at_coverage(score: np.ndarray, y: np.ndarray, coverage: float) -> tuple[float, int]:
    k = max(1, int(round(coverage * len(score))))
    top = np.argsort(-score, kind="mergesort")[:k]
    return float(y[top].mean()), k


def side_matched_lift(sel: np.ndarray, y: np.ndarray, side: np.ndarray,
                      rng: np.random.Generator, n_boot: int = 2000) -> dict:
    """측면 매칭 귀무 대비 초과 정밀도.

    이 두 창은 둘 다 하락장이라 SHORT 기저율이 LONG 보다 훨씬 높다(W2: 58.6% vs 25.0%).
    따라서 "숏을 더 많이 고르는 필터"는 아무 선별 실력이 없어도 pooled 기저율을 이긴다 —
    저장소의 반복 함정(측면 비대칭은 같은 측면 귀무와 함께 봐야 한다). 여기서는 선택된
    집합의 **측면 구성을 그대로 보존한 채** 각 측면 풀에서 무작위로 뽑았을 때의 기대
    정밀도를 귀무로 삼고, 그 차이를 부트스트랩 CI 와 함께 낸다. CI 가 0 을 포함하면
    "이 필터가 한 일은 숏을 고른 것뿐"이다.
    """
    k = int(sel.sum())
    is_s = side < 0
    n_s = int((sel & is_s).sum())
    n_l = k - n_s
    pool_s, pool_l = y[is_s], y[~is_s]
    observed = float(y[sel].mean())
    null_mean = float((n_s * pool_s.mean() + n_l * pool_l.mean()) / k) if k else float("nan")

    draws = np.empty(n_boot)
    for b in range(n_boot):
        acc = 0.0
        if n_s:
            acc += rng.choice(pool_s, size=n_s, replace=False).sum()
        if n_l:
            acc += rng.choice(pool_l, size=n_l, replace=False).sum()
        draws[b] = acc / k
    lo, hi = np.quantile(observed - draws, [0.025, 0.975])
    return {"k": k, "short_share": float(n_s / k) if k else float("nan"),
            "precision": observed, "side_matched_null": null_mean,
            "lift_pp": (observed - null_mean) * 100.0,
            "boot_ci_pp": [float(lo * 100.0), float(hi * 100.0)],
            "excludes_zero": bool(lo > 0.0)}


def main() -> int:
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fee, slip = omega._load_fee_slip()
    base_cols = torch.load(BUNDLE, map_location="cpu", weights_only=False)["base_cols"]
    feat_cols = [c for c in base_cols if c not in LEVEL_COLS]
    headline_cols = feat_cols + PARENT_COLS + ["side_feat"]
    FEATURESETS = (("headline", headline_cols), ("ablation_no_parent", feat_cols))

    print(f"[설정] device={device} fee={fee} slip={slip} 왕복비용={2*(fee+slip)*1e4:.1f}bp", flush=True)
    print(f"[설정] headline {len(headline_cols)}개 = base {len(feat_cols)} (원시수준 "
          f"{len(base_cols)-len(feat_cols)}개 제외) + 부모출력 {len(PARENT_COLS)} + side | "
          f"ablation {len(feat_cols)}개", flush=True)

    data = {}
    for tag, spec in FRAMES.items():
        frame = load_frame(spec, base_cols)
        preds = load_predictions(spec["pred"])
        merged, cand = build_meta_labels(frame, preds, fee, slip)
        base_rate = cand["label"].mean()
        long_rate = cand.loc[cand.side > 0, "label"].mean()
        short_rate = cand.loc[cand.side < 0, "label"].mean()
        print(f"\n[{tag}] {spec['label']}  후보={len(cand)}  기저 승률={base_rate*100:.2f}%", flush=True)
        print(f"      always_long={long_rate*100:.2f}% (n={(cand.side>0).sum()})  "
              f"always_short={short_rate*100:.2f}% (n={(cand.side<0).sum()})", flush=True)
        trunc = int(cand["horizon_truncated"].sum())
        print(f"      해소 사유: {cand['reason'].value_counts().to_dict()}  "
              f"중앙 보유={int(cand['hold_bars'].median())}봉", flush=True)
        print(f"      lookahead 절단 후보={trunc} ({trunc/len(cand)*100:.1f}%) "
              f"[프레임 끝 경계, 라벨은 TP/SL 선터치면 여전히 유효]", flush=True)
        data[tag] = {"merged": merged, "cand": cand,
                     "base_rate": float(base_rate),
                     "always_long": float(long_rate), "always_short": float(short_rate),
                     "horizon_truncated": trunc}
        # 라벨 빌드 직후 즉시 기록한다 -- 뒤따르는 긴 TabPFN 루프가 죽어도(2026-09-09 실제로
        # 역방향 셔플 24/32 지점에서 죽었다) 라벨 산출물까지 같이 날아가지 않도록.
        cand.to_csv(OUT / f"candidates_{tag}.csv", index=False)

    results = {"config": {
        "seeds": SEEDS, "ctx_cap": CTX_CAP, "horizon_bars": HORIZON,
        "quality_threshold": QUALITY_THRESHOLD, "device": device,
        "fee": fee, "slip": slip, "roundtrip_cost_bp": 2 * (fee + slip) * 1e4,
        "headline_features": len(headline_cols), "ablation_features": len(feat_cols),
        "parent_output_cols": PARENT_COLS, "excluded_level_cols": LEVEL_COLS,
        "barrier": {"atr_window": ATR_WINDOW, "tp_mult": TP_MULT, "sl_mult": SL_MULT,
                    "min_tp": MIN_TP, "min_sl": MIN_SL, "max_tp": MAX_TP, "max_sl": MAX_SL,
                    "resolution": "intrabar_high_low", "tie": "both->sl (보수적)"},
        "n_shuffle": N_SHUFFLE, "n_random": N_RANDOM,
    }, "population": {t: {k: v for k, v in d.items() if k not in ("merged", "cand")}
                      | {"n_candidates": int(len(d["cand"]))} for t, d in data.items()},
       "arms": {}}

    for direction, (tr_tag, te_tag) in (("W1->W2", ("W1", "W2")), ("W2->W1", ("W2", "W1"))):
        tr, te = data[tr_tag], data[te_tag]
        ytr = tr["cand"]["label"].to_numpy()
        yte = te["cand"]["label"].to_numpy()

        side_te = te["cand"]["side"].to_numpy()
        boot_rng = np.random.default_rng(RNG_MASTER + 7)

        inc = te["cand"]["quality_for_action"].to_numpy()
        inc_auc = auc(inc, yte)
        inc_sel = inc >= QUALITY_THRESHOLD
        inc_cov = float(inc_sel.mean())
        inc_prec = float(yte[inc_sel].mean()) if inc_sel.any() else float("nan")
        inc_lift = side_matched_lift(inc_sel, yte, side_te, boot_rng)

        print(f"\n{'='*72}\n[{direction}] 학습 {tr_tag} (n={len(ytr)}) -> 평가 {te_tag} (n={len(yte)})", flush=True)
        print(f"  incumbent quality_for_action: AUC={inc_auc:.4f}  "
              f"커버리지={inc_cov*100:.1f}%  정밀도={inc_prec*100:.2f}%  (pooled 기저 {te['base_rate']*100:.2f}%)", flush=True)
        print(f"    측면매칭: 숏비중={inc_lift['short_share']*100:.1f}%  귀무={inc_lift['side_matched_null']*100:.2f}%  "
              f"초과={inc_lift['lift_pp']:+.2f}pp CI[{inc_lift['boot_ci_pp'][0]:+.2f},{inc_lift['boot_ci_pp'][1]:+.2f}] "
              f"{'✅' if inc_lift['excludes_zero'] else '❌'}", flush=True)

        arm = {"train_window": tr_tag, "test_window": te_tag,
               "n_train_candidates": int(len(ytr)), "n_test_candidates": int(len(yte)),
               "test_base_rate": te["base_rate"],
               "always_long": te["always_long"], "always_short": te["always_short"],
               "incumbent": {"auc": inc_auc, "coverage": inc_cov, "precision": inc_prec,
                             "side_matched": inc_lift},
               "featuresets": {}}

        for fs_name, cols in FEATURESETS:
            npz_done = OUT / f"scores_{direction.replace('->','_to_')}_{fs_name}.npz"
            if npz_done.exists() and SKIP_DONE:
                print(f"  [{fs_name}] 이미 완료된 arm -- 건너뜀 ({npz_done.name})", flush=True)
                continue
            Xtr_full = tr["merged"].loc[tr["cand"]["row"].to_numpy(), cols].to_numpy(np.float32)
            Xte_all = te["merged"].loc[te["cand"]["row"].to_numpy(), cols].to_numpy(np.float32)
            Xtr_full = np.nan_to_num(Xtr_full, nan=0.0, posinf=0.0, neginf=0.0)
            Xte_all = np.nan_to_num(Xte_all, nan=0.0, posinf=0.0, neginf=0.0)
            sub = stride_subsample(len(Xtr_full), CTX_CAP)
            Xtr, ytr_s = Xtr_full[sub], ytr[sub]
            print(f"  [{fs_name}] 컨텍스트 실사용 {len(Xtr)}행 / 후보 {len(Xtr_full)}행 "
                  f"({'균등솎기' if len(sub) < len(Xtr_full) else '전량'}), 피쳐 {len(cols)}개", flush=True)

            aucs, precs, lifts, scores = [], [], [], []
            k_sel = max(1, int(round(inc_cov * len(yte))))
            for seed in SEEDS:
                p = fit_tabpfn(Xtr, ytr_s, Xte_all, seed, device)
                a = auc(p, yte)
                pr, _ = precision_at_coverage(p, yte, inc_cov)
                sel = np.zeros(len(yte), dtype=bool)
                sel[np.argsort(-p, kind="mergesort")[:k_sel]] = True
                lf = side_matched_lift(sel, yte, side_te, boot_rng)
                aucs.append(a); precs.append(pr); lifts.append(lf); scores.append(p)
                print(f"      seed {seed}: AUC={a:.4f}  정밀도@커버{inc_cov*100:.1f}%={pr*100:.2f}%  "
                      f"숏비중={lf['short_share']*100:.0f}%  측면매칭초과={lf['lift_pp']:+.2f}pp "
                      f"CI[{lf['boot_ci_pp'][0]:+.2f},{lf['boot_ci_pp'][1]:+.2f}] "
                      f"{'✅' if lf['excludes_zero'] else '❌'}", flush=True)

            np.savez_compressed(OUT / f"scores_{direction.replace('->','_to_')}_{fs_name}.npz",
                                scores=np.array(scores), y=yte, side=side_te,
                                incumbent=inc, seeds=np.array(SEEDS))
            entry = {"context_rows_used": int(len(Xtr)), "candidate_rows": int(len(Xtr_full)),
                     "n_features": len(cols),
                     "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                     "auc_seeds": [float(x) for x in aucs],
                     "precision_mean": float(np.mean(precs)), "precision_seeds": [float(x) for x in precs],
                     "side_matched_seeds": lifts,
                     "side_matched_lift_pp_mean": float(np.mean([x["lift_pp"] for x in lifts])),
                     "side_matched_ci_excludes_zero_seeds": int(sum(x["excludes_zero"] for x in lifts)),
                     "short_share_mean": float(np.mean([x["short_share"] for x in lifts]))}

            if fs_name == "headline":
                rng = np.random.default_rng(RNG_MASTER)
                shuf = []
                for b in range(N_SHUFFLE):
                    ys = rng.permutation(ytr_s)
                    shuf.append(auc(fit_tabpfn(Xtr, ys, Xte_all, SEEDS[0] + b, device), yte))
                    if (b + 1) % 8 == 0:
                        print(f"      셔플 귀무 {b+1}/{N_SHUFFLE} …", flush=True)
                entry["shuffle_null"] = {"n": N_SHUFFLE, "mean": float(np.mean(shuf)),
                                         "q95": float(np.quantile(shuf, 0.95)),
                                         "max": float(np.max(shuf))}
                rng2 = np.random.default_rng(RNG_MASTER + 1)
                rnd = [precision_at_coverage(rng2.random(len(yte)), yte, inc_cov)[0] for _ in range(N_RANDOM)]
                entry["random_filter"] = {"n": N_RANDOM, "mean": float(np.mean(rnd)),
                                          "q95": float(np.quantile(rnd, 0.95))}
                k1 = entry["auc_mean"] > inc_auc
                k2 = entry["auc_mean"] > entry["shuffle_null"]["q95"]
                k3 = entry["precision_mean"] > entry["random_filter"]["q95"]
                k4 = entry["side_matched_ci_excludes_zero_seeds"] >= 4  # 5시드 중 4개 이상
                entry["kill_gates"] = {"K1_beats_incumbent_auc": bool(k1),
                                       "K2_beats_shuffle_q95": bool(k2),
                                       "K3_beats_random_precision_q95": bool(k3),
                                       "K4_side_matched_lift_ci_excludes_zero": bool(k4),
                                       "stage0_pass": bool(k1 and k2 and k3 and k4)}
                print(f"      셔플 귀무: 평균={entry['shuffle_null']['mean']:.4f} "
                      f"q95={entry['shuffle_null']['q95']:.4f}", flush=True)
                print(f"      무작위 필터 정밀도: 평균={entry['random_filter']['mean']*100:.2f}% "
                      f"q95={entry['random_filter']['q95']*100:.2f}%", flush=True)
                print(f"      >>> K1={k1} K2={k2} K3={k3} K4={k4}  "
                      f"Stage0 {'PASS' if (k1 and k2 and k3 and k4) else 'FAIL'}", flush=True)

            arm["featuresets"][fs_name] = entry

        results["arms"][direction] = arm

    (OUT / "report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n산출물: {OUT}/report.json", flush=True)

    head = results["arms"].get("W1->W2", {}).get("featuresets", {}).get("headline", {}).get("kill_gates")
    if head:
        print(f"\n{'='*72}\n헤드라인(W1->W2) Stage0: {'PASS' if head['stage0_pass'] else 'FAIL'}", flush=True)
    else:
        print(f"\n{'='*72}\n헤드라인(W1->W2)은 이번 실행에서 건너뜀(SKIP_DONE) -- "
              f"이전 실행 로그의 판정을 참조할 것", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
