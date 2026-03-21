"""
Trading Router — 6-Agent Single-Directional MoE (Restructured)
================================================================================
1. 6개 에이전트: bull(롱), bear(숏), chop_long(롱), chop_short(숏), normal_long(롱), normal_short(숏)
2. 모든 에이전트는 2-Action (0=대기/청산, 1=진입) 으로 통일하여 학습 병목 원천 차단
3. 7-Way GatingNet: [flat, bull, bear, chop_L, chop_S, norm_L, norm_S] 메타 라우팅
4. [신규 적용] 상태 보상(State) 제거 및 델타 보상(Step Delta) 적용 -> 리워드 파밍 완벽 박멸

[융합 모듈 v2]
5. OnlineHMMDetector: 온라인 EM 기반 4-state HMM → GatingNet 입력 보강 (레짐 전환 확률 실시간 제공)
   - 4개 은닉 상태: Bull-Trend / Bear-Trend / High-Vol-Chop / Low-Vol-Range
   - 관측 피처: log_return + garch_vol_z + oi_change_rate (3차원, 정규화)
   - 온라인 EM: 에피소드마다 최근 window(기본 512봉)로 전이행렬·방출분포 갱신
   - 출력: hmm_state_probs (4차원) → STATE_DIM에 통합 → GatingNet이 레짐 전환을 즉시 인식

6. KellyCriterionSizer: CVaR 분위 기반 fractional Kelly 포지션 사이징
   - IQN의 분위 분포에서 win_rate·payoff_ratio를 직접 추정 → f* = (p*b - q) / b
   - Half-Kelly 적용(과적합 방지) + uncertainty penalty (분위 분산 클수록 축소)
   - GatingNet 가중치(확신도)와 곱하여 최종 leverage_rate 결정
   - MAX_LEVERAGE=1.0 하드캡 유지, 최소 0.1 보장

[융합 모듈 v3]
7. MultiTimeframeFeatures: 상위 타임프레임 추세 피처 (CSV 추가 컬럼 불필요)
   - 1h봉 close 컬럼에서 런타임 롤링 계산: 4h(×4봉) / 일봉(×24봉)
   - 피처: [1h_ret, 1h_vol, 1h_trend, 4h_ret, 4h_vol, 4h_trend, htf_alignment]
     * ret   = 해당 윈도우 수익률 (tanh 정규화)
     * vol   = 해당 윈도우 변동성 (표준편차, 정규화)
     * trend = 선형회귀 기울기 (추세 강도, 정규화)
     * alignment = 4h·일봉 방향 일치 여부 (−1/0/+1)
   - 출력: MTF_DIM=7 → STATE_DIM에 추가

8. MarketAttentionEncoder: 피처 그룹 간 Cross-Attention 인코더
   - RobustIQN.feat_extractor 앞단에 삽입 (기존 Linear 레이어는 그대로 유지)
   - 피처 그룹 6개를 토큰(token)으로 취급:
     [pred(7), conf(7), elite(9), alpha(9), regime(5), synth(14)] → 패딩 후 d_model=16 투영
   - 2-head Self-Attention × 2 layer → 그룹 간 상호작용 학습
   - 출력: 6개 토큰 flatten → Linear → 기존 feat_extractor 입력과 동일 차원으로 압축
   - 경량 설계: 파라미터 ~8K 추가, 추론 오버헤드 최소화



"""
import os, sys, logging, random, argparse, gc, copy
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import pytorch_lightning as pl
import warnings

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=DeprecationWarning)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, _SCRIPT_DIR, os.path.join(_ROOT_DIR, 'ensemble'), os.path.join(_ROOT_DIR, 'strategies')]:
    if _p not in sys.path: sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._devnull_out = open(os.devnull, 'w')
        self._devnull_err = open(os.devnull, 'w')
        sys.stdout = self._devnull_out
        sys.stderr = self._devnull_err
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
        finally:
            self._devnull_out.close()
            self._devnull_err.close()

# ═══════════════════════════════════════════════════════════════════════════
# [융합 모듈 ①] OnlineHMMDetector — 온라인 EM 기반 4-state HMM 레짐 감지기
# ═══════════════════════════════════════════════════════════════════════════
class OnlineHMMDetector:
    """온라인 EM(Baum-Welch)으로 학습하는 4-state HMM 레짐 감지기.

    은닉 상태 4개:
        0: Bull-Trend     (상승 추세, 낮은 변동성)
        1: Bear-Trend     (하락 추세, 낮은 변동성)
        2: High-Vol-Chop  (급등락 혼조, 높은 변동성)
        3: Low-Vol-Range  (저변동 횡보, 좁은 레인지)

    관측 벡터 (3차원, 표준화 후 입력):
        [log_return_z, garch_vol_z, oi_change_rate_z]

    출력 피처 (HMM_DIM = 5차원):
        hmm_probs[0..3]  : 현재 스텝의 각 은닉 상태 사후확률 (합=1)
        hmm_entropy      : 상태 분포 엔트로피 (불확실성, 0~log4)

    사용 방법:
        hmm = OnlineHMMDetector()
        hmm.fit(df_train)                    # 초기 학습 (전체 훈련 데이터)
        feat = hmm.get_features(row_dict)    # 5-dim ndarray, 매 스텝 호출
        hmm.update_online(row_dict)          # 온라인 업데이트 (선택, 에피소드 후 호출)
    """

    N_STATES  = 4
    OBS_DIM   = 3
    MIN_STD   = 1e-3
    WINDOW    = 512       # 온라인 업데이트에 사용할 최근 관측 수

    def __init__(self):
        # 전이 행렬: 자기 상태 유지 확률 높게 초기화
        self.A = np.full((self.N_STATES, self.N_STATES), 0.05 / (self.N_STATES - 1))
        np.fill_diagonal(self.A, 0.85)
        self.A /= self.A.sum(axis=1, keepdims=True)

        # 초기 상태 분포
        self.pi = np.ones(self.N_STATES) / self.N_STATES

        # 가우시안 방출: 각 상태별 mu(3차원), sigma(3차원)
        # Bull: 양수 return, 낮은 vol / Bear: 음수 return, 낮은 vol
        # HVChop: 0 return, 높은 vol / LVRange: 0 return, 낮은 vol
        self.mu = np.array([
            [ 0.8, -0.5, 0.3],   # 0: Bull-Trend
            [-0.8, -0.5, -0.3],  # 1: Bear-Trend
            [ 0.0,  1.5,  0.0],  # 2: High-Vol-Chop
            [ 0.0, -1.0,  0.0],  # 3: Low-Vol-Range
        ], dtype=np.float64)
        self.sigma = np.array([
            [0.5, 0.4, 0.5],
            [0.5, 0.4, 0.5],
            [1.0, 0.6, 0.8],
            [0.3, 0.3, 0.3],
        ], dtype=np.float64)

        # 온라인 업데이트용 관측 링버퍼
        self._obs_buffer: deque = deque(maxlen=self.WINDOW)

        # 직전 스텝 알파(forward variable) — Viterbi 대신 forward 알고리즘으로 실시간 추론
        self._alpha: np.ndarray = self.pi.copy()

        # 관측 정규화 통계 (fit()에서 계산)
        self._obs_mean = np.zeros(self.OBS_DIM)
        self._obs_std  = np.ones(self.OBS_DIM)

        self._fitted = False

    # ── 내부 유틸 ──────────────────────────────────────────────────────────
    def _extract_obs(self, row: dict) -> np.ndarray:
        """dict 또는 Series에서 3차원 관측 벡터 추출 후 표준화."""
        raw = np.array([
            float(row.get('log_return',      0.0)),
            float(row.get('garch_vol_z',     0.0)),
            float(row.get('oi_change_rate',  0.0)),
        ], dtype=np.float64)
        return (raw - self._obs_mean) / (self._obs_std + 1e-8)

    def _emission_log_prob(self, obs: np.ndarray) -> np.ndarray:
        """각 은닉 상태에서 obs의 가우시안 로그 확률 (N_STATES,)."""
        diff  = obs[None, :] - self.mu                          # (S, D)
        var   = np.maximum(self.sigma ** 2, self.MIN_STD ** 2)  # (S, D)
        lp    = -0.5 * np.sum((diff ** 2) / var + np.log(2 * np.pi * var), axis=1)
        return lp

    def _forward_step(self, obs: np.ndarray) -> np.ndarray:
        """직전 alpha를 한 스텝 전진시켜 사후확률 반환 (S,)."""
        log_emit  = self._emission_log_prob(obs)
        predicted = self._alpha @ self.A                        # (S,)
        log_joint = np.log(predicted + 1e-300) + log_emit
        log_joint -= log_joint.max()                            # 수치 안정
        alpha_new  = np.exp(log_joint)
        alpha_new /= alpha_new.sum() + 1e-300
        self._alpha = alpha_new
        return alpha_new

    # ── 공개 API ───────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, n_iter: int = 30) -> None:
        """훈련 데이터 전체로 Baum-Welch EM 초기 학습.

        Args:
            df: 훈련 DataFrame (log_return, garch_vol_z, oi_change_rate 필수)
            n_iter: EM 반복 횟수
        """
        # 관측 정규화 통계 계산
        needed = ['log_return', 'garch_vol_z', 'oi_change_rate']
        avail  = [c for c in needed if c in df.columns]
        raw_mat = np.zeros((len(df), 3), dtype=np.float64)
        for i, col in enumerate(needed):
            if col in df.columns:
                raw_mat[:, i] = df[col].fillna(0).values
        self._obs_mean = raw_mat.mean(axis=0)
        self._obs_std  = raw_mat.std(axis=0).clip(min=1e-6)

        obs_seq = (raw_mat - self._obs_mean) / (self._obs_std + 1e-8)
        T = len(obs_seq)

        for _ in range(n_iter):
            # ── E-step: Forward-Backward ──
            log_emit = np.stack([self._emission_log_prob(obs_seq[t]) for t in range(T)])  # (T, S)

            # Forward
            log_alpha = np.zeros((T, self.N_STATES))
            log_alpha[0] = np.log(self.pi + 1e-300) + log_emit[0]
            for t in range(1, T):
                for j in range(self.N_STATES):
                    log_alpha[t, j] = np.logaddexp.reduce(
                        log_alpha[t-1] + np.log(self.A[:, j] + 1e-300)
                    ) + log_emit[t, j]

            # Backward
            log_beta = np.zeros((T, self.N_STATES))
            for t in range(T - 2, -1, -1):
                for i in range(self.N_STATES):
                    log_beta[t, i] = np.logaddexp.reduce(
                        np.log(self.A[i, :] + 1e-300) + log_emit[t+1] + log_beta[t+1]
                    )

            # gamma: (T, S)
            log_gamma = log_alpha + log_beta
            log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # xi: (T-1, S, S)
            log_xi = np.zeros((T - 1, self.N_STATES, self.N_STATES))
            for t in range(T - 1):
                for i in range(self.N_STATES):
                    for j in range(self.N_STATES):
                        log_xi[t, i, j] = (log_alpha[t, i]
                                           + np.log(self.A[i, j] + 1e-300)
                                           + log_emit[t+1, j]
                                           + log_beta[t+1, j])
                log_xi[t] -= np.logaddexp.reduce(log_xi[t].reshape(-1))

            xi = np.exp(log_xi)  # (T-1, S, S)

            # ── M-step ──
            self.pi = gamma[0] / (gamma[0].sum() + 1e-300)
            self.A  = xi.sum(axis=0) / (gamma[:-1].sum(axis=0, keepdims=True).T + 1e-300)
            self.A /= self.A.sum(axis=1, keepdims=True) + 1e-300

            for s in range(self.N_STATES):
                w = gamma[:, s] + 1e-300
                self.mu[s]    = (w[:, None] * obs_seq).sum(axis=0) / w.sum()
                diff          = obs_seq - self.mu[s]
                self.sigma[s] = np.sqrt((w[:, None] * diff ** 2).sum(axis=0) / w.sum()).clip(self.MIN_STD)

        # alpha 초기화: 훈련 데이터 마지막 상태로 워밍업
        self._alpha = gamma[-1]
        self._obs_buffer.extend(obs_seq[-self.WINDOW:].tolist())
        self._fitted = True
        logger.info(f"[HMM] fit 완료 | mu=\n{self.mu.round(3)}")

    def get_features(self, row: dict) -> np.ndarray:
        """현재 스텝의 HMM 피처 벡터 반환 (HMM_DIM = 5차원).

        Returns:
            np.ndarray shape (5,): [p0, p1, p2, p3, entropy]
        """
        obs   = self._extract_obs(row)
        probs = self._forward_step(obs)
        ent   = float(-np.sum(probs * np.log(probs + 1e-300)))  # 0 ~ log(4) ≈ 1.386
        ent_n = ent / np.log(self.N_STATES + 1e-8)              # 0~1 정규화
        self._obs_buffer.append(obs.tolist())
        return np.concatenate([probs, [ent_n]]).astype(np.float32)

    def update_online(self, n_iter: int = 5) -> None:
        """링버퍼의 최근 관측으로 파라미터 온라인 업데이트 (에피소드 종료 후 호출).

        전체 Baum-Welch 대신 단축 EM (n_iter=5) 으로 빠르게 적응.
        전이 행렬은 20% 이내로만 변경 (급격한 망각 방지).
        """
        if len(self._obs_buffer) < 64:
            return
        obs_seq = np.array(self._obs_buffer, dtype=np.float64)
        T = len(obs_seq)

        A_old = self.A.copy()
        for _ in range(n_iter):
            log_emit = np.stack([self._emission_log_prob(obs_seq[t]) for t in range(T)])
            log_alpha = np.zeros((T, self.N_STATES))
            log_alpha[0] = np.log(self.pi + 1e-300) + log_emit[0]
            for t in range(1, T):
                for j in range(self.N_STATES):
                    log_alpha[t, j] = np.logaddexp.reduce(
                        log_alpha[t-1] + np.log(self.A[:, j] + 1e-300)
                    ) + log_emit[t, j]

            log_beta = np.zeros((T, self.N_STATES))
            for t in range(T - 2, -1, -1):
                for i in range(self.N_STATES):
                    log_beta[t, i] = np.logaddexp.reduce(
                        np.log(self.A[i, :] + 1e-300) + log_emit[t+1] + log_beta[t+1]
                    )

            log_gamma = log_alpha + log_beta
            log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            log_xi = np.zeros((T - 1, self.N_STATES, self.N_STATES))
            for t in range(T - 1):
                for i in range(self.N_STATES):
                    for j in range(self.N_STATES):
                        log_xi[t, i, j] = (log_alpha[t, i]
                                           + np.log(self.A[i, j] + 1e-300)
                                           + log_emit[t+1, j]
                                           + log_beta[t+1, j])
                log_xi[t] -= np.logaddexp.reduce(log_xi[t].reshape(-1))
            xi = np.exp(log_xi)

            A_new = xi.sum(axis=0) / (gamma[:-1].sum(axis=0, keepdims=True).T + 1e-300)
            A_new /= A_new.sum(axis=1, keepdims=True) + 1e-300
            # 급격한 망각 방지: 이전 값의 80% + 새 값의 20%
            self.A = 0.8 * A_old + 0.2 * A_new

            for s in range(self.N_STATES):
                w = gamma[:, s] + 1e-300
                new_mu    = (w[:, None] * obs_seq).sum(axis=0) / w.sum()
                diff      = obs_seq - new_mu
                new_sigma = np.sqrt((w[:, None] * diff ** 2).sum(axis=0) / w.sum()).clip(self.MIN_STD)
                # mu, sigma도 점진적 업데이트
                self.mu[s]    = 0.85 * self.mu[s]    + 0.15 * new_mu
                self.sigma[s] = 0.85 * self.sigma[s] + 0.15 * new_sigma

        self._alpha = gamma[-1]


# ═══════════════════════════════════════════════════════════════════════════
# [융합 모듈 ②] KellyCriterionSizer — CVaR IQN 분위 기반 동적 포지션 사이징
# ═══════════════════════════════════════════════════════════════════════════
class KellyCriterionSizer:
    """IQN의 분위 분포에서 Kelly 비율을 직접 추정하여 leverage_rate를 동적으로 결정.

    수식:
        f* = (p * b - q) / b     (전통 Kelly)
        f_half = f* * 0.5        (Half-Kelly, 과적합/드로다운 방지)
        f_final = f_half * confidence * (1 - uncertainty_penalty)

    추정 방법 (IQN 분위 활용):
        - win_rate p  : Q(action=1) > Q(action=0) 인 분위 비율
        - payoff_ratio b : 양수 분위 평균 / |음수 분위 평균|  (손익비)
        - uncertainty : 분위 분산 / 전체 분산 (크면 배팅 축소)

    파라미터:
        half_kelly      (float): Half-Kelly 계수 (기본 0.5)
        min_lev         (float): 최소 leverage_rate (기본 0.1)
        max_lev         (float): 최대 leverage_rate (기본 1.0, MAX_LEVERAGE와 동일)
        uncertainty_cap (float): uncertainty_penalty 최대값 (기본 0.5)
    """

    def __init__(self, half_kelly: float = 0.5,
                 min_lev: float = 0.1, max_lev: float = 1.0,
                 uncertainty_cap: float = 0.5):
        self.half_kelly      = half_kelly
        self.min_lev         = min_lev
        self.max_lev         = max_lev
        self.uncertainty_cap = uncertainty_cap

    def compute(self,
                q_quantiles: torch.Tensor,
                gating_confidence: float = 1.0) -> float:
        """Kelly leverage_rate 계산.

        Args:
            q_quantiles: shape (N_quantiles, n_actions) — IQN forward()에서 얻은 분위 행렬.
                         행: 분위(정렬된 순서), 열: 액션.
                         액션 0 = 대기/청산, 액션 1 = 진입.
            gating_confidence: GatingNet 해당 에이전트 가중치 (0~1).

        Returns:
            leverage_rate (float): [min_lev, max_lev] 범위의 포지션 배율.
        """
        with torch.no_grad():
            q = q_quantiles.float().cpu()   # (NQ, n_actions)
            if q.shape[1] < 2:
                return self.min_lev

            q0 = q[:, 0]  # 대기 분위
            q1 = q[:, 1]  # 진입 분위

            # ── 승률 추정: 진입이 대기보다 유리한 분위 비율 ──
            win_rate = float((q1 > q0).float().mean())

            # ── 손익비 추정: 양수 advantage / |음수 advantage| ──
            adv = q1 - q0
            pos_mask = adv > 0
            neg_mask = adv < 0

            pos_mean = float(adv[pos_mask].mean()) if pos_mask.any() else 0.0
            neg_mean = float(adv[neg_mask].abs().mean()) if neg_mask.any() else 1e-6
            payoff   = pos_mean / (neg_mean + 1e-8)

            # ── Kelly 공식 ──
            p, q_val = win_rate, 1.0 - win_rate
            b        = max(payoff, 0.1)   # 손익비 최소 0.1 보장
            f_star   = (p * b - q_val) / b
            f_half   = max(f_star * self.half_kelly, 0.0)

            # ── 불확실성 패널티: IQN 분위 분산이 클수록 배팅 축소 ──
            total_var = float(q1.var() + 1e-8)
            adv_var   = float(adv.var() + 1e-8)
            uncertainty = min(adv_var / total_var, self.uncertainty_cap)
            f_penalized = f_half * (1.0 - uncertainty)

            # ── GatingNet 확신도 반영 ──
            f_final = f_penalized * float(np.clip(gating_confidence, 0.0, 1.0))

            lev = float(np.clip(f_final, self.min_lev, self.max_lev))
            return lev

    def log_stats(self, q_quantiles: torch.Tensor) -> dict:
        """디버깅용 Kelly 통계 반환."""
        with torch.no_grad():
            q = q_quantiles.float().cpu()
            q0, q1 = q[:, 0], q[:, 1]
            adv      = q1 - q0
            win_rate = float((adv > 0).float().mean())
            adv_pos  = float(adv[adv > 0].mean()) if (adv > 0).any() else 0.0
            adv_neg  = float(adv[adv < 0].abs().mean()) if (adv < 0).any() else 0.0
            payoff   = adv_pos / (adv_neg + 1e-8)
            f_star   = (win_rate * payoff - (1 - win_rate)) / (payoff + 1e-8)
        return {'win_rate': round(win_rate, 3), 'payoff': round(payoff, 3),
                'f_star': round(f_star, 3), 'f_half': round(max(f_star * self.half_kelly, 0), 3)}


# ═══════════════════════════════════════════════════════════════════════════
# [융합 모듈 ③] MultiTimeframeFeatures — 런타임 멀티타임프레임 피처 계산기
# ═══════════════════════════════════════════════════════════════════════════
class MultiTimeframeFeatures:
    """1h봉 close 시계열에서 1h / 4h 추세 피처를 런타임으로 계산.

    CSV에 추가 컬럼 불필요. TradingEnv 초기화 시 close 배열을 받아
    전체 구간의 MTF 피처를 한 번에 선계산(precompute)하고 numpy 배열로 캐싱.

    출력 (MTF_DIM = 7차원):
        [0] 1h_ret       : 직전 1봉 대비 수익률 (tanh 스케일) — 즉각 모멘텀
        [1] 1h_vol       : 최근 4봉 변동성 (로그수익률 std)   — 단기 노이즈 레벨
        [2] 1h_trend     : 최근 4봉 선형회귀 기울기            — 단기 방향성
        [3] 4h_ret       : 4봉 누적 수익률 (tanh 스케일)       — 중기 모멘텀
        [4] 4h_vol       : 4봉 변동성                          — 중기 노이즈 레벨
        [5] 4h_trend     : 4봉 선형회귀 기울기                  — 중기 방향성
        [6] htf_alignment: 1h·4h 방향 일치 부호 (−1 / 0 / +1)

    모든 출력은 [-1, 1] 범위로 clamp.

    설계 근거 (1h/4h 선택):
        - 1h: 현재 봉 자체가 1h → 즉각적 모멘텀·노이즈 레벨 포착
        - 4h: 단타 관점의 중기 추세. 일봉(24봉)은 코인 단타에 너무 느려
              학습 신호보다 노이즈가 많고 레짐 전환 감지가 지연됨
        - alignment: 1h·4h 방향 일치 시에만 진입 고려 → 역추세 필터 역할
    """

    _RET_SCALE       = 50.0   # tanh(ret * 50): 2% 수익 ≈ 0.76
    _VOL_SCALE       = 10.0   # 변동성 정규화 기준
    _VOL_1H_WINDOW   = 4      # 1h 변동성 계산용 내부 윈도우

    def __init__(self, close_arr: np.ndarray,
                 w1h: int = 1, w4h: int = 4):
        self.w1h  = w1h
        self.w4h  = w4h
        self._cache = self._precompute(close_arr.astype(np.float64))

    @staticmethod
    def _linreg_slope(y: np.ndarray) -> float:
        """가격 배열의 정규화된 선형회귀 기울기 (−1~1).
        price_range 최솟값을 평균의 0.1%로 보장 → 극소변동 구간 포화 버그 수정."""
        n = len(y)
        if n < 3:
            return 0.0
        x  = np.arange(n, dtype=np.float64)
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum()
        if denom < 1e-12:
            return 0.0
        slope = ((x - xm) * (y - ym)).sum() / denom
        price_range = max(y.max() - y.min(), abs(ym) * 0.001, 1e-8)
        return float(np.clip(slope * n / price_range, -1.0, 1.0))

    @staticmethod
    def _logret_slope(logret: np.ndarray) -> float:
        """로그수익률 평균으로 단기 추세 방향 추정 (1h용).
        노이즈 필터: 0.1%(1e-3) 미만 평균 수익률은 0 반환.
        기존 1e-5는 너무 낮아 align 중립 비율이 1%대로 낮아지는 원인."""
        if len(logret) < 2:
            return 0.0
        mean_ret = logret.mean()
        if abs(mean_ret) < 1e-3:
            return 0.0
        return float(np.clip(mean_ret * 100.0, -1.0, 1.0))

    def _precompute(self, close: np.ndarray) -> np.ndarray:
        """전체 길이 T에 대한 MTF 피처 행렬 (T, MTF_DIM) 선계산."""
        T      = len(close)
        out    = np.zeros((T, MTF_DIM), dtype=np.float32)
        logret = np.zeros(T, dtype=np.float64)
        logret[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-8))

        for i in range(T):
            # ── 1h 피처 (로그수익률 기반 단기 추세) ──
            ret1   = float(np.tanh(logret[i] * self._RET_SCALE))
            sv     = max(0, i - self._VOL_1H_WINDOW + 1)
            lr1w   = logret[sv:i+1]
            vol1   = float(np.clip(lr1w.std() * self._VOL_SCALE, 0.0, 1.0)) if len(lr1w) > 1 else 0.0
            # 1h 추세: 로그수익률 평균 (가격 회귀와 다른 방식 → 다양성 확보)
            trend1 = self._logret_slope(lr1w) if len(lr1w) >= 2 else 0.0

            # ── 4h 피처 (가격 선형회귀 기반 중기 추세) ──
            s4     = max(0, i - self.w4h + 1)
            c4     = close[s4:i+1]
            lr4    = logret[s4:i+1]
            ret4   = float(np.tanh((c4[-1] / c4[0] - 1) * self._RET_SCALE)) if len(c4) > 1 else 0.0
            vol4   = float(np.clip(lr4.std() * self._VOL_SCALE, 0.0, 1.0))   if len(lr4) > 1 else 0.0
            # 4h 추세: 가격 선형회귀 (최소 3봉 필요)
            trend4 = self._linreg_slope(c4) if len(c4) >= 3 else 0.0

            # ── 1h·4h 방향 일치 ──
            align  = float(np.sign(trend1) * np.sign(trend4)) if (trend1 != 0 and trend4 != 0) else 0.0

            out[i] = [ret1, vol1, trend1, ret4, vol4, trend4, align]

        align_pos = (out[:,6] > 0).mean() * 100
        align_neg = (out[:,6] < 0).mean() * 100
        align_neu = (out[:,6] == 0).mean() * 100
        logger.info(
            f"[MTF] 선계산 완료 | shape={out.shape} | "
            f"1h_ret μ={out[:,0].mean():.3f} σ={out[:,0].std():.3f} | "
            f"4h_ret μ={out[:,3].mean():.3f} σ={out[:,3].std():.3f} | "
            f"align: +{align_pos:.1f}% / 0:{align_neu:.1f}% / -{align_neg:.1f}%"
        )
        return out

    def get(self, idx: int) -> np.ndarray:
        """인덱스 idx의 MTF 피처 벡터 반환 (MTF_DIM,)."""
        if idx < 0 or idx >= len(self._cache):
            return np.zeros(MTF_DIM, dtype=np.float32)
        return self._cache[idx]


# ═══════════════════════════════════════════════════════════════════════════
# [융합 모듈 ④] MarketAttentionEncoder — 피처 그룹 간 Self-Attention 인코더
# ═══════════════════════════════════════════════════════════════════════════
class MarketAttentionEncoder(nn.Module):
    """피처 그룹 6개를 토큰으로 취급해 Self-Attention으로 그룹 간 상호작용 학습.

    입력 구조 (raw 1-frame 피처, pos/HMM/MTF 제외):
        pred  (7) / conf  (7) / elite (9) / alpha (9) / regime (5) / synth (14)
        총 51차원 → 각 그룹을 d_model=16으로 선형 투영 → 6개 토큰

    아키텍처:
        GroupProjection  : 각 그룹 → Linear(group_dim, d_model) + LayerNorm
        SelfAttention ×2 : 2-head, d_model=16 → 그룹 간 의존성 학습
        OutputProjection : 6×d_model=96 flatten → Linear(96, out_dim)

    출력 out_dim은 RobustIQN의 feat_extractor 입력 차원과 동일하게 맞춤.
    pos / HMM / MTF 피처는 attention 통과 후 concat → feat_extractor 입력.

    파라미터 수: ~8K (경량)
    """

    # 그룹 정의: (이름, 시작 오프셋, 길이)
    # STATE_DIM 내 순서: preds(7) confs(7) stats(3) elite(9) alpha(9) regime(5) synth(14) pos(5) hmm(5) mtf(7)
    # stats(3)는 pred/conf에서 파생된 요약 통계 → pred 그룹에 합산
    _GROUPS = [
        ('pred',   0,  7),
        ('conf',   7,  7),
        ('elite', 17,  9),   # stats(3) 건너뜀 → 14부터지만 stats 포함 시 17
        ('alpha', 26,  9),
        ('regime',35,  5),
        ('synth', 40, 14),
    ]
    # stats(3)는 pred 그룹 뒤에 위치: pred(7) + conf(7) = 14, stats는 14~16
    # 실제 오프셋: pred=0~6, conf=7~13, stats=14~16, elite=17~25, alpha=26~34, regime=35~39, synth=40~53

    D_MODEL  = 16
    N_HEADS  = 2
    N_LAYERS = 2

    def __init__(self, out_dim: int, raw_state_dim: int = None):
        """
        Args:
            out_dim     : 출력 차원 (RobustIQN feat_extractor 첫 Linear 입력 차원과 동일)
            raw_state_dim: 1프레임 STATE_DIM (스택 제외)
        """
        super().__init__()
        self.out_dim = out_dim

        # 각 그룹 → d_model 투영
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Linear(g_dim, self.D_MODEL), nn.LayerNorm(self.D_MODEL))
            for _, _, g_dim in self._GROUPS
        ])

        # Self-Attention 레이어 (n_layers개)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D_MODEL, nhead=self.N_HEADS,
            dim_feedforward=self.D_MODEL * 2,
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=self.N_LAYERS)

        # flatten → out_dim 압축
        n_groups = len(self._GROUPS)
        self.out_proj = nn.Sequential(
            nn.Linear(n_groups * self.D_MODEL, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU()
        )

    def forward(self, raw_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_feat: (B, raw_state_dim) — 1프레임의 피처 (pos/HMM/MTF 포함 전체)
                      실제로는 마지막 프레임의 market 피처 부분만 사용

        Returns:
            (B, out_dim) — attention 인코딩 결과
        """
        # 각 그룹 추출 → 투영 → (B, n_groups, D_MODEL) 토큰 시퀀스
        tokens = []
        for (_, start, length), proj_layer in zip(self._GROUPS, self.proj):
            g = raw_feat[:, start:start + length]   # (B, group_dim)
            tokens.append(proj_layer(g))             # (B, D_MODEL)
        tokens = torch.stack(tokens, dim=1)          # (B, n_groups, D_MODEL)

        # Self-Attention
        attended = self.attn(tokens)                 # (B, n_groups, D_MODEL)

        # Flatten + 출력 투영
        flat = attended.flatten(1)                   # (B, n_groups * D_MODEL)
        return self.out_proj(flat)                   # (B, out_dim)


# ═══════════════════════════════════════════════════════════════════════════
# [상수 및 차원 정의]
# ═══════════════════════════════════════════════════════════════════════════
MODEL_PRED = ['pred_timesfm', 'pred_chronos', 'pred_ttm', 'pred_patchtst', 'pred_tide', 'pred_mdjd', 'pred_ridge']
MODEL_CONF = ['conf_timesfm', 'conf_chronos', 'conf_ttm', 'conf_patchtst', 'conf_tide', 'conf_mdjd', 'conf_ridge']

ELITE_COLS = ['sig_whale', 'sig_orderblock', 'sig_oi_divergence', 'sig_ai_squeeze']
ALPHA_7_COLS = ['session_us', 'hour_cos', 'cvp_poc_dist', 'cvp_volume_imbalance', 'fvg_dist', 'breakout_strength', 'oi_change_rate']
REGIME_COLS = ['regime_chop', 'regime_whipsaw', 'regime_bull', 'regime_bear', 'regime_normal']

TARGET_COL = 'log_return'
SYNTHETIC_ALPHA_COLS = ['ofti', 'kel', 'mta_funding', 'svps', 'cada', 'mshd', 'fvci',
                        'wpad', 'fdlv', 'vsdi', 'vebr', 'tlad', 'mtmb', 'fcsz']

STATE_PRED  = ['pred_tide', 'pred_ridge', 'pred_patchtst', 'pred_timesfm', 'pred_chronos', 'pred_ttm', 'pred_mdjd']
STATE_CONF  = ['conf_tide', 'conf_ridge', 'conf_ttm', 'conf_chronos', 'conf_timesfm', 'conf_mdjd', 'conf_patchtst']
STATE_ELITE = ['evt_excess_z', 'sig_orderblock', 'sig_ai_squeeze', 'sig_oi_divergence', 'sig_whale', 'sig_garch_regime', 'jump_z', 'jump_flag', 'evt_tail_flag']
STATE_ALPHA = ['hour_cos', 'garch_vol', 'garch_vol_z', 'breakout_strength', 'fvg_dist', 'cvp_poc_dist', 'session_us', 'oi_change_rate', 'cvp_volume_imbalance']
STATE_SYNTH = ['ou_funding_z', 'fcsz', 'vebr', 'ofti', 'cada', 'tlad', 'svps', 'mshd', 'fdlv', 'wpad', 'fvci', 'kel', 'mtmb', 'ou_halflife']

FEATURE_DIM = len(STATE_PRED) + len(STATE_CONF) + 3 + len(STATE_ELITE) + len(STATE_ALPHA) + len(REGIME_COLS) + len(STATE_SYNTH)
HMM_N_STATES = 4           # Bull-Trend / Bear-Trend / High-Vol-Chop / Low-Vol-Range
HMM_DIM      = HMM_N_STATES + 1   # 4개 상태확률 + 전환 엔트로피 1개 = 5차원
MTF_DIM      = 7           # [1h_ret, 1h_vol, 1h_trend, 4h_ret, 4h_vol, 4h_trend, htf_alignment]
STATE_DIM = FEATURE_DIM + 5 + HMM_DIM + MTF_DIM   # +5: pos, +5: HMM, +7: MTF
STACK_N           = 4
STACKED_STATE_DIM = STATE_DIM * STACK_N

# MTF 윈도우 (1h봉 기준)
MTF_1H_WINDOW = 1    # 1봉 = 1시간 (현재 봉 단독 → 직전 1봉 대비 모멘텀)
MTF_4H_WINDOW = 4    # 4봉 = 4시간

# ═══════════════════════════════════════════════════════════════════════════
# 2. 거래 환경 (TradingEnv)
# ═══════════════════════════════════════════════════════════════════════════
class TradingEnv:
    STATE_DIM = STATE_DIM

    def __init__(self, df, initial_balance=10000.0, fee=0.0005, slip=0.0002, phase='train', agent_role='bull_sniper',
                 hmm_detector=None, mtf_features=None):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee
        self.slip = slip
        self.phase = phase
        self.agent_role = agent_role
        self.hmm_detector = hmm_detector   # OnlineHMMDetector 인스턴스 (None이면 영벡터 대체)
        # MTF: 외부에서 생성된 MultiTimeframeFeatures 인스턴스 또는 None(내부 생성)
        if mtf_features is not None:
            self.mtf = mtf_features
        else:
            close_arr = self.df['close'].values.astype(np.float32)
            self.mtf = MultiTimeframeFeatures(close_arr)

        self.MAX_EPISODE_STEPS = 4096 if phase == 'train' else len(self.df) - 1
        self.MAX_LEVERAGE = 1.0
        self.MAX_HOLD = {'train': 72, 'val': 144, 'test': 288}

        feat_cols = STATE_PRED + STATE_CONF + STATE_ELITE + STATE_ALPHA + REGIME_COLS + STATE_SYNTH
        self._feat_np  = self.df[feat_cols].values.astype(np.float32)
        self._close_np = self.df['close'].values.astype(np.float32)
        self._n_pred, self._n_conf = len(STATE_PRED), len(STATE_CONF)
        self._n_elite, self._n_alpha = len(STATE_ELITE), len(STATE_ALPHA)
        self._n_regime, self._n_synth = len(REGIME_COLS), len(STATE_SYNTH)
        self._frame_stack = deque(maxlen=STACK_N)

        # HMM 관측 컬럼 미리 추출 (get_features 호출용 row dict)
        _hmm_cols = ['log_return', 'garch_vol_z', 'oi_change_rate']
        self._hmm_obs_np = {
            col: self.df[col].fillna(0).values.astype(np.float32)
            if col in self.df.columns else np.zeros(len(self.df), dtype=np.float32)
            for col in _hmm_cols
        }

        self.reset()

    def reset(self, start_idx=None):
        if self.phase == 'train':
            self.start_step = start_idx if start_idx is not None else random.randint(0, len(self.df) - self.MAX_EPISODE_STEPS - 1)
        else:
            self.start_step = 0

        self.current_step = self.start_step
        self.end_step = self.start_step + self.MAX_EPISODE_STEPS

        self.balance = self.initial_balance
        self.pos = None
        self.entry_price = 0.0
        self.entry_idx = 0
        self.current_leverage = 0.0 
        
        self.total_trades = 0
        self.win_trades = 0
        self.active_steps = 0
        
        self.unrealized_pnl = 0.0
        self.prev_unrealized_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_pnl = 0.0
        self.hold_count = 0

        self._frame_stack.clear()
        return self._get_stacked_state(self._build_state(self.current_step))

    def step(self, action, leverage_rate=1.0):
        current_price = self._close_np[self.current_step]
        
        # ── 1. 이전 스텝의 포트폴리오 가치 (기준점) ──
        if self.pos is not None:
            prev_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            prev_portfolio_value = self.balance

        # ── 6-Agent 글로벌 액션 매핑 ──
        global_action = action 
        if self.agent_role in ['bear', 'chop_short', 'normal_short'] and action == 1:
            global_action = 2

        force_close = False
        if self.pos is not None and self.unrealized_pnl <= -0.025:
            force_close = True

        is_entering_long = False
        is_entering_short = False
        is_closing = False
        
        if force_close:
            is_closing = True
        else:
            if global_action == 0:
                if self.pos is not None: is_closing = True
            elif global_action == 1:
                if self.pos is None: is_entering_long = True
                elif self.pos == 'SHORT': is_closing = True
            elif global_action == 2:
                if self.pos is None: is_entering_short = True
                elif self.pos == 'LONG': is_closing = True

        # ── 2. 상태 업데이트 및 수수료/슬리피지 반영 ──
        if is_entering_long:
            self.pos = 'LONG'
            self.entry_price = current_price * (1 + self.slip)
            self.entry_idx = self.current_step
            self.current_leverage = self.MAX_LEVERAGE * leverage_rate
            self.balance -= self.balance * self.fee * self.current_leverage 
            self.active_steps += 1

        elif is_entering_short:
            self.pos = 'SHORT'
            self.entry_price = current_price * (1 - self.slip)
            self.entry_idx = self.current_step
            self.current_leverage = self.MAX_LEVERAGE * leverage_rate
            self.balance -= self.balance * self.fee * self.current_leverage 
            self.active_steps += 1

        elif is_closing:
            if self.pos == 'LONG': realized_pnl = (current_price * (1 - self.slip) - self.entry_price) / self.entry_price
            else: realized_pnl = (self.entry_price - current_price * (1 + self.slip)) / self.entry_price
            
            realized_pnl *= self.current_leverage
            self.balance += self.balance * realized_pnl 
            self.balance -= self.balance * self.fee * self.current_leverage 
            
            self.total_trades += 1
            if realized_pnl > 0: self.win_trades += 1

            self.pos = None
            self.current_leverage = 0.0
            self.hold_count = 0
            self.unrealized_pnl = 0.0
            self.peak_pnl = 0.0
            self.max_drawdown = 0.0
            is_closed = True

        self.current_step += 1
        done = self.current_step >= self.end_step
        next_price = self._close_np[self.current_step] if not done else current_price

        # ── 3. 미실현 손익 업데이트 ──
        if self.pos is not None:
            self.hold_count = self.current_step - self.entry_idx
            
            if self.pos == 'LONG': 
                est_exit_price = next_price * (1 - self.slip)
                raw_pnl = (est_exit_price - self.entry_price) / self.entry_price
            else: 
                est_exit_price = next_price * (1 + self.slip)
                raw_pnl = (self.entry_price - est_exit_price) / self.entry_price

            # [BUG1 FIX] 진입 수수료 이중 차감 제거
            # 진입 시 balance -= balance * fee * lev 로 이미 차감됨
            # unrealized_pnl에서 또 빼면 진입 직후부터 -0.05% bias 누적
            # → 700회 거래 × 에피소드 반복 시 수백%의 허구 손실 신호 생성
            self.unrealized_pnl = raw_pnl * self.current_leverage

            self.max_drawdown = min(self.max_drawdown, self.unrealized_pnl)
            self.peak_pnl = max(self.peak_pnl, self.unrealized_pnl)
            self.active_steps += 1

        # ── 4. 순수 포트폴리오 가치 증감(Delta) 보상 ──
        if self.pos is not None:
            current_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            current_portfolio_value = self.balance

        step_delta = (current_portfolio_value - prev_portfolio_value) / self.initial_balance * 500.0
        reward = step_delta

        # [BUG2+3+4 FIX] trend_bonus / scalp_bonus 전부 제거
        # scalp_bonus: hold_count=0 시 6x 증폭 → reward 클리핑 포화 → 학습 신호 소실
        # trend_bonus: 큰 step_delta에서 1.0 클리핑으로 신호 손실
        # 두 보너스 모두 balance와 무관하게 reward를 부풀려 Rew와 PnL이 역방향으로 분리
        # → chop의 Rew=+100, PnL=-70% 현상의 근본 원인
        # 수수료+슬리피지가 이미 충분한 자연 억제제이므로 추가 bonus 불필요

        # pred_consensus 보너스만 유지 (스케일이 작아 클리핑 영향 없음)
        if self.pos is not None:
            pred_consensus = float(self._feat_np[self.current_step, :self._n_pred].mean())
            if self.unrealized_pnl >= 0 and ((self.pos == 'LONG' and pred_consensus > 0.0) or (self.pos == 'SHORT' and pred_consensus < 0.0)):
                reward += 0.01

        if done and self.pos is not None:
            ep_end_price = self._close_np[min(self.current_step, len(self._close_np) - 1)]
            if self.pos == 'LONG':
                ep_realized = (ep_end_price * (1 - self.slip) - self.entry_price) / self.entry_price
            else:
                ep_realized = (self.entry_price - ep_end_price * (1 + self.slip)) / self.entry_price
            ep_realized *= self.current_leverage
            self.balance += self.balance * ep_realized
            self.balance -= self.balance * self.fee * self.current_leverage
            self.total_trades += 1
            if ep_realized > 0: self.win_trades += 1
            self.pos = None
            self.current_leverage = 0.0
            self.unrealized_pnl = 0.0
            self.hold_count = 0
            self.peak_pnl = 0.0
            self.max_drawdown = 0.0

        reward = float(np.clip(reward, -1.0, 1.0))
        info = {'pnl_pct': (self.balance / self.initial_balance - 1) * 100, 'wr': self.win_trades / max(1, self.total_trades)}
        return self._get_stacked_state(self._build_state(self.current_step)), reward, done, info

    @property
    def win_rate(self): return self.win_trades / max(1, self.total_trades)

    def _get_stacked_state(self, raw_state):
        self._frame_stack.append(raw_state)
        pad = STACK_N - len(self._frame_stack)
        frames = [np.zeros(STATE_DIM, np.float32)] * pad + list(self._frame_stack)
        return np.concatenate(frames)

    def _build_state(self, idx):
        if idx < 0 or idx >= len(self._feat_np):
            return np.zeros(STATE_DIM, dtype=np.float32)

        row = self._feat_np[idx]
        o = 0
        preds  = row[o:o+self._n_pred];   o += self._n_pred
        confs  = row[o:o+self._n_conf];   o += self._n_conf
        stats  = np.array([preds.mean(), preds.std(), confs.mean()], dtype=np.float32)
        elite  = row[o:o+self._n_elite];  o += self._n_elite
        alpha7 = row[o:o+self._n_alpha];  o += self._n_alpha
        regimes= row[o:o+self._n_regime]; o += self._n_regime
        synth  = row[o:]

        close = self._close_np[idx]
        pos_features = np.array([
            1.0 if self.pos == 'LONG' else (-1.0 if self.pos == 'SHORT' else 0.0),
            self.entry_price / close - 1 if self.pos is not None else 0.0,
            np.tanh(self.unrealized_pnl / 0.02),
            np.clip(self.max_drawdown   / 0.05, -1.0, 1.0),
            self.hold_count / 144
        ], dtype=np.float32)

        # ── HMM 피처 (5차원: 4개 상태확률 + 엔트로피) ──
        if self.hmm_detector is not None:
            row_dict = {col: float(self._hmm_obs_np[col][idx]) for col in self._hmm_obs_np}
            hmm_feat = self.hmm_detector.get_features(row_dict)
        else:
            hmm_feat = np.zeros(HMM_DIM, dtype=np.float32)

        # ── MTF 피처 (7차원: 4h/일봉 수익률·변동성·추세·방향 일치) ──
        mtf_feat = self.mtf.get(idx)

        return np.nan_to_num(
            np.concatenate([preds, confs, stats, elite, alpha7, regimes, synth, pos_features, hmm_feat, mtf_feat]),
            0.0
        )

# ═══════════════════════════════════════════════════════════════════════════
# 2-2. 리플레이 버퍼
# ═══════════════════════════════════════════════════════════════════════════
class RegimeReplayBuffer:
    def __init__(self, capacity=300000, target_regimes=None, warmup_steps=14000):
        self.buffer = deque(maxlen=capacity)
        self.target_regimes = target_regimes or []
        self.warmup_steps = warmup_steps
        self._push_count = 0

    def push(self, state, action, reward, next_state, done, current_regimes_dict, in_pos=False):
        self._push_count += 1
        if self._push_count < self.warmup_steps or in_pos:
            self.buffer.append((state, action, reward, next_state, done))
            return
        is_target = any(current_regimes_dict.get(r, 0.0) == 1.0 for r in self.target_regimes)
        if is_target or random.random() < 0.1:
            self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d)

    def __len__(self): return len(self.buffer)

class PrioritizedRegimeReplayBuffer(RegimeReplayBuffer):
    def __init__(self, capacity=200000, target_regimes=None, warmup_steps=14000,
                 alpha=0.8, beta=0.2, beta_anneal_steps=600_000):
        self.target_regimes     = target_regimes or []
        self.warmup_steps       = warmup_steps
        self._push_count        = 0
        self._total_stored      = 0
        self._cap               = capacity
        self._ptr               = 0
        self._size              = 0
        self._buf_s             = None
        self._buf_a             = np.empty(capacity, np.int32)
        self._buf_r             = np.empty(capacity, np.float32)
        self._buf_ns            = None
        self._buf_d             = np.empty(capacity, np.bool_)
        self._priorities        = np.zeros(capacity, np.float32)
        self.alpha              = alpha
        self.beta               = beta
        self._beta_start        = beta
        self._beta_anneal_steps = beta_anneal_steps
        self.max_priority       = 1.0

    def push(self, state, action, reward, next_state, done, current_regimes_dict, in_pos=False):
        self._push_count += 1
        should_add = False
        if self._push_count < self.warmup_steps or in_pos:
            should_add = True
        else:
            is_target = any(current_regimes_dict.get(r, 0.0) == 1.0 for r in self.target_regimes)
            if is_target or random.random() < 0.1:
                should_add = True
        if not should_add:
            return
        if self._buf_s is None:
            sdim = len(state)
            self._buf_s  = np.empty((self._cap, sdim), np.float32)
            self._buf_ns = np.empty((self._cap, sdim), np.float32)
        is_target = any(current_regimes_dict.get(r, 0.0) == 1.0 for r in self.target_regimes)
        init_priority = self.max_priority * (1.5 if (self._push_count < self.warmup_steps and is_target) else 1.0)
        p = self._ptr
        self._buf_s[p]      = state
        self._buf_a[p]      = action
        self._buf_r[p]      = reward
        self._buf_ns[p]     = next_state
        self._buf_d[p]      = done
        self._priorities[p] = init_priority
        self._ptr          = (p + 1) % self._cap
        self._size         = min(self._size + 1, self._cap)
        self._total_stored += 1

    def sample(self, batch_size):
        self.beta = min(1.0, self._beta_start + (1.0 - self._beta_start) * (self._total_stored / self._beta_anneal_steps))
        pri   = self._priorities[:self._size] ** self.alpha
        probs = pri / (pri.sum() + 1e-8)
        indices = np.random.choice(self._size, batch_size, p=probs, replace=True)
        weights = (1.0 / (self._size * probs[indices] + 1e-8)) ** self.beta
        weights = (weights / weights.max()).astype(np.float32)
        return (self._buf_s[indices], self._buf_a[indices],
                self._buf_r[indices], self._buf_ns[indices],
                self._buf_d[indices].astype(np.float32), indices, weights)

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p = float(abs(err) + 1e-6) ** self.alpha
            self._priorities[idx] = p
            if p > self.max_priority:
                self.max_priority = p

    def __len__(self): return self._size

# ═══════════════════════════════════════════════════════════════════════════
# 3. 모델 아키텍처 (RobustIQN, Agent)
# ═══════════════════════════════════════════════════════════════════════════
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))
        self._sigma_init = sigma_init
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self._sigma_init / self.in_features ** 0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self._sigma_init / self.in_features ** 0.5)

    def _f(self, x):
        return x.sign() * x.abs().sqrt()

    def sample_noise(self):
        eps_i = self._f(torch.randn(self.in_features,  device=self.weight_mu.device))
        eps_j = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_epsilon.copy_(eps_j.ger(eps_i))
        self.bias_epsilon.copy_(eps_j)

    def zero_noise(self):
        self.weight_epsilon.zero_()
        self.bias_epsilon.zero_()

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)

class RobustIQN(nn.Module):
    def __init__(self, state_dim, action_dim=2, hidden_dim=128, raw_state_dim=None):
        super().__init__()
        self.action_dim = action_dim
        _raw = raw_state_dim if raw_state_dim is not None else state_dim

        # ── [융합 ④] MarketAttentionEncoder ──
        self.attn_encoder = MarketAttentionEncoder(out_dim=hidden_dim, raw_state_dim=_raw)
        self.feat_extractor = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 64),                     nn.LayerNorm(64),         nn.SiLU()
        )
        self._market_dim    = _raw - 5
        self._raw_state_dim = _raw
        self.context_gate   = nn.Linear(self._market_dim, 64)
        self.phi            = nn.Linear(64, 64)

        self.v_head = nn.Sequential(nn.SiLU(), nn.Linear(64, 1))
        self.a_head = nn.Sequential(nn.SiLU(), NoisyLinear(64, action_dim, sigma_init=0.05))

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()

    def forward(self, state, num_quantiles=8):
        batch_size = state.size(0)

        last_frame_start = state.shape[1] - self._raw_state_dim
        market_feat      = state[:, last_frame_start : last_frame_start + FEATURE_DIM]
        attn_out         = self.attn_encoder(market_feat)

        feat_input = torch.cat([state, attn_out], dim=1)
        feat       = self.feat_extractor(feat_input)

        market_no_pos = state[:, last_frame_start : last_frame_start + self._market_dim]
        gate = torch.sigmoid(self.context_gate(market_no_pos))
        feat = feat * gate

        tau     = torch.rand(batch_size, num_quantiles, 1, device=state.device)
        cos_tau = torch.cos(tau * torch.arange(1, 65, device=state.device).float() * torch.pi)
        phi_x   = self.phi(cos_tau)
        shared  = feat.unsqueeze(1).expand(-1, num_quantiles, -1) * phi_x

        v = self.v_head(shared)
        a = self.a_head(shared)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q, tau


class IQNAgent:
    NUM_QUANTILES   = 32

    def __init__(self, model, lr=5e-5, gamma=0.99, tau=0.005, device='cuda', cvar_threshold=0.25):
        self.model = model
        self.state_dim = model.feat_extractor[0].in_features
        self.target_model = copy.deepcopy(model).to(device)
        self.target_model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.memory = None
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.cvar_threshold = cvar_threshold

    def act(self, state, eps=0.0):
        if eps > 0.0 and random.random() < eps:
            return random.randrange(self.model.action_dim)
        if self.model.training and hasattr(self.model, 'reset_noise'):
            self.model.reset_noise()
        state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_quantiles, tau = self.model(state_ts, num_quantiles=self.NUM_QUANTILES)
            sort_idx = tau[0, :, 0].argsort()
            q_sorted = q_quantiles[0][sort_idx]
            cvar_k   = max(1, int(self.NUM_QUANTILES * self.cvar_threshold))
            q = q_sorted[:cvar_k].mean(dim=0)
        return torch.argmax(q).item()

    def update(self, batch_size):
        if len(self.memory) < batch_size: return
        is_per = isinstance(self.memory, PrioritizedRegimeReplayBuffer)
        if is_per:
            s, a, r, ns, d, per_indices, per_weights = self.memory.sample(batch_size)
            per_w = torch.FloatTensor(per_weights).to(self.device)
        else:
            s, a, r, ns, d = self.memory.sample(batch_size)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d  = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        NQ = self.NUM_QUANTILES
        if hasattr(self.model, 'reset_noise'):
            self.model.reset_noise()
        q, tau_online = self.model(s, num_quantiles=NQ)
        q_a = q.gather(2, a.unsqueeze(1).expand(-1, NQ, -1)).squeeze(2)

        with torch.no_grad():
            self.model.eval()
            next_actions = self.model(ns, num_quantiles=NQ)[0].mean(dim=1).argmax(dim=1, keepdim=True)
            self.model.train()
            q_target, _  = self.target_model(ns, num_quantiles=NQ)
            q_target_a   = q_target.gather(2, next_actions.unsqueeze(1).expand(-1, NQ, -1)).squeeze(2)
            target = r + self.gamma * (1 - d) * q_target_a

        td_error  = target.unsqueeze(1) - q_a.unsqueeze(2)
        huber     = F.huber_loss(td_error, torch.zeros_like(td_error), reduction='none', delta=1.0)
        tau_exp   = tau_online
        indicator = (td_error.detach() < 0).float()
        loss_per_sample = (torch.abs(tau_exp - indicator) * huber).mean(dim=1).mean(dim=1)

        if is_per:
            loss = (loss_per_sample * per_w).mean()
            td_err_np = td_error.detach().abs().mean(dim=(1, 2)).cpu().numpy()
            self.memory.update_priorities(per_indices, td_err_np)
        else:
            loss = loss_per_sample.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        for tp, p in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

# ═══════════════════════════════════════════════════════════════════════════
# 4. REINFORCE 기반 7-Way GatingNet 메타 라우터
# ═══════════════════════════════════════════════════════════════════════════
class GatingNet7(nn.Module):
    """시장 상태 → [flat, bull, bear, chop_L, chop_S, norm_L, norm_S] (7-way softmax)"""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 32), nn.SiLU(),
            nn.Linear(32, 7) 
        )

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits / 0.5, dim=-1)

def _cvar_q_rl(model, state_ts, nq=32, cvar_threshold=0.25):
    with torch.no_grad():
        q_quants, tau = model(state_ts, num_quantiles=nq)
        sort_idx = tau[0, :, 0].argsort()
        k = max(4, int(nq * cvar_threshold))
        return q_quants[0][sort_idx][:k].mean(dim=0)

def _run_gating_trajectory(gating_net, models, df_train, device, n_steps, _ADV_SCALE,
                            hmm_detector=None, mtf_features=None):
    """GatingNet 단일 궤적 실행 → (log_probs, alpha_returns, entropies) 반환.
    궤적을 함수로 분리해 n_trajectories 반복 시 코드 중복 없이 재사용."""
    start    = random.randint(0, max(0, len(df_train) - n_steps - 10))
    env      = TradingEnv(df_train, phase='val', agent_role='neutral', fee=0.0005,
                          hmm_detector=hmm_detector, mtf_features=mtf_features)
    state_np = env.reset(start_idx=start)

    log_probs:   list = []
    rewards:     list = []
    bnh_rewards: list = []
    entropies:   list = []

    for _ in range(n_steps):
        state_ts = torch.FloatTensor(state_np).unsqueeze(0).to(device)

        with torch.no_grad():
            q_bull         = _cvar_q_rl(models['bull'],         state_ts, cvar_threshold=0.60)
            q_bear         = _cvar_q_rl(models['bear'],         state_ts, cvar_threshold=0.40)
            q_chop_long    = _cvar_q_rl(models['chop_long'],    state_ts, cvar_threshold=0.50)
            q_chop_short   = _cvar_q_rl(models['chop_short'],   state_ts, cvar_threshold=0.50)
            q_normal_long  = _cvar_q_rl(models['normal_long'],  state_ts, cvar_threshold=0.50)
            q_normal_short = _cvar_q_rl(models['normal_short'], state_ts, cvar_threshold=0.50)

        def _adv(q):
            best = q.argmax().item()
            if best == 0 or q[best].item() <= 0:
                return torch.tensor(0., device=device)
            return torch.clamp(q[best] - q[0], min=0., max=0.1) * _ADV_SCALE

        w = gating_net(state_ts)[0]  # (7,)
        scores = torch.stack([
            w[0],
            w[1] * (1 + _adv(q_bull)),
            w[2] * (1 + _adv(q_bear)),
            w[3] * (1 + _adv(q_chop_long)),
            w[4] * (1 + _adv(q_chop_short)),
            w[5] * (1 + _adv(q_normal_long)),
            w[6] * (1 + _adv(q_normal_short)),
        ])

        probs    = scores / (scores.sum() + 1e-8)
        probs    = probs * 0.9 + 0.02
        dist     = torch.distributions.Categorical(probs=probs)
        gate_act = dist.sample()
        log_probs.append(dist.log_prob(gate_act))
        entropies.append(dist.entropy())

        g = gate_act.item()
        if   g == 0: env_action = 0
        elif g == 1: env_action = 1 if int(q_bull.argmax().item())         == 1 else 0
        elif g == 2: env_action = 2 if int(q_bear.argmax().item())         == 1 else 0
        elif g == 3: env_action = 1 if int(q_chop_long.argmax().item())    == 1 else 0
        elif g == 4: env_action = 2 if int(q_chop_short.argmax().item())   == 1 else 0
        elif g == 5: env_action = 1 if int(q_normal_long.argmax().item())  == 1 else 0
        else:        env_action = 2 if int(q_normal_short.argmax().item()) == 1 else 0

        cur_idx = env.current_step
        next_state, reward, done, _ = env.step(env_action)
        nxt_idx  = min(env.current_step, len(env._close_np) - 1)
        bnh_step = float((env._close_np[nxt_idx] - env._close_np[cur_idx])
                         / (env._close_np[cur_idx] + 1e-8)) * 500.0

        rewards.append(reward)
        bnh_rewards.append(bnh_step)
        state_np = next_state
        if done:
            break

    if len(rewards) < 2:
        return None

    # BnH 알파 기반 누적 리턴 계산
    G, G_bnh = 0.0, 0.0
    alpha_returns = []
    for r, b in zip(reversed(rewards), reversed(bnh_rewards)):
        G     = r + 0.99 * G
        G_bnh = b + 0.99 * G_bnh
        alpha_returns.insert(0, G - G_bnh)

    ret_t = torch.FloatTensor(alpha_returns).to(device)
    ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

    return torch.stack(log_probs), ret_t, torch.stack(entropies)


def train_gating_step_rl(gating_net, optimizer, models, df_train, device,
                          n_steps=1500, n_trajectories=3, hmm_detector=None, mtf_features=None):
    """REINFORCE로 GatingNet7 학습 (6개 에이전트는 frozen).

    n_trajectories: 궤적 반복 횟수.
    - 단일 궤적(n=1)은 분산이 크고 특정 시장 구간에 과적합될 위험이 있음.
    - 여러 궤적의 loss를 평균 → gradient 분산 감소, 다양한 시장 구간 커버.
    - 에폭당 호출 주기가 짧아진 만큼(ep%10) n_steps를 유지하되 궤적 수로 안정화.
    """
    for m in models.values(): m.eval()
    gating_net.train()

    _ADV_SCALE = 10.0
    total_loss = torch.tensor(0.0, device=device)
    valid_count = 0

    for _ in range(n_trajectories):
        result = _run_gating_trajectory(gating_net, models, df_train, device, n_steps, _ADV_SCALE,
                                        hmm_detector=hmm_detector, mtf_features=mtf_features)
        if result is None:
            continue
        log_probs_t, ret_t, entropies_t = result
        traj_loss = -(log_probs_t * ret_t.detach()).mean() - 0.01 * entropies_t.mean()
        total_loss = total_loss + traj_loss
        valid_count += 1

    if valid_count == 0:
        return 0.0

    loss = total_loss / valid_count  # 궤적 평균 loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gating_net.parameters(), 0.5)
    optimizer.step()
    return loss.item()


class GatingRouter7:
    """GatingNet7 기반 7-Way 메타 라우터 (HMM + Kelly 통합 버전)

    변경 사항:
        - __init__: hmm_detector, kelly_sizer 인자 추가
        - _state_tensor: HMM 피처 5차원 concat (STATE_DIM 확장과 일치)
        - decide: 포지션 진입 시 KellyCriterionSizer로 leverage_rate 결정
                  (기존 adv_n 기반 단순 레버리지 → IQN 분위 기반 Kelly 대체)
        - decide: HMM 상태 정보를 info dict에 추가 (디버깅/모니터링)
    """
    _W_IDX = {'flat': 0, 'bull': 1, 'bear': 2, 'chop_long': 3, 'chop_short': 4, 'normal_long': 5, 'normal_short': 6}
    _CVAR_THRESH = {'bull': 0.60, 'bear': 0.40, 'chop_long': 0.50, 'chop_short': 0.50, 'normal_long': 0.50, 'normal_short': 0.50}

    def __init__(self, models_dict, gating_net, device='cuda',
                 hmm_detector: OnlineHMMDetector = None,
                 kelly_sizer:  KellyCriterionSizer = None,
                 mtf_features: MultiTimeframeFeatures = None):
        self.models       = {k: v.eval() for k, v in models_dict.items()}
        self.gating_net   = gating_net.eval()
        self.device       = device
        self._active_agent = None
        self._frame_stack  = deque(maxlen=STACK_N)
        self.hmm     = hmm_detector
        self.kelly   = kelly_sizer or KellyCriterionSizer()
        self.mtf     = mtf_features

    def _state_tensor(self, features, pos):
        preds   = np.array([features.get(c, 0.) for c in STATE_PRED],   dtype=np.float32)
        confs   = np.array([features.get(c, 0.) for c in STATE_CONF],   dtype=np.float32)
        stats   = np.array([preds.mean(), preds.std(), confs.mean()],    dtype=np.float32)
        elite   = np.array([features.get(c, 0.) for c in STATE_ELITE],  dtype=np.float32)
        alpha7  = np.array([features.get(c, 0.) for c in STATE_ALPHA],  dtype=np.float32)
        regimes = np.array([features.get(c, 0.) for c in REGIME_COLS],  dtype=np.float32)
        synth   = np.array([features.get(c, 0.) for c in STATE_SYNTH],  dtype=np.float32)
        cur_p   = features.get('close', 1.0)
        pt      = pos.get('type')
        pos_arr = np.array([
            1.0 if pt == 'LONG' else (-1.0 if pt == 'SHORT' else 0.0),
            pos.get('entry_price', cur_p) / cur_p - 1 if pt else 0.0,
            pos.get('unrealized', 0.),
            pos.get('mdd', 0.),
            pos.get('hold_norm', 0.)
        ], dtype=np.float32)

        # ── HMM 피처 (5차원) ──
        if self.hmm is not None:
            hmm_feat = self.hmm.get_features(features)
        else:
            hmm_feat = np.zeros(HMM_DIM, dtype=np.float32)

        # ── MTF 피처 (7차원) ──
        if self.mtf is not None:
            # features dict에 현재 스텝 인덱스가 없으므로 close 기반으로 근사
            # val_env에서 직접 호출되는 경우 인덱스를 features에 'step_idx'로 전달
            _step_idx = int(features.get('_step_idx', -1))
            mtf_feat = self.mtf.get(_step_idx)
        else:
            mtf_feat = np.zeros(MTF_DIM, dtype=np.float32)

        raw = np.concatenate([preds, confs, stats, elite, alpha7, regimes, synth, pos_arr, hmm_feat, mtf_feat])
        self._frame_stack.append(raw)
        pad    = STACK_N - len(self._frame_stack)
        frames = [np.zeros(STATE_DIM, np.float32)] * pad + list(self._frame_stack)
        vec    = np.concatenate(frames)
        return torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _cvar_q(self, model, state, agent_name='normal_long'):
        threshold = self._CVAR_THRESH.get(agent_name, 0.50)
        nq = 32
        with torch.no_grad():
            q_quants, tau = model(state, num_quantiles=nq)
            sort_idx = tau[0, :, 0].argsort()
            k = max(4, int(nq * threshold))
            return q_quants[0][sort_idx][:k].mean(dim=0).cpu()

    def _full_quantiles(self, model, state, nq=32):
        """Kelly 계산용 — 전체 분위 행렬 반환 (nq, n_actions)."""
        with torch.no_grad():
            q_quants, tau = model(state, num_quantiles=nq)
            sort_idx = tau[0, :, 0].argsort()
            return q_quants[0][sort_idx].cpu()

    def decide(self, features, pos):
        cur_pos = pos.get('type')
        state   = self._state_tensor(features, pos)

        # ── HMM 상태 스냅샷 (info 기록용) ──
        if self.hmm is not None:
            hmm_probs = self.hmm._alpha.copy()  # (4,) 사후확률
            hmm_state = int(np.argmax(hmm_probs))
            hmm_names = ['bull-trend', 'bear-trend', 'hv-chop', 'lv-range']
            _hmm_info = {'hmm_state': hmm_names[hmm_state],
                         'hmm_probs': hmm_probs.round(3).tolist()}
        else:
            _hmm_info = {}

        # ── 포지션 유지 / 청산 판단 ──
        if cur_pos is not None:
            agent_name = self._active_agent or ('normal_long' if cur_pos == 'LONG' else 'normal_short')
            q_cvar     = self._cvar_q(self.models[agent_name], state, agent_name=agent_name)
            best       = int(q_cvar.argmax().item())

            eval_best = best
            if agent_name in ['bear', 'chop_short', 'normal_short'] and best == 1:
                eval_best = 2

            wants_reverse = (cur_pos == 'LONG' and eval_best == 2) or \
                            (cur_pos == 'SHORT' and eval_best == 1)
            if best == 0 or wants_reverse:
                self._active_agent = None
                return 0, 0.0, {'agent': f'{agent_name}_exit', **_hmm_info}
            else:
                hold_action = 1 if cur_pos == 'LONG' else 2
                return hold_action, 0.0, {'agent': 'HOLD', **_hmm_info}

        # ── 신규 진입 판단 ──
        with torch.no_grad():
            w = self.gating_net(state)[0].cpu()

        q_map = {}
        for name in ['bull', 'bear', 'chop_long', 'chop_short', 'normal_long', 'normal_short']:
            q_map[name] = self._cvar_q(self.models[name], state, agent_name=name)

        def _adv_n(q):
            best = q.argmax().item()
            if best == 0 or float(q[best]) <= 0:
                return 0.0
            return float(min(max(0., q[best] - q[0]), 0.1) * 10.0)

        scores = {
            'flat':         w[0].item(),
            'bull':         w[1].item() * (1 + _adv_n(q_map['bull'])),
            'bear':         w[2].item() * (1 + _adv_n(q_map['bear'])),
            'chop_long':    w[3].item() * (1 + _adv_n(q_map['chop_long'])),
            'chop_short':   w[4].item() * (1 + _adv_n(q_map['chop_short'])),
            'normal_long':  w[5].item() * (1 + _adv_n(q_map['normal_long'])),
            'normal_short': w[6].item() * (1 + _adv_n(q_map['normal_short'])),
        }

        long_edge  = scores['bull'] + scores.get('normal_long', 0.) * 0.5 - scores['flat']
        short_edge = scores['bear'] + scores.get('normal_short', 0.) * 0.5 - scores['flat']
        _edge_info = {'long_edge': long_edge, 'short_edge': short_edge}

        best_name = max(scores, key=scores.get)
        _w_arr = w.numpy()
        if best_name == 'flat':
            return 0, 0.0, {'agent': 'FLAT', **_edge_info, 'kelly': 0.0,
                             'gating_w': _w_arr, **_hmm_info}

        q_sel        = q_map[best_name]
        agent_action = int(q_sel.argmax().item())
        if agent_action == 0:
            return 0, 0.0, {'agent': f'{best_name}_declined', **_edge_info,
                             'kelly': 0.0, 'gating_w': _w_arr, **_hmm_info}

        q_vals = q_sel.float()
        q_std  = q_vals.std().item()
        q_adv  = float(q_vals[agent_action] - q_vals[0])
        q_z    = q_adv / (q_std + 1e-6)

        if q_z < 0.5:
            return 0, 0.0, {'agent': f'{best_name}_low_conviction(z={q_z:.2f})',
                             **_edge_info, 'kelly': 0.0, 'gating_w': _w_arr, **_hmm_info}

        # ── Kelly 사이징 ──
        gating_conf = float(w[self._W_IDX[best_name]].item())
        q_full      = self._full_quantiles(self.models[best_name], state)
        lev         = self.kelly.compute(q_full, gating_confidence=gating_conf)
        kelly_stats = self.kelly.log_stats(q_full)

        val_action = agent_action
        if best_name in ['bear', 'chop_short', 'normal_short'] and agent_action == 1:
            val_action = 2

        self._active_agent = best_name
        return val_action, lev, {
            'agent':    best_name,
            'score':    scores[best_name],
            'kelly':    lev,
            **kelly_stats,
            **_edge_info,
            'gating_w': _w_arr,
            **_hmm_info,
        }



def train():
    CSV_PATH = 'data/ensemble/rl_training_data_full.csv'
    if not os.path.exists(CSV_PATH):
        return logger.error("데이터가 없습니다. --mode generate_csv 실행 요망")

    df = pd.read_csv(CSV_PATH)
    split_idx = int(len(df) * 0.8)
    df_train  = df.iloc[:split_idx].reset_index(drop=True)
    df_val    = df.iloc[split_idx:].reset_index(drop=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df_train_reg = df_train[REGIME_COLS].values.astype(np.float32)

    MAX_EP    = 4096
    _safe_end = len(df_train) - MAX_EP - 1

    agent_configs = {
        'bull':         {'action_dim': 2, 'target_regimes': ['regime_bull'],                   'cvar_threshold': 0.60},
        'bear':         {'action_dim': 2, 'target_regimes': ['regime_bear'],                   'cvar_threshold': 0.40},
        'chop_long':    {'action_dim': 2, 'target_regimes': ['regime_chop', 'regime_whipsaw'], 'cvar_threshold': 0.50},
        'chop_short':   {'action_dim': 2, 'target_regimes': ['regime_chop', 'regime_whipsaw'], 'cvar_threshold': 0.50},
        'normal_long':  {'action_dim': 2, 'target_regimes': ['regime_normal'],                 'cvar_threshold': 0.50},
        'normal_short': {'action_dim': 2, 'target_regimes': ['regime_normal'],                 'cvar_threshold': 0.50},
    }
    agent_names = list(agent_configs.keys())

    ri = {r: REGIME_COLS.index(f'regime_{r}') for r in ['bull', 'bear', 'chop', 'whipsaw', 'normal']}
    
    regime_starts = {
        'bull':         [i for i in range(_safe_end) if df_train_reg[i, ri['bull']] == 1.0],
        'bear':         [i for i in range(_safe_end) if df_train_reg[i, ri['bear']] == 1.0],
        'chop_long':    [i for i in range(_safe_end) if df_train_reg[i, ri['chop']] == 1.0 or df_train_reg[i, ri['whipsaw']] == 1.0],
        'chop_short':   [i for i in range(_safe_end) if df_train_reg[i, ri['chop']] == 1.0 or df_train_reg[i, ri['whipsaw']] == 1.0],
        'normal_long':  [i for i in range(_safe_end) if df_train_reg[i, ri['normal']] == 1.0],
        'normal_short': [i for i in range(_safe_end) if df_train_reg[i, ri['normal']] == 1.0],
    }
    for name in agent_names:
        if len(regime_starts[name]) < 100:
            regime_starts[name] = list(range(_safe_end))

    envs, models, agents = {}, {}, {}

    # ── [HMM 초기화 및 학습] 환경 생성 전에 훈련 데이터로 fit ──
    logger.info("[HMM] OnlineHMMDetector 초기 학습 시작...")
    hmm_detector = OnlineHMMDetector()
    hmm_detector.fit(df_train, n_iter=30)
    kelly_sizer  = KellyCriterionSizer(half_kelly=0.5, min_lev=0.1, max_lev=1.0)
    logger.info("[HMM] 초기 학습 완료. Kelly Sizer 초기화 완료.")

    # ── [MTF 선계산] 훈련/검증 데이터 각각 1회 계산 후 공유 ──
    logger.info("[MTF] 훈련 데이터 멀티타임프레임 피처 선계산 중...")
    mtf_train = MultiTimeframeFeatures(df_train['close'].values.astype(np.float32))
    logger.info("[MTF] 검증 데이터 멀티타임프레임 피처 선계산 중...")
    mtf_val   = MultiTimeframeFeatures(df_val['close'].values.astype(np.float32))
    logger.info("[MTF] 선계산 완료.")

    for name, cfg in agent_configs.items():
        envs[name] = TradingEnv(df_train, phase='train', agent_role=name, fee=0.0005,
                                hmm_detector=hmm_detector, mtf_features=mtf_train)
        models[name] = RobustIQN(STACKED_STATE_DIM, cfg['action_dim'], raw_state_dim=STATE_DIM).to(device)
        agents[name] = IQNAgent(models[name], device=device, cvar_threshold=cfg['cvar_threshold'])
        agents[name].memory = PrioritizedRegimeReplayBuffer(
            200000, target_regimes=cfg['target_regimes'])


    NEP             = 1000
    BATCH           = 512
    UPDATE_FREQ     = 64
    MIN_BUFFER      = 2048
    global_step     = 0
    EPS_START       = 1.0
    EPS_END         = 0.01
    EPS_DECAY_STEPS = 400000

    os.makedirs('data/ensemble', exist_ok=True)
    best_val_pnl   = -float('inf')
    best_val_score = -float('inf')
    val_pnl_history: list = []
    start_ep       = 1
    os.makedirs('data/ensemble/ckpt', exist_ok=True)
    CHECKPOINT_PATH = 'data/ensemble/ckpt/rl_checkpoint.pth'

    gating_net       = GatingNet7(STACKED_STATE_DIM).to(device)
    gating_optimizer = torch.optim.Adam(gating_net.parameters(), lr=1e-3)

    def _save_checkpoint(epoch):
        save_dict = {
            'global_step': global_step, 'best_val_pnl': best_val_pnl,
            'best_val_score': best_val_score, 'val_pnl_history': val_pnl_history, 'epoch': epoch,
            'gating_net': gating_net.state_dict(),
            'gating_opt': gating_optimizer.state_dict(),
        }
        for name in agent_names:
            save_dict[f'model_{name}'] = models[name].state_dict()
            save_dict[f'opt_{name}'] = agents[name].optimizer.state_dict()
        torch.save(save_dict, CHECKPOINT_PATH)

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        arch_ok = True
        for name in agent_names:
            try:
                models[name].load_state_dict(ckpt[f'model_{name}'], strict=False)
                agents[name].target_model.load_state_dict(models[name].state_dict(), strict=False)
                agents[name].optimizer.load_state_dict(ckpt[f'opt_{name}'])
            except RuntimeError as e:
                logger.warning(f"⚠️ [{name}] 아키텍처 불일치로 가중치 스킵 (처음부터 학습): {e}")
                arch_ok = False
        global_step, best_val_pnl = ckpt['global_step'], ckpt['best_val_pnl']
        best_val_score, val_pnl_history = ckpt['best_val_score'], ckpt.get('val_pnl_history', [])
        start_ep = ckpt['epoch'] + 1 if arch_ok else 1
        
        if 'gating_net' in ckpt:
            try:
                gating_net.load_state_dict(ckpt['gating_net'], strict=False)
                gating_optimizer.load_state_dict(ckpt['gating_opt'])
                logger.info("✅ GatingNet7 복원 완료")
            except RuntimeError as e:
                logger.warning("⚠️ GatingNet 아키텍처 불일치. 새로 초기화합니다.")
        else:
            logger.info("⚠️ 체크포인트에 gating_net 없음 → 새로 초기화")
            
        if arch_ok:
            logger.info(f"♻️ [복원] ep={ckpt['epoch']} → {start_ep} | best_pnl={best_val_pnl:.2f}%")
        else:
            logger.info(f"🆕 [아키텍처 변경] 가중치 초기화 후 ep=1 부터 재학습")

    try:
        for ep in range(start_ep, NEP + 1):
            def _sample_start(agent_name):
                pool = regime_starts[agent_name]
                if pool and random.random() < 0.7:
                    return random.choice(pool)
                return random.randint(0, _safe_end)

            start_idx = _sample_start(random.choice(agent_names))

            env_states = {}
            ep_rewards  = {name: 0.0 for name in agent_names}

            for name in agent_names:
                env_states[name] = envs[name].reset(start_idx)

            eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (global_step / EPS_DECAY_STEPS))
            done = False

            while not done:
                global_step += 1
                eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (global_step / EPS_DECAY_STEPS))

                idx = envs[agent_names[0]].current_step
                if idx >= envs[agent_names[0]].end_step or idx >= len(df_train) - 1:
                    break

                for name in agent_names:
                    env, agent, s = envs[name], agents[name], env_states[name]
                    current_regimes = {r: float(df_train_reg[env.current_step, col_i]) for col_i, r in enumerate(REGIME_COLS)}

                    was_in_pos = env.pos is not None
                    a = agent.act(s, eps)
                    ns, r, d, _ = env.step(a)

                    in_pos = was_in_pos or (env.pos is not None)
                    actually_idle = not was_in_pos and env.pos is None
                    if actually_idle:
                        agent.memory.push(s, a, r, ns, d, current_regimes, in_pos=False)
                        ep_rewards[name] += r
                    else:
                        agent.memory.push(s, a, r, ns, d, current_regimes, in_pos=in_pos)
                        ep_rewards[name] += r


                    env_states[name] = ns
                    done = done or d


                if global_step % UPDATE_FREQ == 0:
                    for name in agent_names:
                        if len(agents[name].memory) >= MIN_BUFFER: agents[name].update(BATCH)

            _SIGMA_FLOOR = 0.05 / (64 ** 0.5)
            for name in agent_names:
                env = envs[name]
                pnl = (env.balance / 10000 - 1) * 100
                noisy_layers = [m for m in models[name].modules() if isinstance(m, NoisyLinear)]
                current_floor = _SIGMA_FLOOR if pnl > -3.0 else _SIGMA_FLOOR * 5
                for nl in noisy_layers:
                    nl.weight_sigma.data.clamp_(min=current_floor)
                    nl.bias_sigma.data.clamp_(min=current_floor)
                avg_sigma = sum(m.weight_sigma.abs().mean().item() for m in noisy_layers) / max(1, len(noisy_layers))
                logger.info(
                    f"Ep {ep:04d} [{name:12s}] "
                    f"PnL:{pnl:6.1f}% Tr:{env.total_trades:4d} WR:{env.win_rate*100:4.0f}% "
                    f"Rew:{ep_rewards[name]:7.3f} | "
                    f"buf:{len(agents[name].memory):6d} | eps:{eps:.3f} | σ:{avg_sigma:.4f}"
                )

            # GatingNet 학습: ep%10, ep>=50부터
            # 기존 ep%50(19회/1000ep) → ep%10(95회/1000ep)으로 주기 단축
            # 에이전트 업데이트 밀도(ep당 64회) 대비 GatingNet(ep당 0.02회)의
            # 3,200:1 불균형을 32:1 수준으로 완화
            # ep>=50: 에이전트가 ~3,200회 업데이트된 후 gating 학습 시작 (커리큘럼 유지)
            if ep % 10 == 0 and ep >= 50:
                g_loss = train_gating_step_rl(
                    gating_net, gating_optimizer, models, df_train, device,
                    n_steps=1500, n_trajectories=3, hmm_detector=hmm_detector,
                    mtf_features=mtf_train)
                logger.info(f"    [GATING] ep={ep} loss={g_loss:.4f} (3 trajectories)")

            if ep % 10 == 0:
                router  = GatingRouter7(models, gating_net, device,
                                        hmm_detector=hmm_detector,
                                        kelly_sizer=kelly_sizer,
                                        mtf_features=mtf_val)
                val_env = TradingEnv(df_val, phase='val', agent_role='neutral', fee=0.0005,
                                     hmm_detector=hmm_detector, mtf_features=mtf_val)
                obs, d = val_env.reset(), False
                _REGIME_NAMES = ['chop', 'whipsaw', 'bull', 'bear', 'normal']
                attr_w   = {r: [] for r in _REGIME_NAMES}
                kelly_log = []   # Kelly 통계 수집

                while not d:
                    feat = df_val.iloc[val_env.current_step].to_dict()
                    feat['_step_idx'] = val_env.current_step   # MTF 인덱스용
                    pos_info = {
                        'type': val_env.pos, 'entry_price': val_env.entry_price,
                        'unrealized': val_env.unrealized_pnl, 'mdd': val_env.max_drawdown,
                        'hold_norm': val_env.hold_count / 144
                    }
                    action, leverage_rate, info = router.decide(feat, pos_info)
                    if 'gating_w' in info:
                        regime_vals = [feat.get(c, 0.) for c in REGIME_COLS]
                        dom_idx     = int(np.argmax(regime_vals))
                        dom_regime  = _REGIME_NAMES[dom_idx]
                        attr_w[dom_regime].append(info['gating_w'])
                    # Kelly 통계 수집 (진입 시만)
                    if action in (1, 2) and 'win_rate' in info:
                        kelly_log.append({
                            'lev': leverage_rate, 'wr': info.get('win_rate', 0),
                            'payoff': info.get('payoff', 0), 'f_star': info.get('f_star', 0)
                        })
                    obs, _, d, _ = val_env.step(action, leverage_rate=leverage_rate)

                val_pnl_pct = (val_env.balance / 10000 - 1) * 100
                val_pnl_history.append(val_pnl_pct)
                if len(val_pnl_history) >= 3:
                    _ph = val_pnl_history[-10:]
                    _std = max(float(np.std(_ph)), 0.1)
                    sharpe_est = float(np.clip(np.mean(_ph) / _std, -10.0, 10.0))
                else:
                    sharpe_est = 0.0
                
                if val_pnl_pct > 0:
                    trade_activity = min(val_env.total_trades / 30.0, 1.0) * 5.0
                else:
                    trade_activity = -min(val_env.total_trades / 30.0, 1.0) * 10.0
                    
                val_score = (val_pnl_pct * 5.0) + (val_env.win_rate * 20.0) + (sharpe_est * 5.0) + trade_activity

                logger.info(f"    [VAL] PnL:{val_pnl_pct:.2f}% | Tr:{val_env.total_trades} | WR:{val_env.win_rate*100:.0f}% | Score:{val_score:.2f} (act:{trade_activity:.1f})")

                # HMM 현재 상태 로그
                hmm_s = int(np.argmax(hmm_detector._alpha))
                hmm_names = ['bull-trend', 'bear-trend', 'hv-chop', 'lv-range']
                logger.info(f"    [HMM]  state={hmm_names[hmm_s]} | probs={hmm_detector._alpha.round(3).tolist()}")

                # MTF 현재 상태 로그 (val 마지막 스텝)
                _last_mtf = mtf_val.get(len(df_val) - 1)
                logger.info(
                    f"    [MTF]  1h_ret:{_last_mtf[0]:.3f} 1h_trend:{_last_mtf[2]:.3f} "
                    f"4h_ret:{_last_mtf[3]:.3f} 4h_trend:{_last_mtf[5]:.3f} align:{_last_mtf[6]:.0f}"
                )



                # Kelly 통계 로그
                if kelly_log:
                    avg_lev = np.mean([k['lev'] for k in kelly_log])
                    avg_wr  = np.mean([k['wr']  for k in kelly_log])
                    avg_po  = np.mean([k['payoff'] for k in kelly_log])
                    avg_fs  = np.mean([k['f_star'] for k in kelly_log])
                    logger.info(f"    [KELLY] n={len(kelly_log)} | avg_lev:{avg_lev:.3f} | wr:{avg_wr:.3f} | payoff:{avg_po:.3f} | f*:{avg_fs:.3f}")
                
                for reg, ws in attr_w.items():
                    if ws:
                        mean_w = np.stack(ws).mean(axis=0)
                        logger.info(
                            f"    [ATTR/{reg:7s}] n={len(ws):4d} | "
                            f"flat:{mean_w[0]:.3f} B:{mean_w[1]:.3f} b:{mean_w[2]:.3f} "
                            f"cL:{mean_w[3]:.3f} cS:{mean_w[4]:.3f} nL:{mean_w[5]:.3f} nS:{mean_w[6]:.3f}"
                        )

                if val_score > best_val_score:
                    best_val_score, best_val_pnl = val_score, val_pnl_pct
                    save_dict = {'best_pnl': best_val_pnl, 'epoch': ep,
                                 'gating_net': gating_net.state_dict()}
                    for name in agent_names: save_dict[f'model_{name}'] = models[name].state_dict()
                    torch.save(save_dict, 'data/ensemble/ckpt/best_rl_agents.pth')
                    logger.info(f"    🎉 [NEW BEST] 저장 완료 (PnL:{best_val_pnl:.2f}%)")

                # HMM 온라인 업데이트 — 검증 구간 관측으로 파라미터 점진 갱신
                if ep % 50 == 0:
                    hmm_detector.update_online(n_iter=5)
                    logger.info("    [HMM]  온라인 업데이트 완료")

                _save_checkpoint(ep)

    except KeyboardInterrupt:
        logger.info("⚠️ 학습 중단. 체크포인트 저장 완료.")
        _save_checkpoint(ep)

if __name__ == "__main__":
    train()