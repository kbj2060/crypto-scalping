"""
Arbiter + Gatekeeper — 듀얼 브레인 통합 레이어
================================================================================
역할:
    Brain A (MoE RL)와 Brain B (Trend Transformer)의 출력을 룰 기반으로 통합하여
    최종 진입 가부, Kelly 배율 조정, 포지션 방향을 결정한다.

    ┌──────────────┐     ┌──────────────────┐
    │  Brain A     │     │  Brain B         │
    │  action(0/1/2)│    │  trend_dir(0/1/2)│
    │  kelly_lev   │────▶│  strength        │
    │  agent_name  │     │  rev_prob        │
    └──────────────┘     └──────────────────┘
              │                    │
              └────────┬───────────┘
                       ▼
               ┌───────────────┐
               │   Arbiter     │  (단순 규칙, ML 아님)
               │               │
               │ A-long+B-up   │──→ Kelly × 1.2 (boost)
               │ A-long+B-down │──→ VETO (flat 강제)
               │ A-short+B-down│──→ Kelly × 1.2
               │ A-short+B-up  │──→ VETO
               │ B rev_prob↑   │──→ Kelly × (1 - rev_prob×0.5)
               └───────────────┘
                       │
                       ▼
               ┌───────────────┐
               │  Gatekeeper   │  (6개 게이트, 순차 적용)
               │  Gate1: Trend │  추세 방향 필터
               │  Gate2: Epist │  불확실성 차단 (Brain A 연동)
               │  Gate3: Vol   │  변동성 과열 차단
               │  Gate4: Rev   │  반전 리스크 축소
               │  Gate5: MDD   │  포트폴리오 드로다운 차단
               │  Gate6: Regime│  레짐 불일치 차단
               └───────────────┘
                       │
                       ▼
              (final_action, final_lev, gate_log)

설계 원칙:
    - Arbiter: "두 브레인이 동의하면 베팅을 키우고, 충돌하면 쉰다"
      → 합의 시 Kelly × 1.2 boost, 반대 시 veto
      → 강도(strength) 반영: 강한 추세 신호일수록 boost 폭 확대

    - Gatekeeper: "6개 독립 필터를 모두 통과해야 진입 허용"
      → AND 논리: 하나라도 거부 시 flat 강제
      → 각 게이트는 독립적으로 lev 조정 가능 (누적 곱 적용)
      → 게이트 로그로 차단 원인 추적 가능

    - 전체 파이프라인은 순수 함수 (상태 최소화, 재현성 보장)
      → 유일한 상태: portfolio_mdd (Gate5), running stats (Gate3)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────
# 입력 데이터클래스
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class BrainAOutput:
    """Brain A (MoE RL GatingRouter7)의 출력.

    GatingRouter7.decide()가 반환하는 (action, leverage_rate, info)를
    이 클래스로 래핑해서 Arbiter에 전달한다.
    """
    action      : int    # 0=flat, 1=long, 2=short
    leverage    : float  # Kelly 사이징 결과 [0.1, 1.0]
    agent_name  : str    # 활성 에이전트명 ('bull', 'bear', ...)
    score       : float  # GatingNet 점수
    epist_std   : float  # 앙상블 epistemic 불확실성
    epist_gate  : str    # 'OK' / 'REDUCED' / 'BLOCKED'
    kelly_wr    : float  # Kelly win_rate 추정
    kelly_payoff: float  # Kelly payoff_ratio 추정
    raw_info    : dict = field(default_factory=dict)

    @classmethod
    def from_router_output(cls, action: int, leverage: float, info: dict) -> 'BrainAOutput':
        """GatingRouter7.decide() 반환값에서 직접 생성."""
        return cls(
            action       = action,
            leverage     = leverage,
            agent_name   = info.get('agent', 'unknown'),
            score        = float(info.get('score', 0.0)),
            epist_std    = float(info.get('epist_std', 0.0)),
            epist_gate   = info.get('epist_gate', 'OK'),
            kelly_wr     = float(info.get('win_rate', 0.5)),
            kelly_payoff = float(info.get('payoff', 1.0)),
            raw_info     = info,
        )


@dataclass
class BrainBOutput:
    """Brain B (TrendContextBrain)의 출력.

    TrendContextBrain.predict()가 반환하는 TrendSignal을 직렬화한 dict,
    또는 TrendSignal 인스턴스에서 직접 생성한다.
    """
    trend_dir   : int    # 0=DOWN, 1=FLAT, 2=UP
    strength    : float  # [0, 1] 추세 강도
    rev_prob    : float  # [0, 1] 반전 확률
    p_up        : float  # UP 확률
    p_down      : float  # DOWN 확률
    p_flat      : float  # FLAT 확률

    @classmethod
    def from_signal(cls, signal) -> 'BrainBOutput':
        """TrendSignal 인스턴스에서 생성."""
        return cls(
            trend_dir = signal.trend_dir,
            strength  = signal.strength,
            rev_prob  = signal.rev_prob,
            p_up      = signal.probs[2],
            p_down    = signal.probs[0],
            p_flat    = signal.probs[1],
        )

    @classmethod
    def from_dict(cls, d: dict) -> 'BrainBOutput':
        """to_arbiter_dict() 결과 dict에서 생성."""
        return cls(
            trend_dir = int(d.get('trend_dir', 1)),
            strength  = float(d.get('strength', 0.0)),
            rev_prob  = float(d.get('rev_prob', 0.0)),
            p_up      = float(d.get('p_up', 0.333)),
            p_down    = float(d.get('p_down', 0.333)),
            p_flat    = float(d.get('p_flat', 0.333)),
        )

    @property
    def is_up(self)   -> bool: return self.trend_dir == 2
    @property
    def is_down(self) -> bool: return self.trend_dir == 0
    @property
    def is_flat(self) -> bool: return self.trend_dir == 1


@dataclass
class ArbiterDecision:
    """Arbiter + Gatekeeper의 최종 결정."""
    final_action : int    # 0=flat, 1=long, 2=short
    final_lev    : float  # 최종 Kelly 배율 [0, 1]
    gate_passed  : bool   # 전체 게이트 통과 여부
    arbiter_mode : str    # 'BOOST' / 'NEUTRAL' / 'VETO' / 'FLAT_INPUT'
    lev_multiplier: float # Arbiter가 적용한 Kelly 배율 조정 계수
    gate_log     : Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        sym = {0: '⏸', 1: '🔼', 2: '🔽'}
        return (f"ArbiterDecision({sym.get(self.final_action, '?')} "
                f"action={self.final_action} lev={self.final_lev:.3f} "
                f"mode={self.arbiter_mode} gates={'✅' if self.gate_passed else '❌'})")


# ───────────────────────────────────────────────────────────────────────────
# Arbiter — 두 브레인 합의/충돌 처리
# ───────────────────────────────────────────────────────────────────────────
class Arbiter:
    """두 브레인의 방향 합의·충돌을 룰로 판단하여 Kelly 배율을 조정.

    규칙 (우선순위 순):
        1. Brain A가 flat(0)이면 → flat 유지 (Arbiter 관여 안 함)
        2. 방향 일치 + strength 강함 → BOOST (Kelly × boost_factor)
        3. 방향 완전 반대 → VETO (flat 강제)
        4. Brain B가 FLAT이고 strength 낮음 → NEUTRAL (Brain A 그대로)
        5. 부분 반대(Brain B가 약한 FLAT) → CAUTION (Kelly 소폭 축소)
        6. rev_prob 높음 → 추가 Kelly 축소

    파라미터:
        boost_factor   : 합의 시 Kelly 증폭 계수 (기본 1.2)
        strength_boost : strength에 따른 추가 증폭 최대치 (기본 0.1)
        veto_threshold : Brain B strength가 이 값 이상일 때만 VETO 발동 (기본 0.4)
        caution_factor : 부분 반대 시 Kelly 축소 계수 (기본 0.7)
        rev_penalty_max: rev_prob 패널티 최대치 (기본 0.4)
        max_lev        : 최종 Kelly 하드캡 (기본 1.0)
    """

    def __init__(
        self,
        boost_factor    : float = 1.2,
        strength_boost  : float = 0.1,
        veto_threshold  : float = 0.4,
        caution_factor  : float = 0.7,
        rev_penalty_max : float = 0.4,
        max_lev         : float = 1.0,
    ):
        self.boost_factor    = boost_factor
        self.strength_boost  = strength_boost
        self.veto_threshold  = veto_threshold
        self.caution_factor  = caution_factor
        self.rev_penalty_max = rev_penalty_max
        self.max_lev         = max_lev

        # 통계 추적
        self._boost_count   = 0
        self._veto_count    = 0
        self._neutral_count = 0
        self._caution_count = 0

    def _direction_agree(self, a_action: int, b: BrainBOutput) -> bool:
        """Brain A와 Brain B 방향이 일치하는가."""
        if a_action == 1 and b.is_up:   return True
        if a_action == 2 and b.is_down: return True
        return False

    def _direction_oppose(self, a_action: int, b: BrainBOutput) -> bool:
        """Brain A와 Brain B 방향이 정반대인가."""
        if a_action == 1 and b.is_down: return True
        if a_action == 2 and b.is_up:   return True
        return False

    def adjudicate(self, brain_a: BrainAOutput, brain_b: BrainBOutput) -> Tuple[int, float, str, float]:
        """
        Args:
            brain_a: Brain A 출력
            brain_b: Brain B 출력

        Returns:
            (action, adjusted_lev, mode, lev_multiplier)
        """
        # ── Rule 0: Brain A가 이미 flat ──
        if brain_a.action == 0:
            return 0, 0.0, 'FLAT_INPUT', 1.0

        # ── 방향 관계 판정 ──
        agree  = self._direction_agree(brain_a.action, brain_b)
        oppose = self._direction_oppose(brain_a.action, brain_b)

        base_lev = brain_a.leverage

        # ── Rule 1: 방향 완전 반대 + Brain B 강도 충분 → VETO ──
        if oppose and brain_b.strength >= self.veto_threshold:
            self._veto_count += 1
            return 0, 0.0, 'VETO', 0.0

        # ── Rule 2: 방향 일치 → BOOST ──
        if agree:
            # strength에 비례한 추가 증폭
            extra  = self.strength_boost * brain_b.strength
            factor = min(self.boost_factor + extra, self.boost_factor + self.strength_boost)
            # rev_prob 패널티 차감 (반전 가능성이 높으면 boost를 줄임)
            rev_pen = self.rev_penalty_max * brain_b.rev_prob
            factor  = max(1.0, factor - rev_pen)
            adj_lev = float(np.clip(base_lev * factor, 0.0, self.max_lev))
            self._boost_count += 1
            return brain_a.action, adj_lev, 'BOOST', factor

        # ── Rule 3: Brain B가 FLAT (방향 없음) → NEUTRAL ──
        if brain_b.is_flat or brain_b.strength < 0.2:
            # rev_prob가 높으면 약간 축소
            rev_pen = self.rev_penalty_max * brain_b.rev_prob * 0.5
            factor  = float(np.clip(1.0 - rev_pen, 0.6, 1.0))
            adj_lev = float(np.clip(base_lev * factor, 0.0, self.max_lev))
            self._neutral_count += 1
            return brain_a.action, adj_lev, 'NEUTRAL', factor

        # ── Rule 4: 약한 반대 (oppose but strength < veto_threshold) → CAUTION ──
        if oppose and brain_b.strength < self.veto_threshold:
            t      = brain_b.strength / self.veto_threshold   # 0~1
            factor = float(np.clip(1.0 - (1.0 - self.caution_factor) * t, self.caution_factor, 1.0))
            adj_lev = float(np.clip(base_lev * factor, 0.0, self.max_lev))
            self._caution_count += 1
            return brain_a.action, adj_lev, 'CAUTION', factor

        # ── Fallback: NEUTRAL ──
        self._neutral_count += 1
        return brain_a.action, base_lev, 'NEUTRAL', 1.0

    def stats(self) -> dict:
        total = self._boost_count + self._veto_count + self._neutral_count + self._caution_count
        return {
            'arbiter_boost'  : self._boost_count,
            'arbiter_veto'   : self._veto_count,
            'arbiter_neutral': self._neutral_count,
            'arbiter_caution': self._caution_count,
            'arbiter_total'  : total,
        }


# ───────────────────────────────────────────────────────────────────────────
# Gatekeeper — 6개 순차 게이트
# ───────────────────────────────────────────────────────────────────────────
class Gate1TrendFilter:
    """[Gate1] 추세 방향 필터.

    목적: Brain B의 추세 방향이 진입 방향과 심하게 충돌할 때 차단.
          (Arbiter는 Kelly를 조정하지만 Gate1은 진입 자체를 막음)

    발동 조건:
        - 진입 방향과 반대 trend + strength > HARD_VETO_STRENGTH → BLOCK
        - 진입 방향과 반대 trend + strength in (SOFT_RANGE) → LEV 축소

    파라미터:
        hard_veto_strength : 이 강도 이상이면 완전 차단 (기본 0.7)
        soft_lev_factor    : 약한 반대 방향 시 레버리지 배율 (기본 0.6)
    """

    def __init__(self, hard_veto_strength: float = 0.7, soft_lev_factor: float = 0.6):
        self.hard_veto_strength = hard_veto_strength
        self.soft_lev_factor    = soft_lev_factor
        self._block_count = 0
        self._soft_count  = 0

    def check(self, action: int, lev: float, brain_b: BrainBOutput) -> Tuple[bool, float, dict]:
        """
        Returns:
            (allow: bool, adjusted_lev: float, log: dict)
        """
        # flat은 Gate 적용 불필요
        if action == 0:
            return True, lev, {'gate1': 'SKIP_FLAT'}

        is_long  = (action == 1)
        is_short = (action == 2)

        # 강한 반대 추세 → BLOCK
        if (is_long  and brain_b.is_down and brain_b.strength >= self.hard_veto_strength) or \
           (is_short and brain_b.is_up   and brain_b.strength >= self.hard_veto_strength):
            self._block_count += 1
            return False, 0.0, {
                'gate1': 'BLOCK',
                'reason': f'hard_oppose trend_dir={brain_b.trend_dir} str={brain_b.strength:.2f}'
            }

        # 약한 반대 추세 → 레버리지 축소
        if (is_long  and brain_b.is_down and brain_b.strength > 0.2) or \
           (is_short and brain_b.is_up   and brain_b.strength > 0.2):
            adj_lev = float(np.clip(lev * self.soft_lev_factor, 0.0, lev))
            self._soft_count += 1
            return True, adj_lev, {
                'gate1': 'SOFT_REDUCE',
                'adj_lev': round(adj_lev, 3),
                'reason': f'soft_oppose str={brain_b.strength:.2f}'
            }

        return True, lev, {'gate1': 'PASS'}


class Gate2EpistemicFilter:
    """[Gate2] 앙상블 Epistemic 불확실성 필터.

    Brain A의 EpistemicUncertaintyGate와 중복되지만,
    Arbiter 이후 lev가 달라질 수 있으므로 독립적으로 재확인한다.

    발동 조건:
        - epist_gate == 'BLOCKED' → 완전 차단
        - epist_gate == 'REDUCED' → lev 유지 (Brain A에서 이미 축소됨)
        - epist_std > EXTRA_THRESHOLD → 추가 축소 (Brain A threshold보다 엄격)
    """

    EXTRA_THRESHOLD = 0.12   # Brain A의 HIGH_THRESH(0.15)보다 낮게 설정

    def __init__(self):
        self._block_count = 0

    def check(self, action: int, lev: float, brain_a: BrainAOutput) -> Tuple[bool, float, dict]:
        if action == 0:
            return True, lev, {'gate2': 'SKIP_FLAT'}

        if brain_a.epist_gate == 'BLOCKED':
            self._block_count += 1
            return False, 0.0, {'gate2': 'BLOCK', 'reason': 'epist_gate=BLOCKED'}

        # Brain A에서 이미 처리된 경우라도 extra threshold 초과 시 재축소
        if brain_a.epist_std > self.EXTRA_THRESHOLD:
            t      = (brain_a.epist_std - self.EXTRA_THRESHOLD) / 0.10
            factor = float(np.clip(1.0 - 0.4 * t, 0.5, 1.0))
            adj_lev = float(np.clip(lev * factor, 0.0, lev))
            return True, adj_lev, {
                'gate2': 'EXTRA_REDUCE',
                'epist_std': round(brain_a.epist_std, 4),
                'adj_lev': round(adj_lev, 3),
            }

        return True, lev, {'gate2': 'PASS', 'epist_std': round(brain_a.epist_std, 4)}


class Gate3VolatilityFilter:
    """[Gate3] 변동성 과열 필터.

    GARCH 변동성이 장기 평균 대비 급등하면 포지션 사이즈 축소.
    Brain B의 strength는 방향성이지 변동성 자체가 아니므로 별도 게이트가 필요.

    발동 조건:
        - garch_vol_z > HIGH_Z (기본 2.5) → 진입 차단
        - garch_vol_z > MED_Z  (기본 1.5) → lev × 0.6
        - garch_vol_z > LOW_Z  (기본 0.8) → lev × 0.85

    상태:
        running_vol_z: 최근 50 스텝 vol_z 이동평균 (과열 기준 동적 조정)
    """

    HIGH_Z = 2.5
    MED_Z  = 1.5
    LOW_Z  = 0.8

    def __init__(self):
        self._vol_z_history = deque(maxlen=50)
        self._block_count = 0
        self._reduce_count = 0

    def check(self, action: int, lev: float, garch_vol_z: float) -> Tuple[bool, float, dict]:
        self._vol_z_history.append(garch_vol_z)
        if action == 0:
            return True, lev, {'gate3': 'SKIP_FLAT'}

        # 이동평균 대비 상대 z score
        mean_z = float(np.mean(self._vol_z_history)) if self._vol_z_history else 0.0
        rel_z  = garch_vol_z - mean_z   # 평균 대비 초과분

        if rel_z > self.HIGH_Z:
            self._block_count += 1
            return False, 0.0, {'gate3': 'BLOCK', 'garch_vol_z': round(garch_vol_z, 3), 'rel_z': round(rel_z, 2)}

        if rel_z > self.MED_Z:
            adj_lev = float(np.clip(lev * 0.6, 0.0, lev))
            self._reduce_count += 1
            return True, adj_lev, {'gate3': 'REDUCE_60', 'garch_vol_z': round(garch_vol_z, 3)}

        if rel_z > self.LOW_Z:
            adj_lev = float(np.clip(lev * 0.85, 0.0, lev))
            self._reduce_count += 1
            return True, adj_lev, {'gate3': 'REDUCE_85', 'garch_vol_z': round(garch_vol_z, 3)}

        return True, lev, {'gate3': 'PASS', 'garch_vol_z': round(garch_vol_z, 3)}


class Gate4ReversalFilter:
    """[Gate4] 반전 리스크 필터.

    Brain B의 rev_prob(가격 구조 기반 반전 확률)을 사용해
    반전 가능성이 높은 구간에서 포지션을 축소한다.

    발동 조건:
        - rev_prob > HIGH_REV (기본 0.6) → 진입 차단
        - rev_prob > MED_REV  (기본 0.4) → lev × (1 - rev_prob × 0.5)
        - rev_prob ≤ MED_REV             → 통과 (lev 유지)
    """

    HIGH_REV = 0.6
    MED_REV  = 0.4

    def __init__(self):
        self._block_count = 0
        self._reduce_count = 0

    def check(self, action: int, lev: float, rev_prob: float) -> Tuple[bool, float, dict]:
        if action == 0:
            return True, lev, {'gate4': 'SKIP_FLAT'}

        if rev_prob > self.HIGH_REV:
            self._block_count += 1
            return False, 0.0, {'gate4': 'BLOCK', 'rev_prob': round(rev_prob, 3)}

        if rev_prob > self.MED_REV:
            factor  = float(np.clip(1.0 - rev_prob * 0.5, 0.5, 1.0))
            adj_lev = float(np.clip(lev * factor, 0.0, lev))
            self._reduce_count += 1
            return True, adj_lev, {'gate4': 'REDUCE', 'rev_prob': round(rev_prob, 3), 'factor': round(factor, 3)}

        return True, lev, {'gate4': 'PASS', 'rev_prob': round(rev_prob, 3)}


class Gate5DrawdownFilter:
    """[Gate5] 포트폴리오 드로다운 차단 게이트.

    누적 낙폭이 임계값을 초과하면 신규 진입을 막는다.
    (개별 포지션 MDD가 아닌 전체 포트폴리오 기준)

    발동 조건:
        - portfolio_mdd < HARD_MDD (기본 -8%) → 진입 완전 차단
        - portfolio_mdd < SOFT_MDD (기본 -5%) → lev × 0.5 (방어 모드)
        - portfolio_mdd < MILD_MDD (기본 -3%) → lev × 0.75

    파라미터:
        hard_mdd: 완전 차단 임계 (기본 -8%, 음수)
        soft_mdd: 절반 축소 임계 (기본 -5%, 음수)
        mild_mdd: 경미 축소 임계 (기본 -3%, 음수)
    """

    def __init__(self, hard_mdd: float = -0.08, soft_mdd: float = -0.05,
                 mild_mdd: float = -0.03):
        self.hard_mdd = hard_mdd
        self.soft_mdd = soft_mdd
        self.mild_mdd = mild_mdd
        self._block_count = 0
        self._reduce_count = 0

    def check(self, action: int, lev: float, portfolio_mdd: float) -> Tuple[bool, float, dict]:
        """
        Args:
            portfolio_mdd: 현재 포트폴리오 낙폭 (음수, 예: -0.06 = -6%)
        """
        if action == 0:
            return True, lev, {'gate5': 'SKIP_FLAT'}

        if portfolio_mdd <= self.hard_mdd:
            self._block_count += 1
            return False, 0.0, {
                'gate5': 'BLOCK',
                'portfolio_mdd': round(portfolio_mdd * 100, 2),
                'reason': f'mdd {portfolio_mdd*100:.1f}% < hard_mdd {self.hard_mdd*100:.1f}%'
            }

        if portfolio_mdd <= self.soft_mdd:
            adj_lev = float(np.clip(lev * 0.5, 0.0, lev))
            self._reduce_count += 1
            return True, adj_lev, {
                'gate5': 'REDUCE_50',
                'portfolio_mdd': round(portfolio_mdd * 100, 2)
            }

        if portfolio_mdd <= self.mild_mdd:
            adj_lev = float(np.clip(lev * 0.75, 0.0, lev))
            self._reduce_count += 1
            return True, adj_lev, {
                'gate5': 'REDUCE_75',
                'portfolio_mdd': round(portfolio_mdd * 100, 2)
            }

        return True, lev, {'gate5': 'PASS', 'portfolio_mdd': round(portfolio_mdd * 100, 2)}


class Gate6RegimeFilter:
    """[Gate6] HMM 레짐 불일치 필터.

    Brain A의 활성 에이전트(방향성)와 HMM이 감지한 레짐이 구조적으로
    불일치할 때 진입을 차단하거나 축소한다.

    불일치 매핑:
        bull 에이전트 진입 + HMM=bear-trend  → VETO
        bear 에이전트 진입 + HMM=bull-trend  → VETO
        bull/bear 에이전트 + HMM=hv-chop     → lev × 0.5 (변동성 장세 경계)
        chop 에이전트    + HMM=bull/bear-trend → lev × 0.7 (추세 장세에서 역추세 비용)

    파라미터:
        hmm_state_names: ['bull-trend', 'bear-trend', 'hv-chop', 'lv-range']
        hmm_conf_threshold: HMM dominant 확률이 이 값 이상일 때만 게이트 발동 (기본 0.5)
    """

    HMM_NAMES = ['bull-trend', 'bear-trend', 'hv-chop', 'lv-range']

    # (에이전트 타입, HMM 상태) → (allow, lev_factor)
    # True = 통과, False = 차단
    _RULE_TABLE = {
        # 방향 에이전트 vs 반대 HMM 추세 → VETO
        ('long',  'bear-trend'): (False, 0.0),
        ('short', 'bull-trend'): (False, 0.0),
        # 방향 에이전트 vs HV-Chop → 경계
        ('long',  'hv-chop'):    (True,  0.5),
        ('short', 'hv-chop'):    (True,  0.5),
        # 방향 에이전트 vs LV-Range → 약한 모멘텀, 소폭 축소
        ('long',  'lv-range'):   (True,  0.8),
        ('short', 'lv-range'):   (True,  0.8),
        # Chop 에이전트 vs 강한 추세 → 역추세, 축소
        ('chop',  'bull-trend'): (True,  0.7),
        ('chop',  'bear-trend'): (True,  0.7),
    }

    def __init__(self, hmm_conf_threshold: float = 0.5):
        self.hmm_conf_threshold = hmm_conf_threshold
        self._block_count  = 0
        self._reduce_count = 0

    def _agent_type(self, agent_name: str) -> str:
        """에이전트 이름 → 타입 (long / short / chop / normal / unknown)."""
        name = agent_name.lower()
        if 'chop' in name:    return 'chop'
        if 'normal' in name:
            if 'long'  in name: return 'long'
            if 'short' in name: return 'short'
            return 'normal'
        if 'bull' in name:    return 'long'
        if 'bear' in name:    return 'short'
        return 'unknown'

    def check(self, action: int, lev: float, brain_a: BrainAOutput,
              hmm_state: str, hmm_conf: float) -> Tuple[bool, float, dict]:
        """
        Args:
            hmm_state: 현재 HMM dominant 상태명 ('bull-trend' 등)
            hmm_conf : HMM dominant 상태 사후확률 (0~1)
        """
        if action == 0:
            return True, lev, {'gate6': 'SKIP_FLAT'}

        # HMM 확신도가 낮으면 게이트 비활성화 (불확실한 레짐 판단으로 차단 방지)
        if hmm_conf < self.hmm_conf_threshold:
            return True, lev, {'gate6': 'SKIP_LOW_CONF', 'hmm_conf': round(hmm_conf, 3)}

        agent_type = self._agent_type(brain_a.agent_name)
        key        = (agent_type, hmm_state)

        if key in self._RULE_TABLE:
            allow, factor = self._RULE_TABLE[key]
            if not allow:
                self._block_count += 1
                return False, 0.0, {
                    'gate6': 'BLOCK',
                    'reason': f'agent={agent_type} vs hmm={hmm_state}',
                    'hmm_conf': round(hmm_conf, 3),
                }
            if factor < 1.0:
                adj_lev = float(np.clip(lev * factor, 0.0, lev))
                self._reduce_count += 1
                return True, adj_lev, {
                    'gate6': f'REDUCE_{int(factor*100)}',
                    'agent': agent_type, 'hmm': hmm_state,
                }

        return True, lev, {'gate6': 'PASS', 'agent': agent_type, 'hmm': hmm_state}


# ───────────────────────────────────────────────────────────────────────────
# Gatekeeper — 6개 게이트 직렬 실행기
# ───────────────────────────────────────────────────────────────────────────
class Gatekeeper:
    """6개 게이트를 순서대로 실행. 하나라도 BLOCK이면 flat 강제.

    게이트 실행 순서:
        Gate1 → Gate2 → Gate3 → Gate4 → Gate5 → Gate6

    설계 원칙:
        - 조기 종료: Gate가 BLOCK을 반환하면 이후 게이트 실행 생략
        - lev는 각 게이트를 통과하면서 누적 곱으로 감소 가능
        - gate_log에 각 게이트 결과를 기록 (디버깅/분석)
    """

    def __init__(self, hmm_conf_threshold: float = 0.5):
        self.gate1 = Gate1TrendFilter()
        self.gate2 = Gate2EpistemicFilter()
        self.gate3 = Gate3VolatilityFilter()
        self.gate4 = Gate4ReversalFilter()
        self.gate5 = Gate5DrawdownFilter()
        self.gate6 = Gate6RegimeFilter(hmm_conf_threshold)

    def run(self,
            action     : int,
            lev        : float,
            brain_a    : BrainAOutput,
            brain_b    : BrainBOutput,
            garch_vol_z: float = 0.0,
            portfolio_mdd: float = 0.0,
            hmm_state  : str   = 'lv-range',
            hmm_conf   : float = 0.25,
    ) -> Tuple[bool, float, dict]:
        """6개 게이트 순차 실행.

        Args:
            action       : Arbiter 결정 후 액션 (0/1/2)
            lev          : Arbiter 결정 후 Kelly 배율
            brain_a      : Brain A 출력 (에이전트 정보, epist 정보)
            brain_b      : Brain B 출력 (trend, strength, rev_prob)
            garch_vol_z  : 현재 스텝 GARCH 변동성 z-score
            portfolio_mdd: 현재 포트폴리오 최대 낙폭 (음수)
            hmm_state    : HMM dominant 상태명
            hmm_conf     : HMM dominant 확률

        Returns:
            (gate_passed: bool, final_lev: float, gate_log: dict)
        """
        gate_log = {}

        allow, lev, log1 = self.gate1.check(action, lev, brain_b)
        gate_log.update(log1)
        if not allow:
            return False, 0.0, gate_log

        allow, lev, log2 = self.gate2.check(action, lev, brain_a)
        gate_log.update(log2)
        if not allow:
            return False, 0.0, gate_log

        allow, lev, log3 = self.gate3.check(action, lev, garch_vol_z)
        gate_log.update(log3)
        if not allow:
            return False, 0.0, gate_log

        allow, lev, log4 = self.gate4.check(action, lev, brain_b.rev_prob)
        gate_log.update(log4)
        if not allow:
            return False, 0.0, gate_log

        allow, lev, log5 = self.gate5.check(action, lev, portfolio_mdd)
        gate_log.update(log5)
        if not allow:
            return False, 0.0, gate_log

        allow, lev, log6 = self.gate6.check(action, lev, brain_a, hmm_state, hmm_conf)
        gate_log.update(log6)
        if not allow:
            return False, 0.0, gate_log

        return True, lev, gate_log

    def gate_stats(self) -> dict:
        """각 게이트별 차단/축소 통계 반환."""
        return {
            'gate1_block':  self.gate1._block_count,  'gate1_soft': self.gate1._soft_count,
            'gate2_block':  self.gate2._block_count,
            'gate3_block':  self.gate3._block_count,  'gate3_reduce': self.gate3._reduce_count,
            'gate4_block':  self.gate4._block_count,  'gate4_reduce': self.gate4._reduce_count,
            'gate5_block':  self.gate5._block_count,  'gate5_reduce': self.gate5._reduce_count,
            'gate6_block':  self.gate6._block_count,  'gate6_reduce': self.gate6._reduce_count,
        }


# ───────────────────────────────────────────────────────────────────────────
# DualBrainArbiter — 전체 파이프라인 통합 진입점
# ───────────────────────────────────────────────────────────────────────────
class DualBrainArbiter:
    """Arbiter + Gatekeeper 전체 파이프라인.

    사용 예:
        arbiter = DualBrainArbiter()

        brain_a_out = BrainAOutput.from_router_output(action, leverage, info)
        brain_b_out = BrainBOutput.from_signal(brain_b.predict(candles_48))

        decision = arbiter.decide(
            brain_a=brain_a_out,
            brain_b=brain_b_out,
            garch_vol_z=feat['garch_vol_z'],
            portfolio_mdd=portfolio_mdd,
            hmm_state=info.get('hmm_state', 'lv-range'),
            hmm_probs=info.get('hmm_probs', [0.25]*4),
        )

        env.step(decision.final_action, leverage_rate=decision.final_lev)

    로그 예시:
        DualBrainArbiter → BOOST (1.18x) → Gate1:PASS Gate2:PASS ...Gate6:PASS
        → final_action=1 lev=0.614
    """

    def __init__(self,
                 boost_factor      : float = 1.2,
                 veto_threshold    : float = 0.4,
                 hmm_conf_threshold: float = 0.5):
        self.arbiter    = Arbiter(boost_factor=boost_factor, veto_threshold=veto_threshold)
        self.gatekeeper = Gatekeeper(hmm_conf_threshold=hmm_conf_threshold)
        self._step_count = 0

    def decide(self,
               brain_a      : BrainAOutput,
               brain_b      : BrainBOutput,
               garch_vol_z  : float = 0.0,
               portfolio_mdd: float = 0.0,
               hmm_state    : str   = 'lv-range',
               hmm_probs    : list  = None,
    ) -> ArbiterDecision:
        """
        Args:
            brain_a       : Brain A 출력
            brain_b       : Brain B 출력
            garch_vol_z   : 현재 GARCH 변동성 z-score (features['garch_vol_z'])
            portfolio_mdd : 현재 포트폴리오 최대 낙폭 (음수, TradingEnv.max_drawdown 상당)
            hmm_state     : HMM dominant 상태명 (info['hmm_state'])
            hmm_probs     : HMM 상태 사후확률 리스트 (4개)

        Returns:
            ArbiterDecision
        """
        self._step_count += 1
        hmm_probs = hmm_probs or [0.25, 0.25, 0.25, 0.25]
        hmm_conf  = float(max(hmm_probs))

        # ── Step 1: Arbiter — 두 브레인 합의/충돌 판정 ──
        arb_action, arb_lev, arb_mode, lev_mult = self.arbiter.adjudicate(brain_a, brain_b)

        # Arbiter가 이미 veto했으면 Gatekeeper 생략
        if arb_action == 0 and arb_mode == 'VETO':
            return ArbiterDecision(
                final_action  = 0,
                final_lev     = 0.0,
                gate_passed   = False,
                arbiter_mode  = 'VETO',
                lev_multiplier= 0.0,
                gate_log      = {'arbiter': 'VETO', 'reason': 'direction_conflict'},
            )

        # ── Step 2: Gatekeeper — 6개 게이트 순차 필터 ──
        gate_passed, final_lev, gate_log = self.gatekeeper.run(
            action        = arb_action,
            lev           = arb_lev,
            brain_a       = brain_a,
            brain_b       = brain_b,
            garch_vol_z   = garch_vol_z,
            portfolio_mdd = portfolio_mdd,
            hmm_state     = hmm_state,
            hmm_conf      = hmm_conf,
        )

        final_action = arb_action if gate_passed else 0
        gate_log['arbiter_mode'] = arb_mode
        gate_log['arb_lev']      = round(arb_lev, 3)
        gate_log['lev_mult']     = round(lev_mult, 3)

        return ArbiterDecision(
            final_action  = final_action,
            final_lev     = final_lev,
            gate_passed   = gate_passed,
            arbiter_mode  = arb_mode,
            lev_multiplier= lev_mult,
            gate_log      = gate_log,
        )

    def full_stats(self) -> dict:
        """Arbiter + Gatekeeper 전체 통계."""
        stats = {}
        stats.update(self.arbiter.stats())
        stats.update(self.gatekeeper.gate_stats())
        stats['total_steps'] = self._step_count
        return stats

    def log_summary(self) -> None:
        """로거에 통계 요약 출력."""
        s = self.full_stats()
        logger.info(
            f"[DualBrainArbiter] steps={s['total_steps']} | "
            f"BOOST={s['arbiter_boost']} VETO={s['arbiter_veto']} "
            f"NEUTRAL={s['arbiter_neutral']} CAUTION={s['arbiter_caution']} | "
            f"G1_blk={s['gate1_block']} G2_blk={s['gate2_block']} "
            f"G3_blk={s['gate3_block']} G4_blk={s['gate4_block']} "
            f"G5_blk={s['gate5_block']} G6_blk={s['gate6_block']}"
        )


# ───────────────────────────────────────────────────────────────────────────
# 통합 사용 예시 (val loop에 삽입할 때)
# ───────────────────────────────────────────────────────────────────────────
def make_dual_brain_step(
    router,         # GatingRouter7 인스턴스
    trend_brain,    # TrendContextBrain 인스턴스
    extractor,      # CandleFeatureExtractor 인스턴스
    arbiter,        # DualBrainArbiter 인스턴스
    features: dict,
    pos_info : dict,
    candles_window: np.ndarray,  # (WINDOW, feat_dim) Brain B 입력
    portfolio_mdd: float = 0.0,
    device: str = 'cpu',
) -> ArbiterDecision:
    """두 브레인 + Arbiter를 하나의 함수로 통합한 헬퍼.

    기존 val loop의 router.decide() 호출을 이 함수로 교체하면 됨.

    반환된 ArbiterDecision에서:
        decision.final_action → env.step()의 action 인자
        decision.final_lev    → env.step()의 leverage_rate 인자
    """
    # Brain A 실행
    action_a, lev_a, info_a = router.decide(features, pos_info)
    brain_a = BrainAOutput.from_router_output(action_a, lev_a, info_a)

    # Brain B 실행 (캔들 윈도우 필요)
    signal_b = trend_brain.predict(candles_window, device=device)
    brain_b  = BrainBOutput.from_signal(signal_b)

    # HMM 정보 파싱 (GatingRouter7.decide()가 info에 포함시킴)
    hmm_state = info_a.get('hmm_state', 'lv-range')
    hmm_probs = info_a.get('hmm_probs', [0.25] * 4)
    garch_vol_z = float(features.get('garch_vol_z', 0.0))

    return arbiter.decide(
        brain_a       = brain_a,
        brain_b       = brain_b,
        garch_vol_z   = garch_vol_z,
        portfolio_mdd = portfolio_mdd,
        hmm_state     = hmm_state,
        hmm_probs     = hmm_probs,
    )