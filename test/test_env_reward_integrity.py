"""TradingEnv 보상 함수 무결성 단위 테스트
==========================================
수정한 5개 버그 각각에 대한 결정론적 검증.

  Bug 1: pos_features unrealized_pnl 정규화 범위 [-1, 1]
  Bug 3: Sigma floor = init 값 (0.00625), 10% 가 아님
  Bug 4: Telescoping sum + clawback → 총 delta shaping 기여 = 0
  Bug 5: GatingNet sharpe_scale 비음수 보장 (음수 ep_sharpe 시에도)

실행: pytest test/test_env_reward_integrity.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─── 공통 헬퍼 ────────────────────────────────────────────────────────────────

def _make_df(feat_cols: list, n_rows: int = 50, price: float = 1000.0) -> pd.DataFrame:
    """최소 DataFrame — 모든 피처는 0.0, close만 지정."""
    df = pd.DataFrame(0.0, index=range(n_rows), columns=feat_cols + ["close", "timestamp"])
    df["close"] = price
    df["timestamp"] = pd.date_range("2024-01-01", periods=n_rows, freq="5min")
    return df


def _rl_env(agent_role="bull", n=50, prices=None):
    from ensemble.train_rl_agent import (
        TradingEnv,
        STATE_PRED, STATE_CONF, STATE_ELITE, STATE_ALPHA, REGIME_COLS, STATE_SYNTH,
    )
    feat_cols = STATE_PRED + STATE_CONF + STATE_ELITE + STATE_ALPHA + REGIME_COLS + STATE_SYNTH
    df = _make_df(feat_cols, n, price=1000.0)
    if prices is not None:
        df["close"] = list(prices)[:n]
    return TradingEnv(df, phase="val", agent_role=agent_role)


def _ls_env(agent_role="long", n=50, prices=None):
    from ensemble.train_ls_agent import (
        TradingEnv as LSTradingEnv,
        STATE_PRED, STATE_CONF, STATE_ELITE, STATE_ALPHA, REGIME_COLS, STATE_SYNTH,
    )
    feat_cols = STATE_PRED + STATE_CONF + STATE_ELITE + STATE_ALPHA + REGIME_COLS + STATE_SYNTH
    df = _make_df(feat_cols, n, price=1000.0)
    if prices is not None:
        df["close"] = list(prices)[:n]
    return LSTradingEnv(df, phase="val", agent_role=agent_role)


# ─── Bug 1: pos_features 정규화 ───────────────────────────────────────────────

class TestPosFeatureNormalization:
    """unrealized_pnl 과 max_drawdown 이 pos_features 에서 [-1, 1] 로 클리핑되는지 검증.

    수정 전: 날것 비율 그대로 → 진입 즉시 -0.0007 점프로 Context Gate 오염
    수정 후: np.clip(val / 0.05, -1.0, 1.0) — 5% 기준 정규화
    """

    def test_rl_pos_features_in_range_after_entry(self):
        """진입 후 pos_features[2:4] 가 [-1, 1] 에 속해야 함."""
        env = _rl_env()
        env.step(1)  # enter long (bull, val phase)
        pos = env._build_state(env.current_step)[-5:]
        assert -1.0 <= pos[2] <= 1.0, f"unrealized_pnl 정규화 실패: {pos[2]}"
        assert -1.0 <= pos[3] <= 1.0, f"max_drawdown 정규화 실패: {pos[3]}"

    def test_rl_extreme_loss_clips_to_minus_one(self):
        """unrealized_pnl = -50% → 5% 기준 정규화 후 -1.0 클리핑."""
        env = _rl_env()
        env.step(1)
        env.unrealized_pnl = -0.50  # 50% loss, 정규화 → -10.0 → clip → -1.0
        env.max_drawdown = -0.30
        pos = env._build_state(env.current_step)[-5:]
        assert pos[2] == pytest.approx(-1.0), f"50% 손실이 -1.0 로 클리핑되어야 함: {pos[2]}"
        assert pos[3] == pytest.approx(-1.0), f"30% DD가 -1.0 로 클리핑되어야 함: {pos[3]}"

    def test_rl_extreme_gain_clips_to_plus_one(self):
        """unrealized_pnl = +20% → 정규화 후 +1.0 클리핑."""
        env = _rl_env()
        env.step(1)
        env.unrealized_pnl = 0.20
        pos = env._build_state(env.current_step)[-5:]
        assert pos[2] == pytest.approx(1.0), f"20% 수익이 +1.0 로 클리핑되어야 함: {pos[2]}"

    def test_rl_small_loss_proportional(self):
        """unrealized_pnl = -2.5% → 정규화 후 -0.5 (비례 감쇠, 클리핑 없음)."""
        env = _rl_env()
        env.step(1)
        env.unrealized_pnl = -0.025  # -2.5% / 5% = -0.5
        pos = env._build_state(env.current_step)[-5:]
        assert pos[2] == pytest.approx(-0.5, abs=1e-5)

    def test_ls_pos_features_in_range_after_entry(self):
        env = _ls_env()
        env.step(1)  # enter long (long agent, val phase)
        pos = env._build_state(env.current_step)[-5:]
        assert -1.0 <= pos[2] <= 1.0
        assert -1.0 <= pos[3] <= 1.0

    def test_ls_extreme_loss_clips_to_minus_one(self):
        env = _ls_env()
        env.step(1)
        env.unrealized_pnl = -0.50
        env.max_drawdown = -0.50
        pos = env._build_state(env.current_step)[-5:]
        assert pos[2] == pytest.approx(-1.0)
        assert pos[3] == pytest.approx(-1.0)

    def test_no_position_pos_features_zero(self):
        """포지션 없을 때 unrealized/drawdown 은 0."""
        env = _rl_env()
        pos = env._build_state(env.current_step)[-5:]
        assert pos[2] == pytest.approx(0.0)
        assert pos[3] == pytest.approx(0.0)


# ─── Bug 3: Sigma floor 값 ────────────────────────────────────────────────────

class TestSigmaFloor:
    """Sigma floor 가 sigma_init (0.00625) 이어야 하며, 그것의 10% (0.000625) 가 아님.

    수정 전: 0.05 / sqrt(64) * 0.10 = 0.000625 → 탐험 죽음
    수정 후: 0.05 / sqrt(64) = 0.00625 → init 값 유지
    """

    def test_sigma_init_value(self):
        sigma_init = 0.05 / (64 ** 0.5)
        assert sigma_init == pytest.approx(0.00625, rel=1e-5)

    def test_correct_floor_equals_init(self):
        correct = 0.05 / (64 ** 0.5)
        buggy = 0.05 / (64 ** 0.5) * 0.10
        assert correct == pytest.approx(0.00625, abs=1e-6)
        assert buggy == pytest.approx(0.000625, abs=1e-7)
        assert correct / buggy == pytest.approx(10.0, rel=1e-5)

    def test_rl_source_uses_correct_floor(self):
        """RL 소스에 * 0.10 이 없음을 확인."""
        import inspect
        import ensemble.train_rl_agent as mod
        # 학습 루프에서 _SIGMA_FLOOR 정의 부분 추출
        src = inspect.getsource(mod)
        # 올바른 패턴: `0.05 / (64 ** 0.5)` 뒤에 * 0.10 이 없어야 함
        import re
        matches = re.findall(r"_SIGMA_FLOOR\s*=\s*[^\n]+", src)
        assert matches, "_SIGMA_FLOOR 정의를 찾을 수 없음"
        for m in matches:
            assert "0.10" not in m, f"RL sigma floor에 *0.10 버그 잔존: {m}"

    def test_ls_source_uses_correct_floor(self):
        """LS 소스에 * 0.10 이 없음을 확인."""
        import inspect, re
        import ensemble.train_ls_agent as mod
        src = inspect.getsource(mod)
        matches = re.findall(r"_SIGMA_FLOOR\s*=\s*[^\n]+", src)
        assert matches, "_SIGMA_FLOOR 정의를 찾을 수 없음"
        for m in matches:
            assert "0.10" not in m, f"LS sigma floor에 *0.10 버그 잔존: {m}"


# ─── Bug 4: Telescoping sum + clawback 수학적 무결성 ─────────────────────────

class TestTelescopingClawback:
    """delta shaping 누적합 + clawback = 0 → total_reward = -entry_fee + realized_pnl.

    수정 전: clawback 없음 → total_reward = -fee + 0.01*Φ_N + realized_pnl (이중 계상)
    수정 후: close 시 0.01*(0 - prev_unrealized) 추가 → telescoping sum = 0
    """

    def _collect_trade_rewards(self, env, min_hold, enter_action, hold_action, close_action):
        """진입 → min_hold 스텝 홀드 → 청산. 총 reward 합 반환."""
        total = 0.0
        _, r, _, _ = env.step(enter_action)
        total += r
        entry_price = env.entry_price
        # hold_action 으로 (min_hold - 1) 스텝 홀드
        for _ in range(min_hold - 1):
            _, r, _, _ = env.step(hold_action)
            total += r
        # 청산 직전 current_price 기록
        close_price = env._close_np[env.current_step]
        _, r, _, _ = env.step(close_action)
        total += r
        return total, entry_price, close_price

    def test_rl_telescoping_flat_price(self):
        """횡보: delta shaping 이 0 으로 수렴 → total = -fee + realized."""
        from ensemble.train_rl_agent import MIN_HOLD_TRAIN
        env = _rl_env(prices=[1000.0] * 50)
        total, ep, cp = self._collect_trade_rewards(env, MIN_HOLD_TRAIN, 1, 1, 0)
        lev = env.MAX_LEVERAGE
        realized = (cp * (1 - env.slip) - ep) / ep * lev - env.fee * lev
        expected = -env.fee * lev + realized
        assert abs(total - expected) < 1e-5, (
            f"[flat] telescoping 깨짐: got={total:.7f}, expected={expected:.7f}, diff={total-expected:.2e}"
        )

    def test_rl_telescoping_rising_price(self):
        """상승: 0.01 * Φ_N 이 clawback 으로 정확히 상쇄."""
        from ensemble.train_rl_agent import MIN_HOLD_TRAIN
        prices = [1000.0 + i * 2.0 for i in range(50)]  # +2/스텝
        env = _rl_env(prices=prices)
        total, ep, cp = self._collect_trade_rewards(env, MIN_HOLD_TRAIN, 1, 1, 0)
        lev = env.MAX_LEVERAGE
        realized = (cp * (1 - env.slip) - ep) / ep * lev - env.fee * lev
        expected = -env.fee * lev + realized
        assert abs(total - expected) < 1e-5, (
            f"[rising] telescoping 깨짐: got={total:.7f}, expected={expected:.7f}, diff={total-expected:.2e}"
        )

    def test_rl_telescoping_falling_price(self):
        """하락 (손실 포지션): clawback 이 음수 delta 를 정확히 환수."""
        from ensemble.train_rl_agent import MIN_HOLD_TRAIN
        prices = [1000.0 - i * 1.0 for i in range(50)]  # -1/스텝
        env = _rl_env(prices=prices)
        total, ep, cp = self._collect_trade_rewards(env, MIN_HOLD_TRAIN, 1, 1, 0)
        lev = env.MAX_LEVERAGE
        realized = (cp * (1 - env.slip) - ep) / ep * lev - env.fee * lev
        expected = -env.fee * lev + realized
        assert abs(total - expected) < 1e-5, (
            f"[falling] telescoping 깨짐: got={total:.7f}, expected={expected:.7f}, diff={total-expected:.2e}"
        )

    def test_ls_telescoping_rising_price(self):
        """LS long: 상승 가격에서 telescoping 무결성."""
        from ensemble.train_ls_agent import MIN_HOLD_TRAIN
        prices = [1000.0 + i * 2.0 for i in range(50)]
        env = _ls_env(agent_role="long", prices=prices)
        # val phase: enter=action 1, hold=action 1(pos=LONG → no-op), close=action 0
        total, ep, cp = self._collect_trade_rewards(env, MIN_HOLD_TRAIN, 1, 1, 0)
        lev = env.MAX_LEVERAGE
        realized = (cp * (1 - env.slip) - ep) / ep * lev - env.fee * lev
        expected = -env.fee * lev + realized
        assert abs(total - expected) < 1e-5, (
            f"[LS rising] telescoping 깨짐: got={total:.7f}, expected={expected:.7f}, diff={total-expected:.2e}"
        )

    def test_churn_penalty_still_applied_for_early_close(self):
        """MIN_HOLD_TRAIN 미만 청산 시 churn_rate 패널티가 여전히 적용됨."""
        from ensemble.train_rl_agent import MIN_HOLD_TRAIN
        env_early = _rl_env(prices=[1000.0] * 50)
        env_normal = _rl_env(prices=[1000.0] * 50)

        # 1스텝만 홀드 후 조기 청산
        _, r0, _, _ = env_early.step(1)
        cp_early = env_early._close_np[env_early.current_step]
        _, r1, _, _ = env_early.step(0)
        total_early = r0 + r1

        # MIN_HOLD_TRAIN 스텝 홀드 후 정상 청산
        total_normal, _, _ = self._collect_trade_rewards(env_normal, MIN_HOLD_TRAIN, 1, 1, 0)

        # 조기 청산이 정상 청산보다 불리해야 함 (가격 동일 조건)
        assert total_early < total_normal, "churn 패널티가 조기 청산을 충분히 억제하지 못함"


# ─── Bug 5: GatingNet sharpe_scale 비음수 ─────────────────────────────────────

class TestGatingNetSharpeScale:
    """ep_sharpe < 0 일 때 sharpe_scale = 0 으로 클램프 → ret_t.mean() >= 0.

    수정 전: tanh(음수) = 음수 → z-score + 음수 = 모든 ret_t 음수 → 모든 행동 확률 하락
             → GatingNet [0.2 × N] 균등 붕괴 (최대 엔트로피 수렴)
    수정 후: max(0.0, tanh(ep_sharpe)) → 나쁜 에피소드는 zero-mean 유지
    """

    @staticmethod
    def _build_ret_t(rewards, sharpe_scale):
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        ret_t = torch.FloatTensor(returns)
        ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)
        ret_t = ret_t + sharpe_scale
        return ret_t

    def test_negative_episode_sharpe_yields_zero_scale(self):
        """일관된 손실 에피소드 → sharpe_scale = 0 (음수 차단)."""
        rewards = [-0.01, -0.008, -0.012, -0.009]
        rew_arr = np.array(rewards, dtype=np.float32)
        ep_sharpe = float(rew_arr.mean() / (rew_arr.std() + 1e-8))
        assert ep_sharpe < 0, "전제 조건: ep_sharpe 가 음수여야 함"
        sharpe_scale = max(0.0, float(np.tanh(ep_sharpe)))
        assert sharpe_scale == pytest.approx(0.0), f"음수 ep_sharpe 에서 scale이 0 이어야 함: {sharpe_scale}"

    def test_negative_episode_ret_t_mean_non_negative(self):
        """수정 후: 음수 ep_sharpe → ret_t.mean() ≈ 0 (균등 붕괴 없음)."""
        rewards = [-0.01, -0.008, -0.012, -0.009]
        rew_arr = np.array(rewards, dtype=np.float32)
        ep_sharpe = float(rew_arr.mean() / (rew_arr.std() + 1e-8))
        sharpe_scale = max(0.0, float(np.tanh(ep_sharpe)))
        ret_t = self._build_ret_t(rewards, sharpe_scale)
        assert ret_t.mean().item() >= -1e-5, (
            f"ret_t.mean() 이 음수: {ret_t.mean().item():.6f} — GatingNet 붕괴 조건"
        )

    def test_positive_episode_ret_t_mean_positive(self):
        """좋은 에피소드: sharpe_scale > 0 → ret_t.mean() > 0 → 행동 강화."""
        rewards = [0.02, 0.015, 0.018, 0.022]
        rew_arr = np.array(rewards, dtype=np.float32)
        ep_sharpe = float(rew_arr.mean() / (rew_arr.std() + 1e-8))
        sharpe_scale = max(0.0, float(np.tanh(ep_sharpe)))
        assert sharpe_scale > 0
        ret_t = self._build_ret_t(rewards, sharpe_scale)
        assert ret_t.mean().item() > 0

    def test_bug_reproduction_without_fix(self):
        """수정 전 버그 재현: clamp 없이 음수 sharpe_scale → ret_t 전체 음수."""
        rewards = [-0.01, -0.008, -0.012]
        rew_arr = np.array(rewards, dtype=np.float32)
        ep_sharpe = float(rew_arr.mean() / (rew_arr.std() + 1e-8))
        sharpe_scale_buggy = float(np.tanh(ep_sharpe))  # 수정 전: clamp 없음
        assert sharpe_scale_buggy < 0, "버그 재현 전제: scale 이 음수여야 함"
        ret_t_buggy = self._build_ret_t(rewards, sharpe_scale_buggy)
        assert ret_t_buggy.mean().item() < 0, "수정 전: ret_t.mean() < 0 → GatingNet 붕괴 확인"

    def test_rl_source_uses_clamped_sharpe(self):
        """RL train_gating_step_rl 소스에 max(0.0, ...) 패턴 존재 확인."""
        import inspect, ensemble.train_rl_agent as mod
        src = inspect.getsource(mod.train_gating_step_rl)
        assert "max(0.0" in src, "RL GatingNet: sharpe_scale clamp 없음 — 버그 재발 위험"

    def test_ls_source_uses_clamped_sharpe(self):
        """LS train_gating_step 소스에 max(0.0, ...) 패턴 존재 확인."""
        import inspect, ensemble.train_ls_agent as mod
        src = inspect.getsource(mod.train_gating_step)
        assert "max(0.0" in src, "LS GatingNet: sharpe_scale clamp 없음 — 버그 재발 위험"

    def test_sharpe_scale_bounded_zero_to_one(self):
        """sharpe_scale 은 항상 [0, 1] 범위."""
        for ep_sharpe in [-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0]:
            scale = max(0.0, float(np.tanh(ep_sharpe)))
            assert 0.0 <= scale <= 1.0, f"ep_sharpe={ep_sharpe} 에서 scale={scale} 범위 이탈"
