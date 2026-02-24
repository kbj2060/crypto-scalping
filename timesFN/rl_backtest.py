import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import logging
from rl_sniper import CryptoSniperEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

# 평가용으로 살짝 수정된 환경 (처음부터 끝까지 순차적으로 진행)
class EvalSniperEnv(CryptoSniperEnv):
    def reset(self, seed=None, options=None):
        super(CryptoSniperEnv, self).reset(seed=seed)
        self.current_step = 0  # 🚨 평가 시에는 무작위가 아닌 0번 인덱스부터 시작
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        return self._next_observation(), {}

def main():
    logger.info("🎯 훈련된 RL 스나이퍼 실전 투입 (Out-of-Sample Backtest)")
    
    # 1. 구글 TimesFM 피처가 포함된 데이터 로드
    df = pd.read_csv("data/training_features_with_ttm.csv")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    logger.info(f"📊 검증 데이터 크기: {len(test_df)} 캔들")

    # 2. 평가용 환경 생성
    env = DummyVecEnv([lambda: EvalSniperEnv(test_df)])

    # 3. 방독면(정규화 기준치) 씌우기
    try:
        env = VecNormalize.load("vec_normalize.pkl", env)
        env.training = False 
        env.norm_reward = False 
    except FileNotFoundError:
        logger.error("❌ vec_normalize.pkl 파일을 찾을 수 없습니다.")
        return

    # 4. 훈련된 100만 스텝 스나이퍼의 뇌 로드
    try:
        model = PPO.load("ppo_crypto_sniper")
        logger.info("✅ PPO 모델 로드 완료")
    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {e}")
        return

    # 5. 백테스트 시뮬레이션 실행
    obs = env.reset()
    
    net_worth_history = []
    actions_taken = []
    
    logger.info("⚡ 시뮬레이션 진행 중...")
    
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # 🚨 핵심 수정: info 딕셔너리를 믿지 않고, 환경 객체에서 실시간 잔고를 강제로 뽑아옵니다.
        current_net_worth = env.get_attr('net_worth')[0]
        net_worth_history.append(current_net_worth)
        actions_taken.append(action[0])
        
        # done은 numpy 배열이므로 첫 번째 환경의 종료 여부를 명확히 확인
        if done[0]:
            break

    # 6. 성과 분석
    initial_cap = 10000.0
    final_cap = net_worth_history[-2] if len(net_worth_history) > 1 else net_worth_history[-1]
    
    # 🚨 추가: 최고점과 최저점 파악 (MDD 분석용)
    max_cap = max(net_worth_history[:-1]) # 마지막 리셋값 제외
    min_cap = min(net_worth_history[:-1])

    roi = (final_cap - initial_cap) / initial_cap
    max_roi = (max_cap - initial_cap) / initial_cap
    min_roi = (min_cap - initial_cap) / initial_cap
    
    action_counts = pd.Series(actions_taken).value_counts().to_dict()
    wait_cnt = action_counts.get(0, 0)
    long_cnt = action_counts.get(1, 0)
    short_cnt = action_counts.get(2, 0)
    total_actions = len(actions_taken)

    logger.info("\n" + "="*50)
    logger.info("🏆 [RL 스나이퍼 최종 백테스트 결과]")
    logger.info("="*50)
    logger.info(f"▶ 초기 자본금: ${initial_cap:,.2f}")
    logger.info(f"▶ 최종 자본금: ${final_cap:,.2f}")
    logger.info(f"▶ 💰 최종 순수익률 (ROI): {roi:.2%}")
    logger.info("-" * 50)
    logger.info("🎯 [행동 패턴 분석]")
    logger.info(f"▶ 관망(Wait) 횟수: {wait_cnt}회 ({wait_cnt/total_actions:.1%})")
    logger.info(f"▶ 매수(Long) 횟수 : {long_cnt}회 ({long_cnt/total_actions:.1%})")
    logger.info(f"▶ 매도(Short) 횟수: {short_cnt}회 ({short_cnt/total_actions:.1%})")
    logger.info("="*50)

if __name__ == "__main__":
    main()