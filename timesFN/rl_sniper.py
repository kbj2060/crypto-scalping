import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # 🚨 추가된 부분
from stable_baselines3.common.callbacks import CheckpointCallback
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================
# 1. 커스텀 트레이딩 환경 (Gymnasium)
# =====================================================================
class CryptoSniperEnv(gym.Env):
    """
    XGBoost를 대체할 RL 저격수를 위한 훈련장
    """
    def __init__(self, df, initial_balance=10000.0, fee=0.0008):
        super(CryptoSniperEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee
        
        # RL 에이전트가 볼 수 있는 시야 (상태 공간)
        # 15개 골든 피처 + (미리 계산해둔) TimesFM 예측값 등
        self.feature_cols = [c for c in df.columns if c not in [
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'label'
        ]]
        
        # 행동 공간 (Action Space): 0=관망(Wait), 1=롱(Long), 2=숏(Short)
        self.action_space = spaces.Discrete(3)
        
        # 상태 공간 (Observation Space)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.feature_cols),), dtype=np.float32
        )
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0 # 0: 없음, 1: 롱, -1: 숏
        self.entry_price = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 학습을 위한 랜덤 시작점 지정 (과적합 방지)
        self.current_step = np.random.randint(0, int(len(self.df) * 0.5))
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        
        return self._next_observation(), {}

    def _next_observation(self):
        obs = self.df[self.feature_cols].iloc[self.current_step].values
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        done = self.current_step >= len(self.df) - 1
        if self.net_worth <= self.initial_balance * 0.2: 
            done = True
            
        if done:
            return self._next_observation(), 0, done, False, {}

        current_price = self.df['close'].iloc[self.current_step]
        prev_price = self.df['close'].iloc[self.current_step - 1]
        
        mapped_action = 0
        if action == 1: mapped_action = 1
        elif action == 2: mapped_action = -1

        actual_pnl_pct = 0.0
        trade_cost = 0.0
        slippage = 0.0002 
        
        # 🚨 추가: AI의 뇌에 가할 가상의 '잦은 매매 억제 페널티'
        brain_trade_penalty = 0.0 

        # 포지션 변경 시
        if self.position != mapped_action:
            if self.position != 0 and mapped_action != 0:
                trade_cost = (self.fee + slippage) * 2 
            else:
                trade_cost = self.fee + slippage
                
            # 🚨 잦은 매매를 하면 뇌에 강력한 마이너스 보상을 줌 (수수료의 수십 배 고통)
            # 이 고통을 이겨낼 만큼 확실한 "대추세" 자리에서만 총을 쏘게 만듦
            brain_trade_penalty = -2.0 
                
            self.position = mapped_action
            self.entry_price = current_price

        # 시장 수익률 계산
        if self.position == 1:
            actual_pnl_pct = (current_price - prev_price) / prev_price
        elif self.position == -1:
            actual_pnl_pct = (prev_price - current_price) / prev_price

        # 현실의 통장 잔고 업데이트
        real_step_return = actual_pnl_pct - trade_cost
        self.net_worth += self.net_worth * real_step_return

        # 🚨 AI의 뇌(Reward) 세팅
        # 잔파도(5분봉 노이즈)에 너무 공포를 느끼지 않도록 KT 페널티 완화
        step_reward = actual_pnl_pct * 100 
        
        if step_reward > 0:
            step_reward *= 2.0  # 추세를 타면 수익 도파민 2배
        elif step_reward < 0:
            step_reward *= 1.2  # 잔파도 손실에 대한 공포심 하향 (2.25 -> 1.2)

        # 캔들 수익 보상에 진입/청산 뇌전기충격 페널티를 더함
        total_reward = step_reward + brain_trade_penalty

        obs = self._next_observation()
        info = {'net_worth': self.net_worth}
        
        return obs, total_reward, done, False, info

def main():
    logger.info("🤖 Hugging Face RL 저격수 (PPO) 훈련 개시")
    
    # 1. 데이터 로드
    df = pd.read_csv("data/training_features_with_ttm.csv")
    df = df.iloc[512:].reset_index(drop=True) 
    
    # 🚨 처방 1: 맹독성 데이터(NaN, Inf) 완벽 제거
    logger.info("🧹 데이터 정화 작업 중 (NaN 및 Inf 제거)...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Train 분리
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    
    # 2. Gym 환경 생성
    env = DummyVecEnv([lambda: CryptoSniperEnv(train_df)])

    # 🚨 처방 2: 환경에 '정규화 방독면(VecNormalize)' 씌우기
    # 이 래퍼(Wrapper)가 52억 같은 숫자를 실시간으로 -10 ~ 10 사이의 안전한 Z-Score로 자동 변환해 줍니다.
    env = VecNormalize(
        env, 
        norm_obs=True,     # 관측값(피처) 정규화 활성화
        norm_reward=True,  # 보상값 정규화 활성화 (학습 안정성 극대화)
        clip_obs=10.0      # 너무 튀는 이상치(Outlier)는 10으로 컷오프
    )

    # 3. PPO 에이전트 생성
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=3e-5,  # 그레이디언트 폭발 방지를 위해 학습률을 살짝 낮춤
        n_steps=1024,
        batch_size=256,
        gamma=0.99,       
        ent_coef=0.02,    
        verbose=1,
        tensorboard_log="logs/tensorboard/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='./models/',
        name_prefix='rl_sniper'
    )

    logger.info("🔥 수십만 번의 모의 전투(학습) 시작...")
    model.learn(total_timesteps=300000, callback=checkpoint_callback) 
    
    # 🚨 처방 3: 정규화 통계치도 함께 저장해야 실전에서 쓸 수 있습니다.
    model.save("ppo_crypto_sniper")
    env.save("vec_normalize.pkl") # 정규화 기준치 저장
    logger.info("✅ 훈련 완료 및 모델 저장 (ppo_crypto_sniper.zip, vec_normalize.pkl)")

if __name__ == "__main__":
    main()