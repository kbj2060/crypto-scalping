"""
DDQN 모델 평가 스크립트 (Evaluation Mode)
탐험(Epsilon)을 0으로 설정하여 AI의 '순수 실력'만 테스트합니다.
"""
import sys
import os
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.train_dqn import DDQNTrainer
import config

# 로깅 설정
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluate.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def evaluate(episodes=10):
    """
    DDQN 모델 평가 함수
    
    Args:
        episodes (int): 평가할 에피소드 수
    """
    logger.info("=" * 60)
    logger.info("🧪 DDQN 모델 평가 모드 시작 (탐험률 0.0% - 순수 실력 검증)")
    logger.info("=" * 60)

    # 1. 학습 환경과 동일하게 트레이너 초기화
    # (XGBoost 피처 선택 등 모든 전처리 과정을 동일하게 수행)
    try:
        trainer = DDQNTrainer(force_recalculate_strategies=False)
    except Exception as e:
        logger.error(f"트레이너 초기화 실패: {e}", exc_info=True)
        return

    # 2. 학습된 모델 가중치 로드
    model_path = config.DDQN_MODEL_PATH
    if os.path.exists(model_path):
        try:
            trainer.agent.load_model(model_path)
            logger.info(f"💾 모델 로드 완료: {model_path}")
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}", exc_info=True)
            return
    else:
        logger.error(f"❌ 학습된 모델 파일이 없습니다: {model_path}")
        return

    # 3. [핵심] 탐험(Epsilon)을 강제로 0으로 설정
    trainer.agent.epsilon = 0.0
    trainer.agent.epsilon_end = 0.0  # epsilon_end도 0으로 설정
    
    # 4. 신경망을 평가 모드로 전환 (Dropout 비활성화 등)
    trainer.agent.policy_net.eval()
    trainer.agent.target_net.eval()
    
    # 5. [핵심] 학습을 비활성화하기 위해 train_step을 임시로 오버라이드
    original_train_step = trainer.agent.train_step
    def no_train_step():
        """평가 모드에서는 학습하지 않음"""
        return None
    trainer.agent.train_step = no_train_step
    
    eval_rewards = []
    eval_steps = []
    
    # 5. 평가 루프 실행
    logger.info(f"📊 평가 시작: {episodes}개 에피소드")
    logger.info("-" * 60)
    
    for ep in range(1, episodes + 1):
        try:
            # train_episode 함수를 호출하되, agent.epsilon이 0이므로 항상 최적 행동 선택
            # train_step이 오버라이드되어 있으므로 학습은 수행되지 않음
            result = trainer.train_episode(ep, max_steps=1000)
            
            if result:
                reward, steps = result
                eval_rewards.append(reward)
                eval_steps.append(steps)
                logger.info(f"📝 Test Ep {ep}/{episodes}: Score {reward:.2f} | Steps {steps} (100% 실력 매매)")
            else:
                logger.warning(f"⚠️ Ep {ep}: 데이터 부족으로 스킵됨")
                
        except Exception as e:
            logger.error(f"에피소드 {ep} 평가 중 오류: {e}", exc_info=True)
            continue

    if len(eval_rewards) == 0:
        logger.error("❌ 평가 결과가 없습니다. 데이터를 확인해주세요.")
        return

    # 6. 결과 분석
    avg_score = np.mean(eval_rewards)
    std_score = np.std(eval_rewards)
    max_score = np.max(eval_rewards)
    min_score = np.min(eval_rewards)
    positive_episodes = sum(1 for r in eval_rewards if r > 0)
    win_rate = (positive_episodes / len(eval_rewards)) * 100
    
    logger.info("=" * 60)
    logger.info(f"📊 평가 종료 (총 {len(eval_rewards)}회 성공)")
    logger.info(f"🏆 평균 점수: {avg_score:.2f} ± {std_score:.2f}")
    logger.info(f"📈 최고 점수: {max_score:.2f}")
    logger.info(f"📉 최저 점수: {min_score:.2f}")
    logger.info(f"✅ 수익 에피소드: {positive_episodes}/{len(eval_rewards)} ({win_rate:.1f}%)")
    logger.info("=" * 60)
    
    # 진단 메시지
    if avg_score > 50:
        logger.info("✅ 진단: AI가 아주 훌륭한 수익 모델을 구축했습니다!")
    elif avg_score > 20:
        logger.info("✅ 진단: AI가 안정적인 수익을 내고 있습니다.")
    elif avg_score > 0:
        logger.info("⚠️ 진단: 수익을 내고는 있지만, 더 정교한 튜닝이 필요합니다.")
    elif avg_score > -20:
        logger.info("⚠️ 진단: 손실이 발생하고 있습니다. 학습 파라미터 재검토가 필요합니다.")
    else:
        logger.info("❌ 진단: 학습이 제대로 되지 않았습니다. (모델 구조/보상 재검토 필요)")
    
    # 그래프 그리기
    try:
        plt.figure(figsize=(12, 6))
        
        # 서브플롯 1: 에피소드별 점수
        plt.subplot(1, 2, 1)
        colors = ['green' if r > 0 else 'red' for r in eval_rewards]
        plt.bar(range(1, len(eval_rewards) + 1), eval_rewards, color=colors, alpha=0.7)
        plt.axhline(y=avg_score, color='blue', linestyle='--', linewidth=2, label=f'Avg: {avg_score:.2f}')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        plt.title('Evaluation Performance (Zero Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 2: 점수 분포
        plt.subplot(1, 2, 2)
        plt.hist(eval_rewards, bins=min(20, len(eval_rewards)), color='skyblue', alpha=0.7, edgecolor='black')
        plt.axvline(x=avg_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_score:.2f}')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        plt.title('Reward Distribution')
        plt.xlabel('Total Reward')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('logs/evaluation_result.png', dpi=150)
        plt.close()
        logger.info("📊 결과 그래프 저장: logs/evaluation_result.png")
    except Exception as e:
        logger.warning(f"그래프 저장 실패: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DDQN 모델 평가')
    parser.add_argument('--episodes', type=int, default=10, help='평가할 에피소드 수 (기본: 10)')
    
    args = parser.parse_args()
    
    try:
        evaluate(episodes=args.episodes)
    except KeyboardInterrupt:
        logger.info("평가 중단")
    except Exception as e:
        logger.error(f"치명적 오류: {e}", exc_info=True)
