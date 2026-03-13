# 유닛 테스트 (IMPLEMENTATION_SPECIFICATION 기반)

`docs/IMPLEMENTATION_SPECIFICATION.md` 단계별 구현에 대한 유닛 테스트입니다.

## 구조

| 파일 | 대응 명세 | 내용 |
|------|-----------|------|
| `test_trading_env_reward.py` | §1 보상 함수 | DSR, Action Dampening, MDD 방지, reset_reward_states, trade_done 보너스 |
| `test_preprocess.py` | 전처리 | DataPreprocessor Rolling Norm: 상수/극소 입력 시 NaN·Inf 방지(epsilon), 윈도우 통계 정규화 |
| `test_xlstm_network.py` | §2 네트워크 | StabilizedSLSTMCell, HybridBackbone, StrategyAttention(shape + **전략 점수 변화 시 출력 변화**), XLSTMNetwork |
| `test_ppo_agent.py` | §3.4 PPO 에이전트 | put_data 7/8 호환, train_net aux_loss, select_action 반환 |
| `test_train_pipeline.py` | §3 학습 파이프라인 | aux_target 공식, 커리큘럼 인덱스, holding_time_norm, transition 8요소 |
| `test_config.py` | §5 설정 | REWARD_*, PPO_*, TRAIN_*, NETWORK_*, 데이터 분할 |

## 실행 방법

프로젝트 루트에서:

```bash
# pytest 설치 (없는 경우)
pip install pytest

# 전체 테스트
python -m pytest test/ -v

# 특정 파일만
python -m pytest test/test_trading_env_reward.py -v
```

`conftest.py`에서 프로젝트 루트를 `sys.path`에 넣으므로, 반드시 **프로젝트 루트**에서 실행하세요.
