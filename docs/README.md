# 📚 문서 디렉토리

암호화폐 트레이딩 강화학습 프로젝트의 상세 기술 문서입니다.

---

## 📖 문서 목록

### 1. [모델 아키텍처 명세](./model_architecture.md)

**내용**:
- PPO 및 TD3 모델의 전체 구조
- QuantTransformerBackbone 상세 설명
- Elite 8 전략 통합 메커니즘
- 44개 Ultimate Features 명세
- 하이퍼파라미터 및 최적화 기법

**대상**: 모델 구조를 이해하고 싶은 개발자

---

### 2. [데이터 흐름 명세](./data_flow.md)

**내용**:
- 원시 데이터 → 모델 입력까지 전체 파이프라인
- 피처 엔지니어링 세부 과정
- 전략 신호 생성 및 캐싱
- GPU 메모리 최적화
- 데이터 크기 및 메모리 사용량

**대상**: 데이터 처리 과정을 파악하고 싶은 개발자

---

### 3. [학습 파이프라인 명세](./training_pipeline.md)

**내용**:
- PPO Curriculum Learning 전략
- TD3 Replay Buffer 및 CQL
- Sortino Ratio 기반 보상 함수
- 평가 및 벤치마크
- 하이퍼파라미터 튜닝 가이드
- 트러블슈팅

**대상**: 학습 프로세스를 이해하고 튜닝하고 싶은 개발자

---

## 🚀 빠른 시작

### 1. 프로젝트 구조 이해
```
1. model_architecture.md 읽기 → 모델 이해
2. data_flow.md 읽기 → 데이터 파이프라인 이해
3. training_pipeline.md 읽기 → 학습 프로세스 이해
```

### 2. 학습 실행
```bash
# PPO 학습
python .\macroHFT\train_ppo.py

# TD3 학습
python .\TD3\train_td3.py
```

### 3. 평가
```bash
# PPO 평가
python .\macroHFT\evaluate_ppo.py

# TD3 평가
python .\TD3\evaluate_td3.py
```

---

## 📊 핵심 스펙 요약

### 모델
- **PPO**: MoE (3 Experts + Router), Transformer Backbone
- **TD3**: Position-Aware Actor, Twin Critics, CQL
- **공통**: QuantTransformer + Elite 8 Strategies

### 데이터
- **피처**: 44개 Ultimate Features
- **전략**: 8개 Elite Strategies
- **시퀀스**: 60틱 (3분봉 기준 3시간)
- **데이터**: ~175,000 캔들 (약 1년)

### 성능
- **학습 속도**: ~40초/episode (AMP + Torch Compile)
- **GPU 메모리**: ~3GB (FP16)
- **성능 개선**: 2.5배 (최적화 후)

---

## 🔧 최적화 적용 내역

### ✅ AMP (Automatic Mixed Precision)
- FP16 연산으로 메모리 50% 절감
- GradScaler로 안정성 확보
- 학습 속도 2배 향상

### ✅ Torch Compile
- PyTorch 2.0 그래프 최적화
- TorchInductor CUDA 최적화
- 25% 추가 속도 향상

### ✅ GPU 캐싱
- 피처 및 전략 신호 사전 계산
- CPU→GPU 전송 제거
- Episode 시작 시간 95% 단축

### ✅ Sortino Ratio 보상
- 단순 PnL → 위험 조정 수익
- 하방 변동성 페널티
- 안정적인 수익 추구

---

## 📈 성능 벤치마크 (RTX 3070Ti)

| 항목 | 최적화 전 | 최적화 후 | 개선율 |
|------|----------|----------|--------|
| Episode 시간 | ~100초 | ~40초 | **2.5배** |
| GPU 메모리 | ~6GB | ~3GB | 50% 절감 |
| Steps/sec | ~1.2 | ~3.0 | **2.5배** |
| Batch Size | 128 | 256 | 2배 |

---

## 🛠️ 요구사항

### 하드웨어
- **GPU**: NVIDIA RTX 3060 이상 (8GB VRAM)
- **CPU**: 8코어 이상
- **RAM**: 16GB 이상

### 소프트웨어
- **Python**: 3.9+
- **PyTorch**: 2.0+ (Torch Compile 필수)
- **CUDA**: 11.7+

---

## 📝 문서 업데이트 이력

- **2026-02-06**: 초기 문서 작성
  - model_architecture.md
  - data_flow.md
  - training_pipeline.md
- **변경 사항**: AMP, Torch Compile, Sortino Reward 반영

---

## 💡 추가 리소스

- **TensorBoard**: `tensorboard --logdir=logs/tensorboard`
- **학습 로그**: `logs/train_ppo.log`, `logs/train_td3.log`
- **모델 체크포인트**: `data/ppo/`, `data/td3/`
- **피처 캐시**: `data/training_features.csv`

---

**작성자**: AI 개발팀  
**최종 업데이트**: 2026-02-06  
**버전**: 1.0.0
