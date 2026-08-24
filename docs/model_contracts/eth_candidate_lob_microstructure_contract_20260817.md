# ETH 오더북(LOB) 마이크로구조 DL 후보 — 데이터 계약 (2026-08-17)

이 문서는 **공식 Odyssey 계보(Odyssey1~4)에 속하지 않는다** — 확정된 성과가 있을 때만 번호를
올린다는 원칙에 따라, 아직 데이터소스 스코핑 단계이므로 "Odyssey5"로 명명하지 않는다.

## 상태

| 컴포넌트 | 상태 |
|---|---|
| **데이터소스 스코핑** | **완료(2026-08-17).** 핵심 발견: 원시 L2 저장(WS-E)이 이미 설계·격리검증까지 끝났으나 프로덕션 미배선 상태로 한 달간 방치돼 있었음. |
| **프로덕션 배선(WS-E E1)** | **완료·라이브 검증됨(2026-08-17).** git `d047de1` 배포, ETH/BTC/SOL 전부 원시 20레벨(`bids_json`/`asks_json`) 축적 시작. E2(연속 10초 스냅샷)는 의도적으로 보류. 상세는 데이터 리소스 레지스트리 "프로덕션 배선 완료" 절. |
| **모델링** | **1단계(파이프라인 스모크테스트) 완료(2026-08-22, PASS)** — 아래 "1단계 파이프라인 스모크테스트" 절 참고. 2단계(예비 신호점검)는 2026-09-14부터, 3단계(프로모션급)는 2026-11-17부터 — 데이터가 물리적으로 부족해 이번엔 시도 안 함. |
| **부속 축 — 게이트/거부 재질문 cheap-gate → 단일터치 OOS-Q2 실행** | **2026-08-22 판정보류로 종결(표본부족).** cheap-gate(약한 양성, +9~17% 크기신호)에 이어 사용자 승인으로 OOS-Q2 조인까지 실제 실행 — 겹침구간 트레이드 N=8뿐이라 핵심 가설(microstructure가 실제 승패를 가르는가) 확인도 반박도 못 함. OOS-Q2는 이 가설에 대해 소진됨(단일터치). 상세: `docs/experiments/eth_candidate_microstructure_veto_gate_cheap_check_20260822.md` §10. |

## 범위

- 모델 id: `eth_candidate_lob_microstructure_20260817`
- 목적: [DL-for-crypto-trading 전수조사](../deep_learning_for_crypto_trading_literature_survey_20260817.md)
  5절이 지목한, 이 저장소가 유일하게 탐색하지 않은 DL 축(오더북/마켓 마이크로구조)의 실현
  가능성을 데이터 측면에서 먼저 검증한다. TLOB(arXiv:2502.15757)가 비트코인 데이터로 검증된
  드문 사례이고, Wang(2025, arXiv:2506.05764)가 "피처가 깊이를 이긴다"는 이 저장소의 반복된
  결론을 LOB 도메인에서 독립 재현했다는 점이 재탐색 근거다.
- 아키텍처 유형: 미정(데이터 확보 전 단계). 문헌상 후보는 DeepLOB류 CNN+LSTM, TLOB류 이중어텐션
  트랜스포머, 또는 raw LOB 대신 OFI 파생 피처 기반 경량 모델(Kolm/Turiel/Westray 패턴).
- Owner agent: Model Architect(단독, Sonnet).
- 리소스 레지스트리: [`eth_candidate_lob_microstructure_data_resources_20260817.md`](eth_candidate_lob_microstructure_data_resources_20260817.md)
- 관련 문서: [`docs/deep_learning_for_crypto_trading_literature_survey_20260817.md`](../deep_learning_for_crypto_trading_literature_survey_20260817.md) 5절/7절,
  `docs/duckdb_live_data_utilization_design_20260719.md`(기존 라이브 데이터 인벤토리 원본),
  `docs/test_designs_duckdb_live_20260719/ws_e_data_flywheel.md`(원시 L2 저장 설계 원본)

## 데이터소스 스코핑 결론

1. **원시 L2 레벨 저장은 이미 만들어져 있다.** 2026-07-19 WS-E 설계·파일럿에서 round-trip
   자가검증 통과, 격리 연구 DB에서 53.08/72시간 소크(coverage 100%, 오류 0) — 여기서 멈췄다.
   프로덕션(`orderbook_recorder.py`)에는 한 번도 배선되지 않았고, 2026-08-17 서버 직접 조회로
   재확인했다(`bids_json`/`asks_json` 컬럼 없음, `orderbook_periodic_snapshots` 테이블 없음).
2. **신규 수집 없이 바로 쓸 수 있는 자원이 두 개 있다**: (a) ETH 3.5개월치 연속 1분봉
   오더플로우 파생 피처(`microstructure_1m`, 34컬럼 — OFI 파생 피처 패턴, 문헌 5.2절과 정합),
   (b) ETH 96일치 의사결정-조건부 20레벨 L2 요약(`orderbook_decision_snapshots` — 표본이 성기고
   샘플링 편향 있음, 원시 레벨은 없음).
3. 전체 상세는 데이터 리소스 레지스트리 참고.

## 완료된 작업 (2026-08-17)

- **B. 프로덕션 배선**: 완료. `orderbook_recorder.py`에 raw 레벨 저장 추가, git `d047de1` →
  CI 통과 → `deploy_watcher.sh`로 `trading-bot.service` 재시작(60s 헬스체크 통과) → 서버
  직접 조회로 ETH/BTC/SOL 전부 raw 레벨 축적 시작 확인. 데이터 에폭 경계는 `data_epochs.json`에
  확정 시각 기록됨(ETH/BTC/SOL 각 2026-08-17T01:43 KST대).
- **C. Binance bookDepth 스키마 검증**: 완료. 실제 파일 다운로드 결과 원시 레벨이 아니라
  30초 간격 %밴드(±0.2~5%) 누적 집계임을 확인 — raw-LOB 모델링엔 부적합, 보조 참고자료로만
  가치. 상세는 데이터 리소스 레지스트리.

## 부속 축 — 게이트/거부(veto/gate) 재질문 cheap-gate 체크 (2026-08-22)

이 문서의 본 축(원시 L2/raw-LOB 딥러닝)과는 별개로, `microstructure_1m`(위 리소스 레지스트리에
이미 등록된 자원)을 **엔트리 알파가 아니라 zig075/h48qual 기존 진입신호의 거부/게이트**로
재질문하는 저비용 체크를 수행했다 — 리소스 레지스트리 표가 이미 이 방향을 "엔트리 알파가
아니라 게이트/거부/청산타이밍 피처로만 재탐색 가치 있음"이라고 명시했던 것의 실행이다.

**핵심 발견**: 원래 계획한 "zig075/h48qual 신호 시점의 microstructure 값 vs 트레이드 승패" 조인
테스트는 **구조적으로 실행 불가**했다 — `microstructure_1m`은 2026-05-03부터만 존재하는데, 이
프로젝트에 등록된 비-OOS 판정/참고창(2025q1~q3, val=2025-10~12)은 전부 그 이전이라 0% 겹치고,
유일하게 겹치는 구간(2026-05~06)은 OOS-Q2 내부라 이번엔 건드리지 않았다(세션 규율). 대체로
`docs/duckdb_live_data_utilization_design_20260719.md` WS-C가 이미 사전승인해 둔 "조건부
수익률-분포" 방법론(트레이드 원장 대신 원시 가격의 조건부 미래-분포만 검정)으로 탐색 스캔만
수행 — `shadow_toxicity_regime`/`shadow_queue_collapse`가 day-block bootstrap 기준 통계적으로
실재하는 크기/변동성 신호(방향 아님, +9~17% 상대효과)를 보였고, `nif_whale`은 가설과 반대
방향으로 유의했다. `kelly_mult`/`signal_bias`(라이브 대시보드용으로만 계산되고 실제 의사결정
경로엔 전혀 연결 안 된 죽은 파생값으로 확인)도 함께 점검했다.

**판단**: 계속 볼 가치는 있으나(신호가 완전히 죽지는 않음) 지금 게이트/베토를 구현할 근거는
아니다. 상세 방법론·수치·한계: `docs/experiments/eth_candidate_microstructure_veto_gate_cheap_check_20260822.md`.

**⚠️ 2026-08-22 같은 날 후속 — 단일터치 OOS-Q2 실행, 판정보류로 종결**: 사용자가 OOS-Q2
예산을 이 가설에 쓰기로 명시 승인해 같은 날 바로 실행했다(신규 추론 없음, 기존
`portfolio_ledger_oos_q2_odyssey3_baseline.csv` 재사용). zig075/h48qual OOS-Q2 전체 13개
트레이드 중 `microstructure_1m` 커버리지(2026-05-03~)와 겹치는 **8건**만 조인 가능(승3/패5).
`shadow_queue_collapse`(스피어만 −0.333)/`signal_bias`(−0.412)가 cheap-gate의 방향과
정성적으로는 일치했으나, N=8은 통계적으로 어느 방향도 결론 낼 수 없는 표본이다 — **핵심
가설은 확인도 반박도 되지 못한 채 판정보류로 종결**한다. OOS-Q2는 이 가설에 대해 이제
소진됐다(단일터치, 재조회 금지). 표본을 늘리려 BTC/SOL이나 다른 창을 추가로 열지 않았다
(단일터치 취지 보호). 재개하려면 다음 OOS 세대(2026-09-30 이후 TRAIN/OOS 재정의)가 필요하다.
상세: 위 문서 §10.

## raw 레벨 최소 축적 기간 기준 (2026-08-17 확정)

에폭 시작(2026-08-17 01:43 KST대) 시점 실측 장기 평균 속도(`orderbook_decision_snapshots`
전체 이력 기준, 의사결정-조건부라 자산별로 다름): **ETH 148.3행/일, BTC 282.1행/일,
SOL 338.8행/일**. 이 속도를 근거로 3단계 기준을 둔다. 3단계(프로모션급)는 WS-E 설계 문서가
`microstructure_1m`/기존 요약 데이터에 이미 적용한 "3개월 누적 후 학습 해금"
선례(T-E4, BTC/SOL 2026-07-14 시작→2026-10-14 해금)와 **동일한 정책을 raw 레벨에도 그대로
적용**한 것 — 임의로 새로 정한 숫자가 아니다.

| 단계 | 목적 | 최소 기간 | 해금일 | 예상 표본(ETH/BTC/SOL) | 승격 근거 가능 여부 |
|---|---|---|---|---|---|
| 1. 파이프라인 스모크테스트 | raw JSON 파싱→피처추출 코드가 도는지만 확인 | 즉시 | **이미 충족** | WS-E 격리 파일럿 19,110행(53시간)으로 충분, 프로덕션 표본 소량 추가로 경로 차이 확인 | 아니오 — 코드 동작 확인용 |
| 2. 예비 신호존재 점검 | IC/상관 등 방향성 신호가 있는지 러프하게만 확인(탐색용) | 4주 | **2026-09-14** | ETH ~4,150 / BTC ~7,900 / SOL ~9,490행 | 아니오 — 탐색적 결과일 뿐, 이 저장소의 cheap_gate 기준(기존 다개월 VAL 구간 재생)에 못 미침 |
| 3. 프로모션급 N≥5시드 학습 + 워크포워드 | 이 저장소의 정식 결과 기준(N≥5 시드, purged CV, Fresh-Forward 워크포워드) 적용 | 3개월(WS-E T-E4 선례) | **2026-11-17** | ETH ~13,640 / BTC ~25,950 / SOL ~31,170행 | 예 — 이 시점 이후에만 승격/모델선택 근거로 사용 가능 |

**적용 규칙**: 2026-11-17 이전에 나온 raw-레벨 기반 모델링 결과는(설령 헤드라인 지표가 좋아
보여도) 표본 부족을 이유로 리뷰에서 반려한다 — WS-E T-E4가 이미 정한 "그 전 학습 시도는
리뷰에서 반려" 원칙을 그대로 따름. 1~2단계는 파이프라인/방향성 점검 목적으로는 지금 바로
진행 가능하다.

## 1단계 파이프라인 스모크테스트 (2026-08-22, PASS)

사용자가 raw L2/OFI DL 축을 "1번"으로 지정해 즉시 착수 승인 — 위 3단계 기준의 **1단계만**
수행했다(2단계/3단계는 데이터가 물리적으로 부족해 시도하지 않음). 스크립트:
`scripts/eth_candidate_lob_ofi_pipeline_smoke_test_20260822.py`.

**데이터**: WS-E 격리 파일럿(`data/research/ws_e_orderbook_raw_pilot.duckdb`,
`orderbook_periodic_snapshots_eth_soak_20260719`, 19,121행, ETH/USDT 선물 상위20레벨,
~10초 간격, 2026-07-19~21 53시간) — 라이브 프로덕션과 무관한 격리 연구 DB.

**피쳐 파이프라인**: `bids_json`/`asks_json`을 파싱해 Cont, Kukanov & Stoikov(2014)의 레벨별
OFI(order flow imbalance) 이벤트 정의를 순위(rank) 기준으로 레벨 1/5/10에 확장 적용(Cont/
Cucuringu/Zhang 2023, Kolm et al. 2023 "OFI 피쳐가 raw LOB보다 낫다" 결론과 정합하는 설계) +
스프레드 + 기존 저장된 `imbalance_{1,5,10}`. DeepLOB(Zhang/Zohren/Roberts 2019)류 최소
Conv1d+LSTM 골격(SEQ_LEN=50 스냅샷≈8.3분).

**⚠️ 2026-08-22 같은 날 재작성**: 최초판 결과(test_bce=1.2687, 절편전용 하한 0.6931보다
훨씬 나쁨)를 사용자가 "너무 대충 만들어서 학습기법이 전혀 적용 안 된 것 같다"고 지적 — 타당했다.
최초판은 Adam lr=1e-3 고정, dropout/weight-decay 없음, train/test 2분할뿐(val 미관측,
매 epoch 커브 로깅 없이 4개 체크포인트만), 조기종료 없음, 시퀀스 stride=1(인접 시퀀스 98%
중복 — 표본 과잉계산). [[feedback_modern_dl_training_checklist]]/[[reference_dl_layer_design_
training_20260816]](이 프로젝트가 이미 N≥5로 검증해 둔 관행) 확인 후 재작성: AdamW(wd=1e-2)+
dropout=0.2, lr=2e-4→2e-6 cosine(이 프로젝트에서 가장 잘 검증된 lever), Prechelt(1998)
UP₄ strip 조기종료, train/val/test 60/20/20+경계 purge, stride=5로 중복 완화, 매 epoch
train+val 전체 커브 로깅("진단습관" 원칙 적용).

**결과 — PASS(공학적 의미로만), 훨씬 정직한 곡선**: train/val BCE가 200epoch 내내 완만하게
움직이고(train 0.693→0.68 근처, val 0.693→0.690 근처, 폭주 없음) Prechelt UP₄는 이번엔
트리거 안 됨(강한 정규화 하에서 val이 뚜렷이 악화되지 않았다는 뜻). **test_bce=0.6938로
절편전용 이론하한(0.6927)에 거의 근접**(최초판 1.2687 대비 극적 개선). **핵심 교훈: 최초판의
파국적 과적합(1.27)은 데이터 부족(53시간·단일레짐)만이 아니라 학습기법 부재의 몫이 컸다** —
같은 데이터로 기법만 고쳐도 "과적합 폭발"에서 "절편수준에 근접한 정직한 무신호"로 바뀌었다.
`sign(OFI) == sign(다음 mid변화)` 일치율 81.7%도 관찰됐으나, 이는 같은 10초 구간 안에서
order flow와 가격이 거의 동시에 움직이는 **동시성** 관계이지 예측력 주장이 아니다(참고용으로만
기록, "적용 규칙" 문단에 따라 신호 근거로 쓰지 않음). 아키텍처 다이어그램(레이어 구성+
학습레시피+before/after): 사용자에게 아티팩트로 전달됨(세션 기록).

**의미**: 파이프라인(OFI 피쳐엔지니어링+DeepLOB 골격+검증된 학습레시피)이 재사용 가능한
형태로 준비됐다 — 2026-09-14 이후 2단계(예비 신호점검)에 착수할 때 이 스크립트의 피쳐 함수
(`_multilevel_ofi` 등)와 학습레시피(AdamW+cosine+Prechelt+purge)를 프로덕션 데이터
(`orderbook_decision_snapshots`의 raw 레벨, 2026-08-17~ 축적 중)에 그대로 적용하면 된다.

## 1단계 — 트랜스포머 버전(OFI-TLOB-lite) 추가 + N=5 시드검증 + DeepLOB 통제비교 (2026-08-22, 4~6차 수정)

사용자 지시("작은 Conv1d+LSTM보다 트랜스포머가 대세")로 [[feedback_dl_architecture_requires_
user_confirmation]] 원칙(신규 등록, 이 축이 최초 적용 사례)에 따라 **아키텍처를 먼저
아티팩트로 제시해 컨펌받은 뒤** 구현. 스크립트: `scripts/eth_candidate_lob_tlob_transformer_
smoke_test_20260822.py`(기존 DeepLOB판 `..._smoke_test_20260822.py`의 OFI 피쳐엔지니어링을
그대로 import해 재사용, 대체가 아니라 병행 — 둘 다 남겨둠).

**문헌근거**: TLOB(Berti & Kasneci 2025, arXiv:2502.15757 — Bitcoin 데이터 포함 검증, Dual
Attention+Bilinear Norm+MLPLOB)을 주 골격으로, Stochastic Depth(Huang 2016, survival_prob을
54블록 원논문 0.5에서 우리 2블록에 맞춰 0.8로 완화)와 AttentionDrop(2025, arXiv:2504.12088 —
메타데이터만 확보, MHA 내장 dropout으로 같은 지점 타겟) 추가. ⚠️ 정직한 반증 병기: 이 축의
"설계도"로 인용해온 Bieganowski & Ślepaczuk(2026, arXiv:2602.00776)를 재확인하니 실제로는
CatBoost(GBDT)를 쓰지 트랜스포머가 아니었다 — 문헌이 트랜스포머를 압도적으로 지지한다는
착각은 금물, 이 저장소 자체 발견(Kaggle 크립토대회 전부 선형/GBDT 우승)과도 같은 방향.

**컨펌된 각색**(원본 대비): 입력=OFI 7피쳐 유지(40차원 원시레벨 대신), 블록수=2(원본4),
출력=3진(train tercile 기반 경계). 상세 비교표는 스크립트 docstring 참고.

**결과 — PASS(공학적 의미로만)**: 파라미터 32,230개, Stochastic Depth survival_prob=
[1.0, 0.8], Prechelt UP4가 epoch=29에서 자연스럽게 조기종료(best epoch=7). **test_ce=1.1186,
test_acc=0.338** — 3진 균등추측 이론엔트로피(ln3≈1.099)에 근접, 다수클래스 기준선
정확도(0.368)보다도 낮음. DeepLOB판의 최초 파국적 결과(test_bce 1.27, 하한대비 +83%) 같은
폭주는 없음 — 처음부터 검증된 학습기법(early-dropout+Prechelt+purge)을 적용했기 때문.
아키텍처를 트랜스포머로 바꿔도 **데이터량(53시간)이 그대로라 결론은 동일**: 신호/성능 주장
불가, Tier-2(2026-09-14) 대기.

**5차 — N=5 시드검증(2026-08-22, 같은 날)**: 사용자가 "트랜스포머를 최적화해보자"고 요청 →
튜닝 전에 시드분산부터 확인([[tabm_hp_low_signal_pattern]]/CLAUDE.md Seed-Diversity Gate
정신). handoff.sh로 서버에서 시드 5개(20260822/71305/419283/88821/550091) 실행(도중 스크립트의
`ROOT` 하드코딩을 [[reference_dev_server_handoff]]의 기존 gotcha와 동일 패턴으로 발견·수정,
동적 경로로 교체). **결과**: test_ce 5개 전부 [1.1035, 1.1186] 구간에 밀집(mean=1.1106,
std≈0.0058), 전부 이론엔트로피(ln3≈1.0986) 대비 +0.45%~+1.8% 이내 — 튜닝으로 밀어낼 여지가
실질적으로 없다는 뜻(under-training 아니라 이미 floor 근접). test_acc([0.317, 0.357],
mean=0.341)는 다수클래스 기준선(0.368, test셋 고정이라 5개 시드 동일값)보다 5/5 전부 낮지만,
이는 결함이 아니라 비교 자체의 구조적 편향으로 판단 — `majority_acc`는 고정 test셋의 **사후**
라벨분포를 아는 값인데 train tercile로 캘리브레이션된 모델은 53시간 단일창의 국소적 라벨쏠림을
미리 알 수 없어 구조적으로 불리하다. **결론: 현재 데이터(Tier-1, 53시간)에서는 하이퍼파라미터
/아키텍처를 더 튜닝해도 이 수치는 실질적으로 움직이지 않는다** — 벽은 학습기법이 아니라
데이터량이다. 상세: [[eth_candidate_lob_ofi_pipeline_smoke_test_20260822]] "5차" 절.

**6차 — DeepLOB vs 트랜스포머 통제비교(2026-08-22, 같은 날)**: "트랜스포머가 CNN+LSTM보다
나은가?" 질문에 답하기 위해 TLOB를 DeepLOB와 동일한 이진 타겟+BCELoss로 맞추고(사용자 제안,
아키텍처는 불변) 동일 5시드로 재비교. **결과**: DeepLOB test_bce mean=0.6957±0.0038, TLOB
mean=0.6964±0.0042(둘 다 하한 0.6927 대비 +0.5% 이내) — **평균차(0.0007)가 양쪽 시드-std보다
작아 유의한 차이 없음**. 53시간 데이터에서는 두 아키텍처 다 같은 정보이론적 벽에 부딪혀 있어
어느 쪽이 나은지 가릴 실익이 없다(TLOB 원논문의 우위는 FI-2010/NASDAQ 같은 훨씬 큰 데이터셋
기준). Tier-2(09-14) 재비교 전까지 아키텍처 우열 주장은 보류. 상세:
[[eth_candidate_lob_ofi_pipeline_smoke_test_20260822]] "6차" 절.

## 다음 단계

- 지금 바로 시작 가능한 것: (a) ~~`microstructure_1m` 오더플로우 피처(3.5개월 축적됨)로 게이트/
  거부 피처 탐색~~ — **cheap-gate + 단일터치 OOS-Q2 실행까지 완료(2026-08-22), 표본부족(N=8)
  판정보류로 종결**, 위 "부속 축" 절 및
  `docs/experiments/eth_candidate_microstructure_veto_gate_cheap_check_20260822.md` §10 참고.
  재개는 다음 OOS 세대(2026-09-30 이후) 대기. (b) WS-E 격리 파일럿 19,110건(53시간)으로
  DeepLOB류 파이프라인 스모크테스트(1단계), (c) 2026-09-14 이후 예비 신호존재 점검(2단계).
- **A. WS-E 72h 소크 재개**(격리 연구 DB, 낮은 리스크): 이번 라운드에선 보류. 필요시 나중에.
- **E2(연속 10초 스냅샷) 프로덕션 배선**: 별도 봇 지연 회귀 테스트가 선행돼야 하는 더 큰
  변경이라 이번엔 하지 않았다 — 별도 세션/승인 필요.
- **2026-11-17**: 3단계(프로모션급) 해금일 — 이 시점에 축적량 재확인 후 본격 모델링 착수
  여부 결정.

## Open Issues

- WS-E 격리 파일럿의 72h 소크가 왜 53.08h에서 멈췄는지(의도적 중단 vs 세션 종료) 미확인 —
  낮은 우선순위.
- ~~게이트/거부 재질문의 핵심 가설이 미검증 상태로 남아 있다~~ — **해소(2026-08-22 같은 날
  후속)**: 사용자 승인으로 OOS-Q2 단일터치 실행함. 단 결과가 표본부족(N=8)으로 판정보류라
  가설 자체는 여전히 미확인 — 차이는 "검증 시도를 안 했음"에서 "검증 시도했으나 표본이
  결론을 못 낼 만큼 작았음"으로 바뀐 것. OOS-Q2는 이제 이 가설에 대해 소진됨, 재개는 다음
  OOS 세대(2026-09-30 이후) 필요.
- Tardis.dev 등 유료 소스는 프로덕션 축적으로 부족하다고 판단될 때만 재검토 — 현재는 불필요.
