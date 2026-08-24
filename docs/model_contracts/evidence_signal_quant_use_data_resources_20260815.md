# 증거 신호 퀀트 활용 — 데이터 및 리소스 관리 (2026-08-15)

이 문서는 증거 신호 퀀트 활용 서브프로젝트(`docs/model_contracts/evidence_signal_quant_use_contract_20260815.md`)에서 실제로 만지거나 검토한 모든 데이터 소스/리소스 목록이다. 계약 문서는 상태와 결과 요약만 다루고, 위치·커버리지·함정은 여기서 관리한다.

**새 리소스를 만질 때마다 그 턴에 행을 추가/갱신할 것.**

## 패널 데이터 (재사용, 신규 수집 없음)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| 60코인 공통 피처 패널 | `data/panel/features/*.parquet` (60개, 2.2GB) | 심볼당 272,4xx~272,6xx bar, 2023-12-31 ~ 2026-08-04, 5분봉 | 횡단면 IC·이벤트 스터디의 유일한 입력 | 검증 완료 — 긍정 결과(사용 가능) | **원시 `volume`/`taker_buy_base` 없음** — `rvol_48`(=volume/rolling48평균, **clip(0,20)**)과 `taker_buy_ratio`(**quote 기준**, base 아님)만 있음. 주문흐름 항은 프록시로 계산해야 함(아래 행 참조) |
| 유니버스 정의 | `data/splits/panel_universe_symbols_20260804.json` | 60심볼 | 심볼 목록 + 편향 기록 | 활성 | 자체 문서화된 **유동성 룩어헤드**(2026-08-04 거래대금 기준 상위 60) + **생존편향**(2024-01-01 이전 상장 & 현재 거래중만). 보정 안 함 — 성과는 시장중립 초과수익으로만 주장 |
| 원시 ETH klines | `data/splits/year_oos/training_features_2025.csv` | 2025 전체, `volume`/`taker_buy_base` 포함 | 프록시 검증 정답 | 검증 완료 | Odyssey 계열이 쓰는 것과 동일 파일(별도 사본 안 만듦) |
| 원시 BTC klines | `data/btc_5m_1year.csv` | 224,353행, `volume`/`taker_buy_base` 포함 | 프록시 검증 정답(두 번째 심볼) | 검증 완료 | 파일명은 "1year"이나 실제 커버리지가 더 김 |

## 파생 신호

| 리소스 | 정의 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|
| 주문흐름 항 패널 프록시 | `z288(rvol_48 · (2·taker_buy_ratio − 1))` | 원시 `delta_z = z288(2·taker_buy_base − volume)` 대체 | 검증 완료 — 긍정 결과 | ETH Spearman **0.9697** / BTC **0.9641**, 이벤트 Jaccard 0.41~0.45. 사전등록 게이트(≥0.80 / ≥0.30) 통과. **`rvol_48`의 clip(0,20)이 거래량 클라이맥스 bar를 절단**하므로 완전 동치는 아님 |
| `votes` 점수 | 6신호 바닥투표 − 천장투표(ETH 단독 백테스트와 동일 구성) | 검증된 구성에 대한 충실도 | 검증 완료 — **부정 결과** | 반전 중립화 후 IC 부호가 뒤집힘(−0.0035~−0.0118) — **폐기 권고** |
| `continuous` 점수 | 같은 성분의 연속 표준화 합 | 순위 해상도 확보 | 검증 완료 — 부분 긍정 | 반전 중립화 후에도 +0.0103(t=9.7, h=12) 유지. 피처 후보로 사용 시 이쪽만 |

## 코드

| 파일 | 용도 | 상태 |
|---|---|---|
| `scripts/research_evidence_signal_cross_sectional_ic_60coin_20260815.py` | 0단계 프록시 검증 게이트 + 횡단면 IC(4 호라이즌 × 2 점수) + 순열/반전벤치마크/반전중립화/기간분할 대조 + 5분위 롱숏 | 완료 |
| `scripts/research_evidence_signal_cross_sectional_event_study_60coin_20260815.py` | 이벤트 구동 시장중립 스터디(K=2/3 × h=12/48/144, 왕복비용 정확 회계, 랜덤 이벤트 대조). 위 스크립트의 `build_signals`를 import 재사용 | 완료 |
| `scripts/research_evidence_signal_breadth_risk_gate_60coin_20260815.py` | 후보 #1: 브레드스 극값 에피소드 분석(방향/변동성/꼬리, 랜덤 에피소드 부트스트랩 1000회, 4기간 부호 일치, 사전등록 킬 기준 기계적 판정) + 노출 게이트 경제성. `build_breadth`는 후속 스크립트가 import 재사용 | 완료 — 부정 결과 |
| `scripts/research_evidence_signal_liquidity_bucket_ic_cost_60coin_20260815.py` | 후보 #1(경로 B): 유동성 버킷(T10/T11_30/T31_60)별 버킷 내부 IC + 5분위 롱숏 손익분기 bp + 이벤트 초과수익(버킷 자체 EW 대비) + 기간 부호 일치 | 완료 — 부정 결과(경로 B 종결), 단 **유동성-알파 단조 관계는 재사용 가치 있는 발견** |
| `scripts/research_evidence_signal_breadth_vol_forecast_60coin_20260815.py` | 후보 #1 결정 단계: 브레드스가 **과거 실현변동성 대비** 증분 예측력을 갖는지(2024 학습 → 2025H1/2025H2/2026 OOS ΔR²) + 변동성 타겟 사이징 경제성(1bar 지연 강제, 복리 경로 기준) | 완료 — 부정 결과(ΔR² 기준 미달) |

## 산출물

| 리소스 | 위치 | 용도 | 상태 |
|---|---|---|---|
| IC 스터디 리포트 | `tmp/causal_regen_20260516/evidence_signal_cross_sectional_ic_60coin_20260815/report.json` | 프록시 게이트 + 호라이즌별 IC/중립화/기간분할/롱숏/순열 전부 | 완료 |
| 이벤트 스터디 리포트 | `tmp/causal_regen_20260516/evidence_signal_cross_sectional_event_study_60coin_20260815/report.json` | 6개 (h,K) 칸의 raw/초과/손익분기bp/랜덤대조/기간분할 | 완료 |
| 브레드스 게이트 리포트 | `tmp/causal_regen_20260516/evidence_signal_breadth_risk_gate_60coin_20260815/report.json` | K∈{1,2,3}×P∈{0.95,0.99}×h∈{48,144,288,864}×양측 에피소드 분석 + 킬 기준 판정 행 + 노출 게이트 경제성 | 완료 |
| 유동성 버킷 분해 리포트 | `tmp/causal_regen_20260516/evidence_signal_liquidity_bucket_ic_cost_60coin_20260815/report.json` | 버킷×호라이즌×점수별 IC/롱숏 손익분기/이벤트 초과/기간분할 + 버킷 멤버 목록 + 미래참조 편향 기록 | 완료 |
| 브레드스 변동성 증분 리포트 | `tmp/causal_regen_20260516/evidence_signal_breadth_vol_forecast_60coin_20260815/report.json` | 호라이즌별 OOS R²/ΔR²/기간분할/학습계수 + 비용 그리드(0~50bp)별 변동성 타겟 성과 | 완료 |

## 미검증 후보 / 보류

- **브레드스 리스크 게이트**(계약 문서 다음 후보 1순위) — 패널은 이미 확보돼 있어 추가 수집 불필요.
- **패널 DL 모델 피처 추가**(2순위) — host는 `data/panel/ckpt/rho1_panel_backbone_*.pt` 계열. 피처셋 정의는 `scripts/train_rho1_panel_backbone_20260804.py`/`research_cross_sectional_panel_dl_60coin_20260810.py`의 컬럼 목록 확인 필요(미확인).
- 시점기준(point-in-time) 유니버스 재구축 — 미착수, 브레드스 후보 진행 시 편향 점검 필요.
