# 사전등록: BTC·SOL 박스 유지 확률 — 코인 전용 HGB(30분·1h·4h) + 1h 굴림 Platt90 (2026-10-31)

**등급: 되돌리기 쉬운 예측 개선 — 단측 90%** (화면 표시 전용 확률 · 매매·크기 입력 아님 · 수수료 없음) + 완화 «통과 후보» 등급 병기.
**이 문서는 계산 전에 커밋한다.** 스크립트: `scripts/research_btc_sol_box_hold_20261031.py`(`--selftest` · `--run` 한 번만, report.json 있으면 거절).
사용자 «일부러 그대로 둔 것들도 모두 각각 코인에 맞게 재학습 및 최적화 진행해줘» — 박스 유지는 지금 ETH·XRP 만 서빙(`dashboard/box_hold.py` `HGB` 표).

## 질문
- **Q0 (HGB)** BTC·SOL 각각 XRP(10-20)와 **같은 정의·같은 레시피·같은 관문**으로 코인 전용 박스 유지 HGB 를 만들면 30분·1h·4h 각 척도가 관문을 넘나?
- **Q1 (1h 굴림 Platt90)** Q0 에서 1h 가 엄격 통과한 코인에 ETH·XRP 와 같은 «굴림 Platt90»(매월 1일 00:00 UTC, 직전 90일 해결 라벨로 Platt 재적합)을 얹으면 더 정확한가?
- 범위 밖(다음 라운드): BTC·SOL 4h 의 QX4/QX4I σ 등조 경로 — BTC·SOL 4h σ 예측(변동성 경주 동결 예측)이 아직 없다.

## 왜 (과거 실패 교훈 반영)
- XRP 10-20: ETH 모델을 그대로 XRP 에 쓰는 대신 코인 전용으로 다시 학습해 30분·1h 통과, 4h 는 2025 보정 6.3pp 로 deployable False → 같은 스크립트(`scripts/research_xrp_box_hold_20261020.py`)를 import 해 심볼만 바꾼다. 라벨 `research_eth_range_hold_multiscale_20261016.build` · 특징 27칸 · HGB 하이퍼파라미터 · 비교 모델 · 관문 전부 그대로(고를 것 없음).
- 플레이북 «HGB 함정»: early_stopping='auto' 는 1만 행 넘으면 무작위 10% 검증 → 시드 의존. 배포 레시피(random_state=0)로 판정하고, **시드 1 로 다시 맞춘 AUC·보정을 보조로 보고**한다(판정에 안 씀).
- 플레이북 «보정층은 배포 대상 그 자체로 잰다»: Q1 은 이번에 내보낸 그 HGB(2022~24 학습) 출력에 얹어 잰다. ETH 30분처럼 이미 보정된 모델에 얹으면 악화할 수 있다 → 1h 만, 사전 고정 V1~V5.
- 같은 구간 재사용 공개: BTC·SOL 1분봉 2025~2026 은 박스 유지 라벨로는 처음 채점한다. 다만 10-22 메이커 박스 순환 사다리 4코인 CPCV(전 설정 음수)가 BTC·SOL 박스 확률 «켜기»를 2022~25 에서 썼다(이 HGB 와 다른 모델). 예상 우연 통과 ≈ 코인 2 × 척도 3 × .10 + Q1 2 × .10 ≈ .8회(관문이 AUC·보정 수준 조건이라 실제론 더 적다).

## 양성 대조 (판정 전 필수)
같은 코드 경로로 **XRP 를 먼저 다시 돌려** 10-20 산출과 맞춘다(1분봉 끝을 10-19 23:59 로 자름):
- P0a 특징: 새로 낸 XRP 특징 = `tmp/xrp_parity_20261020/box_hold/features.parquet`(메인 체크아웃) 겹치는 행 최대 차 ≤ 1e-9.
- P0b 모델: 새로 학습한 XRP HGB 30분·1h·4h 의 예측 = 배포 `dashboard/box_hold_hgb_xrp_20261020.joblib` 예측(2025~ 행) 최대 차 ≤ 1e-9, 관문 결과·AUC 가 10-20 report 와 같다(30분 2025 .777469 등).
- P0c Q1: 배포 XRP 1h + 같은 Q1 함수(END 2026-10-20) → pooled ΔBrier = 10-31 XRP 결과 `docs/experiments/xrp_box_hold_platt90_qx4_20261031_report.json` 값(±1e-12).
- 하나라도 실패하면 BTC·SOL 채점 안 함.

## 데이터·시각 규약
- 1분봉 = `data/binance_vision/klines1m/<SYM>-1m-YYYY-MM.parquet`(2022-01~2026-09) + `klines1m_daily`(2026-10-01~10-27). **10-28 일 파일이 없어 1분봉 끝 = 2026-10-27 23:59 로 자른다**(빈칸 0 assert). 2022-01-01 부터 빈칸 없음 확인.
- 결정 = 5분봉 마감 분(30분·1h 5분마다, 4h 정시 직전 분 hh:59). 박스 = 종가 ± ½·지난 W분 고저폭. 유지 = 다음 W분(i+1..i+W) 1분봉 고가 최대 < 상단 & 저가 최소 > 하단. 특징은 i 분 마감까지(사건 라벨 경계 계약: 피쳐 끝 i < 라벨 시작 i+1). 결정 분 뒤 1분봉을 망가뜨려도 V·R·Q 특징 불변(누수 점검 assert, 3척도). S(요일×시각)·M(발표 일정) 칸은 미리 아는 달력.
- 요일×시각 표(prof) = 그 코인 2022~24 1분 수익² (코인별).
- 발표 일정 = ETH 연구 캐시 events.json(md5 fe960e7b, 자산 무관).
- 서빙 패리티: 연구 행 40개(발표 낀 10 포함) → `dashboard/box_hold.features/payload(asset=<코인>)` 와 특징 1e-6 · 확률 1e-4 일치 assert.
- Q1 라벨 관문 L1: 1h 행 라벨을 `research_eth_box_hold_calibration_drift_20261025.labels()` 로 1분봉에서 독립 재계산 → 100% 일치.

## 구간
| 구간 | 기간 | 쓰임 |
|---|---|---|
| 학습 | 2022-01-01 ~ 2024-12-31 | HGB 적합(배포 ETH·XRP 와 같음) |
| 판정 2025 | 2025-01-01 ~ 12-31 | Q0 관문 |
| 판정 2026 | 2026-01-01 ~ 10-27(라벨 해결분) | Q0 관문 |
| Q1 J1 / J2 | 2025-02-01~12-31 / 2026-01-01~10-27 | Q1 판정(합침) |
| 보고 | 2026-01~05 · 2026-06~10-27 | 사용자 분할 검증·시험만 따로(판정에 안 씀) |
- 🔴분할 이탈 공개: 공통 규칙 분할(학습 2024~25 · 검증 2026-01~05 · 시험 2026-06~)이 아니라 **XRP·ETH 배포 모델과 같은 2022~24 학습 + 2025·2026 독립 두 해 관문**을 쓴다. 이유: (1) 과제가 «XRP 와 같은 관문», (2) 고정 레시피라 검증 창에서 고를 것이 없음, (3) 서빙 표의 다른 코인 모델과 학습 구간을 맞춤. 사용자 분할 시험 창(2026-06~)은 보조로 따로 낸다. 2025·2026 은 «준-표본 밖»(같은 레시피가 ETH·XRP 에서 이 구간으로 이미 채점됨).

## 지표
AUC · 10분위 보정 최대 차(pp) · 로그손실 · BSS(학습 기저율 대비) · «깨질 쪽(<학습 q20)»·«머물 쪽(>q80)» 실제 유지율과 비중 · n · 독립 일수 · 실현 유지율. ΔAUC = 일 블록 부트스트랩 B=200 95% CI(XRP 와 같은 `block_ci`). Q1: Brier · ΔBrier `day_boot`(1일 블록 2,000회, alpha .2 → 단측 90% 경계) · ECE · AUC · 치우침.

## 주 판정 (사전에 고정)
**Q0** — 코인 × 척도(30분·1h·4h) 각각, XRP 관문 그대로:
- 2025·2026 둘 다 AUC ≥ .72 · 보정 최대 차 ≤ 6pp · ΔAUC(HGB − G 기하 로지스틱) 95% CI 하한 > 0.
- 셋 다 → **통과(그 척도 deployable True)**. 아니면 deployable False(화면에 «검정 불통과»).
- 통과 후보(완화): CI 하한 대신 ΔAUC 점추정 > 0, 나머지 같음 → 결과 문서만.

**Q1** — Q0 1h 엄격 통과 코인만, ETH 8라운드·XRP 10-31 V1~V5 그대로(J1+J2 합침):
- V1 ΔBrier(B − A) 단측 90% 상한 < 0 · V2 로그손실 B ≤ A · V3 ECE B ≤ A · V4 AUC B ≥ A − .002 · V5 J1·J2 각각 ΔBrier < 0.
- 다섯 다 → **통과(그 코인 1h 굴림 Platt90 서빙)**. 통과 후보: V1 대신 ΔBrier 점추정 < 0, V2~V5 ✓.

## 보조 (판정에 안 씀)
- 비교 모델(XRP 와 같음): G 기하 · F 로지스틱 6칸 · ETH 배포 모델 그대로(ETH 계절표) · 4h 진단 HGB 27칸.
- 사용자 분할 2026-01~05 · 2026-06~10-27 AUC·보정(Q0) / ΔBrier(Q1).
- 시드 1 재적합 HGB 의 2025·2026 AUC·보정.
- Q1 월별 치우침 A→B · 화면 라벨 바뀜 비율.

## 시행 기록
주 판정: Q0 6개(2코인 × 3척도) + Q1 최대 2개. 양성 대조 3개(P0a~c). 보조 분석 수는 결과 문서에 전부 적는다. 결과를 본 뒤 레시피·창·관문을 바꾸지 않는다 — 바꾸면 새 사전등록.

## 한계 (미리 적음)
- 1h 5분마다 겹치는 창 · 며칠 가는 변동성 국면 → 1일 블록 CI 가 좁을 수 있다.
- BTC 는 2022~24 학습 구간 동안 시장 구조(현물 ETF 2024-01)가 바뀌었다 — 보정 관문이 이것을 잡는다.
- 라이브 REST 점검(XRP 10-20 의 live_check)은 하지 않는다(로컬 REST = 서버와 IP 공유). 서버 배포 뒤 `/api/box-hold?asset=btc|sol` 로 확인.

## 참고문헌
- Platt, J. (1999). Probabilistic outputs for SVMs and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*.
- Gupta, C., & Ramdas, A. (2023). Online Platt Scaling with Calibeating. *ICML* (ETH 사전등록 `docs/eth_box_hold_calibration_drift_prereg_20261025.md` 재인용).
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS* — sklearn HistGradientBoosting 의 히스토그램 방식 원형.
- Brier, G. W. (1950). *Monthly Weather Review* 78(1).
