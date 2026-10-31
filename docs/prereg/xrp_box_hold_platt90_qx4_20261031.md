# 사전등록: XRP 박스 유지 — 1h 굴림 Platt90 · 4h QX4 σ 등조 지도 (2026-10-31)

**등급: 되돌리기 쉬운 예측 개선 — 단측 90%** (화면 표시 전용 확률 · 매매·크기 입력 아님 · 수수료 없음) + 완화 «통과 후보» 등급 병기.
**이 문서는 계산 전에 커밋한다.** 스크립트: `scripts/research_xrp_box_hold_platt90_qx4_20261031.py`(`--selftest` · `--run` 한 번만, report.json 있으면 거절).
사용자 «모두 진행해줘»(ETH↔XRP 대시보드 차이 전수조사 `docs/xrp_dashboard_gap_audit_20261031.md` B #2 · A-1 #3).

## 질문
- **Q1 (B #2)** XRP 1h 박스 유지 확률(배포 `dashboard/box_hold_hgb_xrp_20261020.joblib`, 2022~24 학습)에 ETH 와 같은 «굴림 Platt90»(매월 1일 00:00 UTC, 직전 90일 해결된 라벨로 Platt 재적합)을 얹으면 더 정확한가?
- **Q2 (A-1 #3)** XRP 4h 박스 유지(배포 HGB 4h 는 2025 보정 최대 차 6.3pp 로 deployable False → 화면 꺼짐)를 ETH 처럼 «QX4 4h σ → 두 벽 무접촉 확률(반사 원리) → 2024·25 등조 지도»로 내면 HGB 4h 보다 정확하고, XRP 배포 관문(AUC·보정)을 통과하나?

## 왜 (과거 실패 교훈 반영)
- ETH 1h 굴림 Platt90 은 8라운드(2024·2025 독립 창) 5/5 통과로 조건부 채택 → 배포(10-26 A1). 층의 가치 = **배포 모델 보정이 어긋났을 때**(ETH 30분은 이미 보정돼 있어 오히려 +.00014 악화) → XRP 1h 도 같은 레시피 그대로, 고르기 없이 한 번만 잰다. 치우침(평균 예측 − 실현)을 같이 보고한다.
- ETH 9라운드: 4h 굴림 Platt90 은 2024 창에서 악화 → 4h 는 Platt 층이 아니라 QX4 σ 경로로만 본다.
- ETH 4h QX4I 연구(10-29): 등조 지도 없는 원판은 치우침 −12pp(AUC 는 좋음) → 원판은 보조로만, 주 후보는 등조판. 학습 창 QX4 는 표본 안 예측이 없어 굴림 원점 W1·W2 **표본 밖** 예측으로 지도를 맞춘다(같은 방식).
- XRP 는 옵션 IV 보정 계수(QX4I)가 없다 → **IV 없는 QX4** 만. 서빙 QX4 XRP 눈금 c = 4라운드 report_r4_XRP QX4 c(0.21421, 라이브 패리티 P1 이 아티팩트 c 와 1e-4 안 일치 확인한 값).
- XRP 4h HGB 가 떨어진 이유가 «보정»이었으므로 Brier 우위만으론 부족 → XRP 배포 관문(AUC ≥ .72 · 10분위 보정 최대 차 ≤ 6pp)을 그대로 같이 건다.
- 같은 구간 재사용 공개: XRP 2025·2026 은 10-20 XRP 박스 모델 판정(AUC·보정)에 한 번 쓰였고, XRP QX4 시험 창(2026-06~10-23)은 변동성 경주 4·5·7·9·10·11·14라운드에서 XRP 복제 채점에 쓰였다(최소 7회). 박스 라벨 위 QX4 σ 비교는 XRP 에선 처음. 예상 우연 통과 ≈ 주 판정 2개 × .10 = .2회 + 재사용 몫.

## 데이터·시각 규약
- 행·라벨·피쳐 = 10-20 XRP 박스 연구 산출 `tmp/xrp_parity_20261020/box_hold/features.parquet`(메인 체크아웃, 배포 아티팩트를 낸 같은 실행 · 서빙 패리티 최대 차 7.8e-12) 그대로 → 사본 `tmp/xrp_box_20261031/in/xrp_rows.parquet`(md5 4b361896). 1분봉 끝 2026-10-19 23:59.
  - 1h 행 = 5분마다(봉 시작 분 %5 == 4), 4h 행 = 정시 직전 분(hh:59). 라벨 `hold` = 다음 W분(i+1..i+W) 고가 최대 < 종가 + ½(지난 W분 고저폭) 그리고 저가 최소 > 종가 − ½폭. 피쳐는 i 분 마감까지(경계 계약).
  - **라벨 관문 L1**(결과 보기 전): 1분봉(data.binance.vision XRPUSDT parquet)에서 `research_eth_box_hold_calibration_drift_20261025.labels()` 로 1h·4h 라벨을 독립 재계산 → 겹치는 행 100% 일치. 실패면 채점 안 함.
- 1h A = 배포 XRP 1h HGB `predict_proba`(27칸). B = A + `rolling("platt", ts, p, y, W=60, N=90, t0="2025-01-01")`(ETH 연구 함수 그대로 — 해결된 행만(ts+W+1분 ≤ 맞춤 시각, L5 assert), 28일 미만·100행 미만이면 항등). 배포 XRP 모델 학습 끝 2024-12-31 → 2025-01-01 부터 표본 밖.
- 4h QX4 = 동결 예측(변동성 경주, `chart-indicators-tuning-ecfa25/tmp/vol4h_forecast_20261025` 사본, md5 `tmp/xrp_box_20261031/inputs.md5`). 인덱스 = 5분봉 시작 hh:55 → **결정 시각 τ = 인덱스 + 5분**, 박스 행 τ = ts + 1분. 피쳐는 τ 에 마감한 5분봉까지.
  - 검증·시험 σ: `pred_{val,test}_r4_XRP` 의 `QX4` + c_r4(report_r4_XRP). 재현 관문 P1: 시험 QX4 QLIKE 평균 = report_r4_XRP QX4 test_qlike 0.2165861749 ±1e-9.
  - 등조 지도: `IsotonicRegression(y_min=.001, y_max=.999, out_of_bounds="clip")` 를 **학습 창에서만** — x = W1·W2 표본 밖 QX4(`pred_{val,test}_r5W{1,2}_XRP`, 각 창 c = 그 창 report QX4 c) 의 P_stay, y = 같은 행 박스 라벨. τ+4h ≤ 2026-01-01 assert. (2024·25 의 1~5·6~10월만 — 11~12월 예측 없음.)
- σ → 유지 확률 = ETH 연구 `p_stay` 그대로: s = √exp(ŷ + c) / a, a = exp(log_rng)/2e4(= ½·폭/종가), P = (4/π) Σ_{k<400} (−1)^k/(2k+1)·exp(−(2k+1)²π²s²/8), s < .02 면 1.
- 4h HGB = 배포 XRP 4h(6칸) `predict_proba`. 상수 = 그 채점 창 자신의 실현 유지율(사후 기후값).

## 구간
| 구간 | 1h (Q1) | 4h (Q2) | 쓰임 |
|---|---|---|---|
| 학습 | 2022~24(배포 모델, 이미 고정) | 2024-01-01~2025-10-23 W1·W2 표본 밖 QX4 행 | Q2 등조 지도 적합만 |
| 판정 창 J1 | 2025-02-01~2025-12-31 | — | Q1 주 판정(합침) |
| 판정 창 J2 | 2026-01-01~2026-10-19 | — | Q1 주 판정(합침) |
| 검증 | (보고) 2026-01-01~05-31 | 2026-01-01~05-31 | Q2 관문 G3 · 보조 |
| 시험 | (보고) 2026-06-01~10-19 | 2026-06-01~라벨 끝 ≤ 2026-10-19 23:59 | Q2 주 판정 1회 |
- 🔴Q1 분할 이탈 공개: 굴림 Platt90 은 ETH 8라운드에서 고정된 레시피라 **고를 것이 없다**(검증 창의 역할 없음). 배포 XRP 모델은 2024 까지 학습 → 2025~ 전부 표본 밖이라, ETH 8라운드처럼 독립 두 해(2025·2026)를 판정 창으로 쓴다. 2025-01 은 굴림층이 항등(맞출 자료 < 28일)이라 J1 에서 뺀다. 사용자 분할의 시험(2026-06~)만의 결과는 보고한다(판정에 안 씀).
- 행 조건: 라벨 창 끝(ts + W분) ≤ 창 끝 · 피쳐 유한 · 폭 > 0. 모든 비교 모델은 정확히 같은 행.

## 지표
Brier(주) · 로그손실 · ECE(10 등빈도 칸, pp) · 보정 최대 차(pp) · AUC · 치우침(pp) · n · 독립 일수 · 실현 유지율. 짝 차 CI = `day_boot`(1일 블록 2,000회): alpha .2 → [10%, 90%] = 단측 90% 경계, alpha .05 = 양측 95%(참고). ΔAUC = `auc_boot` 95%.

## 주 판정 (사전에 고정)
**Q1 (1h)** — ETH 8라운드 V1~V5 그대로(J1+J2 합침):
- V1 ΔBrier(B − A) 단측 90% 상한 < 0 · V2 로그손실 B ≤ A · V3 ECE B ≤ A · V4 AUC B ≥ A − .002 · V5 J1·J2 각각 ΔBrier < 0.
- 다섯 다 → **통과(1h XRP 굴림 Platt90 배포 후보)**. 아니면 불통과.
- 통과 후보(완화): V1 대신 «ΔBrier 점추정 < 0» 이고 V2~V5 ✓.

**Q2 (4h)** — 시험 창:
- G1 ΔBrier(QX4 등조 − HGB 4h) 단측 90% 상한 < 0.
- G2 ΔBrier(QX4 등조 − 상수) 단측 90% 상한 < 0.
- G3 XRP 배포 관문: 검증·시험 둘 다 AUC ≥ .72 그리고 보정 최대 차(10 등빈도 칸) ≤ 6pp.
- 셋 다 → **통과(XRP 4h 를 QX4 등조로 켬)**. 아니면 불통과(화면 4h 꺼진 채).
- 통과 후보(완화): G1·G2 점추정 < 0 이고 G3 ✓.

## 보조 (판정에 안 씀)
- Q1: 사용자 분할 시험(2026-06-01~10-19)만의 ΔBrier·단측 90% · 2026-01~05 · 월별 치우침 A→B · 화면 |변화| 평균·«깨질 쪽/머물 쪽» 라벨(배포 q20·q80) 바뀜 비율.
- Q2: QX4 원판 − HGB · **ETH 등조 지도 그대로 옮긴 판**(`dashboard/box_hold_qx4i_iso_20261029.json` x·y, 같은 XRP σ) − HGB · 검증 창 같은 표 · ΔAUC · 섞은 것(로짓 평균 HGB·등조) · 월별 ΔBrier.

## 시행 기록
주 판정 2개(Q1·Q2). 보조 분석 수는 결과 문서에 전부 적는다. 사후에 창·반폭·보정 방식·관문을 바꾸지 않는다 — 바꾸면 새 사전등록.

## 한계 (미리 적음)
- XRP 1분봉 끝 10-19 → 시험 창이 ETH(10-23)보다 4일 짧다.
- 1일 블록은 겹치는 창(1h 5분마다 · 4h 정시마다)과 며칠 가는 변동성 국면 상관을 다 못 잡아 CI 가 좁을 수 있다.
- QX4 XRP c 는 검증 창(2026-01~05)에서 적합 → Q2 검증 수치는 σ 눈금에 대해 표본 안이다(등조 지도는 2024·25 라 표본 밖).
- 반사 원리 = 표류 0·연속 경로·일정 σ 가정. 점프·꼬리는 등조판이 단조 변환으로만 흡수.

## 참고문헌
- Platt, J. (1999). Probabilistic outputs for SVMs and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*.
- Gupta, C., & Ramdas, A. (2023). Online Platt Scaling with Calibeating. *ICML* — 창 Platt(WPS) 근거(ETH 사전등록 `docs/eth_box_hold_calibration_drift_prereg_20261025.md` 에서 확인한 것 재인용).
- Feller, W. (1971). *An Introduction to Probability Theory and Its Applications*, Vol. II, §X.5 — 두 흡수 벽 생존 확률 급수(ETH 4h 연구 `docs/prereg/box_hold_vs_qx4i_20261029.md` 재인용).
- Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *KDD* — 등조 보정.
- Brier, G. W. (1950). *Monthly Weather Review* 78(1).
