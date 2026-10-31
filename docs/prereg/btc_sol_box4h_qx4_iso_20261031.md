# 사전등록: BTC·SOL(+XRP 재채점) 박스 4h — QX4 σ → 두 벽 무접촉 확률 → 등조 지도 (자기 지도 vs ETH 지도) (2026-10-31)

**등급: 되돌리기 쉬운 예측 개선 — 단측 90%** (화면 표시 전용 확률 · 매매·크기 입력 아님 · 수수료 없음) + 완화 «통과 후보» 등급 병기.
**이 문서는 계산 전에 커밋한다.** 스크립트: `scripts/research_box4h_qx4_iso_coins_20261031.py`(`--selftest` · `--run` 한 번만, report.json 있으면 거절).
사용자 «일부러 그대로 둔 것들도 모두 각각 코인에 맞게 재학습 및 최적화 진행해줘» — BTC·SOL 4h 박스 HGB 는 10-31 2026 보정 8.43/9.78pp 로 꺼졌다(`docs/experiments/btc_sol_box_hold_20261031.md`). 이번에 BTC·SOL 4h σ(QX4)가 생겼다(`docs/experiments/btc_sol_vol4h_qx4_20261031.md`, HAR 대비 엄격 통과).

## 질문
코인(BTC·SOL·XRP)마다 4h 박스 유지 확률을 ETH 처럼 «QX4 4h σ → 반사 원리 무접촉 확률 P_stay → 등조 지도»로 내면, 코인 HGB 4h 보다 정확하고 배포 관문(AUC·보정)을 넘나? 지도는 두 팔 중 **검증 창으로 하나를 고른다**:
- **팔 S(자기 지도)**: 그 코인 2024·25 W1·W2 표본 밖 QX4 행으로 맞춘 등조(XRP 10-31 Q2 와 같은 방법).
- **팔 E(ETH 지도 그대로)**: 배포 `dashboard/box_hold_qx4i_iso_20261029.json` 의 x·y 를 그 코인 P_stay 에 `np.interp`. XRP 10-31 결과 문서의 사후 메모(«ETH 지도 이식판이 숫자상 G1~G3 모양을 넘음 → 새 사전등록 필요»)를 정식화한다.

## 왜 (과거 실패 교훈 반영)
- XRP 10-31 Q2: 자기 지도가 순위(ΔAUC +.028)·Brier 는 이겼지만 보정 7.73pp 로 G3 ✗. 원인은 2024·25 창 QX4 눈금 c(W1·W2)와 서빙 r4 c 의 차이로 같은 σ 가 다른 P_stay 로 가는 것 → 눈금 차이가 작은 코인(BTC c .255/.188 vs .259 · SOL .232/.243 vs .221)에선 다를 수 있다. 팔 E 는 지도가 ETH 행 14,220개로 맞춰져 코인 학습 창 눈금 잡음과 무관.
- 결과 본 뒤 고르기 금지: 팔 선택 규칙을 지금 고정하고 시험은 고른 팔만 1회 판정한다. 안 고른 팔의 시험 숫자는 보조로만.
- 10-22 «상한 = 미래 σ»(박스 깨짐 병목 = 변동성) · 10-29 ETH QX4I 등조가 HGB 4h 를 이김(ΔBrier −.00732) → 같은 경로를 코인에.
- 플레이북 «PBO·비교 대상»: 상수(그 창 실현 유지율, 사후 기후값)를 G2 로 함께 건다.
- 같은 구간 재사용 공개: 시험 창 2026-06~10-23 은 변동성 경주 4·5·7·9·10·11·14라운드 + 10-31 BTC·SOL vol4h(≥9회) + 10-31 BTC·SOL 박스 HGB(보조 표 «시험 2026-06~») 에서 썼다. 검증 창 2026-01~05 는 QX4 눈금 c(r4)를 맞춘 구간이라 σ 눈금에 대해 표본 안이다. 예상 우연 통과 ≈ 3코인 × .10 + 재사용 몫.
- 🔴**XRP 는 오염**: XRP 팔 E 의 검증·시험 숫자는 10-31 결과 문서에 이미 나와 있다(시험 Brier .14818 · 보정 5.92pp). 그래서 XRP 판정은 **최대 «통과 후보»**(엄격 통과로 세지 않음, 서빙 안 함). XRP 엄격 판정은 2026-10-20 이후 전진 창에서 새로.

## 양성 대조 (판정 전 필수 · 하나라도 실패면 채점 안 함)
- P0a: 이번 일반화 코드로 XRP 를 돌려 10-31 XRP 결과 `docs/experiments/xrp_box_hold_platt90_qx4_20261031_report.json` Q2 와 맞춘다 — 학습 행 수 14,220 · 검증·시험 HGB·QX4 원판·자기 등조(QX4등조)·ETH 지도(ETH지도) Brier 차 ≤ 1e-12, 자기 등조 ΔBrier vs HGB 평균 같음.
- P0b: 코인마다 시험 QX4 QLIKE 평균 = `report_r4_<coin>.json` QX4 test_qlike ±1e-9(XRP .2165861749 · BTC .2192354041 · SOL .2192817027).
- L1: BTC·SOL 4h 행 라벨을 1분봉에서 `research_eth_box_hold_calibration_drift_20261025.labels(W=240)` 로 독립 재계산 → 100% 일치.

## 데이터·시각 규약
- 박스 행: BTC·SOL = 10-31 박스 연구 산출 `tmp/box_btc_sol_20261031/{btc,sol}/features.parquet`(1분봉 끝 10-27 23:59, 서빙 패리티 ≤1.1e-8) 4h 행 · XRP = `xrp_rows.parquet`(10-20 산출 사본, md5 4b361896, 1분봉 끝 10-19 23:59). ts = hh:59 → **τ = ts + 1분(정시)**. 라벨 hold = 1분봉 i+1..i+240 고가 최대 < c_i + 반폭 그리고 저가 최소 > c_i − 반폭, 반폭 a = ½·(직전 240분 고저폭)/c_i = exp(log_rng)/2e4. 행 조건: HGB 6칸 유한 · 폭 > 0 · ts + 241분 ≤ 1분봉 끝 다음 분.
- HGB 4h = 그 코인 박스 번들 4h(BTC·SOL = `dashboard/box_hold_hgb_{btc,sol}_20261031.joblib`, XRP = `box_hold_hgb_xrp_20261020.joblib`, 모두 2022~24 학습·6칸) `predict_proba`. 화면 «깨질/머물 쪽» 문턱 q20·q80 도 이 번들 값.
- QX4 σ = 변동성 경주 동결 예측(사본 `tmp/box4h_qx4_20261031/in/`, md5 `inputs.md5`) — 인덱스 = 5분봉 시작 hh:55 → τ = 인덱스 + 5분. 피쳐는 τ 에 마감한 5분봉까지, 타깃은 τ 이후 48봉 RV.
  - 검증·시험: `pred_{val,test}_r4_<coin>` 의 `v_QX4`/`t_QX4` + c_r4(`report_r4_<coin>` QX4 c: BTC .25918 · SOL .22141 · XRP .21421).
  - 학습(팔 S 지도만): `pred_{val,test}_r5W{1,2}_<coin>` + 각 창 report QX4 c. τ ∈ [2024-01-01, 2026-01-01), τ+4h ≤ 2026-01-01 assert. (2024·25 의 1~5·6~10월만.)
- P_stay = ETH 연구 `p_stay` 그대로: s = √exp(ŷ + c)/a, P = (4/π)Σ_{k<400}(−1)^k/(2k+1)·exp(−(2k+1)²π²s²/8), s < .02 면 1.
- 팔 S 지도 = `IsotonicRegression(y_min=.001, y_max=.999, out_of_bounds="clip")`. 팔 E = ETH 지도 x·y 고정(다시 맞추지 않음).
- 같은 창 안 모든 모델은 정확히 같은 행(박스 행 ∩ σ 있는 행).

## 구간
| 구간 | τ | 쓰임 |
|---|---|---|
| 학습 | 2024-01-01 ~ 2025-10-23 (W1·W2 표본 밖 QX4) | 팔 S 지도 적합 |
| 검증 | 2026-01-01 ~ 2026-05-31 | **팔 고르기** |
| 시험 | 2026-06-01 ~ 2026-10-23 (XRP 는 라벨 ≤ 10-19 23:59) | 고른 팔 1회 판정 |

## 지표
Brier(주) · 로그손실 · AUC · ECE · 보정 최대 차(10 등빈도 칸, pp) · 치우침(pp) · n · 독립 일수 · 실현 유지율. 짝 차 ΔBrier CI = `day_boot`(1일 블록 2,000회, alpha .2 → [10%, 90%] = 단측 90%, alpha .05 = 95% 참고). ΔAUC = `auc_boot` 95%.

## 팔 고르기 (검증 창, 사전 고정)
1. 검증에서 AUC ≥ .72 그리고 보정 최대 차 ≤ 6pp 인 팔 중 Brier 가 낮은 팔.
2. 둘 다 못 넘으면 보정 최대 차가 작은 팔(→ 시험 G3 에서 걸릴 가능성 높음, 그대로 판정).

## 주 판정 (코인마다, 고른 팔, 시험 창)
- G1 ΔBrier(팔 − HGB 4h) 단측 90% 상한 < 0.
- G2 ΔBrier(팔 − 상수) 단측 90% 상한 < 0.
- G3 검증·시험 둘 다 AUC ≥ .72 그리고 보정 최대 차 ≤ 6pp.
- 셋 다 → **통과**(BTC·SOL: 그 코인 4h 를 이 지도로 켬). XRP 는 셋 다여도 **통과 후보**(오염, 위 «왜»).
- 통과 후보(완화): G1·G2 점추정 < 0 이고 G3 ✓.

## 보조 (판정에 안 씀)
- 안 고른 팔의 검증·시험 같은 표 · QX4 원판 · ΔAUC vs HGB · 월별 ΔBrier(고른 팔 − HGB) · 팔 S 지도 학습 행 수·유지율·원판 평균.

## 시행 기록
주 판정 3개(BTC·SOL·XRP 각 1, XRP 는 후보 상한). 양성 대조 P0a·P0b·L1. 보조 분석 수는 결과 문서에 전부 적는다. 결과를 본 뒤 팔·창·관문을 바꾸지 않는다 — 바꾸면 새 사전등록.

## 서빙 (통과 시에만)
`dashboard/box_hold.py` `with_qx4i` 를 코인 표(지도 JSON)로 넓힌다 — 코인은 IV 없는 QX4 눈금(c_r4)만(BTC QX4I 는 지도가 QX4 σ 로 판정됐으므로 쓰지 않음). 서버 `qx4_state["last"]` 를 코인별로. HGB 4h 가 deployable False 인 코인도 지도가 있으면 4h 칸을 지도 확률로 낸다.

## 한계 (미리 적음)
- 반사 원리 = 표류 0·연속·일정 σ 가정. 점프·꼬리는 등조 단조 변환으로만 흡수.
- 팔 E 는 ETH 의 σ→유지 관계가 다른 코인에도 같다는 가정(눈금 c 는 코인 자기 것).
- 정시 결정 4h 창 겹침 · 며칠 가는 변동성 국면 → 1일 블록 CI 가 좁을 수 있다.
- 검증 창은 c_r4 에 대해 표본 안.

## 참고문헌
- Feller, W. (1971). *An Introduction to Probability Theory and Its Applications*, Vol. II, §X.5 — 두 흡수 벽 생존 확률 급수(ETH `docs/prereg/box_hold_vs_qx4i_20261029.md` 재인용).
- Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *KDD* — 등조 보정.
- Brier, G. W. (1950). *Monthly Weather Review* 78(1).
