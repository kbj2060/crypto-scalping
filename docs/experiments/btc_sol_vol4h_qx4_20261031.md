# 결과: BTC·SOL 4h 변동성 QX4 (HAR 대비) · BTC 옵션 IV 보정 QX4I 재현 (2026-10-31)

사전등록 `docs/prereg/btc_sol_vol4h_qx4_20261031.md` (계산 전 커밋 a734280f) · 스크립트 `scripts/research_btc_sol_vol4h_qx4_20261031.py --run` 1회 · 산출 `tmp/vol4h_btcsol_20261031/report.json`(md5 2d2fd9d8) · 입력 md5 `tmp/vol4h_btcsol_20261031/inputs.md5`(36 파일).

## 한 줄
**BTC·SOL QX4 = 엄격 통과**(HAR 대비 시험 ΔQLIKE 단측 90% 상한 BTC −.125 · SOL −.060, 처음 읽은 굴림 3창 3/3). **BTC QX4I = 엄격 통과**(12b 재현 1e-17 · 단측 90% 상한 −.0027). **SOL·XRP IV 보정 = 데이터 부족.** 서빙: 서버 vol4h 코인 표에 btc·sol 추가 + QX4I 계수 표에 BTC.

## 사실 확인 (지시문과 다른 점)
- QX4 는 처음부터 4코인(ETH·XRP·BTC·SOL) 모델이다(코인별 M3·M6 + 4코인 합쳐 PX3·PX6). 서버 `data/live/vol4h_qx4.joblib`(10-25 학습, 서버에 있음)에 BTC·SOL 구성원이 이미 있다 → **새 학습 없음**. 학습 재현은 10-25 패리티 P1(BTC 5.3e-7 · SOL 4.7e-7)로 이미 보였다.
- XRP 10-20 판(R² .680 > HAR .568)은 QX4 가 아니라 배포 HGB 계열의 ETH+XRP 합동 학습이고, 서버에선 QX4 실패 시 대체로만 쓰인다.

## P0 양성 대조 (채점 전 관문) — 통과
같은 채점 코드로 ETH·XRP 4라운드 «QX4 − M0» ΔQLIKE [평균, 2.5%, 97.5%] 와 시험 QLIKE 를 report_r4 와 **차 0.0**(1e-9 문턱)으로 재현. BTC·SOL 입력 시험 QLIKE 도 차 0.0.

## Q1 — QX4(팔 A, 합동) vs HAR(M0) · 시험 2026-06-01~10-23 · 정시 결정 3,477개 · 독립 146일
| 코인 | QLIKE A / M0 | ΔQLIKE A−M0 [95%] · 단측 90% 상한 | Δ제곱로그오차 · 상한90 | 로그 RV R² A / M0 | G4 굴림 3창(점추정<0) | 판정 |
|---|---|---|---|---|---|---|
| BTC | .2192 / .3619 | −.1424 [−.1707, −.1169] · −.1254 | −.2642 · −.2357 | **.693 / .484** | 3/3 (W0 −.121 · W1 −.112 · W2 −.111) | **엄격 통과** |
| SOL | .2193 / .2938 | −.0744 [−.0946, −.0524] · −.0599 | −.1566 · −.1343 | **.628 / .470** | 3/3 (W0 −.053 · W1 −.084 · W2 −.097) | **엄격 통과** |
| ETH(참고) | .2750 / .3939 | −.1184 [−.1478, −.0910] · −.1003 | — | .614 / .418 | 3/3 | 통과 |
| XRP(참고) | .2166 / .3191 | −.1037 [−.1288, −.0800] · −.0872 | — | .699 / .555 | 2/3 (W2 +.039) | 통과 |
- 굴림 창 BTC W0(2023) 95% [−.199, −.040] · SOL W0 [−.122, .000](상한이 0 에 닿음) — 창별 CI 는 보조.
- 월별 ΔQLIKE(A−M0) 6/6 음수(BTC −.10~−.17 · SOL −.02~−.10, 8월 SOL 가장 약함).
- **G1~G3 은 사전 열람한 재사용 창**(4라운드에서 이미 계산된 숫자의 단측 90% 재기재). 증거의 무게는 처음 읽은 G4 3창.

### 보조 — 코인 단독 팔 B(M3+M6) vs 합동 A
| 코인 | 검증 QLIKE A / B (검증이 고른 팔) | 시험 ΔQLIKE B−A [95%] | Δ제곱로그오차 B−A [95%] | 배포형 HGB(M3) − M0 |
|---|---|---|---|---|
| BTC | .2478 / .2497 (A) | +.0014 [−.0025, +.0053] | +.0014 [−.0059, +.0079] | −.1350 |
| SOL | .2438 / .2471 (A) | **+.0069 [+.0011, +.0134]** | **+.0304 [+.0164, +.0476]** | −.0529 |
→ 검증도 두 코인 모두 A(합동)를 골랐다. SOL 은 합동 학습 몫이 유의(코인 단독이 더 나쁨), BTC 는 동률. 사전 고정한 A 와 검증 선택이 일치.

σ4h(중앙값형 %, 시험 창 분위 5·25·50·75·95): BTC .27 · .50 · .65 · .86 · 1.27 / SOL .55 · .84 · 1.07 · 1.40 · 1.96.

## Q2 — BTC QX4I (QX4 + 옵션 ATM IV 잔차 보정)
- BTC IV 이력 = Deribit BTC 옵션 체결(메인 체크아웃 `tmp/vol4h_r12b_btc_20261026/trades`, 2025-12-01~2026-10-23 일 파일 327개, 12b 가 Deribit 공개 history API 로 받음 · 바이낸스 아님). 새로 받은 것 없음.
- **R0 재현 통과**: 12라운드 코드를 이 브랜치에서 다시 돌려 ETH(12라운드)·BTC(12b) report.json 과 최대 차 5.6e-17 · 8.7e-18.
- 2단 기준 다시 읽기(시험 3,477 결정 · 146일):
| 코인 | b (IV 격차 계수) | ΔQLIKE QX4I−QX4 [95%] | 단측 90% 상한 | 판정 |
|---|---|---|---|---|
| BTC | +.119 | −.0055 [−.0092, −.0011] | **−.0027** | **엄격 통과**(월 6/6 개선) |
| ETH(참고) | +.176 | −.0066 [−.0144, +.0035] | −.0004 | 통과(기존 «조건부 채택»과 같은 근거) |
- 서빙 패리티 S1: 라이브 `atm_iv`·`iv_now`(coin="BTC") = 연구 `hourly()` BTC, 표본 6일 1e-12 안(`test/test_vol4h_qx4i_20261026.py`).

## Q3 — SOL·XRP IV 이력: **데이터 부족**
서버 `data/live/deribit_options.duckdb`(읽기 전용 조회 10-31): `option_summary` iv30 = 2026-09-28 10:51 UTC 부터 790시간(SOL·XRP 각 4,733행, dvol 없음) · `option_trades` SOL_USDC 47,491건·XRP_USDC 11,877건 = 2026-08-06 부터. 검증 창(2026-01~05)을 전혀 덮지 못해 계수를 맞출 수 없다. 필요한 것: SOL_USDC·XRP_USDC 옵션 체결 2026-01~ 백필(Deribit history API) — XRP 는 시간당 ~5건이라 1h ATM(1시간~10일·델타 .35~.65) 칸이 비는 시간이 많을 것.

## 서빙 (엄격 통과분만)
- `dashboard/server.py`: `VOL4H_ARTIFACTS` 에 `btc`·`sol` 추가(경로 = `data/live/{btc,sol}_sizing_vol_model.joblib` — HGB 대체 아티팩트는 아직 없어 QX4 실패 시 `artifact_missing`) · QX4I 칸을 «ETH 만» 대신 계수 표 `BETA_I` 에 있는 코인(ETH·BTC) · IV 시간 캐시 코인별 · `box_hold.with_qx4i` 가 읽는 `last` 는 ETH 만(그대로) · 파일 없는 HGB 경로에서 `va["art"]` KeyError → `va.get("art")`.
- `scripts/live_vol4h_qx4_20261025.py`: `BETA_I` = 코인 표 {ETH, BTC(12b 동결)} · `atm_iv`·`fetch_option_trades`·`iv_now`·`qx4i_log` 에 `coin`(기본 "ETH", 기존 호출 그대로).
- 서버에서 켜려면: 새 파일·cron 불필요(QX4 아티팩트 `data/live/vol4h_qx4.joblib` 이미 4코인) — 코드 배포 + 대시보드 재시작만. BTC QX4I 는 매시 Deribit BTC 옵션 체결 1시간치(하루 ~7천 건 → 시간당 1쪽)를 더 받는다.
- **화면: `VOL4H_ASSETS`(app.js)에 btc·sol 추가 필요** · 부채꼴 σ 칸·BTC «옵션 IV» 칸(`sig4h_iv_pct`) 표시도 화면 작업(/impeccable). 이번에 화면은 안 건드림.

## 박스 4h 연구가 쓸 σ 재현법
- 연구(동결 예측): `<chart-indicators-tuning-ecfa25>/tmp/vol4h_forecast_20261025/pred_{val,test}_r4_{BTC,SOL}.parquet`(사본 `tmp/vol4h_btcsol_20261031/in/`) 열 `t_QX4`/`v_QX4` = 로그 RV ŷ(다음 4h 5분 로그수익 제곱합의 ln). 인덱스 = 5분봉 시작 hh:55 → 결정 시각 τ = 인덱스 + 5분. σ4h(중앙값형) = exp(ŷ/2), 평균 RV 눈금 = exp((ŷ + c)/2), c = `report_r4_{coin}.json` models.QX4.c(BTC .25918 · SOL .22141). 굴림 창 W0·W1·W2 = `pred_*_{r14W0,r5W1,r5W2}_{coin}`.
- 새로 계산: `scripts/live_vol4h_qx4_20261025.py` — `load_frames()`(= `research_vol4h_forecast_horserace_20261025.frame` + `add_mkt`, 4코인 같은 색인) → `predict_log(art, coin, F)`; 아티팩트 학습은 `fit(fr, masks(...))`. 라이브 klines 에선 `qx4_from_klines(rows_by_coin, art, now_ms, coin)` → `sig4h_pct`·`log_rv`. BTC IV 보정 = `qx4i_log(log_rv, iv, "BTC")`, 연구판 = `research_vol4h_round12_option_iv_20261025.qx4i` 식에 12b `beta_I`.

## 못 한 것
- 10-24 이후 새 자료 전진 채점: 피쳐에 5분봉 체결 건수가 필요한데 로컬 1분봉 parquet 엔 없고 바이낸스 REST 대량 호출 금지 → data.binance.vision 5분 klines(체결 건수 포함)로 받으면 가능(10-28 일 파일 404).
- BTC·SOL HGB 대체 아티팩트(XRP 10-20 식): QX4 실패 시에만 쓰여 이번엔 만들지 않음.
- BTC QX4I 전진 판정 기록(ETH 는 voliv 2027-01-22): 필요하면 같은 사양으로 사전등록.

## 시행 기록
주 판정 2개(Q1 BTC·SOL) 둘 다 통과. Q2 재현 1회(ETH·BTC), Q3 자료 확인. 보조: 팔 B·M3·월별·창별 CI·σ 분위·ETH/XRP 참고 행. 같은 시험 창 누적 사용 ≥ 9회(이번 포함).
