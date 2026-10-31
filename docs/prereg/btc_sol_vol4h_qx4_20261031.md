# 사전등록: BTC·SOL 4h 변동성 QX4 (HAR 대비) · BTC 옵션 IV 보정 QX4I 재현 (2026-10-31)

**등급: 되돌리기 쉬운 예측 개선 — 단측 90%** (⑤ 4h 변동성 칸·부채꼴 σ 표시 전용 · 매매·크기 입력 아님 · 수수료 없음) + 완화 «통과 후보» 등급 병기.
**이 문서는 계산 전에 커밋한다.** 스크립트: `scripts/research_btc_sol_vol4h_qx4_20261031.py`(`--selftest` · `--run` 한 번만, report.json 있으면 거절).
사용자 «일부러 그대로 둔 것들도 모두 각각 코인에 맞게 재학습 및 최적화 진행해줘»(공통 규칙 `coin_retrain_rules.md`, 10-31).

## 질문
- **Q1** BTC·SOL 4h 실현 변동성 예측 QX4 가 같은 코인 HAR-RV 보다 정확한가(XRP 10-20 판정과 같은 «HAR 대비» 기준)?
- **Q2** BTC 에 ETH 와 같은 옵션 ATM IV 보정(QX4I)을 얹는 근거가 이 브랜치 코드로 재현되는가? (판정 자체는 12b 에서 이미 사전등록·채점됨 — 아래 «왜».)
- **Q3** SOL·XRP 는 IV 보정을 검정할 IV 이력이 있는가(확인만)?

## 왜 (과거 실패 교훈 반영 · 사실 확인)
- 🔴**QX4 는 처음부터 4코인(ETH·XRP·BTC·SOL) 모델이다.** 구성 = 코인별 M3(배포형 HGB)·M6(MLP) + 4코인 합쳐 PX3·PX6(+시장 MKT 피쳐), 로그 RV 동일 평균(`scripts/live_vol4h_qx4_20261025.py`). 서버 아티팩트 `data/live/vol4h_qx4.joblib`(10-25 `--train`, 서버에 있음 확인 10-31)에 BTC·SOL 구성원·눈금 c·확대 로짓이 이미 들어 있다 → **새로 «재학습»할 것이 없다.** 학습 재현은 10-25 패리티 P1(`chart-indicators-tuning-ecfa25/tmp/vol4h_qx4/parity.json`: 4라운드 창으로 다시 학습한 아티팩트가 연구 t_QX4 를 BTC 5.3e-7 · SOL 4.7e-7 안 재현, c 1e-4 안)로 이미 보였다.
- 과제 지시문의 «XRP = ETH+XRP 합동 학습 QX4» 는 사실과 다르다: XRP 10-20 판(2025~ R² .680 > HAR .568)은 **배포 HGB(`live_eth_sizing_vol_model_20260912`) 계열의 ETH+XRP 합동 학습**이고, 서버에선 QX4 가 먼저 돌고 그 HGB 는 QX4 실패 시 대체(`dashboard/server.py` `api_vol_forecast`). 여기서는 «XRP 와 같은 방식» = **코인 단독 팔 vs 합동 학습 팔을 사전에 고정** + **HAR 대비 판정**으로 옮긴다.
- **팔 고정: 주 판정 팔 = A «QX4(합동, 서빙 아티팩트 그대로)».** 이유: 서빙되는 것이 A 이고, 2라운드에서 4코인 합쳐 학습이 알트 3/3 개선. 코인 단독 팔 B = 평균(M3, M6)(합쳐 학습 구성원 PX3·PX6 뺀 것)은 **보조**로만 보고한다 — 검증이 B 를 고르더라도 이번 판정·서빙은 A(B 서빙은 새 아티팩트 구조가 필요해 새 사전등록).
- 🔴**같은 구간 재사용·사전 열람 공개**: 시험 창 2026-06-01~10-23 은 변동성 경주 4·5·7·9·10·11·12·13라운드에서 최소 8회 쓰였고, **이 문서를 쓰기 전에 4라운드 report_r4 의 BTC·SOL «QX4 − M0» ΔQLIKE(95% CI)를 이미 봤다**(BTC·SOL 모두 0 배제). 그러므로 G1 은 맹검 판정이 아니라 기존 결과의 단측 90% 재기재다. 이를 보완하려고 **이 문서 작성 시점에 결과를 보지 않은** 굴림 원점 3창(W0 2023 · W1 2024 · W2 2025, 5·14라운드 동결 예측)의 BTC·SOL «QX4 − M0» 를 복제 관문 G4 로 건다. 예상 우연 통과 ≈ 주 판정 2코인 × .10 = .2회 + 재사용 몫.
- 실패 교훈: 14라운드(2023 창)에서 ETH «QX4 > HAR» 가 충격 하루(08-17) 몫으로 단측 상한을 못 넘었다 → G4 는 «점추정 < 0» 창 수로 보고, 창별 CI 는 보조. 10-24·25 급락 직전 예측이 실제의 ~0.3배(점프 못 봄) → 한계에 적는다.
- QX4I(Q2): 12b(사전등록 `docs/prereg/vol4h_round12b_btc_iv_20261026.md` c5467f3c · 결과 2bd4cd94, 둘 다 `claude/chart-indicators-tuning-ecfa25` 브랜치)에서 **BTC 를 같은 사양으로 이미 채점**했다: QX4I − QX4 ΔQLIKE −.0055 [95% −.0092, −.0011] · 월 6/6 · b +.12 → «유의 복제». 같은 시험 창을 다시 판정하면 재사용만 늘어난다 → **새 판정 없이 재현 관문 R0 만** 돌리고, 12b 숫자를 2단 기준(단측 90%)으로 다시 읽는다. BTC IV 이력 = Deribit BTC 옵션 체결 `tmp/vol4h_r12b_btc_20261026/trades`(메인 체크아웃, 2025-12-01~2026-10-23 일 파일 327개, 12b 가 Deribit 공개 history API 로 받음).
- 새 자료(10-24~)는 쓰지 않는다: 피쳐에 5분봉 체결 건수(n)가 필요한데 로컬 1분봉 parquet 엔 없고, 바이낸스 REST 대량 호출 금지. 서빙 아티팩트(눈금 끝 10-23)의 진짜 전진 판정은 이번 범위 밖.

## 데이터·시각 규약
- 입력 = 변동성 경주 동결 예측(`.claude/worktrees/chart-indicators-tuning-ecfa25/tmp/vol4h_forecast_20261025/`)의 `pred_{val,test}_{r4,r5W1,r5W2,r14W0}_{ETH,XRP,BTC,SOL}.parquet` + `report_r4_*.json` → 사본 `tmp/vol4h_btcsol_20261031/in/`(md5 목록 `inputs.md5`).
- 행 = 정시 결정(5분봉 시작 hh:55 인덱스, 결정 시각 τ = 인덱스 + 5분). 타깃 y = ln Σ_{k=i+1}^{i+48} r_k²(다음 4h 5분 로그수익 제곱합). 피쳐는 봉 i 마감까지(경주 S1~S4 자체 점검 · 4라운드 S4 통과).
- HAR = M0: 코인별 OLS ln RV ~ lrv48·lrv288·lrv2016, 팔 A 와 **같은 학습 창**(4라운드: 2024-01-01 ~ 라벨 끝 ≤ 2025-12-31 23:55).
- 눈금 상수 c(모델마다) = 검증 창 ln mean exp(y − ŷ) (경주와 같은 규칙) → QLIKE 는 ŷ + c 로.

## 구간
| 구간 | 기간 | 쓰임 |
|---|---|---|
| 학습 | 2024-01 ~ 2025-12 | 4라운드에서 이미 적합(동결 예측) |
| 검증 | 2026-01-01 ~ 2026-05-31 | 눈금 c · 보조(팔 A·B 검증 QLIKE) |
| 시험 | 2026-06-01 ~ 2026-10-23 20:00 | Q1 G1~G3 (재사용·사전 열람 공개, 위) |
| 복제 W0·W1·W2 | 시험 2023·2024·2025 각 06-01~10-23 (학습 직전 2년 · 검증 1~5월) | G4 |
2026 은 준-표본 밖. 데이터 끝이 10-23 인 이유: 동결 예측의 끝(위 «새 자료» 참조).

## 지표
QLIKE = exp(e) − e − 1, e = y − (ŷ + c) · 제곱 로그 오차 (y − ŷ)² · 로그 RV R² = 1 − Σ(y − ŷ)²/Σ(y − ȳ)²(ŷ 원값, XRP 10-20 과 같은 정의 — ln σ 눈금과 같은 R²) · 상관. 짝 차 CI = 일 평균 계열의 5일 블록 부트스트랩 4,000회(seed 7, `research_eth_dir4h_reverify_20261019.block_ci` 와 같은 재표집) — 분위 .90 = 단측 90% 상한, [.025, .975] = 양측 95%(참고). 분모 = 시험 결정 수(정시 n) · 독립 일수.

## 주 판정 (사전에 고정, 코인별 BTC·SOL)
- **P0 양성 대조(채점 전 관문)**: 같은 코드로 ETH·XRP 의 4라운드 숫자를 다시 낸다 — QX4·M0 시험 QLIKE 와 «QX4 − M0» ΔQLIKE [평균, 2.5%, 97.5%] 가 `report_r4_{ETH,XRP}.json` 과 1e-9 안 일치. BTC·SOL 은 시험 QLIKE 가 report 와 1e-9 안(입력 무결성). 하나라도 어긋나면 채점 안 함.
- **G1** ΔQLIKE(A − M0) 단측 90% 상한 < 0.
- **G2** Δ제곱 로그 오차(A − M0) 단측 90% 상한 < 0.
- **G3** R²(A) > R²(M0) (시험 창, XRP 기준).
- **G4** 복제: W0·W1·W2 중 ΔQLIKE(A − M0) 점추정 < 0 인 창 ≥ 2/3.
- 넷 다 → **엄격 통과**(서버 vol4h 코인 표에 BTC·SOL 추가). 통과 후보(완화) = G1·G2 점추정 < 0 이고 G3·G4. 그 밖 불통과.

**Q2 (BTC QX4I)** — 판정은 12b 그대로(새 판정 아님):
- **R0 재현 관문**: `research_vol4h_round12_option_iv_20261025.main()` 을 이 브랜치에서 다시 돌려 ETH → `tmp/vol4h_r12_20261025/report.json`, BTC → `tmp/vol4h_r12b_btc_20261026/report.json` 의 beta_I·모델별 test_qlike·dq_vs_QX4 가 1e-9 안 일치.
- R0 통과 시 2단 기준으로 다시 읽기: BTC ΔQLIKE(QX4I − QX4) 단측 90% 상한 < 0 이고 b > 0 → **엄격 통과**(서빙: BTC 계수 = 12b beta_I 동결). 단측 90% 상한은 같은 일 계열에서 위 부트스트랩으로 기재.
- **S1 서빙 패리티**: 라이브 `atm_iv`·`iv_now`(BTC)를 12b BTC 체결 표본일에 돌린 시간 IV = 연구 `hourly()` 1e-12 안(시험 파일에 남김).

**Q3** — 확인만: 서버 `data/live/deribit_options.duckdb`(읽기 전용) `option_summary` iv30·`option_trades` SOL_USDC·XRP_USDC 이력 시작일. 검증 창(2026-01~05)을 덮지 못하면 «데이터 부족»(계산 없음).

## 보조 (판정에 안 씀)
- 팔 B(코인 단독 M3+M6) − A · B − M0 · 검증 QLIKE A vs B(검증이 무엇을 골랐을지) · M3(배포형 HGB) − M0 · 월별 ΔQLIKE · W0·W1·W2 창별 CI · σ4h 시험 분위(중앙값형 %).

## 시행 기록
주 판정 2개(Q1 BTC·SOL). Q2 는 재현(12b 판정 재기재), Q3 는 자료 확인. 보조 분석 수는 결과 문서에 적는다. 사후에 창·지표·관문을 바꾸지 않는다 — 바꾸면 새 사전등록.

## 한계 (미리 적음)
- G1~G3 는 사전 열람한 재사용 창 → 증거의 무게는 G4(처음 읽는 3창)에 있다.
- 1일 블록은 며칠 가는 변동성 국면 상관을 다 못 잡아 CI 가 좁을 수 있다.
- 점프(급락) 직전 과소 예측은 QX4·HAR 공통 — 이 비교가 다루지 않는다.
- 서빙 아티팩트는 4라운드 창이 아닌 최근 2년 + 5개월 눈금(`--train`)이라, 판정 수치는 «레시피»의 성적이다(ETH 와 같은 처지).

## 참고문헌
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics* 7(2) — HAR-RV(경주 M0 재인용).
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics* 160(1) — QLIKE 견고성.
- Busch, T., Christensen, B. J., & Nielsen, M. Ø. (2011). The role of implied volatility in forecasting future realized volatility and jumps. *Journal of Econometrics* 160(1) — IV 보정(12라운드 재인용).
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics* 17(3) — 블록 부트스트랩.
