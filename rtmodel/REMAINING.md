# rtmodel — 아직 Rust 로 안 옮긴 «모델·예측 서빙» (2026-11-21)

출처: `docs/rtcore_port_scope_20261120.md`(«쓰임» 판정 · 표 1·2·3) · `scripts/live_*.py`·`dashboard/*.py` 의 모델 읽기(grep `joblib.load`·`predict(`) ·
화면 폴링(`dashboard/live/app.js`) · 개발기 파일(메인 체크아웃 `data/live/`·`tmp/`, 읽기만). 서버 crontab 사본은 저장소에 없다 — 주기는 범위 문서 표 2 와 스크립트 머리말.
**개발기 모델** = 개발기 파일이 서버 배포본과 같은지는 해시를 대조해 본 적이 없다(서버 접속 금지). (예외: QX4 qx4_2026-10-23 · 띠 안·이탈 regime_band_h48 은 2026-11-21 서버와 md5 같음 확인)

## 옮긴 것

| 서빙 | Rust | 대조 |
|---|---|---|
| 박스 유지(`dashboard/box_hold.py` · `/api/box-hold` 1분) | `box_hold.rs` (🔴서버 서빙 꺼짐 · 2026-11-21 사용자 결정 -- Rust 는 그대로 둠) | 175벌 · payload 비트 |
| 레벨 5분(`live_sr_lean_20261118.py` · 상주) | `sr_lean.rs` · `sr_live.rs` · `bin/rt_sr_lean` | 27벌 jsonl 글자 · 실행 파일 가짜 거래소 |
| 4h 방향(`live_eth_dir4h_20261015.py` · `/api/dir4h` 2분) | `dir4h.rs` (피쳐 65 + HGB 회귀 + payload) | SOL·XRP 14벌 · 피쳐 64열 비트(P_ac144 6e-14) · payload 비트 |
| 15분·1시간 방향(`live_eth_dir_short_20261028.py` · `/api/dir-short`) | `lgb.rs` (LightGBM 추론 + 시드 앙상블) | 3만 행 · 모델 80개 비트 |
| 위험 레이더 4코인(`live_risk_radar_20261115.py` · cron 매분) | `risk_radar.rs` (σ̂·띠·앞 1시간·급변·IV 1시간 판 + latest.json·forward.jsonl 조립) | 18벌 · 상태·참거짓·None 전부 같음 · 숫자 최대 3.5e-15 상대(🔴X@β 가 numpy BLAS 합 순서 -- 서버 BLAS 기준 글자는 3/18벌만 같음 · 아래 «서버 판과의 차이») |
| 15분·1시간 방향 payload(65열판 BTC·SOL·XRP) | `dirshort.rs` | 9벌 payload 비트 · 반올림 전 예측 비트 · np.quantile 1,500개 비트(분위 씨앗은 ETH 대역) |
| 위험 레이더 SOXL(주식 세션 HAR-S · `compute_session`) | `risk_radar::session_out` (서머타임·NYSE 휴장 표 포함) | 18벌 문자열 같음·숫자 1e-12 상대 · 세션 시계 7,920칸 같음(서머타임 전환·휴장·개장/마감 경계) |
| 위험 레이더 cron 한 회차(`update_cache`·`update_dvol`·`main`) | `bin/rt_radar` (캐시 = 자기 이진 형식 · 그림자 디렉터리) | 가짜 HTTP 5회차(백필→꼬리→DVOL 500→다시) 요청 253개 순서·쿼리 같음 · latest·forward 숫자 최대 1.9e-15 상대 |
| 4h 변동성 QX4(+QX4I · `live_vol4h_qx4_20261025.py` · `/api/vol-forecast` 5분) | `qx4.rs` (피쳐 41열 · HGB 셋 · numpy f32 MLP · 4코인 평균 · payload · QX4I 덧붙이기 · atm_iv · REST 이어 받기) | 개발기 사본 qx4_2026-10-23(서버 모델과 md5 같음 2fce43f5… · 11-21 확인) · 8벌 × 4코인 payload 비트 · 피쳐 41열 × 끝 40봉 비트 · HGB 구성원 비트 · 🔴MLP 구성원 1ulp(f32 · 상대 2.4e-7 -- OpenBLAS sgemm 가장자리 타일 순서) → log_rv 상대 6e-8 · atm_iv 40칸 · 이어 받기 4회차 요청 83개 순서 같음 · Deribit 옵션 체결 받기(iv_now·fetch_option_trades)는 아직 |
| 4h·15분/1시간 REST 조립 + 적중 기록 + 5분 루프 | `dirrest.rs` · `bin/rt_dirpred` | 가짜 거래소 25단계 요청 1,330개 순서·쿼리 같음 · pred_record 410줄 글자 · 실행 파일 한 회차 요청 545개·기록 135줄 |
| 띠 안·이탈(regime band · `live_regime_band_*_20261102.py` · cron 매시) | `regime_band.rs` (청산맵 레벨 compute_spliced_levels 계열 · 피쳐 27열 · HGB 분류 · 기록·채점·요약 · 크론 한 회차) · `bin/rt_regime` | 모델 = 개발기 사본(서버와 md5 같음 f62774c6) · 합성 4코인 11회차(빈 봉·같은 값 구간·3시간 건너뜀·자료 지연·묵은 자료 FAIL) 출력·기록·채점·요약 파일 글자 · 요청 순서 · 레벨 가격 14,389칸 · 청산 칸 번호·무게 30,454칸(넣은 순서까지) · 피쳐 29,916칸 · 확률 1,108개 비트 · 실행 파일 가짜 HTTP |
| 상대 강도(xsec · `pred_record.xsec_record` · 서버 5분 루프 끝) | `dirrest::xsec_px`·`Record::xsec` · `bin/rt_dirpred` 회차 끝 | 12회차 요청 108개·기록 글자·스냅숏(rel_bp) 같음 · 실행 파일 한 회차 |

## 남은 것

| # | 서빙 | 스크립트(줄) | 주기 | 모델 종류 | 입력 원천 | 출력 | 읽는 곳(화면 칸) | 개발기에 모델 | 전진 검정 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 30분 카드(card30) | `dashboard/server.py` `_card30_pred`(서버 안) | 5분봉이 바뀔 때 | HGB 3개 평균 + logit 섞기(식은 `rtmodel::card30` 에 있음) | 서버 메모리 5분봉 | 상황 카드 | `/api/situation`(1초)·SSE `situation` | **없음**(서버 `tmp/card30_direction_20260926/` 전용) | 없음 |
| 6 | 15분·1시간 방향 ETH 75열판(호가 깊이) | `live_eth_dir_short_20261028.py` `book_row`(sqlite 읽기 -- 상태 판정 `dirshort::book_status` 만 옮김 · `book_feats`·`dataset` 은 학습 몫) | 5분(서버 루프) | (65열판은 옮김) | REST(4h 와 같은 `rest_frames`) + 호가 깊이 워커 sqlite 10열 | `/api/dir-short` | ⑤ 예측 줄 15분·1시간 | 65열판 eth·btc·xrp 있음 · **ETH 75열판 없음** · oos 씨앗은 ETH 만 | **fwd_dirshort 2027-01-08** |
| 8 | 방향 전진 기록기 | `live_eth_direction_forward_recorder_20261010.py`(95) | 매일 00:31 UTC | (dir4h 예측을 적음) | `/api/dir4h` 꼴 | `eth_direction_forward.jsonl` | `dashboard/dir_signals.py` · 판정 건강 | — | **dir4h_p2 2027-03-14 · dir3asset 2027-04-08** |
| 9 | 4h 변동성(화면 «지평 사다리») | `dashboard/dir_signals.py`(199) → `live_eth_sizing_vol_model_20260912.py`(358) | 요청 때 | GBM 22피쳐 8시드 + sigmoid 확대 확률 | 5분봉 | `/api/dir-signals` | 풋프린트 머리 «지평 사다리»(10분) | ETH `data/live/eth_sizing_vol_model.joblib` 있음 · BTC·XRP 는 서버(server.py 가 btc 경로를 씀) | 없음(참고용) |
| 10 | 사이징 변동성·MAE 분위 | `live_eth_sizing_vol_model_20260912.py` · `live_eth_mae_quantile_model_20260913.py` | — | — | — | — | — | — | **안 옮김(꺼짐)** -- 서버 `SIZING_RISK_MODEL_ENABLED=False` · 주문 범위가 sltp 만(2026-11-21) |
| 11 | 차트 표식(극값 탐지) | `live_eth_chart_markers_20260909.py` · `live_eth_extreme_detector_20260909.py` | — | — | — | — | — | — | **안 옮김** -- 사용자 결정(2026-11-21) |

**안 옮김(안 쓰임 판정 · 범위 문서)**: `live_eth_sweep_v_rebound_signal_20260829.py`(610, 읽는 곳 0) · `live_eth_vol_forecast_20260910.py`(168, 연구만 import) ·
`live_eth_breakout_detector_*`(화면 0) · `live_eth_maxpain_oos_ledger`(판정 닫힘).

**주기 순(«실시간 값을 빨리»)**: 위험 레이더(매분 · 화면 30초) → card30(5분 · 모델 없음) → dir4h·dir_short·QX4(요청 때 · 화면 2~5분) → regime band(매시) → 기록기(매일).
개발기에서 대조 가능한 것 중 가장 짧은 = 위험 레이더(4코인 + SOXL + cron 한 회차) — 2026-11-21 에 옮겼다. 다음 = #10 사이징·MAE(주문 지연) → #4 QX4 → #5 → #7 -- 셋 다 옮김(11-21) · 남은 = #3 card30(서버 안) · #6 ETH 75열판 · #8 · #9.

## 서버 판과의 차이 (2026-11-21 · 서버 읽기 전용 확인)

서버 = 파이썬 3.11.14 · numpy 2.2.5 · pandas 2.2.3 · scikit-learn 1.8.0 · lightgbm 4.6.0 · numpy BLAS = scipy-openblas 0.3.28 · CPU Ryzen 5 5600X(Zen 3 · AVX2).
개발기 기본 = numpy 2.4.3 · pandas 2.3.3 · lightgbm 4.7.0 · MKL · Ryzen 5 7500F(AVX-512). 정답지는 `tools/server_env.sh` 의 venv(서버와 같은 판)로 만든다.

- **pandas 2.2.3 대 2.3.3 — 차이 없음**: rolling var/std 62,265칸 · mean 12만 · median 6만 · quantile 12만 · ewm 6만 칸이 두 판 모두 옮긴 식과 비트 같다(tools/check_rollvar.py · check_pandas_ops.py).
- **lightgbm 4.6.0 대 4.7.0 — 차이 없음**: 4.6.0 predict 로 만든 정답지에서 모델 하나 80만 칸 · 앙상블 4만 칸 비트 같음.
- **AVX-512 끄기 이름은 numpy 판마다 다르다**: 2.4 = `X86_V4`(옛 `AVX512F` 등은 무시) · 2.2 = `AVX512F AVX512CD AVX512_SKX …`(`X86_V4` 없음 → 무시). 모르는 이름은 두 판 모두 오류 없이 무시하므로
  모든 명령이 둘을 함께 준다(함께 주면 두 판 모두 np.log 가 libm 과 20만 개 중 0개 다름 · 안 끄면 19개). 서버 CPU 엔 AVX-512 가 없어 원래 상관없다.
- **BLAS 를 타는 자리 셋**(numpy 가 BLAS 로 곱한다 · Rust 는 차례 합):
  1. 위험 레이더 σ̂ = c·exp(2·X@β)(gemv · 60·62·13열) → 서버 BLAS(OpenBLAS · `OPENBLAS_CORETYPE=ZEN` = Haswell 커널) 기준 숫자 최대 3.5e-15 상대 · latest 글자 같은 벌 3/18 · forward 줄 40/88 · 상태·참거짓·None 전부 같음.
  2. 4h 방향 P_ac144(np.corrcoef 내적) → 169/720칸 비트 같음 · 최대 3.6e-14 상대 · 예측·payload 는 같음.
  3. 레벨 5분 로지스틱(sklearn X@coef) → 확률 비트 같음 145/164 · 최대 1.1e-16 · 구분·jsonl 줄은 글자 같음.
  - OpenBLAS 합 순서는 본 커널(dgemv_t 4행 묶음 · 4갈래 FMA · (l0+l2)+(l1+l3))만 찾았다(열 수가 4의 배수면 400/400행) -- 남는 행·남는 열 커널은 못 찾아 Rust 에 넣지 않았다. 시험은 1e-12 상대.
  - 🔴서버 BLAS 스레드 수(OPENBLAS_NUM_THREADS)는 미확인 -- 여러 스레드면 행을 나누는 자리가 바뀌어 남는 행 위치도 달라질 수 있다.
