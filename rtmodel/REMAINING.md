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
| 4h 변동성 QX4(+QX4I · `live_vol4h_qx4_20261025.py` · `/api/vol-forecast` 5분) | `qx4.rs` (피쳐 41열 · HGB 셋 · numpy f32 MLP · 4코인 평균 · payload · QX4I 덧붙이기 · atm_iv · REST 이어 받기) | 개발기 사본 qx4_2026-10-23(서버 모델과 md5 같음 2fce43f5… · 11-21 확인) · 8벌 × 4코인 payload 비트 · 피쳐 41열 × 끝 40봉 비트 · HGB 구성원 비트 · 🔴MLP 구성원 1ulp(f32 · 상대 2.4e-7 -- OpenBLAS sgemm 가장자리 타일 순서) → log_rv 상대 6e-8 · atm_iv 40칸 · 이어 받기 4회차 요청 83개 순서 같음 · QX4I 옵션 체결 받기(`qx4::iv_now`·`option_trades` · 2026-11-22) = 원본을 가짜 deribit_get 으로 12회차(쪽 나눔·같은 trade_id 겹침·블록·청산·빈 시간 → 앞 시간·받기 실패) 요청 URL 30개·(iv, 정시)·기억 칸 같음 · `handlers::vol_forecast_qx4` 가 붙임 |
| 4h·15분/1시간 REST 조립 + 적중 기록 + 5분 루프 | `dirrest.rs` · `bin/rt_dirpred` | 가짜 거래소 25단계 요청 1,330개 순서·쿼리 같음 · pred_record 410줄 글자 · 실행 파일 한 회차 요청 545개·기록 135줄 |
| 띠 안·이탈(regime band · `live_regime_band_*_20261102.py` · cron 매시) | `regime_band.rs` (청산맵 레벨 compute_spliced_levels 계열 · 피쳐 27열 · HGB 분류 · 기록·채점·요약 · 크론 한 회차) · `bin/rt_regime` | 모델 = 개발기 사본(서버와 md5 같음 f62774c6) · 합성 4코인 11회차(빈 봉·같은 값 구간·3시간 건너뜀·자료 지연·묵은 자료 FAIL) 출력·기록·채점·요약 파일 글자 · 요청 순서 · 레벨 가격 14,389칸 · 청산 칸 번호·무게 30,454칸(넣은 순서까지) · 피쳐 29,916칸 · 확률 1,108개 비트 · 실행 파일 가짜 HTTP |
| 상대 강도(xsec · `pred_record.xsec_record` · 서버 5분 루프 끝) | `dirrest::xsec_px`·`Record::xsec` · `bin/rt_dirpred` 회차 끝 | 12회차 요청 108개·기록 글자·스냅숏(rel_bp) 같음 · 실행 파일 한 회차 |
| 방향 전진 기록기(`live_eth_direction_forward_recorder_20261010.py --record` · 00·08·16 UTC 31분) | `signals::decide`·`record` · `bin/rt_dirfwd` | 가짜 시계·가짜 klines 7회차(00:31·08:40·16:31 · 03:00 거부 · 첫 30분 거래 0 · SOL 00:25 봉 빠짐 KeyError → 앞 두 줄은 남음 · votes 90일 창 경계) 요청 36개 순서·URL · 기록 17줄 글자 같음 · 실행 파일 가짜 HTTP |
| 지평 사다리 · ⑤ 판 보조(`dashboard/dir_signals.py`: `vol4h_from_klines`(QX4 실패 때 대체) · `open_rev_from_klines` · `dir_signals_payload` 코인베이스 7일·FRED 4~13주) | `signals::vol4h`·`vol4h_rows`·`open_rev`·`dir_signals` · 피쳐 22열 = `qx4::svm_features`(QX4 와 공용으로 뺌) | 4h 변동성 2벌(+ expand_logit 다른 판 · 봉 부족 문구) · 개장 되돌림 5벌(대기·00:31·05:00·23:59·서버 실수 시계) · 신호 2벌(FRED 키 있음·없음) 요청 10개 URL · 전부 JSON 글자 같음 · 모델 = 서버 사본 eth 722f7bc8(train_end 2026-10-14 · expand_logit 있음) · xrp 085161de(2026-10-19) -- 코디네이터가 md5 같음 확인(개발기 data/live 의 fc984e6e 는 낡은 판) · 4h 변동성 4벌(eth·xrp × expand_logit 있음·없음 · XRP 는 4자리 달러) · 서버 HGB 쪽 klines 10쪽 URL = 서버 원문과 같음(코디네이터 확인 · 쿼리 symbol·interval·limit=1000·endTime) · 원천 예외 문구는 키 없음만 같은 글자 |
| 15분·1시간 방향 ETH 75열판(`live_eth_dir_short_20261028.py` `payload_from_features` ETH 갈래 · 호가 깊이 sqlite 10열) | `dirshort::payload_book`(+ `book_status`) · 65열판과 한 함수 | 서버 모델 사본 eth_dirshort_model 124d3c86(2026-11) · 임시 sqlite 4벌(전부 온전 · 없는 봉·warm·빈 칸 · 지금 봉 warm → pred null · 전부 없음 → hist 빔) payload 비트 · 반올림 전 예측 비트(온전한 봉만) · 표본 밖 분위 씨앗 = 개발기 eth_dirshort_oos_seed.parquet(서버와 md5 같음 3885488c) · sqlite 읽기 = `dirshort::book_sqlite`(rusqlite 0.37 · rtcollect 와 같은 판 · 읽기 전용 · bars 표 없으면 원본처럼 실패) -- 같은 4벌을 진짜 sqlite 파일로 읽어도 payload 같음 · `bin/rt_dirpred` 가 ETH 도 돈다(2026-11-22) |
| 30분 카드 방향·닿음(card30 · `dashboard/flow_read.card30_features` + server.py `_card30_pred`) | `card30::features`·`predict`(보정 = `lib::card30`) | 서버 모델 사본 model b4994b68 · model_reach 6409c409 · 창 309벌 + 빈 봉 · 피쳐 7,852칸 · 확률 604개 비트(거래 0·고=저 구간 포함) · 보정 6줄은 server.py 안 닫힌 함수라 정답지 생성기에 글자 그대로 옮겨 부름 |

## 남은 것

| # | 서빙 | 스크립트(줄) | 주기 | 모델 종류 | 입력 원천 | 출력 | 읽는 곳(화면 칸) | 개발기에 모델 | 전진 검정 |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 사이징 변동성·MAE 분위 | `live_eth_sizing_vol_model_20260912.py` · `live_eth_mae_quantile_model_20260913.py` | — | — | — | — | — | — | **안 옮김(꺼짐)** -- 서버 `SIZING_RISK_MODEL_ENABLED=False` · 주문 범위가 sltp 만(2026-11-21) |
| 11 | 차트 표식(극값 탐지) | `live_eth_chart_markers_20260909.py` · `live_eth_extreme_detector_20260909.py` | — | — | — | — | — | — | **안 옮김** -- 사용자 결정(2026-11-21) |
| 12 | 4h 방향 **ETH 판** | `live_eth_dir4h_20261015.py`(SOL·XRP 와 같은 코드 · 팔 S) | 5분(서버 루프)·요청 때 | HGB 회귀(65피쳐) | REST(rest_frames) | `/api/dir4h?asset=eth` · pred_record ETH «4» 줄 | ⑤ 예측 판 «4h 방향» | **옮김(2026-11-22)** -- 서버 사본(11-22 가져옴 · 대조 뒤 지움)으로 export → 피쳐·예측 6벌(normal·gaps·zero_vol·nan_metrics·missing_coin·exact) 65열 비트(P_ac144 만 1.1e-13)·payload 같음 · 짧은 패널 예외 같음 · 핸들러(d4.live_payload) 2회차 요청 147개·payload 비트 | dir4h_p2 2027-03-14 |

### 2026-11-22 서빙 쪽 파이썬 모델 코드 전수(재학습 스크립트 = 연구 몫이라 뺌)
근거 = `scripts/live_*.py` 중 모델을 읽는 것(joblib·pickle·TabPFN·LightGBM·torch 글자) 10개 + 대시보드가 import 하는 모듈(dashboard/server.py 머리 import 줄을 Read 로) + 대시보드 안 모델 읽기(dashboard/*.py).
서버 crontab 사본은 저장소에 없다 -- cron 몫은 SHADOW.md 의 넘긴 목록(regime·radar·dirfwd)과 스크립트 머리말 기준.

| 파이썬 | 쓰임(서빙 경로) | Rust |
|---|---|---|
| live_eth_dir4h_20261015 | 대시보드 루프·`/api/dir4h` | dir4h.rs · dirrest.rs -- ETH 판 포함(#12 · 11-22) |
| live_eth_dir_short_20261028 | 대시보드 루프·`/api/dir-short` | dirshort.rs(65열판 + ETH 75열판·sqlite) · 핸들러 정답지 ETH 2회차 요청 147개·payload 비트 |
| live_vol4h_qx4_20261025 | `/api/vol-forecast` 먼저 | qx4.rs(+ QX4I iv_now) |
| live_eth_sizing_vol_model_20260912 | `/api/vol-forecast` 대체(HGB) · 사이징은 꺼짐 | signals.rs vol4h(서버 사본 eth·xrp) · 사이징 #10 안 옮김 |
| live_regime_band_20261102(+cron·recorder) | cron 매시 | regime_band.rs · rt_regime -- **넘김(11-21 23:46)** |
| live_sr_lean_20261118 | 상주 워커 | sr_lean.rs·sr_live.rs · rt_sr_lean -- 그림자 준비 중 |
| live_risk_radar_20261115 | cron 매분(계수 JSON · 모델 파일 없음) | risk_radar.rs · rt_radar -- **넘김(11-22 00:27)** |
| live_eth_direction_forward_recorder_20261010 | cron 하루 3회(규칙) | signals.rs · rt_dirfwd -- **넘김(11-22)** |
| live_eth_extreme_detector_20260909 · chart_markers | 대시보드 차트 표식 | #11 사용자 결정으로 안 옮김 |
| live_eth_mae_quantile_model_20260913 | 사이징(꺼짐) | #10 |
| live_eth_sweep_v_rebound_signal_20260829 · live_eth_vol_forecast_20260910 | 읽는 곳 0 · 연구만 | 범위 밖 |
| 대시보드 안: card30(_card30_pred) · box_hold | 상황 카드 dir_p·reach_p · 박스(서빙 꺼짐) | card30.rs · box_hold.rs |
| 대시보드 안: dir_signals(7일·4~13주·개장 되돌림) | `/api/dir-signals`·`/api/open-rev` | signals.rs · handlers.rs |
남은 서빙 모델 코드 = **없음**(2026-11-22 · #12 ETH 4h 옮김). 나머지는 붙이기(rtserve 핸들러 · rt_dirpred·rt_sr_lean 넘기기) 몫.

**안 옮김(안 쓰임 판정 · 범위 문서)**: `live_eth_sweep_v_rebound_signal_20260829.py`(610, 읽는 곳 0) · `live_eth_vol_forecast_20260910.py`(168, 연구만 import) ·
`live_eth_breakout_detector_*`(화면 0) · `live_eth_maxpain_oos_ledger`(판정 닫힘).

**주기 순(«실시간 값을 빨리»)**: 위험 레이더(매분 · 화면 30초) → card30(5분 · 모델 없음) → dir4h·dir_short·QX4(요청 때 · 화면 2~5분) → regime band(매시) → 기록기(매일).
개발기에서 대조 가능한 것 중 가장 짧은 = 위험 레이더(4코인 + SOXL + cron 한 회차) — 2026-11-21 에 옮겼다. 다음 = #10 사이징·MAE(주문 지연) → #4 QX4 → #5 → #7 -- 셋 다 옮김(11-21) · 남은 = #3 card30(서버 안) · #6 ETH 75열판 · #8 · #9.

## card30 서빙 경로 (2026-11-21 조사 · server.py 읽기 전용)

| 무엇 | 어디 | 주기 | 쓰는 곳 | 판정 |
|---|---|---|---|---|
| 방향(dir)·닿음(reach) 확률 | `compute_situation`(코인마다 매초 도는 상황 계산) → 스레드에서 `_card30_pred`(봉이 바뀔 때만 · ETH 모델을 SOL·XRP 등에도 그대로) | 5분봉 마감마다 한 번(캐시 = 봉 시각) | `_flow_read_ctx` 의 `dir_p`·`reach_p` → `fr.read`(융합 카드 «30분 안 닿음/방향») · ETH 융합 장부 `flow_read_fused.jsonl` 의 dir_p·reach_p · 시장 맥락 카드 | **쓰임 → 옮김**(위 표) |
| 입력 5분봉 | `load_5m_day`(REST 300개 · 60초 캐시 · 형성 봉 시각으로 거름) 또는 카드 캐시 | — | 같은 봉 목록을 상황 카드·권장 배수가 같이 쓴다 | 봉 받기는 rtserve/rtcore 몫 |
| 원본 `run.py`·`live_parity.py` | 저장소에 없음 · 학습·내보내기 = `scripts/research_eth_card30_direction_hgb_tabpfn_20260926.py`(저장소에 있음) | — | — | 서빙엔 필요 없음 |

## 응답 서버(rtserve)에 붙일 공개 함수 (2026-11-21 · `rtmodel::handlers`)

| 경로 | 원본 | rtmodel 함수 | 입력 → 출력 |
|---|---|---|---|
| /api/dir4h | dir4h_cached → d4.live_payload | `handlers::dir4h_body` | Rest 캐시 + Get → Value(키 = 원본 순서 · preserve_order 크레이트에서 직렬화) |
| /api/dir-short | dir_short_cached → ds.live_payload | `handlers::dir_short_body` | Rest(4h 와 따로) + Get + 표본 밖 분위 + (ETH) 호가 깊이 한 봉 콜백 → Value |
| /api/vol-forecast | qx4_payload → 실패면 HGB | `handlers::vol_forecast_qx4` → None 이면 `handlers::vol_forecast_hgb` | QX4 행 상태 + fetch → Value(+asset) · HGB = fetch → 글자(+asset) · QX4I = (시간 칸 기억, Deribit get) 을 넘기면 iv_now → add_iv |
| /api/open-rev | api_open_rev | `handlers::open_rev_body` | fetch(1일봉 93 · 5분봉 600) + 실수 시계 → 글자 |
| /api/dir-signals | dir_signals_payload | `handlers::dir_signals_body`(= `signals::dir_signals`) | fetch + FRED 키 → 글자 |
| /api/regime-band · /api/risk-radar · /api/sr-lean · /api/pred-record | 파일 읽기만 | rtserve/src/api.rs 에 이미 있음 | — |
| 상황 카드 dir_p·reach_p | _card30_pred | `card30::predict` | 완결 5분봉 → (dir, reach) |

모델 다시 읽기(mtime)·캐시 시간·밴 가드·예외를 `{"ok": false, "reason"}` 로 감싸기는 부르는 쪽(rtserve)이 한다.

## pandas 가속 꾸러미(bottleneck·numexpr)에 따라 갈리는 자리 (2026-11-21 · 서버 확인함: 둘 다 없음 · pandas 2.2.3)

pandas 의 Series/DataFrame 줄이기(mean·std·var·min·max·median)는 bottleneck 이 깔려 있으면 그쪽(차례 합)을, 없으면 numpy(짝 합)를 탄다. rolling·groupby·resample 은 pandas 자체 cython 이라 무관.
서버에 bottleneck 이 없으므로 아래는 모두 «numpy 짝 합» 으로 옮긴 것이 맞다. 서버에 bottleneck 이 생기면 이 자리부터 다시 정답지를 뜬다.

| 자리 | 원본 | Rust | 영향 |
|---|---|---|---|
| 상대 강도 스냅숏 평균 | `pred_record.xsec_record` `rel.mean()`(9코인 실수) | `dirrest::Record::xsec` `npsum / 9` | rel_bp 끝자리(0.1bp 반올림 뒤 거의 안 보임) · xsec.jsonl 줄은 무관 |
| 띠 안·이탈 요약 적중률 | `recorder.score`·`summarize` `.mean()`(정수 0/1) | `regime_band::summarize` 정수 합 ÷ n | 없음(정수 합은 어느 길이든 정확) |
| (나머지) | dir4h·dir_short·QX4·위험 레이더·박스·레벨 5분·신호 -- numpy 함수(np.mean·np.std·np.nanmedian…)를 직접 부르거나 rolling 만 씀 | — | 무관 |

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
