# 레벨 5분 그림자 운영 (rt_sr_lean · 2026-11-21)

Rust 실행 파일 `rt_sr_lean` 을 서버에서 **파이썬 워커(`scripts/live_sr_lean_20261118.py`) 옆에** 돌려, 같은 거래소 스트림으로 쓴 기록 줄이 같은지 대조한다.
파이썬 워커·`data/live/sr_lean/`·전진 검정은 건드리지 않는다. 전환(파이썬을 끄고 Rust 가 원래 디렉터리를 이어 씀)은 대조 통과 뒤 사용자 승인으로.

## 1. 실행 파일과 모델

- 개발기에서 빌드: `cd rtmodel && cargo build --release -j 2 --bin rt_sr_lean` → `rtmodel/target/release/rt_sr_lean`.
  서버에서 빌드하지 않는다(라이브 서버 CPU). 옮긴 뒤 `md5sum` 을 개발기 것과 맞춘다. TLS 는 rustls(OpenSSL 없음).
- 모델: 서버에서 `python rtmodel/tools/export_hgb.py data/live/rt_sr_lean` → `data/live/rt_sr_lean/sr_lean.json`(얼린 joblib 의 sha256 이 파일 안에 있다).
  joblib 이 바뀌면(재학습) 다시 내보낸다. 실행 파일은 scikit-learn 1.8.0 으로 내보낸 파일만 연다.

## 2. 띄우기

```
cp rtmodel/target/release/rt_sr_lean bin/rt_sr_lean       # md5 를 개발기 것과 맞춘다
bash scripts/ops/supervisor_rt_sr_lean.sh                  # 로그 logs/supervisor/rt_sr_lean_YYYYMMDD.log
```

- 기본 출력은 `data/live/rt_sr_lean/`(ethusdt.jsonl · ethusdt_feat.jsonl · ethusdt_state.json). 🔴`data/live/sr_lean` 을 가리키면 종료코드 2 로 거부한다
  (전환 때만 `RT_SR_TAKEOVER=1`). 대시보드 `/api/sr-lean` 은 파이썬 디렉터리만 읽으므로 그림자는 화면에 안 나온다.
- REST(번호 메움 aggTrades · 1분봉 klines)는 공용 밴 파일(`RT_SR_BAN_FILE`, 기본 `data/live/binance_ban_until` = scripts/binance_ban_guard.py)을 지킨다:
  밴 중엔 보내지 않고 실패로 친다(번호 메움 실패 → 구멍 · 1분봉은 1분 뒤 다시). 418·429 를 받으면 해제 시각을 그 파일에 적는다.
  🔴원본 파이썬 워커는 aiohttp 라 이 가드를 안 탄다 -- 그림자 기간엔 REST 호출이 두 배(시작 때 klines 1회 + 2분 뒤 1회 + 번호 빠질 때만 aggTrades · 분당 최대 30회).
- 죽으면(종료코드 1 = 깨진 메시지 등 원본도 죽는 경우) 감독 스크립트 `scripts/ops/supervisor_rt_sr_lean.sh` 가 다시 띄운다(`_supervise.sh` 규칙 ·
  같은 출력 디렉터리의 rt_sr_lean 이 돌면 켜지 않음 · 모델 파일이 없으면 켜지 않음). crontab 줄은 그 파일 머리에 적어만 뒀다 -- 등록은 사람.
- (재)시작 뒤 약 1시간은 준비(파이썬과 같다). 두 프로세스는 시작 시각이 달라 처음 1시간의 «warm» 건너뜀이 다르다.

## 3. 대조 (매일 · 읽기 전용)

```
python rtmodel/tools/shadow_sr_lean.py data/live/sr_lean data/live/rt_sr_lean --since <Rust 준비 끝 epoch>
```

- «양쪽 다 채점한 id» 끼리 ev·res·feat 줄을 글자로 비교한다. 한쪽에만 있는 id 는 수만 센다(분모 병기). 종료코드 0 = 전부 같음.
- 달라질 수 있는 정당한 이유: 한쪽만 끊겼다(구멍 → 1시간 다시 준비) · 시작 시각 · 같은 초 체결의 도착 순서(1초 묶음 종가 pc).
  이 셋이 아닌데 다르면 멈추고 원인을 찾는다(먼저 `tests/sr_lean.rs`·`tests/sr_live.rs` 를 서버 파이썬 버전으로 다시 만든 정답지로).
- 🔴서버 파이썬이 3.12 이상이면 `sum()` 이 보정 합이라 분봉 거래량 끝자리가 다를 수 있다(s_tb 열) -- 서버 버전을 먼저 확인한다.

## 4. 끝낼 때

그림자를 끄고 `data/live/rt_sr_lean/` 을 지운다(백업·봉인 대상 아님). 전환이면 파이썬을 끄고 `RT_SR_OUT=data/live/sr_lean RT_SR_TAKEOVER=1` 로 띄운다 --
원본과 같은 파일에 이어 쓰므로 `/api/sr-lean` 은 그대로 읽는다.

---

# 4h·15분/1시간 방향 예측 루프 그림자 (rt_dirpred · 2026-11-21)

대시보드 서버의 `pred_record_loop`(5분마다 코인별 예측 → `data/live/pred_record/{코인}.jsonl` · 전진 검정 fwd_dirshort 가 읽는 기록)를 따로 돈다.

- 빌드: `cd rtmodel && cargo build --release -j 2 --bin rt_dirpred` → `bin/rt_dirpred`(md5 대조).
- 모델(서버에서 · 매달 재학습 cron 뒤 다시): `python rtmodel/tools/export_dir4h.py data/live data/live/rt_dirpred` · `python rtmodel/tools/export_dirshort.py data/live/rt_dirpred`.
- 띄우기: `bash scripts/ops/supervisor_rt_dirpred.sh`(실행 파일 `bin/rt_dirpred` · 모델 없으면 안 켬 · 같은 출력 디렉터리의 rt_dirpred 가 돌면 안 켬 ·
  로그 logs/supervisor/rt_dirpred_YYYYMMDD.log · @reboot 줄은 그 파일 머리에 적어만 뒀다). 🔴`RT_DIR_OUT=data/live` 는 종료코드 2(파이썬 기록 자리 · 전환 때만 `RT_DIR_TAKEOVER=1`).
- REST: 원본과 같은 요청 열(대상 5분봉 9,000 + metrics 5종 30.5일 → 다음부터 꼬리 300·0.5일 · 19코인 340 · 1일봉 120)을 **파이썬 서버와 따로 한 벌 더** 보낸다. 공용 밴 파일을 지킨다.
- 대조: 두 `pred_record/{코인}.jsonl` 의 같은 (k, t) 줄을 글자로 비교한다(시작 시각이 달라 처음 몇 줄은 한쪽에만 있다).
- 회차 끝 상대 강도: 9코인 5분봉 60개(코인마다 1회 · limit 61) → `pred_record/xsec.jsonl`(원본 xsec_record 와 같은 글자) + `xsec_latest.json`
  (원본은 서버 프로세스 안 `_xs_latest` -- 화면 «4h 상대 강도» 칸이 읽는 스냅숏). 대조: 두 xsec.jsonl 의 같은 t 줄을 글자로.
- 🔴아직 못 하는 것: ETH 15분·1시간(75열판 · 호가 깊이 sqlite) -- 파이썬 서버 기록만 있다.

---

# 위험 레이더 그림자 (rt_radar · 2026-11-21)

원본 `scripts/live_risk_radar_20261115.py` 처럼 **cron 이 매분 한 번** 부르고 끝나는 실행 파일. 파이썬 cron 은 그대로 두고 한 줄을 더한다.

- 빌드: `cd rtmodel && cargo build --release -j 2 --bin rt_radar` → `bin/rt_radar`(md5 대조).
- crontab(사람이 등록): `* * * * * cd <저장소> && bash scripts/ops/supervisor_rt_radar.sh` -- 실행 파일·계수 확인 → `flock -n`(첫 회차 31일 백필 1~2분이
  다음 분 회차와 겹치지 않게 · 겹치면 종료코드 3 으로 건너뜀) → 한 회차(nice 10) · 로그 logs/supervisor/rt_radar_YYYYMMDD.log. 상주가 아니라 재기동 루프는 없다.
- 쓰는 곳: `data/live/rt_risk_radar/`(risk_radar_latest.json · risk_radar_forward.jsonl · cache/ 의 1분봉·DVOL 이진 캐시). 🔴`RT_RADAR_OUT=data/live` 는 종료코드 2(전환 때만 `RT_RADAR_TAKEOVER=1`).
  `/api/risk-radar` 는 파이썬 파일만 읽으므로 그림자는 화면에 안 나온다. 첫 회차는 31일 백필(코인마다 1분봉 45쪽 · 쪽 사이 0.5초)이라 1~2분 걸린다.
- 계수: `RT_RADAR_PARAMS`(기본 dashboard/risk_radar_params_20261116.json) · `RT_RADAR_PARAMS_SESS`(SOXL, 기본 dashboard/risk_radar_params_soxl_20261117.json) -- 파이썬과 같은 파일.
- REST: 바이낸스 1분봉(평소 코인마다 1회 · limit 99) + Deribit DVOL(ETH·BTC 시간에 한 번). 바이낸스만 공용 밴 파일을 지킨다(원본 가드와 같은 범위). 파이썬과 따로 한 벌 더 보낸다.
- 대조: 두 latest.json 의 코인 칸을 비교한다 -- 문자열·참거짓·None 은 같아야 하고 숫자는 1e-12 상대 안(X@β 의 BLAS 합 순서 때문에 끝자리는 갈린다 · REMAINING.md «서버 판과의 차이»).
  분이 어긋나 읽으면 다른 회차를 비교하게 되니 `updated` 가 같은 것끼리만.
- 종료코드: 0 정상 · 1 한 코인이라도 오류(원본과 같음 -- cron 은 다음 분에 다시) · 2 설정 거부.

---

# 띠 안·이탈 그림자 (rt_regime · 2026-11-21)

원본 `scripts/live_regime_band_cron_20261102.py` 처럼 **cron 이 매시 02분 한 번** 부르고 끝나는 실행 파일. 파이썬 cron 은 그대로 두고 한 줄을 더한다.

- 빌드: `cd rtmodel && cargo build --release -j 2 --bin rt_regime` → `bin/rt_regime`(md5 대조).
- 모델(서버에서): `python rtmodel/tools/export_regime.py data/live/regime_band_h48_model.joblib data/live/rt_regime_band/regime_band.json`(원본 joblib sha256 이 파일 안에 있다).
- crontab(사람이 등록): `2 * * * * cd <저장소> && flock -n data/live/.rt_regime.lock nice -n 10 bin/rt_regime >> logs/supervisor/rt_regime_$(date +\%Y\%m\%d).log 2>&1`
- 쓰는 곳: `data/live/rt_regime_band/`(regime_band_forward.jsonl · _scored.jsonl · _summary.json -- 원본과 같은 이름·글자). 🔴`RT_REGIME_OUT=data/live` 는 종료코드 2
  (전환 때만 `RT_REGIME_TAKEOVER=1`). `/api/regime-band` 는 파이썬 파일만 읽으므로 그림자는 화면에 안 나온다.
- 종료코드: 0 정상(출력 JSON 한 줄 = 원본 print 와 같은 글자) · 1 실패(`FAIL …` · 아무것도 안 씀 · 묵은 자료 70분 초과 포함) · 2 설정 거부.
- 대조: 두 regime_band_forward.jsonl 의 같은 (ts, coin) 줄을 글자로. 채점·요약은 기록이 같으면 같다(같은 패널 · 같은 규칙).

---

# 그림자에서 켜면 REST 가 얼마나 느나 (2026-11-21 · 바이낸스 IP 한도 = 가중치 2,400/분 · 서버와 같은 IP)

그림자는 파이썬과 **같은 요청을 한 벌 더** 보낸다(캐시·REST 를 나눠 쓰지 않는다) -- 켠 항목의 몫이 그대로 2배가 된다. 횟수는 정답지(가짜 거래소 요청 열) 실측 · 가중치는 바이낸스 표(klines: limit <100 → 1 · <500 → 2 · ≤1000 → 5 · >1000 → 10).

| 켜는 것 | 파이썬 몫(평소) | 그림자 켜면 | 처음(재기동) 몰림 | 공용 밴 파일 |
|---|---|---|---|---|
| rt_sr_lean | 1분봉 klines 시작 1회 + 2분 뒤 1회 · aggTrades 번호 빠질 때만(분당 ≤ 30) | ×2 | 작음 | Rust 만 지킴(파이썬 워커는 aiohttp) |
| rt_dirpred(4h·15분/1시간) | 5분마다 약 108회(코인·모듈마다 5분봉 301 + metrics 6쪽 + 1일봉 · 19코인 341 은 모듈마다 한 번) ≈ 분당 가중치 ~40 + futures/data 분당 ~10회 | ×2 | 코인·모듈마다 5분봉 9쪽(1000) + metrics 91쪽 ≈ 회차 하나에 ~880회(코인 사이 3초) | 둘 다 |
| rt_dirpred 상대 강도(xsec) | 5분마다 9회(limit 61 · 가중치 1) | ×2 | 같음 | 둘 다 |
| rt_radar | 분마다 코인 수만큼 1회(limit 99 · 가중치 1) + Deribit DVOL 시간에 1회 | ×2 | 코인마다 1분봉 45쪽(쪽 사이 0.5초) | 바이낸스만 |
| rt_regime | 시간마다 8회(코인마다 limit 1500 두 쪽 · 가중치 10) = 시간당 80 | ×2(시간당 160) | 같음 | 둘 다 |
| rt_dirfwd | 00·08·16시 6회(1일봉 93 · 5분봉 320 · 가중치 2) | ×2 | 같음 | Rust 만(원본은 urllib) |
| rtduck(`RTDUCK_REST=1`) | 분당 36회 | ×2(분당 72) | — | rtcollect/SHADOW.md |

- 다 켜도 평소 몫은 분당 가중치 수백 안(한도 2,400)이다. 위험은 **재기동 몰림**이 겹칠 때 -- rt_dirpred 첫 회차(~880회)·rt_radar 백필을 같은 분에 띄우지 않는다(하나씩 켜고 첫 회차가 끝난 걸 로그로 본 뒤 다음).
- 418·429 를 받으면 Rust 쪽이 해제 시각을 공용 밴 파일에 적고 그동안 보내지 않는다. 파이썬 대시보드(binance_ban_guard)도 같은 파일을 본다.

---

# 방향 전진 기록기 그림자 (rt_dirfwd · 2026-11-21)

원본 `scripts/live_eth_direction_forward_recorder_20261010.py --record` 처럼 **cron 이 00·08·16 UTC 31분 뒤** 한 번 부르고 끝나는 실행 파일. 전진 판정(dir4h_p2 2027-03-14 · dir3asset 2027-04-08)이 읽는 줄이다.

- 빌드: `cd rtmodel && cargo build --release -j 2 --bin rt_dirfwd` → `bin/rt_dirfwd`(md5 대조). 모델 없음(규칙).
- crontab(사람이 등록 · 파이썬 줄과 같은 분): `31 0,8,16 * * * cd <저장소> && nice -n 10 bin/rt_dirfwd >> logs/supervisor/rt_dirfwd_$(date +\%Y\%m\%d).log 2>&1`
- 쓰는 곳: `data/live/rt_dirfwd/eth_direction_forward.jsonl`(RT_DIRFWD_OUT 디렉터리). 🔴`data/live` 는 종료코드 2(전환 때만 `RT_DIRFWD_TAKEOVER=1`).
- REST: 자산 셋 × (1일봉 93 + 5분봉 320) = 회차당 6회 · 하루 18회(가중치 각 2) -- 그림자면 ×2.
- 대조: 두 파일의 같은 (sym, date, hour) 줄에서 recorded_ms 를 뺀 나머지를 글자로(두 cron 이 같은 분에 돌아도 초가 다르다). 종료코드 0 정상 · 1 실패(원본처럼 앞 자산 줄은 남음 · 시각 밖 포함) · 2 설정 거부.

---

# 서버에서 그 자리에서 대조하고 넘기기 (2026-11-21 · 사용자 «기다리지 말고 바로바로 반영 배포»)

대기 기간(7일 병행·24시간 관찰)을 두지 않는다. **같은 입력에 같은 줄 글자임을 그 자리에서 보이고 곧바로 넘긴다** -- 대조 자체는 줄이지 않는다.
전진 판정이 걸린 기록(fwd_band · dir4h_p2 · fwd_dirshort · xsec)도 같다: Rust 가 같은 파일에 이어 쓰고, 넘기는 순간 같은 시각 줄이 두 번 들어가거나 빠지지 않는 것이 조건.

```
bash rtmodel/tools/takeover.sh <이름> export     # 모델 JSON(data/live/rtmodel/) -- rt_regime · rt_dirpred · rt_sr_lean
bash rtmodel/tools/takeover.sh <이름> check      # Rust 한 회차(그림자 /mnt/data2/rtcollect_shadow/model/<이름>/) vs 파이썬이 실제로 쓴 줄 → «칸 N · 어긋남 M · 짝 없음 K» · 통과면 check.ok
bash rtmodel/tools/takeover.sh <이름> go         # check.ok(2시간 안)일 때만 → crontab 백업·파이썬 줄 주석·Rust 줄 추가(줄 수 검증) → (상주형) 파이썬 감독·워커 끔 → Rust 감독 → 한 회차 지켜봄
bash rtmodel/tools/takeover.sh <이름> back       # 되돌리기(태그 줄만 · 상주형은 Rust 끄고 파이썬 감독 다시)
bash rtmodel/tools/takeover_selftest.sh          # 가짜 저장소·가짜 crontab·가짜 워커로 check·go·back 자체 시험(진짜 crontab·프로세스 안 건드림)
```

| 이름 | check 가 견주는 것(빼는 칸) | check 의 REST | go 의 끊김·중복 막기 | go 뒤 확인 |
|---|---|---|---|---|
| rt_regime | 파이썬 마지막 결정 시각 T 의 4줄 vs Rust 한 회차(그 결정 줄만 뺀 기록 사본에 이어 씀) 글자 · 요약 파일 글자(빼는 칸 없음) · T 뒤 55분 안에 | 8 | (ts, coin) 중복 제거(원본·Rust 둘 다) · 정시 0~5분엔 거부 | 즉시 한 회차(새 줄 0 · 요약 다시 씀) → 요약 파일 갱신 |
| rt_dirfwd | 파이썬 마지막 회차 3줄 vs Rust(같은 recorded_ms) 글자 · 회차 뒤 90분 안에 | 6 | 회차 시각(00·08·16시 29~40분) 거부 -- 원본 기록기는 중복 제거가 없다 | 다음 회차 뒤 `status` |
| rt_radar | latest.json 칸별(updated·data_age_s 빼고 · 실수 1e-12 상대 = X@β BLAS 끝자리) · 파이썬 latest 90초 안 | 처음: 코인마다 1분봉 45쪽 + DVOL · 다음부터 코인마다 1 | 분의 20초 뒤에만 바꿈 + 그림자 1분봉 캐시를 data/live/cache 로 옮겨 첫 회차 백필 없음 | 다음 분 latest 갱신 |
| rt_dirpred | pred_record/{코인}.jsonl·xsec.jsonl 의 같은 (k, t) 줄 글자 | 약 890(처음 회차) | — | 🔴go 불가: 파이썬 몫이 대시보드 서버 프로세스 안(pred_record_loop) -- server.py 에 루프를 끄는 스위치가 있어야 한다 |
| rt_sr_lean | `shadow_sr_lean.py`(양쪽 다 채점한 id 의 줄 글자) -- 그림자가 없으면 띄우고 65분 뒤 다시(원본도 재시작 뒤 1시간 준비) | 웹소켓 1 + klines | 파이썬 감독·워커를 /proc cmdline(저장소 경로·락 파일까지)으로 찾아 감독 먼저 끔 → Rust 감독(같은 폴더 · TAKEOVER) · 재시작이라 1시간 준비(원본 재시작과 같다) | Rust 워커 pid · 상태 파일 갱신 |
| — 요청 때 계산(vol-forecast · open-rev · dir-signals · dir-short · dir4h) | rtserve 몫 | — | — | **rtserve 전환 때**(rtmodel::handlers) |

- 🔴check 의 «같은 시각» 조건: Rust 한 회차는 REST 최근 창을 다시 받는다 -- 파이썬 회차 직후(위 표의 «안에»)에 돌려야 같은 봉을 본다. 늦으면 도구가 거부한다.
- rt_dirpred 의 4h·15분/1시간은 Rest 캐시의 첫 봉이 파이썬(재기동 뒤 이어 붙인 32일)과 다르다 -- rolling 합 끝자리가 갈려 반올림한 칸이 드물게 어긋날 수 있다(어긋남은 그대로 보인다).
- 모델이 매달 바뀌는 것 따라가기: Rust 는 원본 joblib 이 모델 JSON 보다 새것이면(재학습 뒤 내보내기 안 함) **값을 내지 않는다** -- rt_regime 은 종료코드 1(`RT_REGIME_SRC_DIR`) ·
  rt_dirpred 는 그 모델만 건너뜀(`RT_DIR_SRC_DIR` · 로그 «모델 낡음»). 규칙 = `rtmodel::model_stale`(JSON 의 source·sources 이름 → mtime).
  crontab 후보(재학습 cron 시각을 몰라도 되게 매시): `17 * * * * cd <저장소> && bash rtmodel/tools/takeover.sh all export-if-stale >> logs/supervisor/rt_export.log 2>&1`

## card30 · 상황 카드에 dir_p·reach_p 싣기 (결론)

**응답 서버(rtserve)가 합친다.** 원본도 `sit.classify`(코어의 Situation 계산)는 card30 을 쓰지 않고, 그 뒤 서버의 `_flow_read_ctx`·`fr.read` 단계에서 dir_p·reach_p 를 붙인다.
rtserve 가 5분봉 마감 때 한 번 `rtmodel::card30::predict`(완결 5분봉 → (dir, reach) · HGB 10개 · 봉당 수 ms)를 돌려 (봉 시각, dir, reach) 를 들고 있다가 시장 맥락·융합 카드와 융합 장부를 낼 때 싣는다.
코어(rtcore)가 rtmodel 을 path 의존성으로 쓰면 rtmodel 의 실행 파일 의존(ureq·tokio-tungstenite·rustls)과 모델 JSON 읽기가 매초 미는 허브로 번진다 -- 코어는 봉·상황만, 모델 값은 응답 서버.

---

# 서버와 같은 파이썬 환경으로 정답지 만들기 (tools/server_env.sh)

정답지(시험의 «원본 값»)는 서버와 같은 판으로 만들어야 서버에서 도는 파이썬과 비교한 것이 된다. 서버(2026-11-21 확인) = 파이썬 3.11 · numpy 2.2.5 · pandas 2.2.3 ·
scikit-learn 1.8.0 · lightgbm 4.6.0 · numpy BLAS = scipy-openblas 0.3.28 · CPU Zen 3(AVX2 · AVX-512 없음).

```
bash rtmodel/tools/server_env.sh ~/venv_srv                       # 한 번(PyPI 에서 받는다 -- 거래소 접속 아님)
source <(bash rtmodel/tools/server_env.sh ~/venv_srv --env)       # PY · numpy AVX-512 끄기(2.2 이름) · OPENBLAS_CORETYPE=ZEN · OPENBLAS_NUM_THREADS=1
( ulimit -v 6000000; $PY rtmodel/tools/export_hgb.py && $PY rtmodel/tools/gen_sr_fixture.py )   # 다른 생성기도 같은 $PY 로(시험 머리말 순서)
cd rtmodel && cargo test -j 2 --release
```

- 🔴AVX-512 를 끄는 이름이 numpy 판마다 다르다: 2.4 = `X86_V4`(옛 `AVX512F` 등은 조용히 무시) · 2.2 = `AVX512F`·`AVX512_SKX` 등(`X86_V4` 없음 → 조용히 무시).
  `--env`·시험 머리말 명령은 둘을 함께 준다(모르는 이름은 오류 없이 무시 · 함께 주면 두 판 모두 np.log 가 libm 과 같음 -- 2026-11-21 실측).
- 서버 판이 바뀌면(재설치·업그레이드) server_env.sh 의 판 번호를 서버와 맞추고 정답지를 다시 만든다.
