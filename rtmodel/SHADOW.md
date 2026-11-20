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
- 띄우기: `RT_DIR_OUT=data/live/rt_dirpred nice -n 10 bin/rt_dirpred`. 🔴`RT_DIR_OUT=data/live` 는 종료코드 2(파이썬 기록 자리 · 전환 때만 `RT_DIR_TAKEOVER=1`).
  감독 스크립트는 아직 없다(`supervisor_rt_sr_lean.sh` 를 본떠 만든다).
- REST: 원본과 같은 요청 열(대상 5분봉 9,000 + metrics 5종 30.5일 → 다음부터 꼬리 300·0.5일 · 19코인 340 · 1일봉 120)을 **파이썬 서버와 따로 한 벌 더** 보낸다. 공용 밴 파일을 지킨다.
- 대조: 두 `pred_record/{코인}.jsonl` 의 같은 (k, t) 줄을 글자로 비교한다(시작 시각이 달라 처음 몇 줄은 한쪽에만 있다).
- 🔴아직 못 하는 것: ETH 15분·1시간(75열판 · 호가 깊이 sqlite) · 상대 강도(xsec.jsonl) -- 이 둘은 파이썬 서버 기록만 있다.
