# 서버 부하 전수 조사와 섀도우 2축 은퇴 (2026-09-15)

사용자 요청 "새도우나 수집기 등 현재 서버에 어떤 부하를 주고 있는 작업 리스트를 모두 조사"에서
시작해, 후속 지시로 두 축을 은퇴시키고 무동작 코드 하나를 제거한 기록.

커밋: `6da437d` `82fa4a0` `362b96f` `37ec23a` `03c0b8d`
재현 스크립트: `scripts/audit_maker_fill_shadow_ledgers_20260915.py`

---

## 0. 출발점 — 부하 baseline (09-15 06:02 KST)

```
load avg 1.16 / 0.89 / 1.11 (12코어)   mem 18Gi/31Gi   swap 1.6Gi   disk 410G/1007G
repo 관련 프로세스 68개 (감독 래퍼 포함)
```

CPU 는 여유였고 **유일한 압박 신호는 swap 1.6Gi** 였다. 메모리 상위 8개가 18Gi 중 16Gi 를 썼다.

| RSS | CPU | 프로세스 |
|---|---|---|
| 3.55G | 8.0% | `trading_bot.py` |
| 2.36G | 10.4% | `live_eth_v_rebound_econ_shadow_runner_20260902.py` ← **은퇴** |
| 2.20G | 4.7% | `live_signal_worker.py` regime_gbm3 |
| 2.15G | 1.3% | `live_signal_worker.py` evidence_metalabel |
| 2.10G | 3.0% | `live_signal_worker.py` v_rebound sweep |
| 2.08G | 9.8% | `live_eth_extreme_detector_worker_20260910.py` |
| 912M | 0.4% | `dashboard/server.py` |
| 682M | 0.0% | `live_signal_worker.py` macro_calendar |

그 외: 수집기 13종(l2_anomaly ×5 · duckdb_persist ×4 · orderflow_raster · book_ticker ·
depth_diff · liq_magnet), 섀도우 7종(maker_fill ×5 · v_rebound econ · exit_ladder),
cron 8줄 + `@reboot` 26줄.

---

## 1. V자반등 경제라벨 섀도우 은퇴 (`6da437d`, 정정 `82fa4a0`)

### 판정 — 유의하게 음수

09-05 리셋 이후 단일 설정(`3seed@0.8221` · `backlog_entry_blocked` · `exit_basis=bar_high_low`,
계측 섞임 없음) **139건 / 9.54일**:

```
기대값 -20.68bp   t -2.71   95%CI [-35.6, -5.7]  ← 0 배제
누적 -2874bp   최대DD -3249bp   승률 64.0% 인데 손익비 0.260
```

한 건짜리 사고가 아니다:

| 절단 | 결과 |
|---|---|
| 최악 1/3/5/10건 제외 | -17.6 / -13.2 / -9.9 / **-2.68bp** — 7%를 잘라도 양수 안 됨 |
| 전반69 / 후반70 | -18.8 / -22.5 (둘 다 음수, 악화) |
| 롱95 / 숏44 | -26.7 / -7.7 (양 측면 음수) |
| 일별 8일 | 양수 2 / 음수 6 |

낙관 편향 여지도 없다 — 유일한 양수 기여인 `stop_infeasible` 84건(60.4%) +26.5bp 는 09-07
「걸 수 없는 스톱」 현실화 경로라 **이미 유리한 쪽** 계측이고, 손실은 `stop` 53건 -87.9bp 에서 난다.

### ⚠️기준선 정정 — 기각이 아니라 확인이다

러너 `--report` 가 찍는 「HOLDOUT +6.09bp」를 그대로 인용해 "라이브가 백테스트를 기각"이라고
먼저 보고했는데 **틀렸다**. 그 상수는 09-08 에 폐기된 legacy `sim_exit` 값이고, 09-07 회계 수정
(`infeasible="exit"`) 후 실제 기준선은 **-8.42bp/건**이다
(`docs/experiments/eth_v_rebound_econ_hold_cap_removal_20260908.md:41`). 대시보드 카드와 실험
문서는 **둘 다 갱신돼 있었고 러너 docstring/리포트만 뒤처져 있었다.**

⇒ 올바른 문장: **이미 음수였던 백테스트를 라이브가 더 나쁘게 확인했다**(-8.42 는 라이브 CI 안).
결론은 같고 근거는 더 강하다 — 라이브 CI 가 0 을 배제한다.

---

## 2. 대시보드 카드 제거 (`362b96f`)

러너가 죽었으므로 원장이 -2874bp 에서 동결된다. 카드를 두면 죽은 숫자를 60초마다 폴링한다.
`server.py`(payload/핸들러/라우트) · `app.js`(폴링·렌더·헬퍼) · `styles.css`(`.vshadow-*` 전량) ·
`index.html`(섹션) 에서 **485줄 삭제**.

- **손대지 않은 것**: 위쪽 **V자반등 칩**(`modelChipVRebound`/`latestVRebound`/`refreshVReboundSignal`)
  은 라벨 정의부터 다른 별개 모델이다.
- **유일하게 조심한 곳**: ≤430px 공유 미디어쿼리에서 `.vshadow-trade-ts` 가 선언 블록의 `{` 를
  들고 있는 마지막 선택자였다. 두 선택자만 빼고 `.detail-toggle` 이 블록을 물려받게 했다
  (과거 `},,` 하나로 대시보드가 두 번 죽은 자리).
- 검증: `py_compile` · `check_appjs_dict_integrity_20260911.py` 3종 · CSS 중괄호 608/608 ·
  잔존 참조 0 · sw.js/manifest 0 · 배포 후 `/api/v-rebound-econ-shadow` **404** 확인.

---

## 3. maker_fill_shadow 비ETH 4종 은퇴 (`37ec23a`)

### ⭐체결분만 보면 정반대 결론이 나온다

첫 집계가 `filled == True` 만 평균내서 "static 이 압도적으로 싸다"고 나왔다. **틀린 비교다** —
static 은 체결률이 8~10% 낮고, 그 미체결분의 taker 폴백 비용이 빠져 있었다.

| 심볼 | peg 왕복 (체결만 / 전량) | static 왕복 (체결만 / 전량) | static 미체결비용 | 전량 peg−static |
|---|---|---|---|---|
| ETH | 5.75 / **5.83** | 3.96 / **5.86** | 11.96bp (9.5%) | **-0.03** |
| BTC | 5.24 / **5.41** | 3.99 / **5.43** | 8.71bp (10.8%) | **-0.03** |
| SOL | 5.31 / **5.38** | 3.02 / **5.24** | 12.59bp (10.0%) | **+0.13** |
| XRP | 5.28 / **5.31** | 3.28 / **5.05** | 12.97bp (7.8%) | **+0.26** |
| HYPE | 6.41 / **6.41** | 3.88 / **6.04** | 16.97bp (7.2%) | **+0.38** |

**전량 기준 두 정책은 동률**이다. 원장은 미체결 leg 에도 `cost_bp` 를 taker 폴백가로 남기므로
`filled` 필터만 빼면 된다. 「결과로 걸러진 부분집합이 지표를 부풀린다」의 집행판 재현.

### 은퇴 근거 3가지

1. **수렴 완료.** 각 ~11,556 legs / 10.6일, CI 반폭 ±0.07~0.12bp. 마지막 5일(표본 47% 추가)이
   옮긴 폭 BTC +0.06 / SOL +0.01 / XRP +0.06 / HYPE +0.08bp. 전·후반 Δ 는 4종 다 CI 에 0 포함.
   **⭐ETH 는 반대** — 전반 6.26 → 후반 5.39, Δ **-0.87bp (±0.12) 유의**. 그래서 ETH 는 남겼다.
2. **소비처 0.** dashboard / ops_watchdog / trading_bot 어디도 이 4개 duckdb 를 읽지 않는다.
3. **비거래 심볼.** 실계좌 최근 체결 11건 전부 ETHUSDT, 열린 포지션도 ETHUSDT 하나.

### ⚠️데이터 보존 — WAL 이 체크포인트돼 있지 않았다

본체 mtime 이 09-04 에 멈춰 있어 하마터면 "11일간 안 쓰고 있다"로 읽을 뻔했다. 실제로는
**본체 12KB / WAL 7.9MB** 였다 — 워커에 SIGTERM 핸들러도 `con.close()` 도 없어 체크포인트가
한 번도 안 됐던 것이다. 데이터 유실은 없지만(다음 open 에서 DuckDB 가 replay), 자립 파일로
남기려면 **정지 후 각 DB 를 한 번 열어 `CHECKPOINT`** 해야 한다.
실행 후 각 1.01MB · WAL 소멸 · legs 11,555~11,560 · heartbeat 2,884 확인.
원장은 `data/live/maker_fill_shadow_{btc,sol,xrp,hype}.duckdb` 에 그 자리 보존.

---

## 4. 결정시점 동기화 arm 제거 (`03c0b8d`)

2026-08-24 에 넣은 축인데 leg 를 거의 못 만들었다. 원인이 **셋**이고 심볼별로 다르다.

| 심볼 | 원인 | 고칠 수 있나 |
|---|---|---|
| ETH | **전이가 실제로 0.17건/일** | 고장 아님 — 전제가 틀림 |
| BTC/SOL | 테이블에 `final_action` 필드가 **없음** | 봇이 그 심볼 최종결정을 기록하기 전엔 불가 |
| (증폭) | 빈 `last_ts` 가 영구 ConversionException | 가드 부재 |

1. **ETH — arm 은 정상이고 잡을 게 없었다.** 워커 가동 이후 `final_governor_decision` 6,408행,
   행 간격 중앙 300s, 액션 분포 `{2: 6,374 (99.5%), 0: 34}`. 23일간 전이 **4건**뿐이고 워커는
   그 4건을 전부 잡았다(4×4 = **leg 16건**, 원장과 정확히 일치). 놓친 전이 0, "too old" 0.
   그 16건의 대가로 10초마다 봇 duckdb(292MB)에 락을 걸어 **5,705회** 충돌했다
   (`DECISION_POLL_S=10` vs 스냅샷 기록 주기 300s = **30배 과잉 폴링**).
2. **BTC/SOL — 구조적으로 불가능.** `orderbook_decision_snapshots_{btc,sol}` 은
   `record_reason='omega4_6_1_shadow_decision'` 만 담고 키가 `['asset','price','record_reason']`
   뿐이다. 필터 상수 문제가 아니라 **필요한 데이터가 그 테이블에 없다.**
3. **그게 영구 에러 루프로 증폭됐다.** init 의 tail 이 비면 `last_ts=""` 로 남는데, `last_ts` 는
   쿼리 성공 후 행 루프 안에서만 전진한다. 그래서 `recorded_at_kst > ''` 가 영구
   ConversionException 이 된다(BTC 86,417 · SOL 86,418회). **재시작 없이는 회복 불가능한 구조.**

제거: `DECISION_*` 설정 6개 · `Leg` 의 `trigger`/`decision_*` 필드 4개 ·
`_decision_rows_sync`/`_spawn_decision_legs`/`decision_loop` · `run()` gather 항목 (−126/+17줄).
원장 호환은 유지 — `trigger` 컬럼과 마이그레이션은 남기고 새 행은 항상 `'schedule'`(과거 ETH
decision 16건이 계속 읽힌다). `decision_*` 3컬럼은 새 DB 에서 더 이상 만들지 않는다.
자체점검: 임시 DB 2종(신규 / 구스키마 4컬럼 보유) 에 실제로 써서 insert 인자수(21=21)와
과거 행 보존을 확인.

---

## 5. ⭐틀렸던 판단 2건 — 이게 이 문서의 핵심이다

두 번 다 **"CPU 를 안 쓰는데 자원을 잡고 있다"를 낭비로 읽은** 같은 오류다.

### (a) `live_signal_worker` 4개를 "줄일 순서 ①"로 올렸다

`ps` 의 `%CPU` 는 **가동시간 평균**이라 주기 워커는 원래 낮게 찍힌다. 누적 CPU 시간으로 보면
regime 4.7% / v_rebound 3.0% / metalabel 1.4% 로 실제로 일하고 있고, 여섯 개 상태파일이
**전부 대시보드가 읽는 것**이며 전부 신선했다(0~12분).

더 중요한 건 **그 구조 자체가 실장애 대응으로 만들어졌다**는 점이다. `live_signal_worker.py`
상단 기록: 2026-09-10 에 대시보드 요청 경로에서 모델을 돌리다 `to_thread` 16스레드 풀이 고갈돼
대시보드가 멈췄고, 09-13 실측으로 재시작 직후 첫 요청이 V자 **137초** / 증거신호 **160초**를
기다렸다(두 번째가 더 느린 경우도 — 경합). **상주 메모리 7.1GB 는 낭비가 아니라 그 지연을
안 겪는 대가다.**

### (b) depthdiff 를 "보존 정책을 봐야 할 축"이라고 했다

커밋 `79ffcb8`(09-15 00:23), 가동 **6시간**, 파일 8개. "하루 ~0.5GB, 한 달 ~15GB"는 6시간
표본의 외삽이었고 시간별 편차가 12.6M~36.4M 로 3배 널뛴다. 게다가 이건 방치된 수집기가 아니라
**같은 날 착수한 연구 데이터 수집**이다 — 호가벽 수명 중앙값 0.61초라 30초 패널이 94.5%를
못 봤다는 발견 때문에 켠 것이고, 커밋이 말하는 "달력 대기"는 **쌓이기를 기다리는 것 자체가
목적**이라는 뜻이다.

**교훈**: 자원을 쥔 프로세스를 낭비로 분류하기 전에 ① 누적 CPU 시간(평균 아님) ② 산출물의
소비처 ③ 그 구조가 생긴 이유(코드 주석·커밋 메시지) 를 먼저 읽는다. 이번에 실제로 정리할 게
있었던 건 처음 두 축뿐이었다.

---

## 6. 재사용 가능한 절차

### 섀도우 은퇴는 4단이 한 쌍이다
`ops_watchdog.py` 의 SHADOW_RUNNERS 주석이 직접 경고하는 실수이고 09-05·09-14 두 번 재발했다.
1. 러너 정지 — **감독(`_supervise.sh`)을 먼저** TERM, 그 다음 자식. 순서가 반대면 되살아난다.
2. `crontab -l | grep -v <supervisor> | crontab -` (백업 먼저)
3. `scripts/ops_watchdog.py` SHADOW_RUNNERS 표에서 줄 제거 → main 머지 시 deploy_watcher 가
   `ops-watchdog.service` 를 자동 재시작. **안 지우면 영구 CRITICAL.**
4. 원장 보존(필요하면 `CHECKPOINT`).

### 프로세스 정지는 명시 PID 로만 한다
`pgrep -f maker_fill_shadow_worker.py` 가 **5개**를 반환했다 — 워커 + 감독(argv 에 스크립트명이
들어간다) + **작업 쉘 자신 3개**(bash -c 명령문에 그 문자열이 있어서). "1개가 아니면 중단"
가드가 없었다면 감독까지 죽여 ETH 섀도우가 조용히 사라졌을 것이다. `/proc/<pid>/cmdline` 으로
`argv[0]` 이 python 이고 스크립트 경로를 포함하는 프로세스만 특정한 뒤 명시 PID 로 TERM 한다.

### `scripts/*.py` 워커는 deploy_watcher 가 재시작하지 않는다
재시작 대상은 `trading_bot.py` / `ops_watchdog.py` / `prometheus_exporter.py` / `dashboard/**`
뿐이다. 워커 코드를 고쳤으면 **자식만 TERM** 해서 `_supervise.sh` 가 15초 뒤 새 코드로 올리게 한다.
새 코드가 실제로 도는지는 **관측 가능한 변화**로 확인한다(이번엔 로그 형식이
`leg done: schedule peg ...` → `leg done: peg ...` 로 바뀌는 것).

---

## 7. 최종 상태 (09-15 07:03 KST)

```
mem 16Gi/31Gi (available 12 → 15Gi)     load avg 0.87 (조사 시작 1.16)
정지된 상주 프로세스 5개 — v_rebound econ 섀도우 1 + maker_fill 비ETH 4
```

| 은퇴 | 근거 | 회수 |
|---|---|---|
| V자반등 경제라벨 섀도우 | 139건 -20.68bp, t -2.71, CI 가 0 배제 | RSS 2.36GB · CPU 10.4% |
| maker_fill_shadow 비ETH 4종 | 수렴 완료 · 소비처 0 · 비거래 심볼 | CPU ~6% |
| 결정시점 동기화 arm | 23일 leg 16개 · BTC/SOL 구조적 불가 | 봇 duckdb 락 경합 제거 |

배포 검증: 서버 HEAD `03c0b8d` · ops-watchdog active(SHADOW_RUNNERS 빈 튜플 서빙) ·
대시보드 8787 LISTEN · 제거 엔드포인트 404 · ETH 워커 pid 1965022(ppid 448) 신형식 로그 ·
원장 trigger 분포 `schedule 25,742 / decision 16`.
