# 지지·저항 «가까워짐» 텔레그램 알림 (2026-11-04)

사용자 «닿는 게 아니라 가까워지면 텔레그램 · 워커와 크론탭까지». 뚫림 알림(서버 `sr_break_loop`, 10-25)은 그대로 두고 «다가옴»만 따로.

## 구성
| 조각 | 위치 | 역할 |
|---|---|---|
| 데이터 | 서버 `GET /api/sr-watch` (`dashboard/server.py` · `sr_watch_view`) | `sr_break_loop` 이 10초마다 코인별 {1초 마지막 체결가 px · 시각 t · 지지1~3 · 저항1~3(화면 srLevelsLive 와 같은 목록)} 를 채운다. 읽기 전용·계산 0·거래소 REST 0 |
| 워커 | `scripts/live_sr_proximity_alert_20261104.py` | 10초마다 위 API 를 읽어 판정 → `ops_watchdog.send_telegram` (`.env` TELEGRAM_BOT_TOKEN·CHAT_ID) |
| 감시 | `scripts/ops/supervisor_sr_proximity.sh` | `_supervise.sh` 크래시 재시작(푸시 알림 데몬과 같은 패턴) |
| 시험 | `test/test_sr_proximity_20261104.py` | 판정·재무장·쿨다운·낡은 가격·서버 보기 |

## 판정
- 코인별 기준거리 `THR_BP`(ETH 20 · BTC 15 · SOL 25 · XRP 25 · SNDK 45 · SOXL 90bp — 1시간 고저폭 중앙값의 약 1/3).
- 거리 ≤ 기준 + «무장» → 한 번 알리고 무장 해제. 거리 ≥ 기준×2 면 다시 무장. 같은 레벨 30분 쿨다운.
- 처음 본 레벨이 이미 기준 안이면 조용히(재시작·매시간 바뀌는 청산맵 때문에 폭주하지 않게). 1초 체결가가 180초 넘게 멈춘 코인은 건너뛴다.
- 문구는 서술형(매매 지시 아님). 상태는 메모리 -- 재시작 직후 첫 폴링은 기준만 잡는다.

## 운영
- crontab(권한 분류기가 ssh 로 crontab 쓰기를 막는다 → 사용자 수동). 서버 기존 줄(`supervisor_push_notifier`·`ob5m`)과 같은 모양:
  ```
  @reboot sleep 90 && cd /home/llewyn/crypto-scalping && bash scripts/ops/supervisor_sr_proximity.sh >> logs/supervisor/sr_proximity_reboot.log 2>&1 &
  ```
  `sleep 90` = 대시보드가 먼저 뜬 뒤(안 떠 있어도 워커는 다음 주기에 다시 시도하므로 필수는 아님).
- 지금 바로 띄우기(서버, 배포 뒤): `cd /home/llewyn/crypto-scalping && nohup bash scripts/ops/supervisor_sr_proximity.sh >> logs/supervisor/sr_proximity_reboot.log 2>&1 &`
- 로그 `logs/supervisor/sr_proximity_YYYYMMDD.log` · 끄기 = supervisor 먼저 종료(`pkill -f supervisor_sr_proximity` 는 supervisor 도 죽인다 -- 의도한 정지일 때만).
- 기준 조정 = `THR_BP` · 너무 잦으면 `COOLDOWN_S` 를 늘린다.
