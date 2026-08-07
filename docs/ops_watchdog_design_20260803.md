# 24시간 운영 감시 프로그램 설계

## 목적

`ops_watchdog`는 매매 판단과 분리된 단일 감독 프로세스다. 트레이딩봇,
데이터 수집, Tau1/Sigma6 새도우, 대시보드 산출물 및 저장소의 진행 상태와
무결성을 감시하고 상태 변화만 Telegram으로 알린다.

## 구성

```text
live processes / exchange / data/live artifacts
                    |
                    v
         ops_watchdog (30초 주기)
                    |
        +-----------+-----------+
        |                       |
        v                       v
health_snapshot.json     incidents.sqlite / Telegram
```

감시기는 주문을 제출하거나 데이터/모델 계약을 자동 보정하지 않는다. 초기 단계에는
자동 재시작도 하지 않으며, 원인을 드러내는 감시와 경보만 한다.

## 감시 항목과 타이트한 임계값

| 항목 | 정상 진행 기준 | WARN | CRITICAL |
|---|---|---:|---:|
| 5분 시장 스냅샷 | `decision_feature_snapshot.jsonl` 마지막 시장 시각 | 7분 | 12분 |
| 봇 heartbeat | `trading_bot_decision_heartbeat.json` 기록 시각 | 6분 | 10분 |
| 데이터 파이프라인 | `data_pipeline_health.json`의 raw ETH 마지막 시각 | 7분 | 12분 |
| 대시보드 상태 | `dashboard_state.json` cycle timestamp | 7분 | 12분 |
| Tau1 1시간봉 진행 | `sigma6_regime_tiebreak_shadow/state.json`의 `last_processed_bar_ts` | 70분 | 90분 |
| Tau1 equity 출력 | `equity_curve.jsonl` 마지막 `bar_ts`가 state와 일치 | 즉시 | 즉시 |
| 프로세스 | 등록된 명령행 signature와 heartbeat | 즉시 | 즉시 |
| JSON/JSONL 계약 | 필수 필드·마지막 행 파싱 | 즉시 | 즉시 |

파일 수정 시각만으로 정상 여부를 판단하지 않는다. 예를 들어 state 파일이 다시
저장되어도 입력의 시장 timestamp 또는 `last_processed_bar_ts`가 전진하지 않으면
`stale`로 판정한다.

## 상태와 알림

상태는 `OK`, `WARN`, `CRITICAL`, `BLOCKED`다. 각 check의 이전 상태를
`data/live/ops_watchdog/state.json`에 저장한다.

| 상태 전이 | Telegram 동작 |
|---|---|
| `OK -> WARN` | 즉시 1회, 동일 경고는 2시간 후 재알림 |
| `WARN -> CRITICAL` 또는 `* -> BLOCKED` | 즉시 1회, 30분마다 재알림 |
| `WARN/CRITICAL/BLOCKED -> OK` | 복구 알림 1회 |
| 동일 상태 | 위 재알림 주기 외에는 발송하지 않음 |

메시지에는 severity, component, 감지 시각(KST), 마지막 정상/진행 시각,
지연 시간, 영향 범위, 원인 후보, incident ID를 포함한다. 토큰, 주문 세부 정보,
개인 정보는 메시지나 이력에 기록하지 않는다.

Tau1 예시:

```text
CRITICAL tau1_shadow_progress_stale
마지막 처리 봉: 2026-08-02 21:00 KST
현재 지연: 15시간
원인 후보: decision_feature_snapshot 입력 정지
```

## 구현 구조

```text
trading_bot_modules/telegram_notifier.py  # 트레이딩봇과 watchdog 공용 Telegram 전송기
scripts/ops_watchdog.py                   # 30초 주기 검사, 상태 전이, Telegram 발송
data/live/ops_watchdog/
  health_snapshot.json                    # 대시보드용 최신 상태
  incidents.sqlite                        # 상태 변화와 발송 이력
  state.json                              # dedupe/재알림 상태
  watchdog_heartbeat.json                 # watchdog 자체 heartbeat
```

감시 대상은 코드에 암묵적으로 찾지 않고 명시적 component 등록 목록으로 관리한다.
각 component는 process signature, 입력/출력 경로, 진행 timestamp field, 임계값,
그리고 `live` 또는 `shadow` 모드를 갖는다.

## 검증과 롤아웃

1. `ops_watchdog.py --once --dry-run`으로 현재 상태를 Telegram 발송 없이 검사한다.
2. 고의로 오래된 fixture를 사용해 WARN, CRITICAL, JSON 계약 오류, 복구 전이를 검증한다.
3. `--interval-seconds 30`으로 24시간 실행하되 첫 주는 자동 재시작을 활성화하지 않는다.
4. 대시보드가 `health_snapshot.json`을 표시하고 Telegram 경보와 incident ID가 일치하는지 확인한다.

