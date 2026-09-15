# 증거신호 8칩 제거 — 복원 안내서 (2026-09-16)

사용자 지시: *"증거신호 8칩 제거 진행해줘. 관련 백그라운드 리소스 쓰고 있는것도 제거해줘.
대신에 나중에 백업할 수도 있으니 참고 자료는 남겨줘."*

## 왜 지웠나 — 측정 요약 (2026-09-16 세션)
시작은 *"DeMarker·SMT·칼만·피보나치 말고 나머지는 너무 못 맞추는 것 같다"* 였다.
전수 측정 결과 **그 4/4 분할은 데이터에 없었고**, 8종 전부 쓸 용도가 없다는 쪽으로 닫혔다.

| 용도 | 넘어야 할 벽 | 결과 |
|---|---|---|
| 진입(방향) | 왕복 5.5bp | 건당 −0.81 ~ +0.28bp — 비용 이전에 이미 0 |
| 진입(청산 9종 스윕) | 비용 0 가정 | 56칸 gross 최고 +1.4bp · 오라클 상한도 «변동성» |
| 청산 타이밍 | **0**(추가 비용 없음) | 8/8 CI 0 포함(−0.90 ~ +0.44bp) |
| 청산 파라미터(크기·손절폭) | 증분 > 0 | −0.145%, t −1.05 (양성대조는 +12~57% 검출) |
| 레짐 조건부(저변동·횡보) | — | 승률 50.7%(6레짐 중 4위)·건당 순 −5.30bp, **가장 불리** |
| 극점 탐지기 피쳐 | 기여 > 0 | 빼면 **좋아진다**(정밀도 +8.5~+12.3pp) → 09-16 배포 완료 |

상세: `eth_extreme_detector_drop_evidence_signals_20260916.md`, 메모리
`evidence8_chip_ab_split_not_in_data_20260916`.

## 무엇을 지웠나 / 무엇을 남겼나

**지운 것**
- 화면: `dashboard/live/index.html` 의 «증거 신호 (ETH 5분봉)» 칩 행 8개 + 배지 2개,
  `dashboard/live/app.js` 의 라벨 사전 3종(ETH/BTC/XRP)·렌더·폴링.
- 서버: `/api/evidence-signals`, `/api/evidence-signals-provisional` 엔드포인트와 핸들러,
  `compute_signals()` 호출(측정 8.22초/회 — 대시보드 최고 비용 항목).
- 백그라운드: TabPFN 메타라벨 워커
  (`live_signal_worker.py --compute live_evidence_signal_metalabel_20260829:...`, 60초 주기)
  와 그 supervisor, crontab `@reboot` 등록.

**🔴남긴 것 (지우면 안 되는 것)**
- `scripts/live_evidence_signal_dashboard_20260823.py` — **극점 탐지기가 `compute_signals` 를
  import 한다**(`live_eth_extreme_detector_20260909.py:36`). `p_fast/p_slow/delta_z/vol_z/
  wick/ret3_z/atr_pct/dem/kalman_dev_z` 는 이 모듈이 준다. 발동 플래그만 안 쓰는 것이지
  지표 계산은 살아 있어야 한다.
- 대시보드 서버의 klines 수집 경로 — `load_market_history_from_evidence_cache()` 가
  **차트 캔들을 이 캐시에서 슬라이스**한다(ETH/BTC 1500봉). 신호 계산만 뺐고 수집은 남겼다.
- `evidenceStripSvg` 계열 렌더 함수 — v_rebound·모델 지표 칩과 **공유**한다.
- 연구 산출물 전부: `tmp/eth_causal_population_metalabel_20260902/`(발동 모집단+라벨+확률),
  `data/labels/eth_5m_evidence_chip_causal_20260904/`(TabPFN 컨텍스트),
  `data/live/eth_evidence_metalabel_state.json`(마지막 상태).

## 복원 방법
1. 이 커밋을 `git revert` 한다 (아래 «커밋» 절). 화면·서버가 그대로 돌아온다.
2. 백그라운드 워커를 다시 띄운다:
   `bash scripts/ops/supervisor_signal_worker.sh evidence_metalabel \
      live_evidence_signal_metalabel_20260829:compute_eth_evidence_metalabels_snapshot \
      data/live/eth_evidence_metalabel_state.json 60`
   그리고 crontab 에 `@reboot` 줄을 되살린다(아래 원문).
3. BTC/XRP 칩까지 되살리려면 `git log -S compute_btc_evidence_signals_panel` 로 2026-09-14
   제거 커밋을 찾아 함께 되돌린다.

### 제거한 crontab 줄 (원문 보존)
```
@reboot sleep 40 && cd /home/llewyn/crypto-scalping && bash scripts/ops/supervisor_signal_worker.sh evidence_metalabel live_evidence_signal_metalabel_20260829:compute_eth_evidence_metalabels_snapshot data/live/eth_evidence_metalabel_state.json 60 >> logs/supervisor/signal_worker_evidence_metalabel_reboot.log 2>&1 &
```

## ⚠️복원 전에 읽을 것
이 칩들은 **매매 근거로 쓸 수 없다는 게 측정으로 확인된 상태**다(위 표). 되살린다면
«사람이 보는 맥락» 이상으로 쓰지 말 것. 화면 %는 신호마다 라벨 난이도가 달라
(0.7ATR/40분 ~ 4.2ATR/6시간) **서로 비교할 수 없다** — 이번 세션의 출발점이 그 착시였다.

## 실행 기록 (2026-09-16)
- (1/3) 화면 `6091e51` — index.html 66줄 + app.js 764줄 제거
- (2/3) 서버 `97f3ac7` — server.py −324/+28줄. **차트 캔들 갱신 주체 소실 버그를 같이 고쳤다**
  (frames 를 «None 일 때만» 데우고 있었고 주기 갱신이 증거신호 엔드포인트에 얹혀 있었다).
- (3/3) 백그라운드 — TabPFN 메타라벨 워커(PID 557035, **RSS 1.82GB**, 47시간 상주)와 그
  supervisor 종료, crontab `@reboot` 줄 제거. 서버 백업:
  `data/live/backups/crontab_before_evidence_retire_20260916.txt`.
  🔴순서가 중요하다: crontab -> supervisor -> worker. supervisor 를 먼저 끄지 않으면 워커가
  재기동되고, `pkill -f '<워커>.py'` 는 supervisor 명령줄에도 걸린다(같은 날 극점 워커에서 겪음).

### 복원 시 체크리스트
1. 위 세 커밋을 revert → 화면·서버 복귀
2. 워커 재기동 + crontab 줄 복원(위 원문 또는 서버 백업 파일)
3. `load_market_history_from_evidence_cache()` 의 «매번 부른다» 주석을 보고, 갱신 주체가
   둘이 되지 않는지 확인(엔드포인트가 돌아오면 중복 호출은 swr_cached 가 흡수한다)
