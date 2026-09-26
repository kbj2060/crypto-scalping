# BTC·SOL·XRP·HYPE 실시간 수집기 확장 (2026-09-26)

ETH 전용으로 돌던 연속 수집기(바이낸스·OKX·하이퍼리퀴드)를 BTC·SOL·XRP·HYPE 로 넓힌다.
이 문서는 **코드 수정 내역 · 용량 판단 · 기동 절차**를 한곳에 둔다.
도구: `scripts/ops/multicoin_collectors_20260926.py` (`plan` / `start` / `resume` / `check`).

⚠️ 이 세션은 서버·Pi 에 SSH 가 없고 거래소 REST/WS 도 네트워크 정책으로 막혀 있었다. 아래 수치 중
**RSS 는 이 세션 실측**(x86_64, 네트워크 없이 기동 후 15초)이고, **용량은 ETH 실측의 비례 추정**이다.
하루 운용 뒤 `check` 로 실제 값을 다시 잰다(§4).

## 1. 환경변수만 바꿔 띄우면 깨지던 곳 → 고친 내역

| # | 수집기 | 문제 | 수정 |
|---|---|---|---|
| 1 | HL 고래 포지션 | `clearinghouseState` 는 한 주소의 **전 코인**을 한 번에 준다. 코인마다 띄우면 같은 주소를 코인 수만큼 조회 → 5개면 1,500/분 > 한도 1,200 | `HL_POS_COINS=BTC,SOL,XRP,HYPE` **한 프로세스**가 코인별 상위 300 주소의 합집합을 돌고 응답 하나에서 전 코인을 적는다. ETH 프로세스는 그대로(사전등록 검정의 2.5분 바퀴). 합계 600/분(50%) |
| 2 | HL 체결 | `HL_COINS=ETH,BTC` 면 폴더가 `ETH_BTC/` → 포지션 수집기·ETH 연속성 깨짐. supervisor 는 두 번째 인스턴스를 무조건 막았다 | supervisor 를 코인별 락으로. 콤마 목록은 거부 |
| 3 | OKX 컨텍스트 | 청산 채널이 `instType=SWAP` 전 종목 → 종목마다 띄우면 같은 청산이 중복 저장 | 청산 구독은 `ETH-USDT-SWAP` 프로세스만(전 종목이 이미 거기 있다) |
| 4 | OKX 3종 | `CT_VALS`/`BUCKETS` 에 ETH·BTC·SOL 만 | XRP(ctVal 100)·HYPE(0.1) 추가. 🔴**거래소 미실측** -- `start` 가 REST 로 대조하고 실패·불일치면 띄우지 않는다(수집기 자체 대조는 «REST 안 되면 통과») |
| 5 | 공유 duckdb | 테이프·OKX 테이프·OKX 컨텍스트·HL 컨텍스트가 파일 하나 → 5개 writer 가 번갈아 열면 읽는 쪽과 잠금 충돌 5배 | ETH 가 아니면 코인별 파일이 기본값(`trade_tape_btc.duckdb`, `okx_context_xrp.duckdb`, `hyperliquid_context_hype.duckdb` …). ETH 경로는 불변 |
| 6 | 래스터 보관 뷰 | `book` 뷰가 전 심볼 폴더를 글롭하는데 symbol·빈 폭 열이 없어 ETH·BTC 빈이 섞인다 | 뷰에 `symbol`(경로에서), parquet 에 `bin_size` 열. 이전 파일은 `bin_size` NULL(= ETHUSDT 0.5 시절) |
| 7 | 래스터 빈 폭 | `OF_BIN_SIZE`=$0.50 은 ETH 가격대 기준 | `start` 가 ETH 와 같은 상대 폭(0.5/ETH가격)으로 코인별 계산, 틱 배수로 반올림해 crontab 줄에 **고정** |
| 8 | 테이프 자동 복구 | 망 복구 뒤 모든 심볼·호스트가 동시에 aggTrades(쪽당 가중치 20)로 메운다 — 봇과 같은 IP 한도 2,400/분 | ETH 외는 한 주기 1분만 |
| 9 | 감시 | 새 수집기에 신선도 검사 없음 | 워치독이 **호스트별 매니페스트**(`data/live/multicoin_collectors.json`)에 적힌 것만 본다. 안 띄운 호스트에서 영구 BLOCKED 가 나지 않고, 배포가 기동보다 먼저 닿아도 안 울린다. 기동 직후 유예 15분(포지션 30분) |

## 2. 용량 (판단 근거)

**메모리 — 이번에 새로 잰 값이 앞선 추정(70MB/개)보다 크다.** duckdb 를 import 하는 수집기는 그것만으로 ~150MB 다.

| 수집기 | RSS(MB) | | 수집기 | RSS(MB) |
|---|---|---|---|---|
| BN bookTicker | 26 | | HL 호가+컨텍스트 | 151 |
| BN depthDiff | 26 | | HL 포지션(전체 1개) | 151 |
| BN 래스터 | 42~67 | | OKX 호가 | 37 |
| BN 체결 테이프 | 155 | | OKX 컨텍스트 | 155 |
| HL 체결 | 64 | | OKX 체결 테이프 | 156 |

- 1단계(호가·HL·OKX 컨텍스트): 코인당 ~459MB → 4코인 + 포지션 ≈ **2.0GB**
- 2단계(래스터·테이프 둘): 코인당 ~378MB → 4코인 ≈ **1.5GB**
- 합계 ≈ **3.5GB** (+ 기존 ETH 세트 ≈ 0.85GB). Pi 4GB 로는 불가, 8GB 면 가능하지만 여유가 작다.
  `start` 는 가용 메모리에서 이 합을 빼고 512MB 가 안 남으면 띄우지 않는다(`--force` 로 무시).

**디스크** — ETH 1코인 실측 ≈ 0.75GB/일(bookTicker 168 · depthDiff 456 · 래스터 보관 26 · OKX 45 · HL 23 · 테이프 수십 MB).
4코인이 ETH 급이면 합계 ≈ 3.7GB/일 → 서버 여유 545GB(09-16) 기준 **약 4~5개월**에 재검토선(100GB).
BTC 는 ETH 보다 클 가능성이 높고 XRP·HYPE 는 작을 수 있다 — `check` 가 «ETH 대비» 배율을 같이 찍는다.

**거래소 한도** — 평상시 무시할 수준. 위험은 복구 폭주(#8)와 HL 조회(#1)였고 둘 다 코드로 막았다.
서버와 Pi 가 같은 공유기 뒤라 **봇과 공인 IP 를 나눈다**고 보고 설계했다.

## 3. 기동 절차 (수집 호스트에서, quant_ai 파이썬으로)

```bash
git pull   # 이 커밋이 있어야 한다
PY=~/miniforge3/envs/quant_ai/bin/python      # 서버는 ~/miniconda3/...
$PY scripts/ops/multicoin_collectors_20260926.py plan  --phase 1               # 사전점검·crontab 줄만
$PY scripts/ops/multicoin_collectors_20260926.py start --phase 1 --install-cron # 소급 불가한 것부터
# 하루 뒤 check 가 깨끗하면
$PY scripts/ops/multicoin_collectors_20260926.py start --phase 2 --install-cron
```

- **1단계**: BN bookTicker·depthDiff, HL 체결·호가+컨텍스트·포지션(1개), OKX 호가·컨텍스트 — 전부 소급 불가.
- **2단계**: BN 래스터·체결 테이프, OKX 체결 테이프 — 테이프는 거래소 덤프로 소급 가능해서 뒤로 뒀다.
- `--tail-risk-btc-sol`: 09-19 에 멈춘 BTC·SOL 청산(forceOrder) 1분 워커를 되살린다. **서버에서만**
  (기존 `tail_risk_btc_sol.duckdb` 가 서버에 있다). ⚠️09-25 커밋 `671c0b1` 로 청산 수량 정의가 바뀌어
  USD 값이 ~6배다 — 09-19 이전 행과 섞지 말 것.
- **어느 호스트? — 실제 배치(2026-09-26 기동): 1단계 = Pi, 2단계 = 서버.** 한 스트림은 **한 호스트에서만** 띄운다
  (두 곳이면 IP 한도를 두 번 쓴다). Pi(3.8GB)는 1단계 실측 RSS 합 2.5GB 로 2단계(+1.5GB)가 안 들어가서 갈랐다.
  - 1단계 orderflow `.gz`(bookTicker·depthDiff·OKX/HL 호가·HL 체결 일별)는 기존 `ship_orderflow_archive` 가 그대로 서버로 보낸다.
  - 2단계 테이프·래스터는 처음부터 서버에 있으므로 테이프 복제 cron 이 필요 없다.
  - Pi 의 OKX/HL 컨텍스트·HL 포지션 duckdb 는 복제 경로가 **아직 없다**(ETH 도 마찬가지 — Pi 에만 남는다).
  - 워치독(`ops_watchdog.py`)은 서버 매니페스트(2단계)만 본다. Pi(1단계)는 §4 의 `check` 로 본다.
  - Pi 의 `check` 디스크 줄은 재검토선 100GB(서버 기준)라 117GB SSD 에선 늘 음수 일수로 나온다 — Pi 는 전송 cron(48h 뒤 삭제)이 지킨다.

## 4. 하루 뒤 점검

```bash
$PY scripts/ops/multicoin_collectors_20260926.py check --hours 24
```

수집기별 pid·RSS·CPU·마지막 쓰기·MB/일·ETH 대비 배율, 합계 RSS·GB/일, 디스크 재검토선까지 일수, 부하.
죽었거나 critical 을 넘은 게 있으면 🔴 와 함께 종료코드 1. 죽은 것은 `resume` 이 매니페스트대로 다시 띄운다.

## 5. 은퇴

러너 정지 + crontab `@reboot` 줄 제거 + **매니페스트에서 항목 삭제**가 한 쌍이다
(`ops_watchdog.py` SHADOW_RUNNERS 주석의 사고 — 한쪽만 하면 영구 CRITICAL).
