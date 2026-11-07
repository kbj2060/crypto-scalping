# 데이터 파이프라인·저장 지도 (2026-11-07)

현재 저장 구조의 기준 문서이자 11-07 최적화 기록이다(«2026-11-07 최적화 기록» 절). **지금 값**(크기·수집기·위치)은 손으로 적지 않는다:
서버 `data/live/ops_watchdog/data_map.md` 를 본다(매일 11:30 자동 생성, `scripts/ops/gen_data_map_20261107.py`).

## 원칙

1. **원본과 사본은 항상 다른 디스크.**
2. **자주 쓰고 지연에 민감한 것은 C:(NVMe), 한 번에 크게 읽는 것과 원시 호가는 6TB.**
3. **용량 판단은 Windows 쪽 여유로 한다.** WSL 안의 `df` 는 vhdx 의 가상 크기(1T)를 보여 줄 뿐이다.
   2026-11-07 실측: WSL 안 여유 371GB, 실제 D: 여유 102GB, 소진 7~8GB/일.
4. **라이브 DB 파일 하나를 심볼릭 링크로 옮기지 않는다. 디렉터리만 링크한다.** 디스크가 안 붙으면 디렉터리 링크는
   깨져서 `mkdir`·`open` 이 실패한다. 파일 링크나 안 붙은 빈 폴더는 새 빈 DB 를 조용히 만든다.

## 디스크별 역할

| 디스크 | 경로 | 담는 것 |
|---|---|---|
| C: NVMe | WSL 루트 `C:\wsl\Ubuntu\ext4.vhdx` | 코드, conda, 작은 `data/*`. 라이브 쓰기는 거의 없다 |
| C: NVMe VHDX 200GB (`C:\wsl\fast\fast.vhdx`) | `/mnt/nvme` | `data/hot`, `data/live`(DuckDB·상태·원장·래스터), `logs`, 라이브 실행 환경, 임시 파일. 3.6GB/s 쓰기 |
| D: HDD 1TB (ST1000DM010, 33,018시간) | `/mnt/d` | 파이프라인에서 뺐다(11-07 사용자 결정, 다른 용도) |
| 6TB USB (ext4, 라벨 data2) | `/mnt/data2` | 원시 호가 원본, lake, 학습용 대용량, 연구 산출물, hot 의 1차 백업. 209MB/s 쓰기 |
| dev | `~/backups/crypto-scalping-server` | 전부의 사본(다른 기계) |

## 링크 지도 (전부 디렉터리 링크라 코드 경로는 그대로다)

| 코드가 쓰는 경로 | 실제 위치 |
|---|---|
| `data/hot`, `data/live`, `logs` | `/mnt/nvme/{hot,live,logs}` |
| `~/miniconda3/envs/quant_ai` (라이브 실행 환경) | `/mnt/nvme/envs/quant_ai`. import 18.9초 → 1.8초(HDD 경합 중 실측) |
| `data/live/orderflow/raster` | `/mnt/nvme/live/orderflow/raster` (실물) |
| `data/live/orderflow/{bookticker,depthdiff,hyperliquid,hyperliquid_bookticker,okx_bookticker,raster_archive}` | `/mnt/data2/orderflow/…` |
| `data/lake`, `data/splits` | `/mnt/data2/{lake,splits}` |
| `binance_data/aggTrades`, `~/gpu_jobs`, `~/.cache/huggingface`, 안 쓰는 실행 환경, `tmp` 큰 폴더들 | `/mnt/data2/research/…` |

### WSL 본체 C: 이전 (2026-11-07 완료)

`scripts/ops/windows/migrate_wsl_to_c_20261107.ps1`. 🔴**대시보드가 손절을 감시한다(거래소 주문이 아니다, `dashboard/orders.py::bracket_tick`).** 대시보드를 끄면 그동안 손절이 안 나간다.

| 회차 | 정지 | 결과 |
|---|---|---|
| 1차 | 10:12:40 ~ 12:00 (약 1시간 48분) | 압축 1시간 45분, 658 → 212GB. 그대로 옮기면 C: 여유 34.7GB 라 복사 안 함. fstrim 은 경로 오류로 안 돌았다 |
| 2차 (사용자가 손절을 직접 보기로 함) | 14:10:50 ~ 15:06 (약 55분) | 압축 39분, 212 → 156.9GB. 복사 14분. `C:\wsl\Ubuntu\ext4.vhdx` |

ext4 실사용 96GB 인데 vhdx 가 157GB 다. 압축은 통째로 빈 블록만 돌려주므로 fstrim 뒤에도 약 60GB 차이가 남았다.
🔴**C: 여유 91GB(DISM 뒤).** `fast.vhdx`(지금 54GB)는 200GB 까지 커질 수 있어 최악이면 C: 가 찬다. `check_host_disks` 가 C: 50GiB WARN·20GiB CRITICAL 로 잡는다. 줄일 때는 큰 폴더를 6TB 로 옮기고 다시 fstrim·압축한다.
`wsl --manage --set-sparse` 는 쓰지 않는다(WSL 이 데이터 손상 가능성 때문에 막아 둔 기능이다).

**긴 정지 뒤에는 일일 cron 을 확인한다. 소급 실행되지 않는다.** 11-07 정지(10:12~12:00)로 `seal_to_lake`(11:05)·`live_macro_bias_forward`(토 10:30, 일요일이 재시도)·`gen_data_map`(11:30)이 빠져 손으로 돌렸다.

`raster` 만 NVMe 에 둔다. 대시보드가 실시간으로 읽고, 14일 보존이다.

저장 계층의 뜻(10-01 재설계): **hot** = SQLite WAL 최근 8~16일(`data/hot`) · **lake** = 끝난 날짜의 일별 Parquet(`scripts/seal_to_lake.py`,
읽기는 `scripts/data_store.py`) · **state** = 상태 JSON · **ledger** = jsonl 원장.

## 백업 (`scripts/ops/backup_live_data.sh`, 매일 04:00)

| 무엇 | 원본 | 사본 1 | 사본 2 |
|---|---|---|---|
| hot (`data/live`, `data/hot` 사본, 그 밖의 `data/`) | NVMe·루트 | 6TB `crypto-scalping-backups` | dev |
| cold (`orderflow`·`lake`·`splits`) | 6TB | dev | 없음 |

- 제외: `data/hot`(온라인 백업 API 사본을 보낸다) · `tmp` · `test` · `binance_vision`(다시 받을 수 있다) · `raster`(14일 보존) · 진행 중 시각 파일 `.bt`/`.jsonl` · `*.wal`.
- 🔴cold 의 사본은 dev 하나다(11-07 D: 사본 삭제, 사용자 결정). 원시 호가는 다시 받을 수 없으니 6TB·dev 가 함께 잘못되면 잃는다.
- `--delete` 를 안 쓴다. 사본은 늘기만 한다.
- 백업 전에 2시간 넘게 안 바뀐 `.jsonl`/`.bt` 를 압축한다. 수집기는 시각이 바뀔 때만 압축해서, 재시작하면 쓰던 시각 파일이 그대로 남고 사본에서 빠졌다. 같은 시각 `.gz` 가 이미 있으면 부분본은 `/mnt/data2/orderflow_partial/` 로 치운다.
- `data/live/backups/crontab_latest.txt` 가 함께 간다(crontab 은 저장소에 없다).
- `6TB` 의 `crypto-scalping-backups` 에는 11-07 까지 D: 에 있던 사본 전체(지금은 원본이 지운 파일 포함)가 들어 있다.
- `/mnt/data2/research` 는 백업하지 않는다(연구 산출물, 다시 만들 수 있다).

## 2026-11-07 최적화 기록

### 전후 구조

| | 전 | 후 |
|---|---|---|
| WSL 본체 | D: HDD, vhdx 658GB | C: NVMe, vhdx 157GB |
| hot DB·상태 파일·로그 | D: HDD | C: NVMe VHDX |
| 라이브 실행 환경 | D: HDD | C: NVMe VHDX |
| 원시 호가·lake·학습 데이터·연구 산출물 | D: HDD | 6TB |
| 백업 | D:(원본과 같은 디스크) + dev | hot → 6TB + dev · cold → dev |
| 수집기 | 서버 + Pi | 서버만 |
| 여유(11-07 16:12) | D: 102GB, 소진 7~8GB/일 | C: 91GiB · NVMe 148GiB · D: 740GiB · 6TB 4.6TiB |

C: 와 NVMe 는 같은 SSD 다. NVMe 여유는 `fast.vhdx` 안의 여유라 C: 여유와 더하면 안 된다.

### 처음 찾은 문제 8개

| # | 문제 | 해결 | 확인 |
|---|---|---|---|
| 1 | 용량 예측이 틀렸고 감시기가 가짜 여유(WSL 안)만 봤다 | `check_host_disks`: C:·D:·NVMe·6TB 실제 여유 + 소진 속도 | 시험 · 서버 감시 OK |
| 2 | 백업 실패를 아무도 몰랐다 · 복원 시험·crontab 사본 없음 | 결과 JSON + `check_backup` · 월 복원 시험 · crontab 사본 | 11-07 13:31 전체 성공, 복원 시험 통과 |
| 3 | 원본과 백업이 같은 디스크 | hot → 6TB + dev · cold → dev(D: 사본은 11-07 사용자 결정으로 삭제) | 같음 |
| 4 | hot 이 가장 느린 디스크에 있었다 | NVMe 로 이동 | 디스크 대기(PSI full) 약 27% → 2.5% |
| 5 | 수집기가 동기 DB 작업으로 이벤트 루프를 막아 정지 | DB 작업 `asyncio.to_thread` · 초기화 잠금(`95cfb4e0`, `1535ca97`) | 전환 뒤 확인 구간에 정지 없음. 장기 효과는 더 봐야 한다 |
| 6 | 보존 결정이 백업에서 안 지켜졌다(raster·압축 전 파일) | raster 제외 · 백업 전 시각 파일 닫기 | 시험 · 13개 278MB 처리. 옛 raster 18GB 는 사본에 남겼다 |
| 7 | 현재 구조 문서가 없었다 | 이 문서 · `data_map.md` 매일 자동 생성 | 11:30 cron |
| 8 | 안 붙은 6TB 에 백업이 쌓이는 함정 · 부팅 마운트 | `mountpoint` 가드 · `wsl_boot_recover.ps1` · cron 마운트 대기 | 시험 · WSL 재시작 2회와 Windows 재부팅 1회 복구 |

### 재점검에서 새로 찾은 문제

| 문제 | 해결 | 커밋 |
|---|---|---|
| 재시작 뒤 시장 맥락이 1분 넘게 안 뜸: 기준선 계산이 읽기 연결 4칸을 쥐었다(스택 표본 3개 전부) | 아래 «속도 규칙» | `834d1f10`, `58cb2ed7` |
| SQLite 쓰기가 커밋마다 fsync | `synchronous=NORMAL` | `a17b7a93` |
| lake 봉인이 11-05·06 에 죽어 Deribit 봉인·hot 정리가 멈췄고 경보가 없었다 | 스트림별 실패 처리 · `check_seal`. 11-07 재실행 실패 0. 원인은 쓰는 중 복사한 사본으로 추정(같은 파일의 11-07 사본은 전 열 정상) | `71116c67` |
| 방향 예측이 같은 교차 코인 klines 를 코인마다 받았다 | 5분봉당 한 번 받아 공유 | `ea6becd3` |
| 재시작이 남긴 압축 안 된 시각 파일이 백업에서 빠졌다 | `seal_stale` | `3bee81d4` |
| Windows 재부팅 때 터널이 NVMe 보다 먼저 돌아 실패·재시도 없음 | NVMe 대기. 스크립트 재현 시험 통과, 실제 재부팅 재시험은 안 했다 | `c88c0fcb`, `11cd95a3` |
| 정지 중 일일 cron 은 소급되지 않는다 | 손으로 실행 · 위 규칙 | 없음 |

### 속도 규칙

**🔴대시보드 프로세스에서 큰 결과를 행으로 받지 않는다.** `sqlite3` 는 행마다 GIL 을 놓았다가 다시 잡는다. CPU 를 쓰는 스레드가 있으면 행마다 기다린다. dev 실측, 바쁜 스레드 3개, 2,016행: 한가할 때 0.11초 → 전환 간격 5ms 191초 → 0.5ms 12초.
수만 행이 넘으면 SQL 에서 접거나(`GROUP BY`) 글자 한 덩어리로 받아 pandas C 파서로 읽는다.

| 파싱 방법 (57만 행, 바쁜 스레드 2개, dev) | 총 시간 | 이벤트 루프 지연 최대 |
|---|---|---|
| pandas C 파서 (채택) | 0.20초 | 8ms |
| `json.loads` | 0.12초 | 108ms (GIL 을 쥔다) |
| `np.fromstring` | 60초 넘음 | 숫자마다 GIL 을 다시 잡는다 |

### 전후 실측

| 바꾼 것 | 전 | 후 | 조건 |
|---|---|---|---|
| 재시작 뒤 시장 맥락 표시 | 10:10 +104초 넘음 → 12:20 +73.6초 → 12:55 +25.1초 | 13:05 +6.9초 | 서버 재시작 1회씩. 10:10·12:20 은 다른 작업 영향 가능. 13:05 는 페이지 캐시가 찬 상태 |
| 풋프린트·청산지도 표시 | 12:20 +18초·+23.8초 | 13:05 +2.5초·+6.8초 | 같음 |
| 대시보드 CPU(재시작 직후 약 100초) | 108% | 10% | 표본 창의 시작 시점이 조금 다름 |
| 파이썬 import | 18.9초 | 1.8초 | 전 값은 복사 경합 중 |
| 디스크 대기(PSI full) | 약 27% | 2.5% | 둘 다 복사 없음 |
| SQLite 커밋 p50·p99 | 3.42·6.0ms | 0.09·0.17ms | NVMe 빈 표, 초당 약 45회 |
| NVMe flush | 22.4회/초 | 5.8회/초 | 서버 60초 표본, 후 표본은 복사 중 |
| 이벤트 루프 지연 p99 (`setswitchinterval` 0.5ms) | 227ms | 26ms | dev, 바쁜 스레드 3개. CPU 처리량 −2% |
| 방향 예측 REST (4코인·두 판, 5분마다) | 약 216회 | 약 56회 | 측정한 라운드당 27회로 계산. 배포 뒤 실측 안 함 |
| lake 봉인 | 11-04 43분 (161개 파일) | 11-07 12분 (190개 파일) | 쓰는 파일 수 다름 |
| hot 백업 | 07:49 49분(HDD, 복사 경합) | 13:22 1분 44초 | 다른 복사 없음 |

`synchronous=NORMAL` 은 전원이 끊기면 마지막 커밋을 잃을 수 있다(DB 는 안 깨진다). 원장·주문 기록은 SQLite 가 아니다.
옛 방식과 값 대조: 바이낸스 OI 5분 끝값 1,994칸 일치 · 기준선 초 합계 ETH 485,843초·SOXL 308,666초 상대 차 5e-15 미만.

### 재봐서 안 바꾼 것

- `mmap_size`·`cache_size`: 테이프 8일 조회 0.8초 그대로.
- 청산지도: 2~4초는 재시작 뒤 첫 계산뿐이다. 평소 ETH 0.3초 미만(그 이상만 기록되는 로그로 확인).
- 대시보드 정상 상태: CPU 10% · 이벤트 루프 표본 25번 중 24번 대기(재시작 약 20초 뒤부터 100초 표본 1회).
- 6TB read-ahead(이미 8MB) · Windows 전원 구성(이미 고성능) · I/O 스케줄러(`none`, 조정 불가).
- 루트 디스크: 라이브가 쓰는 파일 없음(3분 표본).

### 빨라지지 않은 것·남은 한계

- 가격 갱신 주기(1초 설정), 주문 체결, 거래소까지의 네트워크 지연은 건드리지도 재지도 않았다.
- WSL 본체 C: 이전의 속도 효과는 재지 않았다. hot 은 이미 NVMe 였으므로 작을 것으로 본다. 주된 이득은 낡은 HDD 에서 OS 를 뺀 것이다.
- 🔴C: 여유 91GB 가 빠듯하다(위 «WSL 본체 C: 이전»).
- 6TB 모델·SMR 여부를 모른다(도킹이 SMART 를 막는다).

### 이전 뒤 드러난 문제 (11-07 16시대)

| 문제 | 원인 | 조치 | 확인 |
|---|---|---|---|
| 서버 시계가 33초마다 1.4초씩 벌어졌다 | 재부팅 뒤 Windows 시계가 0.81초 틀리고 동기화도 안 됐다. WSL 은 호스트를 따라가고 timesyncd 는 NTP 를 따라가 서로 싸웠다(`tick` 9606~10437 은 WSL 이 움직이는 값이라 고쳐도 되돌아온다. 시계 소스 전환도 효과 없음) | Windows 시계 1회 보정 · 부팅 때 자동 보정(`Sync-WindowsClock`) · 감시 `check_clock` | 보정 뒤 5분 점프 0회, 거래소 차이 최대 190ms, timesyncd 차이 8ms. 부팅 자동 보정은 실제 재부팅 시험 전 |
| 진입 주문이 `-2019` 로 거부되고 SL/TP 도 안 걸렸다 | 수량을 가용 잔고가 아니라 순자산으로 쟀다. 이번 이전과 무관한 기존 동작 | `build_entry_plan` 이 가용 잔고의 98% 안으로 자른다 | 재현 시험 · 미리보기 100%: ETHUSDC 4.339개(증거금 973, 가용 993) · 실제 주문은 안 냈다 |

**11-07 약 15:20~16:50 수집 시각은 최대 약 1.5초 어긋날 수 있다.** 재부팅 전 구간에도 있었는지는 재지 못했다. 이 구간을 초 단위로 쓰는 연구는 주의한다.

### 대가

대시보드·손절 감시 정지: 10:12~12:00 · 14:10~15:06 · 15:18~15:23. 수정 배포마다 재시작.
작업 중 사고: 일괄 종료 조건이 감독 50개를 죽임(대시보드 2개 약 3분) · 동시 초기화 잠금으로 OKX 수집 약 16분 공백 · fstrim 경로 오류로 1차 이전 실패 · 터널 부팅 순서.

## 감시

| 무엇 | 어디서 | 문턱 |
|---|---|---|
| C:·D: 실제 여유 | `check_host_disks` | WARN 50GiB · CRITICAL 20GiB |
| NVMe 마운트·여유 | `check_host_disks` | 안 붙음 CRITICAL · WARN 30GiB · CRITICAL 10GiB |
| 6TB 마운트·쓰기·여유 | `check_host_disks` · 데드맨 | 안 붙음·쓰기 실패·30초 멈춤 = CRITICAL · WARN 200GiB · CRITICAL 50GiB |
| 백업 결과·나이 | `check_backup` | 26h WARN · 50h CRITICAL · 실패 CRITICAL |
| 복원 시험 | `scripts/ops/backup_restore_check_20261107.py`, 월 1회 | 실패 CRITICAL · 40일 WARN |
| lake 봉인 결과 | `check_seal` (`seal_status.json`) | 26h WARN · 50h CRITICAL · 실패 스트림 있으면 WARN |
| 서버 시계 | `check_clock` (timesyncd 와 NTP 차이) | 0.5초 WARN · 2초 CRITICAL |
| 소진 속도 | `check_host_disks` 가 `morning_audit` 일별 기록을 읽는다 | 7일 평균과 어제 대비 중 느린 쪽으로 남은 일수 14일 아래 WARN |

## 디스크가 안 붙으면

- 부팅: `CryptoWSLBootRecover`(`scripts/ops/windows/wsl_boot_recover.ps1`)가 60초마다 확인하고 `wsl --mount` + `mount` 를 한다. 6TB 는 번호가 아니라 «USB·5TB 초과»로 고른다. SYSTEM 계정으로 돌리면 안 된다(`wsl --mount` 는 호출 사용자의 VM 에 붙는다).
- cron 은 `/mnt/nvme` 가 붙을 때까지 최대 5분 기다린다(`/etc/systemd/system/cron.service.d/wait-data-disks.conf`). 안 기다리면 `@reboot` 감독들이 깨진 링크에서 죽고 다시 안 뜬다.
- 터널(`CryptoScalpingPublicDashboard`)도 부팅 때 바로 도는데 `data/live` 가 NVMe 링크라, `start_public_dashboard.sh` 가 `/mnt/nvme` 를 5분까지 기다린다(11-07 재부팅 때 먼저 돌아 터널이 안 떴다, 고친 뒤 실제 재부팅 시험 전).
- 재부팅 뒤 로그인 전에는 WSL 안에서 `powershell.exe` 가 `Invalid argument` 로 실패한다. `WSL_INTEROP=/run/WSL/<keepalive>_interop` 를 주면 된다. crontab 은 Windows 실행 파일을 안 부른다.
- 손으로 복구: 관리자 PowerShell 에서 `powershell -ExecutionPolicy Bypass -File C:\Users\kbj20\crypto-ops\wsl_boot_recover.ps1 -MountOnly`
- 6TB 가 빠지면: 원시 수집기는 링크가 깨져 재시작을 반복한다(**그 시간의 호가는 비고, 소급 불가**). 봉인·백업의 6TB 단계도 건너뛰고 `check_backup` 이 경보한다. 대시보드의 실시간 풋프린트(hot)는 계속된다. ob5m·iceberg 처럼 호가 diff 를 읽는 워커는 함께 멈춘다.

## 수집기 위치

대시보드가 쓰는 수집기는 전부 서버에 있다. 목록은 `data_map.md` 의 매니페스트 표와 서버 crontab(`data/live/backups/crontab_latest.txt`).
**Pi 는 파이프라인에서 빠졌다.** 수집기·crontab·감시를 2026-11-07 16:01~16:07 에 모두 없앴다. 서버 코드가 Pi 에 의존하지 않는다(`.from_pi.duckdb` 는 서버에 있는 얼은 파일이고 봉인이 계속 읽는다).
**🔴이제 서버가 멈추면 그 시간의 호가는 빈다.** 11-07 정지 때는 Pi 가 ETH·BTC·SOL·XRP 호가 diff 와 ETH·BTC HL 체결을 메웠다. SNDK·SOXL 호가는 10:12~12:01, 14:10~15:06, 15:18~15:21 이 비었다.
BTC HL 포지션은 Pi 끝(11-07 06:08)과 서버 시작(06:22) 사이 약 14분이 빈다.

`/mnt/data2/research/pi_archive_20261107/` (1.4GB, 백업 대상 아님): Pi 의 DuckDB(`live/`) · 그 밖의 상태 파일(`live_other/`) · 닫히지 않은 원시 파일(`orderflow_leftovers/`) · 정지 구간이 든 11-07 HL 체결(`hyperliquid/`, 서버 파일과 이름이 같아 합치지 않았다) · 로그 · 저장소 미커밋 변경 diff · 전송 스크립트. `.gz` 는 전부 서버로 보낸 뒤 Pi 에서 지웠다. Pi 의 `data/ensemble` 은 서버와 같다(크기 대조 차이 0).

Pi 를 치울 때: 서버 `~/.ssh/authorized_keys` 의 Pi 키 제거 · `handoff.hosts.conf` 의 `collector` 줄 제거 · Pi 디스크 초기화(실키 파일이 남아 있을 수 있다) · `~/plantpoint/plantpoint-automation` 은 이 저장소와 무관한 프로젝트다.

Pi 에서 옮긴 기준(사용자 «대시보드에 사용하지 않는 것들은 제거»):

| Pi 수집기 | 처리 | 이유 |
|---|---|---|
| 호가 diff BTC | 서버로 | `live_ob5m --symbol BTCUSDT` 가 대시보드 BTC 호가 칸의 원천인데 서버에 실시간 파일이 없어 비어 있었다 |
| HL 체결·포지션 BTC | 서버로 | 대시보드 BTC HL 칸(`HL_LIQ_BY_ASSET`). 포지션이 체결 파일에서 주소를 고른다 |
| HL 맥락 BTC | 서버 hot(`HL_BT_CTX_ONLY=1`) | 시장 맥락 카드. 원시 호가 `.bt` 는 안 받는다 |
| 호가 diff·HL 체결 ETH·SOL·XRP | Pi 쪽만 끔 | 서버가 이미 받고 있었다(중복) |
| bookTicker 5종 · OKX 호가 2종 · HYPE 전부 · HL 원시 호가 BTC·HYPE | 제거 | 대시보드와 라이브 워커 어디도 읽지 않는다 |

쌓인 파일은 지우지 않았다(`/mnt/data2/orderflow`). 호가는 소급이 안 되므로, 연구에 다시 필요하면 그때부터 다시 쌓인다.

새 수집기 규칙: 원시 시각 파일은 `data/live/orderflow/<스트림>/<코인>/` 아래에 쓴다(이미 6TB 로 간다). 새 스트림 폴더라면 `move_to_data2_20261107.sh` 로 링크를 만든다.
