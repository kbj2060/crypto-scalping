# 데이터 저장 지도 (2026-11-07)

현재 저장 구조의 기준 문서다. **지금 값**(크기·수집기·위치)은 손으로 적지 않는다:
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
| D: HDD 1TB (ST1000DM010, 33,018시간) | `/mnt/d` | `crypto-scalping-cold`: 6TB 원본의 사본만 |
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
🔴**이전 뒤 C: 여유 88GB.** `fast.vhdx`(지금 54GB)는 200GB 까지 커질 수 있어 최악이면 C: 가 찬다. `check_host_disks` 가 C: 50GiB WARN·20GiB CRITICAL 로 잡는다. 줄일 때는 큰 폴더를 6TB 로 옮기고 다시 fstrim·압축한다.
`wsl --manage --set-sparse` 는 쓰지 않는다(WSL 이 데이터 손상 가능성 때문에 막아 둔 기능이다).

**긴 정지 뒤에는 일일 cron 을 확인한다. 소급 실행되지 않는다.** 11-07 정지(10:12~12:00)로 `seal_to_lake`(11:05)·`live_macro_bias_forward`(토 10:30, 일요일이 재시도)·`gen_data_map`(11:30)이 빠져 손으로 돌렸다.

`raster` 만 NVMe 에 둔다. 대시보드가 실시간으로 읽고, 14일 보존이다.

저장 계층의 뜻(10-01 재설계): **hot** = SQLite WAL 최근 8~16일(`data/hot`) · **lake** = 끝난 날짜의 일별 Parquet(`scripts/seal_to_lake.py`,
읽기는 `scripts/data_store.py`) · **state** = 상태 JSON · **ledger** = jsonl 원장.

## 백업 (`scripts/ops/backup_live_data.sh`, 매일 04:00)

| 무엇 | 원본 | 사본 1 | 사본 2 |
|---|---|---|---|
| hot (`data/live`, `data/hot` 사본, 그 밖의 `data/`) | NVMe·루트 | 6TB `crypto-scalping-backups` | dev |
| cold (`orderflow`·`lake`·`splits`) | 6TB | D: `crypto-scalping-cold` | dev |

- 제외: `data/hot`(온라인 백업 API 사본을 보낸다) · `tmp` · `test` · `binance_vision`(다시 받을 수 있다) · `raster`(14일 보존) · 진행 중 시각 파일 `.bt`/`.jsonl` · `*.wal`.
- `--delete` 를 안 쓴다. 사본은 늘기만 한다.
- 백업 전에 2시간 넘게 안 바뀐 `.jsonl`/`.bt` 를 압축한다. 수집기는 시각이 바뀔 때만 압축해서, 재시작하면 쓰던 시각 파일이 그대로 남고 사본에서 빠졌다. 같은 시각 `.gz` 가 이미 있으면 부분본은 `/mnt/data2/orderflow_partial/` 로 치운다.
- `data/live/backups/crontab_latest.txt` 가 함께 간다(crontab 은 저장소에 없다).
- `6TB` 의 `crypto-scalping-backups` 에는 11-07 까지 D: 에 있던 사본 전체(지금은 원본이 지운 파일 포함)가 들어 있다.
- `/mnt/data2/research` 는 백업하지 않는다(연구 산출물, 다시 만들 수 있다).

## 속도 규칙과 실측 (2026-11-07)

**🔴대시보드 프로세스에서 큰 결과를 행으로 받지 않는다.** `sqlite3` 는 행마다 GIL 을 놓았다가 다시 잡는다. CPU 를 쓰는 스레드가 하나라도 있으면 행당 전환 간격만큼 기다린다. dev 실측, 바쁜 스레드 3개, 2,016행: 한가할 때 0.11초 → 5ms 간격 191초 → 0.5ms 간격 12초.
수만 행이 넘으면 SQL 에서 접거나(`GROUP BY`) 글자 한 덩어리로 받아 pandas C 파서로 읽는다. `np.fromstring` 은 숫자마다 GIL 을 다시 잡아 같은 병이 되고, `json.loads` 는 GIL 을 쥔 채 파싱해 이벤트 루프를 100ms 쯤 세운다.

| 바꾼 것 | 전 | 후 | 조건 |
|---|---|---|---|
| 재시작 뒤 시장 맥락 표시 | 10:10 +104초 넘음 → 12:20 +73.6초 → 12:55 +25.1초 | 13:05 +6.9초 | 전부 서버 실측 재시작 1회씩. 10:10·12:20 은 다른 작업 영향 가능. 13:05 는 페이지 캐시가 찬 상태 |
| 풋프린트·청산지도 표시 | 12:20 +18초·+23.8초 | 13:05 +2.5초·+6.8초 | 같음 |
| 대시보드 CPU(재시작 직후 약 100초) | 108% | 10% | 표본 창이 재시작 후 시점 조금 다름 |
| SQLite 커밋 p50·p99 (`synchronous=NORMAL`) | 3.42·6.0ms | 0.09·0.17ms | NVMe 빈 표에 초당 약 45회 |
| NVMe flush | 22.4회/초 | 5.8회/초 | 서버 60초 표본, 후 표본은 복사 중 |
| 이벤트 루프 지연 p99 (`setswitchinterval` 0.5ms) | 227ms | 26ms | dev, 바쁜 스레드 3개. CPU 처리량 −2% |

| 방향 예측 REST (4코인·두 판, 5분마다) | 약 216회 | 약 56회 | 라운드 측정 27회(이 중 20회가 교차 코인 klines) × 8 → 공유 후 계산값. 배포 뒤 실측 안 함 |
| lake 봉인 | 11-04 43분 (161개 파일, HDD) | 11-07 12분 (190개 파일) | 쓰는 파일 수 다름 |
| hot 백업(온라인 사본 + 6TB) | 11-07 07:49 49분(HDD, 복사 경합) | 13:22 1분 44초 | 다른 복사 없음 |

`synchronous=NORMAL` 은 전원이 끊기면 마지막 커밋을 잃을 수 있다(DB 는 안 깨진다). 원장·주문 기록은 SQLite 가 아니다.

재봐서 **안 바꾼 것**: `mmap_size`·`cache_size`(테이프 8일 조회 0.8초 그대로) · 6TB read-ahead(이미 8MB) · Windows 전원 구성(이미 고성능) · I/O 스케줄러(`none`, 조정 불가).

## 감시

| 무엇 | 어디서 | 문턱 |
|---|---|---|
| C:·D: 실제 여유 | `check_host_disks` | WARN 50GiB · CRITICAL 20GiB |
| NVMe 마운트·여유 | `check_host_disks` | 안 붙음 CRITICAL · WARN 30GiB · CRITICAL 10GiB |
| 6TB 마운트·쓰기·여유 | `check_host_disks` · 데드맨 | 안 붙음·쓰기 실패·30초 멈춤 = CRITICAL · WARN 200GiB · CRITICAL 50GiB |
| 백업 결과·나이 | `check_backup` | 26h WARN · 50h CRITICAL · 실패 CRITICAL |
| 복원 시험 | `scripts/ops/backup_restore_check_20261107.py`, 월 1회 | 실패 CRITICAL · 40일 WARN |
| lake 봉인 결과 | `check_seal` (`seal_status.json`) | 26h WARN · 50h CRITICAL · 실패 스트림 있으면 WARN |
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
Pi 수집기는 2026-11-07 16:01 에 전부 껐다(`multicoin_collectors.json` 비움, crontab 비움, Pi 감시 끔). 마지막까지 돈 호가 diff 4개·HL 체결 2개는 WSL 이전과 재부팅 공백을 메웠고, `.gz` 는 전부 서버로 보낸 뒤 Pi 에서 지웠다.
**이제 서버가 멈추면 그 시간의 호가는 빈다.**
Pi 의 얼은 DB 와 정지 구간이 든 11-07 HL 체결 파일은 `/mnt/data2/research/pi_archive_20261107/` 에 보관했다. 서버 쪽 HL 일별 파일과 이름이 같아 일부러 합치지 않았다.
SNDK·SOXL 원시 호가는 Pi 에 없어 10:12~12:01, 14:10~15:06, 15:18~15:21 이 비어 있다.

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
