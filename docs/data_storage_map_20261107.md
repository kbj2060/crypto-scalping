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
| C: NVMe 475GB | WSL 본체 `ext4.vhdx`(이전 후) | 코드, conda, hot SQLite, `data/live` 의 DuckDB·상태 파일, 모델 |
| 6TB USB (ext4, 라벨 data2) | `/mnt/data2` | 원시 호가 원본, lake, 학습용 대용량, 연구 산출물, hot 의 1차 백업 |
| D: HDD 1TB (ST1000DM010) | `/mnt/d` | cold 의 사본. 원본을 두지 않는다 |
| dev | `~/backups/crypto-scalping-server` | 전부의 사본(다른 기계) |

## 6TB 로 옮긴 경로 (전부 디렉터리 링크라 코드 경로는 그대로다)

| 코드가 쓰는 경로 | 실제 위치 |
|---|---|
| `data/live/orderflow/{bookticker,depthdiff,hyperliquid,hyperliquid_bookticker,okx_bookticker,raster_archive}` | `/mnt/data2/orderflow/…` |
| `data/lake` | `/mnt/data2/lake` |
| `data/splits` | `/mnt/data2/splits` |
| `tmp/causal_regen_20260516`, `~/gpu_jobs`, `~/.cache/huggingface` | `/mnt/data2/research/…` |

`data/live/orderflow/raster` 는 로컬에 둔다. 대시보드가 실시간으로 읽고, 14일 보존이다.

저장 계층의 뜻(10-01 재설계): **hot** = SQLite WAL 최근 8~16일(`data/hot`) · **lake** = 끝난 날짜의 일별 Parquet(`scripts/seal_to_lake.py`,
읽기는 `scripts/data_store.py`) · **state** = 상태 JSON · **ledger** = jsonl 원장.

## 백업 (`scripts/ops/backup_live_data.sh`, 매일 04:00)

| 무엇 | 원본 | 사본 1 | 사본 2 |
|---|---|---|---|
| hot (`data/` 전체, 제외 목록 뺌) | 루트 디스크 | 6TB `crypto-scalping-backups` | dev |
| cold (`orderflow`·`lake`·`splits`) | 6TB | D: `crypto-scalping-cold` (vhdx 가 D: 를 떠난 뒤부터) | dev |

- 제외: `data/hot`(온라인 백업 API 사본을 보낸다) · `tmp` · `test` · `binance_vision`(다시 받을 수 있다) · `raster`(14일 보존) · 진행 중 시각 파일 `.bt`/`.jsonl` · `*.wal`.
- `--delete` 를 안 쓴다. 사본은 늘기만 한다.
- `data/live/backups/crontab_latest.txt` 가 함께 간다(crontab 은 저장소에 없다).
- `6TB` 의 `crypto-scalping-backups` 에는 11-07 까지 D: 에 있던 사본 전체(지금은 원본이 지운 파일 포함)가 들어 있다.
- `/mnt/data2/research` 는 백업하지 않는다(연구 산출물, 다시 만들 수 있다).

## 감시

| 무엇 | 어디서 | 문턱 |
|---|---|---|
| C:·D: 실제 여유 | `check_host_disks` | WARN 50GiB · CRITICAL 20GiB |
| 6TB 마운트·쓰기·여유 | `check_host_disks` | 안 붙음·쓰기 실패·10초 멈춤 = CRITICAL · WARN 200GiB · CRITICAL 50GiB |
| 백업 결과·나이 | `check_backup` | 26h WARN · 50h CRITICAL · 실패 CRITICAL |
| 복원 시험 | `scripts/ops/backup_restore_check_20261107.py`, 월 1회 | 실패 CRITICAL · 40일 WARN |
| 디스크 추세 | `morning_audit` 일별 기록 | 사람이 본다 |

## 6TB 가 안 붙으면

- 부팅: `CryptoWSLBootRecover`(`scripts/ops/windows/wsl_boot_recover.ps1`)가 60초마다 확인하고 `wsl --mount` + `mount /mnt/data2` 를 한다.
  디스크는 번호가 아니라 «USB·5TB 초과»로 고른다. SYSTEM 계정으로 돌리면 안 된다(`wsl --mount` 는 호출 사용자의 VM 에 붙는다).
- 손으로 복구: 관리자 PowerShell 에서 `powershell -ExecutionPolicy Bypass -File C:\Users\kbj20\crypto-ops\wsl_boot_recover.ps1 -MountOnly`
- 그동안: 원시 수집기는 링크가 깨져 재시작을 반복한다(**그 시간의 호가는 비고, 소급 불가**). 봉인·백업의 6TB 단계도 건너뛰고 `check_backup` 이 경보한다.
  대시보드의 실시간 풋프린트(hot)는 계속된다. ob5m·iceberg 처럼 depthdiff 를 읽는 워커는 함께 멈춘다.

## 수집기 위치

2026-11-07 부터 전 수집기가 서버에 있다. 목록은 `data_map.md` 의 매니페스트 표와 서버 crontab(`data/live/backups/crontab_latest.txt`).
Pi 를 쓰지 않으므로 **서버가 멈추면 그 시간의 호가는 빈다**.

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
