"""
통합 데이터 파이프라인 (update_features.py)

기간을 지정하면:
  1. training_features_5m.csv 에서 빠진 구간 확인
  2. binance_data/ 로컬 파일 확인
  3. 없으면 Binance에서 다운로드
  4. 피처 생성 → 기존 CSV와 병합 저장

사용법:
  python scripts/update_features.py --start 2024-01-01 --end 2024-12-31
  python scripts/update_features.py          # 기본: 2024-01-01 ~ 오늘

[중요 -- 2026-07-13 재현성 버그 및 수정 이력]
--start 기본값이 2024-01-01이라, check_gaps()의 gap_start = min(start, ts_min)은 거의 항상
2024-01-01로 잡힌다. 즉 이 스크립트를 실행할 때마다 "새로 추가된 구간"만 계산하는 게 아니라
2024년부터 전체 raw klines/funding/metrics를 다시 로드해서 GARCH/OU 재귀 피처
(garch_vol/garch_vol_z/ou_halflife 등, features/elite.py VolatilityModelEngine)를 처음부터
다시 계산한다. 개별 피처 공식 자체는 과거만 보는 causal rolling window라 문제 없지만, raw
데이터가 재수집 때마다 완전히 동일하지 않으면 재계산 결과가 미세하게 달라질 수 있다.
과거에는 merge_and_save()가 `concat([new_df, existing])` 순서로 합친 뒤
drop_duplicates(keep='first' 기본값)를 적용해서, 방금 재계산한 값이 기존 캐시값을
"조용히" 덮어썼다 -- 그래서 같은 과거 구간을 다시 계산할 때마다 옛날 백테스트 숫자가
재현되지 않는 사고가 있었다(project memory: project-portfolio-3asset-design.md 참고).
2026-07-13에 concat 순서를 `[existing, new_df]`로 바꿔 기존 캐시값이 항상 우선하도록
고쳤다 -- 앞으로는 이 스크립트를 몇 번을 다시 돌려도 이미 저장된 과거 피처값은 절대
바뀌지 않는다. 피처 계산 로직 자체의 버그를 고쳐서 과거 값을 의도적으로 전부
재생성하고 싶은 경우에만 `data/splits/year_oos/training_features_2026_rebuilt.csv`를
먼저 삭제하고 실행할 것 (그래야 진짜 전체 재빌드가 됨을 명확히 인지한 상태로 하게 됨).

[중요 -- 2026-07-30 원본 metrics zip 무결성 고정]
07-13 fix는 feature 병합 로직만 다뤘다. 그런데 실제로는 그보다 위쪽, 원본
`binance_data/metrics/ETHUSDT-metrics-*.zip` 자체가 조용히 바뀌는 사고가 있었다: 2026-07-02에
Jan-Jun 2026 구간 zip의 78개(약 43%)가 새로 채워졌고(03-11/03-16/04-14에도 각각 반복),
그 결과 `sum_open_interest_value`/`sum_toptrader_long_short_ratio`/`count_long_short_ratio`/
`whale_retail_ratio` 값이 이전에 이 파일들로 만든 feature 대비 89%의 행에서 달라졌다
(자세한 내용: `docs/pipeline_integrity_and_research_redesign_20260730.md`,
project memory `project-omega461-baseline-drift-bisection-20260730`). `_download_file`은 파일이
이미 존재하면 절대 재다운로드하지 않으므로 코드 자체가 덮어쓰지는 않지만, 파일이 없던 날짜가
나중에 채워지거나 로컬 파일이 삭제된 뒤 다시 받아지면 이전에 그 파일로 만들어진 feature 값과
새로 받은 파일로 만든 feature 값이 소리 없이 달라진다.
이를 막기 위해 `ensure_metrics()`가 각 zip 파일의 sha256을
`binance_data/RAW_SOURCE_MANIFEST.json`에 기록한다: 처음 보는 파일은 등록하고, 이미 등록된
파일의 내용이 바뀌면(해시 불일치) 즉시 실패한다(AGENTS.md의 fail-fast 계약과 동일 방향) --
조용한 보정이나 자동 허용은 하지 않는다.
"""

import hashlib
import json
import os
import sys
import io
import zipfile
import argparse
from datetime import datetime, timedelta

import pandas as pd
import requests

# ── 경로 설정 ─────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from features.engineering import FeatureEngineer
from features.schema import prune_to_active_feature_keep

DATA_DIR         = os.path.join(_ROOT_DIR, 'data')
BINANCE_DIR      = os.path.join(_ROOT_DIR, 'binance_data')
METRICS_DIR      = os.path.join(BINANCE_DIR, 'metrics')
FUNDING_RATE_DIR = os.path.join(BINANCE_DIR, 'funding_rate')
ETH_KLINES_DIR   = os.path.join(BINANCE_DIR, 'klines', 'ETHUSDT')
BTC_KLINES_DIR   = os.path.join(BINANCE_DIR, 'klines', 'BTCUSDT')
RAW_SOURCE_MANIFEST = os.path.join(BINANCE_DIR, 'RAW_SOURCE_MANIFEST.json')

ETH_CSV      = os.path.join(DATA_DIR, 'eth_5m_1year.csv')
BTC_CSV      = os.path.join(DATA_DIR, 'btc_5m_1year.csv')
METRICS_CSV  = os.path.join(DATA_DIR, 'TOTAL_ETHUSDT_metrics.csv')
# Funding is symbol-scoped and must come from ETHUSDT monthly funding zips.
# Do not mix legacy single-file funding CSVs here; an ETHFIUSDT CSV previously
# shared this path and contaminates ETHUSDT features.
FUNDING_CSV  = None
FEATURES_CSV = os.path.join(DATA_DIR, 'training_features_5m.csv')

KLINES_COLS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades',
    'taker_buy_base', 'taker_buy_quote', 'ignore',
]


# ══════════════════════════════════════════════════════════════════
# 1. 인자 파싱
# ══════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description='training_features_5m.csv 업데이트')
    parser.add_argument('--start', default='2024-01-01',
                        help='시작일 (YYYY-MM-DD), 기본값: 2024-01-01')
    parser.add_argument('--end',   default=datetime.today().strftime('%Y-%m-%d'),
                        help='종료일 (YYYY-MM-DD), 기본값: 오늘')
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════
# 2. training_features_5m.csv gap 분석
# ══════════════════════════════════════════════════════════════════
def check_gaps(start: datetime, end: datetime) -> tuple[datetime, datetime] | None:
    """
    요청 기간 [start, end] 중 training_features_5m.csv에 없는 구간을 반환.
    전체가 있으면 None 반환.
    """
    if not os.path.exists(FEATURES_CSV):
        print(f"  ℹ️  training_features_5m.csv 없음 → 전체 생성")
        return (start, end)

    df = pd.read_csv(FEATURES_CSV, usecols=['timestamp'], parse_dates=['timestamp'])
    ts_min = df['timestamp'].min().to_pydatetime().replace(tzinfo=None)
    ts_max = df['timestamp'].max().to_pydatetime().replace(tzinfo=None)

    print(f"  기존 training_features_5m: {ts_min.date()} ~ {ts_max.date()}")

    # 요청 기간이 기존 범위에 완전히 포함되면 종료
    if ts_min <= start and end <= ts_max:
        print(f"  ✅ 요청 기간 [{start.date()} ~ {end.date()}]이 이미 포함됨 → 건너뜀")
        return None

    # 아직 없는 구간만 계산
    gap_start = min(start, ts_min)
    gap_end   = max(end,   ts_max)

    # 실제로 빠진 부분만 (기존 범위 밖)
    need_before = start < ts_min   # 앞쪽 gap
    need_after  = end   > ts_max   # 뒤쪽 gap

    if need_before and not need_after:
        return (start, ts_min - timedelta(minutes=5))
    if need_after and not need_before:
        return (ts_max + timedelta(minutes=5), end)

    # 양쪽 모두 필요하면 전체 재생성 범위
    return (gap_start, gap_end)


# ══════════════════════════════════════════════════════════════════
# 3. 다운로드 유틸
# ══════════════════════════════════════════════════════════════════
def _download_file(url: str, save_path: str) -> bool:
    """단일 파일 다운로드. 이미 있으면 스킵. 성공 여부 반환."""
    if os.path.exists(save_path):
        return True
    try:
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  ✅ {os.path.basename(save_path)}")
            return True
        else:
            print(f"  ❌ HTTP {resp.status_code}: {url}")
            return False
    except Exception as e:
        print(f"  ❌ 다운로드 실패 ({url}): {e}")
        return False


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_raw_source_manifest() -> dict:
    if os.path.exists(RAW_SOURCE_MANIFEST):
        with open(RAW_SOURCE_MANIFEST, 'r') as f:
            return json.load(f)
    return {"schema_version": "raw_source_manifest_v1", "files": {}}


def _save_raw_source_manifest(manifest: dict) -> None:
    with open(RAW_SOURCE_MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _verify_and_register_raw_source(manifest: dict, rel_path: str, abs_path: str) -> None:
    """rel_path의 현재 내용을 해싱해 manifest와 대조한다.

    - manifest에 없으면 지금 상태를 기준선으로 등록한다.
    - manifest에 있고 해시가 같으면 그대로 둔다.
    - manifest에 있는데 해시가 다르면 원본 소스가 조용히 바뀐 것이므로 즉시 실패한다
      (fail-fast -- AGENTS.md 계약, 2026-07-30 발견된 metrics zip 재수집 drift 참고).
    """
    entry = manifest["files"].get(rel_path)
    digest = _sha256_file(abs_path)
    if entry is None:
        manifest["files"][rel_path] = {
            "sha256": digest,
            "size_bytes": os.path.getsize(abs_path),
            "first_seen": datetime.utcnow().isoformat() + "Z",
        }
        return
    if entry["sha256"] != digest:
        raise RuntimeError(
            f"Raw source drift detected: {rel_path} content changed since it was first "
            f"registered ({entry['first_seen']}, sha256={entry['sha256']}) -- now sha256={digest}. "
            f"Refusing to silently use a changed raw source file. See "
            f"docs/pipeline_integrity_and_research_redesign_20260730.md."
        )


def ensure_metrics(gap_start: datetime, gap_end: datetime):
    """gap 구간의 metrics 일별 ZIP 확인 및 다운로드. 모든 파일은 sha256으로 고정된다."""
    os.makedirs(METRICS_DIR, exist_ok=True)
    print(f"\n  [Metrics] {gap_start.date()} ~ {gap_end.date()} 확인 중...")
    manifest = _load_raw_source_manifest()
    curr = gap_start.replace(hour=0, minute=0, second=0)
    count_dl = 0
    while curr <= gap_end:
        date_str  = curr.strftime('%Y-%m-%d')
        fname     = f'ETHUSDT-metrics-{date_str}.zip'
        save_path = os.path.join(METRICS_DIR, fname)
        if not os.path.exists(save_path):
            url = (f'https://data.binance.vision/data/futures/um/daily'
                   f'/metrics/ETHUSDT/{fname}')
            if _download_file(url, save_path):
                count_dl += 1
        if os.path.exists(save_path):
            rel_path = os.path.relpath(save_path, BINANCE_DIR)
            _verify_and_register_raw_source(manifest, rel_path, save_path)
        curr += timedelta(days=1)
    _save_raw_source_manifest(manifest)
    print(f"  → metrics {count_dl}개 신규 다운로드")


def ensure_funding(gap_start: datetime, gap_end: datetime):
    """gap 구간의 funding rate 월별 ZIP 확인 및 다운로드."""
    os.makedirs(FUNDING_RATE_DIR, exist_ok=True)
    print(f"\n  [Funding] {gap_start.date()} ~ {gap_end.date()} 확인 중...")
    curr = datetime(gap_start.year, gap_start.month, 1)
    end_m = datetime(gap_end.year, gap_end.month, 1)
    count_dl = 0
    while curr <= end_m:
        month_str = curr.strftime('%Y-%m')
        fname     = f'ETHUSDT-fundingRate-{month_str}.zip'
        save_path = os.path.join(FUNDING_RATE_DIR, fname)
        if not os.path.exists(save_path):
            url = (f'https://data.binance.vision/data/futures/um/monthly'
                   f'/fundingRate/ETHUSDT/{fname}')
            if _download_file(url, save_path):
                count_dl += 1
        next_month = curr.month % 12 + 1
        next_year  = curr.year + (curr.month // 12)
        curr = datetime(next_year, next_month, 1)
    print(f"  → funding {count_dl}개 신규 다운로드")


def ensure_klines(gap_start: datetime, gap_end: datetime):
    """gap 구간 klines를 Binance API로 다운로드."""
    for symbol, klines_dir, csv_path in [
        ('ETHUSDT', ETH_KLINES_DIR, ETH_CSV),
        ('BTCUSDT', BTC_KLINES_DIR, BTC_CSV),
    ]:
        os.makedirs(klines_dir, exist_ok=True)
        api_csv = os.path.join(klines_dir, f'{symbol}-5m-api.csv')
        print(f"\n  [{symbol} klines] API 다운로드 중 ({gap_start.date()} ~ {gap_end.date()})...")

        # 기존 api.csv가 있으면 가장 최신 날짜 이후부터만 받음 (이어받기)
        target_start = gap_start
        if os.path.exists(api_csv):
            tmp = pd.read_csv(api_csv, usecols=['timestamp'], parse_dates=['timestamp'])
            if not tmp.empty:
                api_max = tmp['timestamp'].max().to_pydatetime().replace(tzinfo=None)
                target_start = max(gap_start, api_max - timedelta(minutes=5))
                if api_max >= gap_end:
                    print(f"  ℹ️  기존 파일이 범위를 커버합니다. 마지막 봉을 재확인합니다.")
                else:
                    print(f"  ℹ️  기존 파일 존재({api_max.date()}까지). 마지막 봉부터 이어서 다운로드합니다.")

        base_url = 'https://fapi.binance.com/fapi/v1/klines'
        target_end = gap_end + timedelta(days=1)
        if target_start >= target_end:
            print(f"  ⏭️  {symbol} api.csv가 이미 범위( ~ {gap_end.date()}) 커버 → 건너뜀")
            continue
        start_ms = int(target_start.timestamp() * 1000)
        end_ms   = int(target_end.timestamp() * 1000)
        all_rows = []
        curr_ms  = start_ms

        while curr_ms < end_ms:
            try:
                resp = requests.get(base_url, params={
                    'symbol': symbol, 'interval': '5m',
                    'startTime': curr_ms, 'endTime': end_ms, 'limit': 1500,
                }, timeout=30)
                resp.raise_for_status()
                batch = resp.json()
            except Exception as e:
                print(f"  ❌ API 오류: {e}")
                break
            if not batch:
                break
            all_rows.extend(batch)
            curr_ms = batch[-1][0] + 1
            print(f"  수집 중... {len(all_rows):>7}봉  "
                  f"({datetime.fromtimestamp(batch[-1][0]/1000).strftime('%Y-%m-%d')})")

        if all_rows:
            df = pd.DataFrame(all_rows, columns=KLINES_COLS)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            # 기존 api.csv가 있으면 합치기
            if os.path.exists(api_csv):
                old = pd.read_csv(api_csv, parse_dates=['timestamp'])
                df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp')
            df.to_csv(api_csv, index=False)
            print(f"  ✅ {symbol} api.csv 저장 ({len(df):,}봉)")
        else:
            print(f"  ⚠️ {symbol} 수집된 데이터 없음")


# ══════════════════════════════════════════════════════════════════
# 4. 전체 소스 로드
# ══════════════════════════════════════════════════════════════════
def _load_zips_df(directory: str, ts_col_candidates: list[str],
                   extra_drop: list | None = None) -> pd.DataFrame:
    """directory 안의 zip 파일들을 모두 읽어 합친 DataFrame 반환."""
    dfs = []
    if not os.path.isdir(directory):
        return pd.DataFrame()
    for fname in sorted(f for f in os.listdir(directory) if f.endswith('.zip')):
        try:
            with zipfile.ZipFile(os.path.join(directory, fname)) as zf:
                raw = zf.read(zf.namelist()[0])
                df  = pd.read_csv(io.BytesIO(raw))
                
                # 여러 후보 중 존재하는 컬럼 탐색
                found_col = None
                for c in ts_col_candidates:
                    if c in df.columns:
                        found_col = c
                        break
                
                if found_col:
                    ct = df[found_col]
                    ct_num = pd.to_numeric(ct, errors='coerce')
                    if ct_num.notna().mean() > 0.9:
                        df['timestamp'] = pd.to_datetime(ct_num, unit='ms', errors='coerce')
                    else:
                        df['timestamp'] = pd.to_datetime(ct, errors='coerce')
                    df = df.dropna(subset=['timestamp'])
                    drop_cols = ([found_col] if found_col != 'timestamp' else []) + (extra_drop or [])
                    df = df.drop(columns=drop_cols, errors='ignore')
                    dfs.append(df)
        except Exception as e:
            print(f"  ⚠️ zip 읽기 실패 ({fname}): {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_all_sources(gap_start: datetime, gap_end: datetime):
    """ETH/BTC klines, metrics, funding 전체 소스를 로드 & gap 구간 필터링."""
    print("\n  소스 데이터 로드 중...")

    # ── ETH klines ──
    dfs_eth = []
    if os.path.exists(ETH_CSV):
        df = pd.read_csv(ETH_CSV, parse_dates=['timestamp'])
        dfs_eth.append(df)
    api_eth = os.path.join(ETH_KLINES_DIR, 'ETHUSDT-5m-api.csv')
    if os.path.exists(api_eth):
        df = pd.read_csv(api_eth)
        if pd.api.types.is_integer_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        dfs_eth.append(df)
    eth_df = (pd.concat(dfs_eth, ignore_index=True)
              .drop_duplicates(subset=['timestamp'], keep='last')
              .sort_values('timestamp').reset_index(drop=True))
    eth_df['timestamp'] = eth_df['timestamp'].astype('datetime64[us]')
    eth_df = eth_df[(eth_df['timestamp'] >= pd.Timestamp(gap_start).tz_localize(None))
                    & (eth_df['timestamp'] <= pd.Timestamp(gap_end).tz_localize(None))].copy()

    # ── BTC klines ──
    dfs_btc = []
    if os.path.exists(BTC_CSV):
        df = pd.read_csv(BTC_CSV, parse_dates=['timestamp'])
        dfs_btc.append(df)
    api_btc = os.path.join(BTC_KLINES_DIR, 'BTCUSDT-5m-api.csv')
    if os.path.exists(api_btc):
        df = pd.read_csv(api_btc)
        if pd.api.types.is_integer_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        dfs_btc.append(df)
    btc_df = (pd.concat(dfs_btc, ignore_index=True)
              .drop_duplicates(subset=['timestamp'], keep='last')
              .sort_values('timestamp').reset_index(drop=True))
    btc_df['timestamp'] = btc_df['timestamp'].astype('datetime64[us]')
    btc_df = btc_df[['timestamp', 'close', 'volume', 'quote_volume']]

    # ── Metrics ──
    dfs_met = []
    if os.path.exists(METRICS_CSV):
        df = pd.read_csv(METRICS_CSV)
        df['timestamp'] = pd.to_datetime(df['create_time'], errors='coerce')
        df = df.dropna(subset=['timestamp']).drop(columns=['create_time', 'symbol'], errors='ignore')
        dfs_met.append(df)
    zip_met = _load_zips_df(METRICS_DIR, ts_col_candidates=['create_time'], extra_drop=['symbol'])
    if not zip_met.empty:
        dfs_met.append(zip_met)
    metrics_df = (pd.concat(dfs_met, ignore_index=True)
                  .drop_duplicates(subset=['timestamp'])
                  .sort_values('timestamp').reset_index(drop=True))
    metrics_df['timestamp'] = metrics_df['timestamp'].astype('datetime64[us]')

    # ── Funding ──
    dfs_fund = []
    if FUNDING_CSV and os.path.exists(FUNDING_CSV):
        df = pd.read_csv(FUNDING_CSV)
        df['timestamp'] = pd.to_datetime(df['calc_time'], errors='coerce')
        df = df.dropna(subset=['timestamp'])[['timestamp', 'last_funding_rate']]
        dfs_fund.append(df)
    bad_funding_zips = [
        fname for fname in sorted(os.listdir(FUNDING_RATE_DIR))
        if fname.endswith('.zip') and 'ETHUSDT' not in fname
    ]
    if bad_funding_zips:
        raise RuntimeError(
            f"Funding zip contract violation: expected ETHUSDT files only, got {bad_funding_zips[:5]}"
        )
    zip_fund = _load_zips_df(FUNDING_RATE_DIR, ts_col_candidates=['calcTime', 'calc_time'])
    if not zip_fund.empty:
        # Binance zip의 컬럼명 정리
        if 'calcTime' in zip_fund.columns:
            zip_fund = zip_fund.rename(columns={'calcTime': 'timestamp'})
        if 'fundingRate' in zip_fund.columns:
            zip_fund = zip_fund.rename(columns={'fundingRate': 'last_funding_rate'})
        if 'last_funding_rate' in zip_fund.columns:
            dfs_fund.append(zip_fund[['timestamp', 'last_funding_rate']])
    funding_df = (pd.concat(dfs_fund, ignore_index=True)
                  .drop_duplicates(subset=['timestamp'])
                  .sort_values('timestamp').reset_index(drop=True))
    funding_df['timestamp'] = funding_df['timestamp'].astype('datetime64[us]')

    # ── 교집합 구간(Common timeframe)으로 필터링 ──
    # NOTE:
    # metrics/funding 파일은 공개 지연이 자주 발생한다.
    # 최신 캔들 구간을 보존하기 위해 공통 구간 계산은 ETH/BTC 캔들만 기준으로 한다.
    # metrics/funding은 build_features()에서 backward merge_asof로 보수적으로 붙인다.
    start_time = max(
        eth_df['timestamp'].min(),
        btc_df['timestamp'].min(),
    )
    end_time = min(
        eth_df['timestamp'].max(),
        btc_df['timestamp'].max(),
    )

    print(f"\n  [공통 구간 필터링] {start_time} ~ {end_time}")
    
    eth_df = eth_df[(eth_df['timestamp'] >= start_time) & (eth_df['timestamp'] <= end_time)].copy()
    btc_df = btc_df[(btc_df['timestamp'] >= start_time) & (btc_df['timestamp'] <= end_time)].copy()
    # metrics/funding은 전체 히스토리를 유지한다 (backward asof에서 최신 과거값 사용).

    print(f"  ✓ 최종 교집합 데이터 건수:")
    print(f"    - ETH klines : {len(eth_df):,}봉")
    print(f"    - BTC klines : {len(btc_df):,}봉")
    print(f"    - Metrics    : {len(metrics_df):,}건")
    print(f"    - Funding    : {len(funding_df):,}건")

    return eth_df, btc_df, metrics_df, funding_df


# ══════════════════════════════════════════════════════════════════
# 5. 피처 생성
# ══════════════════════════════════════════════════════════════════
def build_features(eth_df: pd.DataFrame, btc_df: pd.DataFrame,
                   metrics_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    """merge_asof → FeatureEngineer.process() → dropna."""
    print("\n  피처 생성 중...")

    # 중복 컬럼 제거
    overlap = [c for c in metrics_df.columns if c in eth_df.columns and c != 'timestamp']
    if overlap:
        eth_df = eth_df.drop(columns=overlap)

    # ETH ← Metrics (8시간 주기 → backward + 9h tolerance)
    eth_merged = pd.merge_asof(
        eth_df.sort_values('timestamp'),
        metrics_df.sort_values('timestamp'),
        on='timestamp',
        direction='backward',
        tolerance=pd.Timedelta('9h'),
    )

    # metrics 필수 컬럼 Null 체크
    required_metrics = [
        'sum_toptrader_long_short_ratio', 'count_long_short_ratio',
        'sum_open_interest_value', 'sum_open_interest',
    ]
    null_pct = eth_merged[required_metrics].isnull().mean()
    for col, pct in null_pct.items():
        if pct > 0.1:
            print(f"  ⚠️ {col} NaN 비율: {pct:.1%} (tolerance 범위 밖 데이터 많음)")

    # ETH ← Funding (backward) to avoid look-ahead leakage.
    # 최신 월 파일 미공개 구간에서도 마지막 관측 funding을 사용할 수 있도록 tolerance를 두지 않는다.
    eth_merged = pd.merge_asof(
        eth_merged.sort_values('timestamp'),
        funding_df.sort_values('timestamp'),
        on='timestamp',
        direction='backward',
    )

    print(f"  ✓ 병합 완료: {len(eth_merged):,}행")

    # 피처 생성
    engineer = FeatureEngineer(candle_minutes=5, keep_only_active=True, include_entry_price=False)
    result = engineer.process(eth_merged, btc_df)
    print(f"  ✓ 피처 생성: {len(result):,}행, {len(result.columns)}컬럼")

    # NaN 제거
    before = len(result)
    result = result.dropna()
    if before != len(result):
        print(f"  ℹ️  NaN 행 제거: {before - len(result):,}행")

    return result


# ══════════════════════════════════════════════════════════════════
# 6. 기존 CSV와 병합 저장
# ══════════════════════════════════════════════════════════════════
def merge_and_save(new_df: pd.DataFrame):
    """기존 training_features_5m.csv와 concat → 중복 제거 → 저장."""
    new_df = prune_to_active_feature_keep(
        new_df,
        include_entry_price=False,
        extra_keep=["timestamp"],
    )
    if os.path.exists(FEATURES_CSV):
        existing = pd.read_csv(FEATURES_CSV, parse_dates=['timestamp'])
        existing['timestamp'] = existing['timestamp'].astype('datetime64[us]')
        existing = prune_to_active_feature_keep(
            existing,
            include_entry_price=False,
            extra_keep=["timestamp"],
        )
        all_cols = sorted(set(new_df.columns).union(set(existing.columns)))
        if "timestamp" in all_cols:
            all_cols.remove("timestamp")
        all_cols = ["timestamp"] + all_cols
        existing = existing.reindex(columns=all_cols)
        new_df = new_df.reindex(columns=all_cols)
        # existing (previously-saved) rows go FIRST so drop_duplicates(keep='first' default)
        # preserves them over a fresh recompute -- gap_start defaults to 2024-01-01 (see
        # check_gaps), so build_features() reprocesses the whole raw history on every run;
        # without this, that full recompute silently overwrote already-cached historical rows,
        # making backtest numbers non-reproducible across extension runs even though the
        # per-row formulas themselves are causal (found 2026-07-13, see project memory).
        result = pd.concat([existing[all_cols], new_df[all_cols]], ignore_index=True)
    else:
        result = new_df

    before = len(result)
    result = (result
              .drop_duplicates(subset=['timestamp'])
              .sort_values('timestamp')
              .reset_index(drop=True))
    result = result.dropna()

    result.to_csv(FEATURES_CSV, index=False)
    print(f"\n  ✅ 저장 완료: {FEATURES_CSV}")
    print(f"     행 수: {len(result):,}  ({result['timestamp'].min()} ~ {result['timestamp'].max()})")
    print(f"     NaN: {result.isnull().sum().sum()}")


# ══════════════════════════════════════════════════════════════════
# 7. 검증 로직
# ══════════════════════════════════════════════════════════════════
def verify_data(eth_df: pd.DataFrame, metrics_df: pd.DataFrame, funding_df: pd.DataFrame):
    """저장된 training_features_5m.csv의 랜덤 행을 추출해 원본 데이터와 일치하는지 검증."""
    print("\n[Step 6] 데이터 무결성 검증 (랜덤 샘플링)...")
    if not os.path.exists(FEATURES_CSV):
        print("  ⚠️ 저장된 파일이 없어 검증할 수 없습니다.")
        return

    df_saved = pd.read_csv(FEATURES_CSV, parse_dates=['timestamp'])
    if df_saved.empty:
        print("  ⚠️ 저장된 데이터가 없습니다.")
        return

    # 이번에 병합된 구간 내에서 무작위 1개 샘플링
    min_ts = eth_df['timestamp'].min()
    max_ts = eth_df['timestamp'].max()
    target_df = df_saved[(df_saved['timestamp'] >= min_ts) & (df_saved['timestamp'] <= max_ts)]
    
    if target_df.empty:
        target_df = df_saved

    sample = target_df.sample(1).iloc[0]
    ts = sample['timestamp']
    print(f"  🔍 검증 대상 시간: {ts}")

    # 1. ETH 가격 검증 (정확히 일치해야 함)
    eth_raw = eth_df[eth_df['timestamp'] == ts]
    if not eth_raw.empty:
        raw_close = eth_raw['close'].values[0]
        saved_close = sample['close']
        print(f"    - [ETH 가격 (close)] 원본: {raw_close} / 저장됨: {saved_close} -> {'✅ 일치' if raw_close == saved_close else '❌ 불일치'}")
    
    # 2. Metrics (backward 병합이므로 ts와 같거나 가장 가까운 과거 값)
    met_raw = metrics_df[metrics_df['timestamp'] <= ts].sort_values('timestamp').tail(1)
    if not met_raw.empty:
        raw_oi_val = met_raw['sum_open_interest_value'].values[0]
        saved_oi_val = sample['sum_open_interest_value']
        match = abs(raw_oi_val - saved_oi_val) < 1e-3 if pd.notna(raw_oi_val) and pd.notna(saved_oi_val) else pd.isna(raw_oi_val) == pd.isna(saved_oi_val)
        print(f"    - [Metrics (OI Value)] 원본: {raw_oi_val:.2f} / 저장됨: {saved_oi_val:.2f} -> {'✅ 일치' if match else '❔ 오차 / 불일치'}")

    # 3. Funding Rate (backward 병합이므로 ts와 같거나 가장 가까운 과거 값)
    fun_raw = funding_df[funding_df['timestamp'] <= ts].sort_values('timestamp').tail(1)
    if not fun_raw.empty:
        raw_fund = fun_raw['last_funding_rate'].values[0]
        saved_fund = sample['last_funding_rate']
        match = abs(raw_fund - saved_fund) < 1e-8 if pd.notna(raw_fund) and pd.notna(saved_fund) else pd.isna(raw_fund) == pd.isna(saved_fund)
        print(f"    - [Funding Rate] 원본: {raw_fund:.6f} / 저장됨: {saved_fund:.6f} -> {'✅ 일치' if match else '❔ 오차 / 불일치'}")


# ══════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    start = datetime.strptime(args.start, '%Y-%m-%d')
    end   = datetime.strptime(args.end,   '%Y-%m-%d')

    print(f"\n{'='*60}")
    print(f"  통합 데이터 파이프라인")
    print(f"  요청 기간: {start.date()} ~ {end.date()}")
    print(f"{'='*60}")

    # Step 1: gap 분석
    print("\n[Step 1] training_features_5m.csv gap 분석...")
    gap = check_gaps(start, end)
    if gap is None:
        print("  모든 데이터가 최신 상태입니다.")
        return
    gap_start, gap_end = gap
    print(f"  → 처리 필요 구간: {gap_start.date()} ~ {gap_end.date()}")

    # Step 2 & 3: 다운로드
    print("\n[Step 2] 로컬 파일 확인 및 다운로드...")
    ensure_metrics(gap_start, gap_end)
    ensure_funding(gap_start, gap_end)
    ensure_klines(gap_start, gap_end)

    # Step 4: 소스 로드
    print("\n[Step 3] 소스 데이터 로드...")
    eth_df, btc_df, metrics_df, funding_df = load_all_sources(gap_start, gap_end)

    if eth_df.empty:
        print("  ❌ ETH klines 데이터가 없습니다. 종료.")
        return

    # Step 5: 피처 생성
    print("\n[Step 4] 피처 생성...")
    new_features = build_features(eth_df, btc_df, metrics_df, funding_df)

    # Step 6: 저장
    print("\n[Step 5] 기존 CSV와 병합 및 저장...")
    merge_and_save(new_features)

    # Step 7: 검증
    verify_data(eth_df, metrics_df, funding_df)

    print(f"\n{'='*60}")
    print("  ✅ 완료!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
