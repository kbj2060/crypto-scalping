#!/usr/bin/env python3
"""ETH 5m: session-split (us / europe / asia / none) feature-vs-price correlation analysis.

목적: 풀링(pooled) 상태에서는 엣지가 안 보이는 ETH 데이터를, 거래 세션으로 쪼갰을 때
피처-가격 상관이 세션별로 갈라지는지(= split이 엣지를 만드는지) 측정한다.

세션 정의 (사용자 선택: live 캘린더 기반)
  - session_us      : live 102-feature 계약의 session_us 컬럼 (NYSE mcal, DST/휴장일 반영)
  - session_europe  : live 102-feature 계약의 session_europe 컬럼 (LSE mcal)
  - session_asia    : features/engineering.py 의 session_japan 과 동일 레시피로 신규 파생
                      (JPX mcal; 102-col 계약에는 없어서 여기서 만든다)
  - none            : 위 셋 중 아무것도 안 열린 바 (주말 / 21~24 UTC / 06~07 UTC 등)
  us 와 europe 은 13~16 UTC 부근에서 겹치므로, 4-way 분할은 us > europe > asia 우선순위로
  배타 할당한다. 겹침 구간 크기는 리포트에 별도로 기록한다.

타깃 (사용자 선택: 둘 다 동등하게)
  A) forward log-return: h in {1,3,6,12,24,72} bar (5m / 15m / 30m / 1h / 2h / 6h)
  B) 동시점 close 레벨

지표는 Spearman rank correlation. forward return이 겹치므로(overlapping) 순진한 p-value는
과대신뢰가 된다 -> p-value 대신 (1) TRAIN/VAL/OOS 부호 일관성, (2) hour-of-day rotation null
두 가지로 판정한다.

Rotation null: 세션 라벨 시계열을 시각(hour) 단위로 k시간 회전시키면 버킷 크기와 블록 구조는
보존되면서 진짜 정렬만 깨진다. k=1..23 회전에서 나오는 세션별 IC 분포가 귀무분포다.

읽기 전용 연구 스크립트. 기존 스크립트/배포 번들/trading_bot_modules 를 수정하지 않는다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

HORIZONS = [1, 3, 6, 12, 24, 72]
SESSIONS = ["us", "europe", "asia", "none"]
ROTATIONS = list(range(1, 24))


def load_frames(cache_dir: Path) -> dict[str, pd.DataFrame]:
    """true 102-feature 프레임 로드. cache_dir 에 parquet 이 있으면 재사용."""
    names = ["train_raw", "val_raw", "oos_raw"]
    if all((cache_dir / f"{n}.parquet").exists() for n in names):
        return {n: pd.read_parquet(cache_dir / f"{n}.parquet") for n in names}

    import eth_odyssey4_true_feature_pipeline_20260816 as tp

    d = tp.prepare_frames_true()
    cols = ["timestamp"] + list(d["feature_cols"])
    out = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        df = d[n][cols].copy()
        df.to_parquet(cache_dir / f"{n}.parquet", index=False)
        out[n] = df
    return out


def build_asia_flag(ts: pd.Series) -> np.ndarray:
    """features/engineering.py 의 session_japan 과 동일 레시피 (JPX mcal, 1min membership)."""
    ts_utc = pd.to_datetime(ts).dt.tz_localize("UTC")
    jpx = mcal.get_calendar("JPX")
    schedule = jpx.schedule(
        start_date=ts_utc.min().date() - pd.Timedelta(days=2),
        end_date=ts_utc.max().date() + pd.Timedelta(days=2),
    )
    minutes = mcal.date_range(schedule, frequency="1min")
    return ts_utc.isin(minutes).to_numpy()


def assign_sessions(df: pd.DataFrame) -> pd.Series:
    """us > europe > asia 우선순위 배타 할당."""
    us = df["session_us"].to_numpy() > 0.5
    eu = df["session_europe"].to_numpy() > 0.5
    asia = build_asia_flag(df["timestamp"])
    label = np.full(len(df), "none", dtype=object)
    label[asia] = "asia"
    label[eu] = "europe"
    label[us] = "us"
    return pd.Series(label, index=df.index, name="session")


def rotate_sessions(df: pd.DataFrame, session: pd.Series, hours: int) -> pd.Series:
    """세션 라벨을 시각 기준 +hours 회전. 5m bar 이므로 12*hours row shift (원형)."""
    shift = 12 * hours
    vals = session.to_numpy()
    return pd.Series(np.roll(vals, shift), index=session.index)


def forward_log_returns(close: np.ndarray, horizons: list[int]) -> dict[int, np.ndarray]:
    logc = np.log(close)
    out = {}
    for h in horizons:
        fwd = np.full(len(logc), np.nan)
        fwd[:-h] = logc[h:] - logc[:-h]
        out[h] = fwd
    return out


def _zranks(x: np.ndarray) -> np.ndarray | None:
    """열 x 의 rank 를 평균 0 / 표준편차 1 로. 상수열이거나 비유한값이 있으면 None."""
    if not np.all(np.isfinite(x)):
        return None
    r = rankdata(x)
    sd = r.std()
    if sd < 1e-12:
        return None
    return (r - r.mean()) / sd


def feature_zranks(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """마스크된 feats (n, k) 의 열별 z-rank 행렬과 유효 열 마스크. 타깃마다 재계산하지 않도록
    마스크당 한 번만 부른다 (모든 타깃이 동일한 행 집합을 쓰기 때문에 가능)."""
    n, k = feats.shape
    z = np.zeros((n, k), dtype=np.float64)
    valid = np.zeros(k, dtype=bool)
    for j in range(k):
        zr = _zranks(feats[:, j])
        if zr is None:
            continue
        z[:, j] = zr
        valid[j] = True
    return z, valid


def spearman_from_zranks(z: np.ndarray, valid: np.ndarray, target: np.ndarray) -> np.ndarray:
    """미리 계산한 feature z-rank 행렬과 target 의 Spearman."""
    out = np.full(z.shape[1], np.nan)
    tr = _zranks(target)
    if tr is None or len(target) < 200:
        return out
    ic = (z.T @ tr) / len(target)
    out[valid] = ic[valid]
    return out


def analyze_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_name: str,
    with_null: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """한 split(TRAIN/VAL/OOS) 에 대해 세션별 IC 표와 rotation-null 요약을 만든다."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    session = assign_sessions(df)
    close_full = df["close"].to_numpy(dtype=np.float64)
    fwd = forward_log_returns(close_full, HORIZONS)

    # 모든 horizon 이 정확히 같은 행 집합을 쓰도록 마지막 max(HORIZONS) 행을 잘라낸다.
    # (그래야 마스크당 feature rank 를 한 번만 계산해서 모든 타깃에 재사용할 수 있고,
    #  horizon 간 IC 비교도 같은 표본 위에서 이뤄진다.)
    keep = len(df) - max(HORIZONS)
    df = df.iloc[:keep].reset_index(drop=True)
    session = session.iloc[:keep].reset_index(drop=True)
    feats = df[feature_cols].to_numpy(dtype=np.float64)
    targets: dict[str, np.ndarray] = {f"fwd{h}": fwd[h][:keep] for h in HORIZONS}
    targets["close_level"] = close_full[:keep]
    fwd_names = [f"fwd{h}" for h in HORIZONS]

    def ic_rows(mask: np.ndarray, tag: dict, target_names: list[str]) -> list[dict]:
        z, valid = feature_zranks(feats[mask])
        rows = []
        for tname in target_names:
            ic = spearman_from_zranks(z, valid, targets[tname][mask])
            for j, col in enumerate(feature_cols):
                rows.append({**tag, "target": tname, "feature": col, "ic": ic[j]})
        return rows

    rows = []
    for sess in SESSIONS + ["pooled"]:
        mask = np.ones(len(df), dtype=bool) if sess == "pooled" else (session == sess).to_numpy()
        n = int(mask.sum())
        if n < 200:
            continue
        rows += ic_rows(mask, {"split": split_name, "session": sess, "n": n}, fwd_names + ["close_level"])
    ic_df = pd.DataFrame(rows)

    null_rows = []
    if with_null:
        for k in ROTATIONS:
            rot = rotate_sessions(df, session, k)
            for sess in SESSIONS:
                mask = (rot == sess).to_numpy()
                if mask.sum() < 200:
                    continue
                null_rows += ic_rows(mask, {"split": split_name, "rotation": k, "session": sess}, fwd_names)
    null_df = pd.DataFrame(null_rows)
    return ic_df, null_df


def main() -> None:
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tmp/session_split_20260817"
    outdir = ROOT / "tmp/session_split_20260817"
    outdir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(cache)
    feature_cols = [c for c in frames["train_raw"].columns if c != "timestamp"]
    print(f"features={len(feature_cols)}", flush=True)

    split_map = {"TRAIN": "train_raw", "VAL": "val_raw", "OOS": "oos_raw"}
    ics, nulls = [], []
    for split_name, key in split_map.items():
        df = frames[key]
        print(f"[{split_name}] n={len(df)} {df.timestamp.min()} .. {df.timestamp.max()}", flush=True)
        ic_df, null_df = analyze_split(df, feature_cols, split_name, with_null=True)
        ics.append(ic_df)
        nulls.append(null_df)
        sess = assign_sessions(df.sort_values("timestamp").reset_index(drop=True))
        counts = sess.value_counts().to_dict()
        us = df["session_us"].to_numpy() > 0.5
        eu = df["session_europe"].to_numpy() > 0.5
        print(f"  session counts={counts} eu_us_overlap={int((us & eu).sum())}", flush=True)

    ic_all = pd.concat(ics, ignore_index=True)
    null_all = pd.concat(nulls, ignore_index=True)
    ic_all.to_parquet(outdir / "session_ic.parquet", index=False)
    null_all.to_parquet(outdir / "session_ic_null.parquet", index=False)

    # rotation-null 대비 z-score: 같은 (split, session, target, feature) 의 실제 IC 를
    # 그 셀의 회전 귀무분포와 비교
    null_stats = (
        null_all.groupby(["split", "session", "target", "feature"])["ic"]
        .agg(null_mean="mean", null_std="std")
        .reset_index()
    )
    merged = ic_all.merge(null_stats, on=["split", "session", "target", "feature"], how="left")
    merged["z_vs_null"] = (merged["ic"] - merged["null_mean"]) / merged["null_std"]
    merged.to_parquet(outdir / "session_ic_with_null.parquet", index=False)

    meta = {
        "sessions": SESSIONS,
        "session_definition": "live mcal columns (session_us=NYSE, session_europe=LSE) + asia=JPX derived; precedence us>europe>asia",
        "horizons_bars": HORIZONS,
        "targets": [f"fwd{h}" for h in HORIZONS] + ["close_level"],
        "rotations": ROTATIONS,
        "splits": {k: [str(frames[v].timestamp.min()), str(frames[v].timestamp.max()), int(len(frames[v]))]
                   for k, v in split_map.items()},
        "n_features": len(feature_cols),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("WROTE", outdir, flush=True)


if __name__ == "__main__":
    main()
