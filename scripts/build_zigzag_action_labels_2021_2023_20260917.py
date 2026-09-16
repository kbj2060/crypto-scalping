#!/usr/bin/env python3
"""E3 -- zigzag 라벨을 2021~2023 으로 확장.

⭐**역공학은 필요 없었다.** 생성기가 저장소에 있었고 이름이 `build_wave3_action_labels_20260531.py`
라서(내부 `DEFAULT_OUT` 이 `zigzag_action_labels_20260531`) 「zigzag」로 찾을 때 안 걸렸다.
커밋된 그 스크립트를 손대지 않고 그대로 돌리니 **2024·2025 가 행 단위로 완전 재현**된다
(일치율 1.000000). 2026 만 다른데, 정본이 2026-05-31 에 02-28 까지의 입력으로 만들어진
구판이고 지금 입력은 그보다 길어서다 -- 생성기 불일치가 아니다.

여기서는 같은 공개 함수 `build_zigzag_action_labels` 를 **기본값 그대로** 아카이브에 건다.
⚠️정본이 **연도별로 따로** 만들어졌으므로(연 경계에서 지그재그 상태가 리셋된다) 여기서도
연도별로 자른다. 이어붙여 한 번에 돌리면 배포 라벨과 다른 라벨이 된다.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT / "scripts"))
import build_wave3_action_labels_20260531 as W  # noqa: E402

ARCHIVE = ROOT / "data/eth_5m_2021_2023_archive.csv"
GOLD = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
REPRO = ROOT / "tmp/omega461_longwindow_20260917/zz_repro"
OUT = ROOT / "tmp/omega461_longwindow_20260917/zigzag_labels_full"


def verify_2026_prefix() -> None:
    """재생성 2026 이 정본 2026 의 «연장»인지 확인 -- 앞 16,897행이 같아야 한다."""
    g = pd.read_csv(GOLD / "zigzag_action_labels_2026.csv")
    n = pd.read_csv(REPRO / "zigzag_action_labels_2026.csv")
    head = n.iloc[: len(g)]
    ts = pd.to_datetime(g.timestamp)
    same_ts = (pd.to_datetime(head.timestamp).to_numpy() == ts.to_numpy()).all()
    assert same_ts, "2026 타임스탬프가 어긋난다"

    # 정본 2026 은 입력이 02-28 16:00 에서 끊긴 상태로 만들어졌다. 지그재그는 **확정된 피벗**
    # 까지만 라벨하므로, 절단면 근처의 「아직 확정 안 된 마지막 파동」은 입력이 길어지면 값이
    # 바뀐다 -- 생성기 불일치가 아니라 라벨 정의 자체의 성질이다. 그래서 「전부 같은가」가
    # 아니라 **「불일치가 절단면 근처에만 있고 그 앞은 완전히 같은가」**를 본다.
    diff = (head.zigzag_action.to_numpy() != g.zigzag_action.to_numpy())
    idx = diff.nonzero()[0]
    cut = ts.iloc[-1]
    edge = cut - pd.Timedelta(days=14)
    first_bad = ts.iloc[idx.min()] if len(idx) else None
    clean = float((~diff[: (idx.min() if len(idx) else len(diff))]).mean())
    print(f"2026 접두 검증: 불일치 {len(idx)}/{len(g)} · 첫 불일치 {first_bad} "
          f"· 절단면 {cut} · 그 앞 {idx.min() if len(idx) else len(diff)}행 일치율 {clean:.6f}",
          flush=True)
    assert clean == 1.0, "절단면 이전 구간이 완전히 재현되지 않는다"
    assert first_bad is None or first_bad >= edge, (
        f"불일치가 절단면에서 14일보다 멀다({first_bad}) -- 끝단 효과로 설명되지 않는다")


def main() -> int:
    verify_2026_prefix()
    OUT.mkdir(parents=True, exist_ok=True)

    # 파라미터를 여기 다시 적지 않는다 -- 방금 정본을 재현한 그 실행의 audit 에서 읽는다.
    audit = json.loads((REPRO / "zigzag_action_label_audit.json").read_text())
    prm = audit["params"]
    kw = dict(min_reversal_pct=float(prm["zigzag_reversal_pct"]),
              min_wave_bars=int(prm["min_wave_bars"]),
              transition_buffer=int(prm["transition_buffer"]),
              atr_window=int(prm["atr_window"]),
              atr_multiplier=float(prm["atr_multiplier"]),
              mae_penalty=float(prm["mae_penalty"]),
              softmax_temperature=float(prm["softmax_temperature"]),
              min_risk_floor=float(prm["min_risk_floor"]))
    print(f"파라미터(재현 실행의 audit 에서): {kw}", flush=True)

    raw = pd.read_csv(ARCHIVE)
    raw["timestamp"] = pd.to_datetime(raw["open_time"].astype("int64"), unit="ms")
    raw = raw[["timestamp", "open", "high", "low", "close"]].dropna(subset=["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    print(f"아카이브 {len(raw)}행  {raw.timestamp.min()} ~ {raw.timestamp.max()}", flush=True)

    for year in (2021, 2022, 2023):
        frame = raw[raw.timestamp.dt.year == year].reset_index(drop=True)
        if frame.empty:
            print(f"{year}: 행 없음 -- 건너뜀", flush=True)
            continue
        labels = W.build_zigzag_action_labels(frame, **kw)
        s = W._summary(labels)
        print(f"{year}: {s['rows']}행 · counts {s['counts']} · ratios "
              f"{ {k: round(v,4) for k,v in s['ratios'].items()} } · segments {s['segments']}",
              flush=True)
        # 관문: 정본 연도들의 CASH 비중은 0.085~0.118 이다. 한참 벗어나면 다른 라벨이다.
        assert 0.04 <= s["ratios"]["0"] <= 0.22, f"{year} CASH 비중 이상: {s['ratios']['0']}"
        assert s["segments"] > 200, f"{year} 세그먼트 부족: {s['segments']}"
        labels.to_csv(OUT / f"zigzag_action_labels_{year}.csv", index=False)

    # 2024~2026 은 재생성본을 정본으로 쓴다(2024/2025 는 행 단위 동일, 2026 은 연장판).
    for year in (2024, 2025, 2026):
        src = REPRO / f"zigzag_action_labels_{year}.csv"
        (OUT / f"zigzag_action_labels_{year}.csv").write_bytes(src.read_bytes())
    print(f"\n저장: {OUT}", flush=True)
    for p in sorted(OUT.glob("*.csv")):
        d = pd.read_csv(p, usecols=["timestamp", "zigzag_action"])
        t = pd.to_datetime(d.timestamp)
        print(f"  {p.name}  {len(d):>6}행  {t.min()} ~ {t.max()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
