#!/usr/bin/env python3
"""증거신호 8종 + 이벤트 트리거를 **한 프레임**에 모은 재료 패널 (2026-09-12).

사용자: *"있는 데이터 모두 모아서 증거신호와 이벤트 트리거까지 신호를 내서 재료들을 모아줘."*
다음 단계(딥러닝/강화학습이 아니라 **알고리즘 조합으로 방향성**)의 입력이다.

새 로직은 없다. 이미 배포·검증된 계산기를 호출해 timestamp 로 붙이는 조립 스크립트다.
  증거신호 8종    `live_evidence_signal_dashboard_20260823.compute_signals`
  앵커(조합) 규약 `build_eth_anchor_label_dataset_20260907.anchor_index` (any3/Wc3 · any2/Wc3 · first_fire)
  문맥 피쳐 29    `live_eth_extreme_detector_20260909._feature_frame`
  재료 텐서 51열  `data/materials/eth_evidence_signal_tensor_20260902` (2026-09-03 OOF 누수수정본)
  돌파 탐지기     `live_eth_breakout_detector_20260911` 의 규칙(2종 AND · z288 · 후행 q90)을 매 봉으로

## 인과 규약
인덱스 i 의 모든 값은 봉 i 의 **종가까지**만 쓴다. **라벨은 이 패널에 없다** — 사건 라벨을 붙일 때
CLAUDE.md 「사건 라벨 경계 계약」대로 피쳐 창 끝 ≤ 라벨 탐색 시작 −1 을 반드시 지킬 것.

## 상한
BTC 5분봉(2026-08-20)이 ETH(09-10)보다 짧다. smt_divergence 가 BTC 고저를 필요로 하므로
(빠뜨리면 조용히 발동 0) 패널은 **BTC 상한에서 자른다**. 펀딩은 2026-07-31 까지라 그 뒤
orthogonal_combo 바닥의 funding OR-leg 가 조용히 꺼진다 → `fund_ok` 컬럼으로 표시한다.

산출: data/materials/eth_signal_trigger_panel_20260912/{panel_5m.parquet,README.json,coverage.csv}
자체점검: `python3 scripts/build_eth_signal_trigger_material_panel_20260912.py --selftest`
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402
import live_eth_extreme_detector_20260909 as X  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

TENSOR = ROOT / "data/materials/eth_evidence_signal_tensor_20260902/eth_evidence_material_5m.parquet"
OUT = ROOT / "data/materials/eth_signal_trigger_panel_20260912"
ANCHORS = [("any3", 3), ("any2", 3), ("first_fire", None)]
QWIN, COMPRESS = 2016, 0.70      # 돌파 탐지기 원문 상수
RANK_W = X.RANK_W                # 2016 (7일) 후행 분위 창
VREV_W, VREV_K = 6, 1.5          # V자 반전: 6봉 안에 1.5×ATR 되돌림이 **이미 끝난** 봉
VEXP_W, VEXP_K = 12, 1.3         # 변동성 확장: 직전 12봉 실현변동성 / 그 앞 12봉
ER_W, ER_Q, ER_CONF = 48, 0.70, 3  # 추세: 효율비 후행 70분위, 3봉 확인


def log(m: str) -> None:
    print(f"[panel {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _rank_pct(x: np.ndarray, w: int = RANK_W) -> np.ndarray:
    """후행 창 안에서의 분위(인과). 봉 i 자신을 포함한다 — 값이 i 종가에 확정되므로 정당하다."""
    return pd.Series(x).rolling(w, min_periods=500).rank(pct=True).to_numpy()


def _triggers(sig: pd.DataFrame, qv: np.ndarray, trades: np.ndarray) -> pd.DataFrame:
    """이벤트 트리거를 **과거만 보는 상태 플래그**로 계산한다.

    ⚠️`research_eth_all_triggers_midterm_20260910` 의 같은 이름들은 **라벨**(미래 창)이다.
    재료 패널에는 라벨을 담지 않으므로, 여기서는 같은 사건을 '이미 일어났는가'로 뒤집어 쓴다.
    """
    hi = sig.high.to_numpy(float); lo = sig.low.to_numpy(float)
    cl = sig.close.to_numpy(float); atr = sig.atr_pct.to_numpy(float)
    T = pd.DataFrame(index=range(len(sig)))

    # 돌파 탐지기(배포 규칙) — 압축 상태 + 거래대금/체결속도 z288 이 후행 q90 동시 초과
    lr = np.diff(np.log(cl), prepend=np.log(cl[0]))
    volexp = (pd.Series(lr).rolling(12).std() / pd.Series(lr).rolling(288).std()).to_numpy()
    T["volexp"] = volexp
    T["compressed"] = (volexp < COMPRESS).astype(float)
    det = np.ones(len(sig), bool)
    for nm, raw in (("qv", qv), ("n", trades)):
        s = pd.Series(raw, dtype=float)
        z = ((s - s.rolling(288).mean()) / s.rolling(288).std()).to_numpy()
        thr = pd.Series(z).rolling(QWIN, min_periods=200).quantile(0.90).to_numpy()
        T[f"brk_z288_{nm}"] = z
        det &= np.isfinite(z) & np.isfinite(thr) & (z >= thr)
    T["trg_breakout"] = det.astype(float)

    # V자 반전 — 창 안 극점이 **내부**(현재 봉 아님)에 있고 **양쪽 다리 모두** K×ATR 이상.
    # ⚠️한쪽 다리만 재면(초판) 91.7회/일로 상시 켜져 트리거 구실을 못 한다 — 실측 후 고쳤다.
    n = len(sig)
    Hs = np.stack([pd.Series(hi).shift(k).to_numpy() for k in range(VREV_W)])   # 행 k = k봉 전
    Ls = np.stack([pd.Series(lo).shift(k).to_numpy() for k in range(VREV_W)])
    ar = np.arange(n)
    older = np.arange(VREV_W)[:, None] > np.zeros((1, n))                      # 자리표시(아래서 갱신)
    scale = np.maximum(cl * atr, 1e-9)
    ok = np.isfinite(Ls).all(axis=0) & np.isfinite(Hs).all(axis=0)

    def _leg(E: np.ndarray, O: np.ndarray, sign: float) -> tuple[np.ndarray, np.ndarray]:
        """sign=+1: 저점(바닥 V) · sign=-1: 고점(천장 Λ). (극점까지의 다리, 극점에서 현재까지의 다리)"""
        j = np.nanargmin(np.where(np.isfinite(E), sign * E, np.inf), axis=0)   # 몇 봉 전이 극점인가
        ext = E[j, ar]
        older[:] = np.arange(VREV_W)[:, None] > j[None, :]                     # 극점보다 더 과거
        opp = np.where(older, sign * O, -np.inf).max(axis=0) * sign            # 그 구간 반대쪽 극값
        return j, np.stack([(opp - ext) * sign / scale, (cl - ext) * sign / scale])

    jb, legb = _leg(Ls, Hs, 1.0)
    jt, legt = _leg(Hs, Ls, -1.0)
    T["vrev_down_atr"] = legb[0]; T["vrev_up_atr"] = legb[1]
    vbot = ok & (jb >= 1) & (legb[0] >= VREV_K) & (legb[1] >= VREV_K)
    vtop = ok & (jt >= 1) & (legt[0] >= VREV_K) & (legt[1] >= VREV_K)
    T["trg_vrev"] = vbot.astype(float) - vtop.astype(float)

    # 변동성 확장 — 직전 창 실현변동성이 그 앞 창의 1.3배 이상
    rv = pd.Series(lr).rolling(VEXP_W).std()
    ratio = (rv / rv.shift(VEXP_W)).to_numpy()
    T["vol_expand_ratio"] = ratio
    T["trg_vol_expand"] = (np.isfinite(ratio) & (ratio >= VEXP_K)).astype(float)

    # 추세 레짐 — 효율비(방향거리/경로거리)가 후행 70분위 이상으로 ER_CONF 봉 연속
    move = np.abs(cl - pd.Series(cl).shift(ER_W).to_numpy())
    path = pd.Series(np.abs(np.diff(cl, prepend=cl[0]))).rolling(ER_W).sum().to_numpy()
    er = move / np.maximum(path, 1e-9)
    er_thr = pd.Series(er).rolling(RANK_W, min_periods=500).quantile(ER_Q).to_numpy()
    hot = np.isfinite(er) & np.isfinite(er_thr) & (er >= er_thr)
    T["er"] = er
    T["trg_regime_trend"] = (pd.Series(hot.astype(float)).rolling(ER_CONF).min().to_numpy())
    return T


def build(start: str = "2024-01-01", tmax: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    kl = B._load_kl(B.ETH_KL)
    btc = B._load_kl(B.BTC_KL)
    fund = B._load_funding()
    cap = min(kl["timestamp"].max(), btc["timestamp"].max())   # smt 가 BTC 고저를 쓴다
    if tmax is not None:
        cap = min(cap, pd.Timestamp(tmax))
    kl = kl[(kl["timestamp"] >= pd.Timestamp(start)) & (kl["timestamp"] <= cap)].reset_index(drop=True)
    btc = btc[btc["timestamp"] <= cap]
    fund = fund[fund["calc_time"] <= cap]
    log(f"kl {len(kl):,}행 {kl.timestamp.min()} ~ {kl.timestamp.max()}")

    sig = compute_signals(kl, btc_df=btc, funding_df=fund)
    log(f"증거신호 계산 완료 ({len(sig):,}행)")

    S, _ = X._feature_frame(sig, btc)
    P = S.drop(columns=[f"f_{s}" for s in B.SIGNALS])           # 발동은 아래에서 ±1 로 다시 붙인다
    ts = pd.to_datetime(sig["timestamp"]).to_numpy()
    P.insert(0, "timestamp", ts)
    for c in ("open", "high", "low", "close", "volume"):
        P[c] = sig[c].to_numpy(float)

    # 증거신호 8종 — 텐서와 같은 규약(+1 바닥/롱우호, −1 천장, 0 무발동)
    fire = {}
    for side in ("bottom", "top"):
        fire[side] = np.stack([sig[f"{side}_{s}"].fillna(False).to_numpy(bool) for s in B.SIGNALS], axis=1)
    for j, s in enumerate(B.SIGNALS):
        P[f"ev_{B.ABBR[s]}"] = fire["bottom"][:, j].astype(float) - fire["top"][:, j].astype(float)
    P["ev_n_bottom"] = fire["bottom"].sum(axis=1).astype(float)
    P["ev_n_top"] = fire["top"].sum(axis=1).astype(float)

    # 앵커(조합) 트리거 — 측면별로 따로 세고 ±1 로 합친다(양측 동시면 0, 개수 컬럼에 남는다)
    for name, wc in ANCHORS:
        tag = name if name == "first_fire" else f"{name}w{wc}"
        col = np.zeros(len(P))
        for side, sgn in (("bottom", 1.0), ("top", -1.0)):
            idx = B.anchor_index(fire[side], name, wc)
            if len(idx):
                col[idx] += sgn
        P[f"trg_{tag}"] = col

    # 이벤트 트리거 상태
    qvt = pd.read_csv(B.ETH_KL, usecols=["timestamp", "quote_volume", "trades"], parse_dates=["timestamp"])
    qvt = qvt.drop_duplicates("timestamp", keep="last").set_index("timestamp").reindex(pd.DatetimeIndex(ts))
    T = _triggers(sig, qvt["quote_volume"].to_numpy(float), qvt["trades"].to_numpy(float))
    P = pd.concat([P, T], axis=1)

    # 재료 텐서에서 **레짐만** 가져온다.
    # 🔴디스크의 텐서(mtime 2026-09-03 00:20)는 누수 수정 **이전** 판이다. 수정본은
    # `data/materials/eth_evidence_signal_tensor_oof_20260903/` 로 갔어야 하는데 그 디렉토리가
    # 없다(커밋 ee59951: "TRAIN 전체 in-sample 이라 하류 모델이 지름길을 배웠다", 누수지표
    # +0.5029 → −0.0430). 그래서 proba/pct/signed/age 는 **일부러 안 붙인다** — 붙이면 그 누수를
    # 그대로 하류로 다시 흘린다. 필요하면 빌더를 `git show ee59951:scripts/build_eth_evidence_
    # signal_material_tensor_oof_20260903.py` 로 복원해 재생성한 뒤 이 패널에 조인할 것.
    joined = 0
    if TENSOR.exists():
        mt = pd.read_parquet(TENSOR)[["timestamp", "regime_eth", "regime_btc"]]
        P = P.merge(mt.rename(columns={"regime_eth": "mt_regime_eth", "regime_btc": "mt_regime_btc"}),
                    on="timestamp", how="left")
        joined = int(P["mt_regime_eth"].notna().sum())
    P["fund_ok"] = (P["timestamp"] <= fund["calc_time"].max()).astype(float)

    meta = {
        "rows": int(len(P)), "cols": int(P.shape[1]),
        "range": [str(P.timestamp.min()), str(P.timestamp.max())],
        "cap_reason": "BTC 5m klines 상한(smt_divergence 가 BTC 고저를 필요로 함)",
        "funding_max": str(fund["calc_time"].max()),
        "tensor_joined_rows": joined,
        "tensor_note": "레짐 2열만 조인. proba/pct/signed/age 는 디스크 텐서가 누수수정 이전 판이라 제외(커밋 ee59951).",
        "causality": "인덱스 i 의 값은 봉 i 종가까지만 사용. 라벨 없음.",
        "signals": [f"ev_{B.ABBR[s]}" for s in B.SIGNALS],
        "triggers": [c for c in P.columns if c.startswith("trg_")],
    }
    return P, meta


def _selftest() -> None:
    """인과성 확인 — 뒤를 잘라도 앞쪽 값이 바뀌면 미래참조다.

    같은 시작점으로 **짧게** 한 번, **길게** 한 번 만들어 겹치는 구간(워밍업 뒤)을 대조한다.
    후행 분위/롤링만 쓰므로 완전히 일치해야 한다.
    """
    short, _ = build("2025-01-01", tmax="2025-06-30")
    long_, _ = build("2025-01-01", tmax="2025-09-30")
    cut = pd.Timestamp("2025-06-30")
    a = short[short.timestamp <= cut].reset_index(drop=True)
    b = long_[long_.timestamp <= cut].reset_index(drop=True)
    assert len(a) == len(b), f"행수 불일치 {len(a)} vs {len(b)}"
    cols = [c for c in a.columns if c.startswith(("ev_", "trg_")) or c in ("volexp", "er")]
    warm = 900                                   # 2016봉 분위의 min_periods=500 + 여유
    bad = []
    for c in cols:
        u, v = a[c].to_numpy()[warm:], b[c].to_numpy()[warm:]
        same = np.isclose(np.nan_to_num(u), np.nan_to_num(v), equal_nan=True)
        if not same.all():
            bad.append((c, int((~same).sum())))
    assert not bad, f"절단에 따라 값이 변한다(미래참조 의심): {bad}"
    assert set(a["ev_taker"].unique()) <= {-1.0, 0.0, 1.0}
    print(f"selftest OK — {len(cols)}개 컬럼이 절단 불변, 겹침 {len(a) - warm:,}행")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--out", default=None,
                    help="산출 디렉토리(기본 OUT). 기간을 바꿔 만들 때는 반드시 다르게 준다 --"
                         "덮어쓰면 그 패널로 낸 기존 결과가 재현 불가가 된다")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return 0
    P, meta = build(a.start)
    out = Path(a.out) if a.out else OUT
    meta["start"] = a.start
    out.mkdir(parents=True, exist_ok=True)
    P.to_parquet(out / "panel_5m.parquet", index=False)
    cov = pd.DataFrame({"column": P.columns, "notna": P.notna().sum().to_numpy(),
                        "nonzero": [int((P[c] != 0).sum()) if pd.api.types.is_numeric_dtype(P[c]) else -1
                                    for c in P.columns]})
    cov.to_csv(out / "coverage.csv", index=False)
    (out / "README.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"저장 {out} — {meta['rows']:,}행 × {meta['cols']}열 {meta['range']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
