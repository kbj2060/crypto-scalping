#!/usr/bin/env python3
"""실계좌 왕복 원장의 **실현 수수료**를 bp 로 — 섀도우 시뮬이 아니라 실제 지불액 (2026-09-14).

사용자 *"비용 쪽으로 연구해줘"*. 이 저장소의 비용 수치는 전부 섀도우 워커(가상주문) 실측이었다.
여기서는 `data/live/account_round_trips.jsonl`(09-12 VWAP 무결성 수리본)의 `commission` 을 쓴다.

용도는 **회계 감사**다 — CLAUDE.md 가 저장 원장 replay 를 diagnostic/accounting/historical
reproduction 전용으로 허용한다. 승격·모델선택·live 후보 성과 근거로 쓰지 않는다.

왕복 수수료(bp) = commission / (진입VWAP × qty_in) × 1e4  — commission 은 양다리 합계다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

RNG_ = np.random.default_rng(20260914)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data/live/account_round_trips.jsonl"
REFS = [(4.00, "메이커 양다리(수수료 하한)"), (5.52, "peg 양다리(섀도우 실측)"),
        (5.88, "static진입+peg청산"), (7.80, "peg진입+테이커청산"), (10.00, "테이커 양다리")]
# peg 진입 배포 2026-09-12 · peg 청산 배포 2026-09-13 (KST). 청산시각 UTC 기준 경계.
PEG_DEPLOY = pd.Timestamp("2026-09-12 00:00:00")


def selftest() -> None:
    # 왕복 수수료 bp 환산: 명목 10,000 에 수수료 8 이면 8bp
    assert abs(8.0 / 10_000.0 * 1e4 - 8.0) < 1e-9
    # 역산 메이커 비중: 수수료 4bp 면 100%, 10bp 면 0%, 7bp 면 50%
    inv = lambda c: (10.0 - c) / (10.0 - 4.0)
    assert abs(inv(4.0) - 1.0) < 1e-9 and abs(inv(10.0)) < 1e-9 and abs(inv(7.0) - 0.5) < 1e-9
    print("selftest OK")


def main() -> int:
    selftest()
    if "--selftest" in sys.argv: return 0
    d = pd.DataFrame([json.loads(l) for l in LEDGER.open()])
    d = d[d.closed == True].copy()                                   # noqa: E712
    d["notional"] = d.entry_price * d.qty_in
    d["cost_bp"] = d.commission / d["notional"] * 1e4
    d["t"] = pd.to_datetime(d.exit_time, unit="ms")
    ver = (d.price_basis == "leg_vwap") & (d.pnl_check_bp.abs() < 1.0)
    print(f"왕복 {len(d)}건 · 회계항등식 검증통과 {int(ver.sum())}건 · "
          f"{d.t.min():%Y-%m-%d} ~ {d.t.max():%Y-%m-%d} · 명목 중앙 ${d['notional'].median():,.0f}")
    c = d.cost_bp
    print(f"\n⭐실현 왕복 수수료 중앙 **{c.median():.2f}bp** · 평균 {c.mean():.2f} · "
          f"p10 {c.quantile(.1):.2f} · p90 {c.quantile(.9):.2f} · [{c.min():.2f}, {c.max():.2f}]")
    m = (10.0 - c) / (10.0 - 4.0)
    print(f"역산 메이커 체결 비중: 중앙 {np.median(m)*100:.0f}% · 테이커 양다리(>9.5bp) {int((c>9.5).sum())}건"
          f" ({(c>9.5).mean()*100:.0f}%)")
    print(f"\n{'참고선':<28}{'왕복bp':>8}{'68왕복 절감':>14}{'순손익 변화':>18}")
    net = d.net_pnl.sum()
    for tgt, lab in REFS:
        save = ((d.cost_bp - tgt).clip(lower=0) / 1e4 * d["notional"]).sum()
        print(f"{lab:<28}{tgt:>8.2f}{save:>14.2f}{f'{net:+.2f} → {net+save:+.2f} ({save/net*100:+.0f}%)':>18}")
    print(f"\n주별")
    for k, s in d.groupby(d.t.dt.to_period("W").astype(str)):
        print(f"  {k}  n={len(s):>3}  중앙 {s.cost_bp.median():>5.2f}bp  "
              f"테이커양다리 {(s.cost_bp>9.5).mean()*100:>4.0f}%")
    print(f"\n합계: 실현 {d.realized_pnl.sum():+.2f} · 수수료 {-d.commission.sum():+.2f} · 순 {net:+.2f} USDT")

    # --- peg 배포 전후 ---
    pre, post = d[d.t < PEG_DEPLOY], d[d.t >= PEG_DEPLOY]
    print(f"\n{'='*72}\npeg 배포(09-12/13) 전후")
    print(f"{'구간':<22}{'n':>4}{'중앙bp':>9}{'평균bp':>9}{'테이커양다리':>12}{'메이커비중':>11}")
    for tag, s_ in (("배포 전 (~09-11)", pre), ("배포 후 (09-12~)", post)):
        if not len(s_): continue
        c_ = s_.cost_bp
        print(f"{tag:<22}{len(s_):>4}{c_.median():>9.2f}{c_.mean():>9.2f}"
              f"{(c_>9.5).mean()*100:>11.0f}%{np.median((10.0-c_)/6.0)*100:>10.0f}%")
    if len(post) >= 2:
        boot = np.array([np.median(RNG_.choice(post.cost_bp.values, len(post))) for _ in range(4000)])
        print(f"\n배포 후 중앙의 부트스트랩 95% CI: [{np.percentile(boot,2.5):.2f}, {np.percentile(boot,97.5):.2f}]"
              f"  ⚠️n={len(post)} — 측정이지 판정이 아니다")
        print(f"배포 후 왕복별: " + " · ".join(f"{v:.2f}" for v in sorted(post.cost_bp.values)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
