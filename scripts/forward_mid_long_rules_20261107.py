#!/usr/bin/env python3
"""달러·순유동성 Z · 코인베이스 프리미엄 전진 판정 (동결 docs/prereg/liq_dollar_netliq_rule_frozen_20261015.json · short_fresh_coinbase_premium_frozen_20261015.json).

규칙은 동결본 그대로다. 판정 전에는 건수만 보여 주고 적중은 계산하지 않는다.
    python scripts/forward_mid_long_rules_20261107.py --status
    python scripts/forward_mid_long_rules_20261107.py --dry      # 전진 전 구간만 -- 연구 기록 재현 대조
    python scripts/forward_mid_long_rules_20261107.py run        # 조건 충족 뒤 1회 (원천을 새로 받은 뒤)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_short_fresh_20261015 as C  # noqa: E402
import research_trend_mid_long_20261015 as R  # noqa: E402

MAIN_TMP = Path("/home/kbj20/crypto-scalping/tmp")   # 원천 캐시는 메인 체크아웃 tmp
R.OUT, R.RAW = MAIN_TMP / "trend_mid_long_20261015", MAIN_TMP / "trend_mid_long_20261015" / "raw"
C.OUT, C.RAW = MAIN_TMP / "short_fresh_20261015", MAIN_TMP / "short_fresh_20261015" / "raw"
FWD = pd.Timestamp("2026-10-15")
OUT = MAIN_TMP / "forward_mid_long_20261107"


def dollar_signal():
    lp = R.load_prices()
    z = R.liq_score(*R.liq_inputs(lp.index))
    return lp, np.sign(z).where(z.abs() >= 1.0).replace(0, np.nan)


def premium_signal():
    cb, bn = C._btc_closes()
    lp = R.load_prices()
    z = C.cb_premium_z(cb, bn).reindex(lp.index)
    return lp, np.sign(z).where(z.abs() >= 1.0).replace(0, np.nan)


def block_ci(x, block, seed=0, B=1000):
    x = np.asarray(x, float)
    k, st = int(np.ceil(len(x) / block)), np.arange(len(x) - block + 1)
    rng = np.random.default_rng(seed)
    v = [np.concatenate([x[i:i + block] for i in rng.choice(st, k)])[:len(x)].mean() for _ in range(B)]
    return [float(np.quantile(v, .025)) * 100, float(np.quantile(v, .975)) * 100]


def score(lp, sig, h, a, b, coins=("ETH", "BTC")):
    out = {}
    for c in coins:
        hit, _ = R._hits(lp, sig, h, [c], a, b)
        x = hit[c].dropna()
        out[c] = {"hit": f"{int(x.sum())}/{len(x)}", "rate": float(x.mean() * 100) if len(x) else None, "ci95": block_ci(x, h) if len(x) >= 3 * h else None}
    return out


def status():
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    _, s1 = dollar_signal()
    n1 = int(s1.loc[FWD:].notna().sum())
    _, s2 = premium_signal()
    n2 = int(s2.loc[FWD:].notna().sum())
    print(f"오늘 {today.date()} · 달러 Z k=1 결정일 {n1}/180(그 뒤 56일) · 코인베이스 예측일 {n2}/120(그 뒤 7일). 원천 끝 {s1.index.max().date()} -- 적중은 계산 안 함")
    return n1, n2


def dry():
    """전진 전 구간 연구 수치 재현. 재현이 깨지면 판정 코드를 믿으면 안 된다."""
    lp, s1 = dollar_signal()
    d = score(lp, s1, 56, "2018", "2026-10-14")
    lp2, s2 = premium_signal()
    c = score(lp2, s2, 7, "2024-01-01", "2026-10-14")
    print(json.dumps({"dollar_z_56d_k1_2018": d, "연구 기록": "ETH 75.4 · BTC 69.9", "cb_premium_7d_2024": c, "연구 기록2": "ETH 65 · BTC 59"}, ensure_ascii=False, indent=1))
    assert abs(d["ETH"]["rate"] - 75.4) < 1.0 and abs(d["BTC"]["rate"] - 69.9) < 1.0, d
    assert abs(c["ETH"]["rate"] - 65) < 1.5 and abs(c["BTC"]["rate"] - 59) < 1.5, c


def run():
    n1, n2 = status()
    OUT.mkdir(parents=True, exist_ok=True)
    lp, s1 = dollar_signal()
    last = lp.index.max()
    if n1 >= 180 and s1.loc[FWD:].dropna().index[179] + pd.Timedelta(days=56) <= last and not (OUT / "dollar_z.json").exists():
        r = score(lp, s1, 56, FWD, last)
        r["pass"] = all(r[c]["rate"] >= 60 and r[c]["ci95"] and r[c]["ci95"][0] > 50 for c in ("ETH", "BTC"))
        (OUT / "dollar_z.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print("달러 Z", r)
    else:
        print("달러 Z: 조건 미충족이거나 이미 판정함")
    lp2, s2 = premium_signal()
    if n2 >= 120 and s2.loc[FWD:].dropna().index[119] + pd.Timedelta(days=7) <= lp2.index.max() and not (OUT / "cb_premium.json").exists():
        r = score(lp2, s2, 7, FWD, lp2.index.max())
        r["pass"] = all(r[c]["rate"] >= 60 and r[c]["ci95"] and r[c]["ci95"][0] > 50 for c in ("ETH", "BTC"))
        (OUT / "cb_premium.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print("코인베이스", r)
    else:
        print("코인베이스: 조건 미충족이거나 이미 판정함")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--status" in a:
        status()
    elif "--dry" in a:
        dry()
    elif "run" in a:
        run()
    else:
        sys.exit(__doc__)
