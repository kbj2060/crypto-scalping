"""횡보/추세 분별(앞으로 4시간 안에 가장 가까운 지지~저항 띠를 종가로 벗어날 확률)의 최신값 (2026-11-02).

⑤ 예측 줄의 «횡보·추세» 칸이 읽는다. 값은 요청 때 계산하지 않는다 -- 전진 기록 크론(scripts/live_regime_band_cron_20261102.py, 매 시 02분)이
data/live/regime_band_forward.jsonl 에 쌓은 줄의 코인별 마지막 행만 돌려준다(파일 꼬리 32KB 만 읽는다).
연구: docs/experiments/regime_acc_20261102.md · regime_zeroshot_20261102.md (4코인 시험 68.7% · 미학습 5코인 제로샷 66.6%).

    python dashboard/regime_band.py --selftest
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

LOG = Path(__file__).resolve().parents[1] / "data" / "live" / "regime_band_forward.jsonl"
TAIL_BYTES = 32768


def regime_band_payload(log: Path = LOG) -> dict:
    """{"ok", "coins": {eth|btc|sol|xrp: {decided_at_ms, p_trend, regime, confidence, support, resistance}}}. 기록이 없으면 ok=False."""
    if not log.exists():
        return {"ok": False, "reason": "기록 없음(전진 기록 크론 대기)"}
    with log.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - TAIL_BYTES))
        lines = f.read().decode("utf-8", "replace").splitlines()
    if size > TAIL_BYTES:
        lines = lines[1:]                               # 잘린 첫 줄은 버린다
    coins: dict = {}
    for ln in lines:
        try:
            r = json.loads(ln)
            coins[str(r["coin"]).lower()] = {
                "decided_at_ms": int(pd.Timestamp(r["ts"], tz="UTC").timestamp() * 1000),
                "p_trend": float(r["p_trend"]), "regime": r["regime"], "confidence": float(r["confidence"]),
                "support": float(r["support"]), "resistance": float(r["resistance"]),
            }
        except (ValueError, KeyError, TypeError):
            continue                                    # 깨진 줄은 건너뛴다(마지막 정상 행이 이긴다)
    return {"ok": True, "coins": coins} if coins else {"ok": False, "reason": "읽을 수 있는 기록 없음"}


def _selftest() -> None:
    d = Path(tempfile.mkdtemp())
    assert regime_band_payload(d / "none.jsonl")["ok"] is False
    p = d / "f.jsonl"
    rows = [dict(ts="2026-11-01 19:00:00", coin=c, p_trend=0.3, regime="횡보", confidence=0.4, support=1.0, resistance=2.0) for c in ("ETH", "XRP")]
    rows.append(dict(ts="2026-11-01 20:00:00", coin="ETH", p_trend=0.9, regime="추세", confidence=0.8, support=3.0, resistance=4.0))   # 같은 코인은 마지막 행이 이긴다
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n{깨진 줄\n")
    out = regime_band_payload(p)
    assert out["ok"] and out["coins"]["eth"]["p_trend"] == 0.9 and out["coins"]["xrp"]["p_trend"] == 0.3, out
    assert out["coins"]["eth"]["decided_at_ms"] == 1793563200000, out["coins"]["eth"]            # 2026-11-01 20:00 UTC
    big = d / "big.jsonl"                                                                          # 꼬리 32KB 보다 큰 파일: 잘린 첫 줄을 버려도 마지막 행은 살아야 한다
    big.write_text("".join(json.dumps(dict(rows[0], ts=f"2026-10-0{1 + i % 9} 00:00:00", coin="BTC", p_trend=i / 1000)) + "\n" for i in range(600)))
    assert big.stat().st_size > TAIL_BYTES and regime_band_payload(big)["coins"]["btc"]["p_trend"] == 0.599
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
