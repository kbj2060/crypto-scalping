"""오메가4.6.1 섀도우 원장 성과 + **배리어 맞춘** 대조군 (2026-09-12).

1판 결함: 귀무가 배리어 없이 원시 수익을 썼다. 실제는 TP 7.5%/SL 4.0% 로 잘리는데
귀무는 34일짜리 무제한 손실을 그대로 먹어 중앙 −43% 라는 비현실적 값이 나왔다.
대조군은 실제 코드 경로를 덮어야 한다 -- 여기서는 같은 배리어·같은 보유상한.

ETH 행은 take_profit/stop_loss 가 0.0 으로 기록됐지만, CLAUDE.md 포지션피쳐 계약에
따르면 라이브 ETH 는 atr_pct 가 floor 미만인 구간이 99.7% 라 실제 배리어가 7.5%/4.0%
다(실현 gross 가 +7.5~9.5 / −3.7~−5.7 로 그 값과 맞는다). 같은 값을 쓴다.
"""
import json, sys, statistics as st
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

TP, SL = 0.075, 0.040
SP = Path(__file__).resolve().parents[1] / "tmp" / "sizing_counterfactual_20260912"
cl = [r for r in (json.loads(l) for l in open(Path(__file__).resolve().parents[1] / "data" / "live" / "trade_journal.jsonl") if l.strip())
      if r.get("pnl_frac") is not None]

px = {}
for s in ("ETHUSDT", "SOLUSDT", "BTCUSDT"):
    b = json.loads((SP / f"{s}_1h.json").read_text())
    px[s] = (np.array([x[0] for x in b], dtype=np.int64), np.array([x[1] for x in b]),
             np.array([x[2] for x in b]), np.array([x[3] for x in b]))   # ms, open, high, low

def P(x): return datetime.fromisoformat(str(x)).astimezone(timezone.utc)

for r in cl:
    r["_sym"] = r.get("symbol") or "ETHUSDT"
    r["_o"], r["_c"] = P(r["actual_opened_at"]), P(r["actual_closed_at"])
    r["_hold_h"] = max(1, round((r["_c"] - r["_o"]).total_seconds() / 3600))
    r["_sgn"] = 1.0 if r["side"] == "LONG" else -1.0
    r["_notional"] = float(r.get("notional_exposure") or 1.0)
    r["_fee"] = float(r.get("fee_cost_frac") or 0.0)

def barrier_move(sym, i0, hold_h, sgn):
    """i0 봉 시가에 들어가 TP/SL 먼저 닿는 쪽으로 나간다. 둘 다 안 닿으면 만기 종가.
    같은 봉에서 양쪽이 닿으면 **SL 우선**(보수적) -- 어느 쪽이 먼저인지 시간봉은 모른다."""
    t, o, h, l = px[sym]
    n = len(t)
    if i0 + 1 >= n: return None
    entry = o[i0]
    end = min(i0 + hold_h, n - 1)
    for i in range(i0, end + 1):
        up, dn = h[i] / entry - 1.0, l[i] / entry - 1.0
        gain, loss = (up, dn) if sgn > 0 else (-dn, -up)
        if loss <= -SL: return -SL
        if gain >= TP: return TP
    return sgn * (o[end] / entry - 1.0)

def stats(moves, notional, fees):
    acct = [m * n - f for m, n, f in zip(moves, notional, fees)]
    eq, peak, mdd = 1.0, 1.0, 0.0
    for x in acct:
        eq *= (1 + x); peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
    return dict(n=len(acct), win=sum(1 for x in acct if x > 0) / len(acct),
                move=st.mean(moves) * 100, per=st.mean(acct) * 100,
                total=(eq - 1) * 100, mdd=mdd * 100)

notional = [r["_notional"] for r in cl]; fees = [r["_fee"] for r in cl]
real = stats([r["gross_return_frac"] for r in cl], notional, fees)
lo = min(r["_o"] for r in cl); hi = max(r["_c"] for r in cl)
print(f"기간 {lo:%Y-%m-%d} ~ {hi:%Y-%m-%d} ({(hi-lo).days}일) · 청산 {len(cl)}건 · 배리어 TP {TP:.1%} / SL {SL:.1%}\n")
print(f"실제  승률 {real['win']:.0%} · 가격변동 {real['move']:+.2f}% · 건당계좌 {real['per']:+.2f}% · "
      f"**누적 {real['total']:+.1f}%** · MDD {real['mdd']:.1f}%")

rng = np.random.default_rng(0); B = 2000
tot, mv, wr = [], [], []
for _ in range(B):
    moves = []
    for r in cl:
        t = px[r["_sym"]][0]
        i0 = int(rng.integers(0, max(1, len(t) - r["_hold_h"] - 1)))
        m = barrier_move(r["_sym"], i0, r["_hold_h"], r["_sgn"])
        moves.append(r["gross_return_frac"] if m is None else m)
    x = stats(moves, notional, fees); tot.append(x["total"]); mv.append(x["move"]); wr.append(x["win"])
tot, mv, wr = map(np.array, (tot, mv, wr))
p = float((tot >= real["total"]).mean())
print(f"\n같은측면·같은배리어 귀무 (진입 시각만 무작위, B={B})")
print(f"   승률   중앙 {np.median(wr):.0%}      (실제 {real['win']:.0%})")
print(f"   가격변동 중앙 {np.median(mv):+.2f}%   (실제 {real['move']:+.2f}%)")
print(f"   누적   중앙 {np.median(tot):+.1f}% · 5~95% [{np.percentile(tot,5):+.1f}, {np.percentile(tot,95):+.1f}]")
print(f"   ⇒ p = {p:.3f}   {'대조군 초과' if p < 0.05 else '**대조군과 구분 불가**'}")

print("\n측면별")
for side in ("LONG", "SHORT"):
    sub = [r for r in cl if r["side"] == side]
    x = stats([r["gross_return_frac"] for r in sub], [r["_notional"] for r in sub], [r["_fee"] for r in sub])
    print(f"   {side:6s} n={x['n']:>2} 승률 {x['win']:.0%} · 가격변동 {x['move']:+.2f}% · 건당계좌 {x['per']:+.2f}%")

print("\n크기 반사실 (방향·시점·청산 그대로, 명목만 균일)")
uni = st.mean(notional)
for label, nn in (("실제 명목", notional), (f"균일 {uni:.3f}", [uni] * len(cl))):
    x = stats([r["gross_return_frac"] for r in cl], nn, fees)
    print(f"   {label:14s} 건당 {x['per']:+.2f}% · 누적 {x['total']:+7.1f}% · MDD {x['mdd']:6.1f}%")
print("\n자산별 명목 평균")
for s in ("ETHUSDT", "SOLUSDT", "BTCUSDT"):
    sub = [r for r in cl if r["_sym"] == s]
    x = stats([r["gross_return_frac"] for r in sub], [r["_notional"] for r in sub], [r["_fee"] for r in sub])
    print(f"   {s:9s} n={x['n']:>2} 명목평균 {st.mean([r['_notional'] for r in sub]):.3f} · "
          f"가격변동 {x['move']:+.2f}% · 누적 {x['total']:+.1f}%")
