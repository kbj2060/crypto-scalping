"""라벨 상한 스크린 — "이 라벨을 100% 맞히면 얼마 버는가".

지금까지 라벨은 학습 가능성(정확도)으로만 평가됐다. 그 결과 `terminal_giveback` 처럼
**잘 배우고 돈을 잃는** 라벨을 골랐다(정확도 0.6947, 결정층 -19.5pp). 상한을 먼저 재면
그런 라벨을 학습 전에 떨어뜨릴 수 있고, 모델이 필요 없어 싸다.

배리어 판정은 `greedy_replay` 와 같은 **종가** 규약을 쓴다(라이브의 intrabar 고저가 아니다).
오라클을 채점할 리플레이가 종가 기준이므로 라벨도 같아야 한다 -- 규약이 어긋나면 라벨이
"TP 먼저"라 해도 리플레이가 동의하지 않는다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import greedy_replay, prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
PRED = OUT / "preds"
DIR_LBL = ROOT / ("tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629"
                  "/label_contracts/zigzag_action_labels_20260531")
POLICIES = [(0.075, 0.040), (0.080, 0.030), (0.030, 0.015)]
H = 4032                       # 14일 시간청산. 실측 보유 q75 는 4~6일이라 여유가 있다.
FIX_MARGIN, FIX_LEVERAGE = 0.225, 2.0     # 진입 선택만 비교하려고 사이징을 고정한다
sidecar._predict_exit_prob_one = lambda *a, **k: 0.0   # 헤드 비활성(실측 0/59,815)


def _tb_side(close, opn, tp, sl, slip):
    """각 봉에서 롱/숏 각각 배리어 승패를 매기고, 이기는 쪽을 side 로 돌려준다."""
    n = len(close)
    win = np.zeros((n, 2), dtype=np.int8)          # [:,0]=롱 [:,1]=숏
    for col, s in ((0, 1), (1, -1)):
        for i in range(n - 2):
            E = opn[i + 1] * (1 + slip if s > 0 else 1 - slip)
            seg = close[i + 1:min(i + 1 + H, n)]
            if s > 0:
                w, l = seg >= E * (1 + tp) / (1 - slip), seg <= E * (1 - sl) / (1 - slip)
            else:
                w, l = seg <= E * (1 - tp) / (1 + slip), seg >= E * (1 + sl) / (1 + slip)
            iw = int(w.argmax()) if w.any() else 1 << 30
            il = int(l.argmax()) if l.any() else 1 << 30
            win[i, col] = 1 if iw < il else 0
    return np.where(win[:, 0] == 1, 1, np.where(win[:, 1] == 1, -1, 0)).astype(np.int64)


def _mfe_side(close, opn, slip, cost):
    """H봉 내 MFE-|MAE| 가 더 큰 쪽. 배리어와 무관한 '경로 품질' 라벨."""
    n = len(close)
    out = np.zeros(n, dtype=np.int64)
    for i in range(n - 2):
        E = opn[i + 1]
        seg = (close[i + 1:min(i + 1 + H, n)] - E) / E
        if len(seg) == 0:
            continue
        up, dn = seg.max(), seg.min()
        gl, gs = up + dn, -dn - up          # 롱 = MFE-|MAE|, 숏 = 대칭
        out[i] = 1 if (gl > gs and up > cost) else (-1 if (gs > gl and -dn > cost) else 0)
    return out


def _fwd_side(close, opn, cost):
    n = len(close)
    out = np.zeros(n, dtype=np.int64)
    j = np.minimum(np.arange(n) + H, n - 1)
    r = (close[j] - opn[np.minimum(np.arange(n) + 1, n - 1)]) / opn[np.minimum(np.arange(n) + 1, n - 1)]
    out[r > cost], out[r < -cost] = 1, -1
    return out


def _zigzag_side(frame):
    fs = sorted(DIR_LBL.glob("zigzag_action_labels_*.csv"))
    lb = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    if "timestamp" not in lb.columns:
        return None, 0
    lb["timestamp"] = pd.to_datetime(lb["timestamp"])
    m = frame[["timestamp"]].merge(lb[["timestamp", "zigzag_action"]].drop_duplicates("timestamp"),
                                   on="timestamp", how="left")
    a = m["zigzag_action"].to_numpy()
    return np.where(a == 1, 1, np.where(a == 2, -1, 0)).astype(np.int64), int(np.isfinite(a.astype(float)).sum())


def _arm(base, side, tp, sl, frame, fee, slip, device):
    """side 배열을 결정으로 심고 배포 실행모델 그대로 리플레이한다."""
    dec = base["dec"].copy()
    dec["side"] = side
    dec["action"] = np.where(side > 0, omega.ACTION_LONG,
                             np.where(side < 0, omega.ACTION_SHORT, omega.ACTION_CASH))
    dec["notional_exposure"] = np.where(side != 0, FIX_MARGIN * FIX_LEVERAGE, 0.0)
    dec["take_profit"], dec["stop_loss"] = tp, sl
    comp = dict(base)
    comp["dec"] = dec
    comp["margin"] = np.full(len(dec), FIX_MARGIN)
    comp["leverage"] = np.full(len(dec), FIX_LEVERAGE)
    _s, lg = greedy_replay(frame, {"zig075": comp}, fee=fee, slip=slip,
                           cost_mult=retest.COST_MULT, device=device)
    if lg.empty:
        return {"pnl": 0.0, "mdd": 0.0, "n": 0, "wr": 0.0, "h1": 0.0, "h2": 0.0}
    r = lg["trade_return"].to_numpy(float)
    eq = np.cumprod(1 + r)
    t = pd.to_datetime(lg["entry_timestamp"], errors="coerce")
    h = [float((np.prod(1 + lg.loc[m, "trade_return"].to_numpy(float)) - 1) * 100) if m.any() else 0.0
         for m in (t < pd.Timestamp("2026-06-01"), t >= pd.Timestamp("2026-06-01"))]
    return {"pnl": float((eq[-1] - 1) * 100), "mdd": float((eq / np.maximum.accumulate(eq) - 1).min() * 100),
            "n": int(len(r)), "wr": float((r > 0).mean() * 100), "h1": h[0], "h2": h[1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-08-30")
    args = ap.parse_args()
    device = parent._device("cpu")
    fee, slip = omega._load_fee_slip()
    frame = retest.load_frame_current(args.start, args.end)
    close = frame["close"].to_numpy(float)
    opn = frame["open"].to_numpy(float)
    cost = 2 * (fee + slip)
    cfg = dict(retest.COMPONENTS["zig075"])
    base = prepare_component(frame, PRED / "zig075" / "predictions_q075.csv", cfg, device)
    print(f"[프레임] {len(frame):,}봉  fee={fee} slip={slip}  왕복비용 {cost*100:.2f}%  "
          f"시간청산 H={H}봉({H*5/60/24:.0f}일)", flush=True)

    zz, nzz = _zigzag_side(frame)
    print(f"[지그재그 라벨] 매칭 {nzz:,}봉 / {len(frame):,}", flush=True)
    mfe = _mfe_side(close, opn, slip, cost)
    fwd = _fwd_side(close, opn, cost)
    dep = pd.to_numeric(base["dec"]["side"], errors="raise").to_numpy(np.int64)
    dep = np.where(omega._active(base["dec"]), dep, 0)

    rows = []
    for tp, sl in POLICIES:
        need = sl / (tp + sl) * 100
        orc = _tb_side(close, opn, tp, sl, slip)
        cands = {"① 오라클(이 배리어에서 이기는 쪽)": orc, "② MFE-|MAE| 우세": mfe,
                 "③ H봉 후 수익 부호": fwd, "④ 지그재그(현행)": zz, "⑤ 배포 모델 실제 결정": dep}
        print(f"\n=== 청산정책 TP {tp*100:.1f}% / SL {sl*100:.1f}%  (RR {tp/sl:.2f} · "
              f"손익분기 정밀도 {need:.1f}%) ===", flush=True)
        print(f"{'라벨':30s} {'진입봉':>8s} {'거래':>5s} {'PnL':>10s} {'MDD':>8s} {'WR':>6s} {'전반':>9s} {'후반':>9s}")
        for nm, sd in cands.items():
            if sd is None:
                continue
            m = _arm(base, sd, tp, sl, frame, fee, slip, device)
            rows.append({"tp": tp, "sl": sl, "need": need, "label": nm, "signal_bars": int((sd != 0).sum()), **m})
            print(f"{nm:30s} {int((sd!=0).sum()):8,d} {m['n']:5d} {m['pnl']:+10.2f}% {m['mdd']:+7.2f}% "
                  f"{m['wr']:5.1f}% {m['h1']:+8.2f}% {m['h2']:+8.2f}%", flush=True)

    print("\n=== 열화 곡선 — 라벨 ②(MFE-|MAE|) 를 정확도 a 로만 맞히면 =====================")
    print("무작위 오염이라 실제 모델 오차(난이도와 상관)보다 낙관적이다 — 상한의 상한이다.")
    tp, sl = POLICIES[0]
    print(f"{'정확도':>7s} {'거래':>5s} {'실현WR':>7s} {'PnL 중앙':>11s} {'시드범위':>22s} {'후반 중앙':>10s}")
    for a in (1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4):
        ps, hs, ns, ws = [], [], [], []
        for sd_i in (615372041, 208844917, 933105268):
            rng = np.random.default_rng(sd_i)
            bad = rng.random(len(mfe)) >= a
            d = np.where(bad, -mfe, mfe)          # 오염 = 반대 방향으로 뒤집기
            m = _arm(base, d, tp, sl, frame, fee, slip, device)
            ps.append(m["pnl"]); hs.append(m["h2"]); ns.append(m["n"]); ws.append(m["wr"])
        print(f"{a:7.2f} {int(np.median(ns)):5d} {np.median(ws):6.1f}% {np.median(ps):+10.2f}% "
              f"[{min(ps):+9.2f},{max(ps):+9.2f}] {np.median(hs):+9.2f}%", flush=True)

    f = OUT / "label_oracle_ceiling.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    print(f"\n산출물: {f}")
    print("주의: 오라클은 결과로 선택된 상한이지 예측이 아니다. 라벨 간 **비교**와 "
          "현실→천장 비율에만 쓴다. 단일 구간·단일 시드 → 승격 근거 아님.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
