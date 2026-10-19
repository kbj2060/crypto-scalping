"""**호메로스 gym** — 배포된 진입·청산 스택 위에서 {홀드, 롱, 숏}만 정하는 에이전트용 환경 (2026-09-14).

사용자: *"이 강화모델 에이전트는 롱숏홀드의 행동만 갖고 진입 및 청산 로직을 따르며 학습하는거야."*
설계 `docs/eth_rl_gym_control_agent_design_20260914.md` §1·§4.

## 무엇을 재구현하지 않는가
체결·손절(봉내)·예산 사다리·만기·펀딩·세 조각 비용은 `research_fresh_forward_random_entry_stack_20260914.walk`
의 순서를 **그대로** 따르고, 상수·함수는 그 모듈과 배포 모듈에서 import 한다. 여기서 새로 짜는 건
«진입을 무작위가 아니라 행동으로 받는다» 한 줄뿐이다. `--selftest` 가 그 사실을 증명한다:
walk 와 같은 rng 순서로 행동을 뽑는 정책을 넣으면 계좌배수·거래수·손절수·사다리수가 1e-9 안에서 같다.

## semi-MDP
슬롯이 빈 봉에서만 결정한다. 보유 중에는 결정이 없다(청산은 배포 로직만). 보상은 청산 봉에서
`ln(청산 후 순자산 / 진입 직전 순자산)` 한 항(사다리 부분청산 포함). 홀드는 0.
🔴학습 창의 가격은 **탈드리프트**(O/H/L/C × exp(−μt), 봉 내부 구조 보존)할 수 있다 -- 평가는 실제 가격.
피쳐는 항상 실제 가격에서 만든다(에이전트는 실제 시장을 보되 보상만 초과분).

ponytail: 벡터화 안 함. walk 가 창(26k봉)당 0.1초라 정책 로짓을 창 전체에 한 번 계산해 두고
numpy 루프로 돌리면 충분하다(정책이 롤아웃 중 고정이고 관측에 포지션 상태가 없어서 가능).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import research_fresh_forward_random_entry_stack_20260914 as H  # noqa: E402
from scripts.live_eth_risk_sizing_policy_20260913 import exit_fraction_required, policy_leverage  # noqa: E402
from scripts.live_eth_trade_plan_20260913 import funding_cost_bp, leverage_setting  # noqa: E402
from scripts.live_manual_peg_entry_20260912 import STOP_LOSS_PCT  # noqa: E402

PANEL = ROOT / "data/materials/eth_signal_trigger_panel_ext_20260913/panel_5m.parquet"
HOLD, LONG, SHORT = 0, 1, 2
# 연속 재료(패널). ev_*/trg_* 이진 발동 플래그는 넣지 않는다(2026-09-12 사용자 지시).
MATERIAL_COLS = [
    "p_fast", "p_slow", "delta_z", "vol_z", "lower_wick_ratio", "upper_wick_ratio", "ret3_z",
    "atr_pct", "dem", "kalman_dev_z", "atr_pctile", "ret12", "pos_in_range12", "dist_lo12",
    "dist_hi12", "ret48", "pos_in_range48", "dist_lo48", "dist_hi48", "ret144", "pos_in_range144",
    "dist_lo144", "dist_hi144", "volexp", "compressed", "brk_z288_qv", "brk_z288_n",
    "vrev_down_atr", "vrev_up_atr", "vol_expand_ratio", "er"]
BTC_COLS = ["btc_ret12", "btc_ret48", "eth_btc_div"]
VOL_COLS = ["rv12", "rv48", "rv288", "park48"]           # klines 에서 여기서 계산
REGIME_COLS = [f"reg_eth_{k}" for k in (-1, 0, 1, 2)] + [f"reg_btc_{k}" for k in (-1, 0, 1, 2)]
TIME_COLS = ["hour_sin", "hour_cos", "wd_sin", "wd_cos"]
FAMILIES = {"material": MATERIAL_COLS, "vol": VOL_COLS, "regime": REGIME_COLS,
            "btc": BTC_COLS, "time": TIME_COLS}
# 확장 피쳐(research_eth_rl_gym_feature_expansion_20260914 가 만든 parquet + 군 목록). 있으면 군으로 등록한다.
EXT_PARQUET = ROOT / "data/research/eth_rl_gym_direction_20260914/ext_features.parquet"
EXT_FAMILIES = ROOT / "data/research/eth_rl_gym_direction_20260914/ext_families.json"
if EXT_FAMILIES.exists():
    import json as _json
    FAMILIES.update(_json.load(open(EXT_FAMILIES)))
DEFAULT_FAMILIES = ("material", "vol", "regime")        # BTC·시간대는 절제 팔로만(방향 증분 음수)


# ── 데이터 ─────────────────────────────────────────────────────────────────
def load_frame() -> pd.DataFrame:
    """klines(하네스와 같은 파일) 에 재료 패널을 timestamp 로 붙인다. 패널 밖 봉은 NaN."""
    d = H.load()
    p = pd.read_parquet(PANEL)
    drop = [c for c in p.columns if c.startswith(("ev_", "trg_")) or c in
            ("open", "high", "low", "close", "volume", "fund_ok")]
    p = p.drop(columns=drop)
    d = d.merge(p, on="timestamp", how="left")
    c = d.close.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.nan)
    s = pd.Series(lr)
    for w in (12, 48, 288):
        d[f"rv{w}"] = s.rolling(w).std().to_numpy()
    hl = np.log(d.high.to_numpy(float) / d.low.to_numpy(float)) ** 2
    d["park48"] = np.sqrt(pd.Series(hl).rolling(48).mean().to_numpy() / (4 * np.log(2)))
    for side in ("eth", "btc"):
        r = d[f"mt_regime_{side}"].to_numpy(float)
        for k in (-1, 0, 1, 2):
            d[f"reg_{side}_{k}"] = (r == k).astype(np.float32)   # NaN(패널 밖·2024 이전)은 전부 0
    h = d.timestamp.dt.hour.to_numpy(float) + d.timestamp.dt.minute.to_numpy(float) / 60
    wd = d.timestamp.dt.weekday.to_numpy(float)
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    d["wd_sin"], d["wd_cos"] = np.sin(2 * np.pi * wd / 7), np.cos(2 * np.pi * wd / 7)
    if EXT_PARQUET.exists():
        # 🔴**타임스탬프로 정렬**한다(위치 정렬 금지). klines 파일이 머신마다 길이가 달라
        # 위치로 붙이면 조용히 어긋난다 -- 2026-09-14 서버 이관에서 단언문이 실제로 잡았다.
        e = pd.read_parquet(EXT_PARQUET)
        keep = [c for c in e.columns if c == "timestamp" or c not in d.columns]
        d = d.merge(e[keep], on="timestamp", how="left")
        cov = float(np.isfinite(d[[c for c in keep if c != "timestamp"][0]].to_numpy(float)).mean())
        assert cov > 0.5, f"확장 피쳐 커버리지 {cov:.2f} -- 타임스탬프가 거의 안 겹친다"
    return d


def state_columns(families=DEFAULT_FAMILIES) -> list[str]:
    return [c for f in families for c in FAMILIES[f]]


def fit_normalizer(d: pd.DataFrame, cols: list[str], lo: int, hi: int) -> dict:
    """🔴정규화 통계는 **학습 창에서만** 낸다(전체표본 정규화 = 누수, RLv2 에서 실측)."""
    x = d[cols].to_numpy(np.float64)[lo:hi]
    med = np.nanmedian(x, axis=0)
    iqr = np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
    return {"cols": cols, "med": np.where(np.isfinite(med), med, 0.0),
            "iqr": np.where(np.isfinite(iqr) & (iqr > 1e-12), iqr, 1.0)}


def build_state(d: pd.DataFrame, norm: dict) -> np.ndarray:
    x = d[norm["cols"]].to_numpy(np.float64)
    z = (x - norm["med"]) / norm["iqr"]
    z = np.tanh(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0) / 2.0)
    return z.astype(np.float32)


def dedrift(d: pd.DataFrame, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """학습 창 [lo,hi) 의 평균 로그수익을 O/H/L/C 에서 뺀 가격. 봉 내부 비율은 보존된다."""
    c = d.close.to_numpy(float)
    mu = float(np.nanmean(np.diff(np.log(c[lo:hi]))))
    f = np.exp(-mu * (np.arange(len(d), dtype=float) - lo))
    return c * f, d.high.to_numpy(float) * f, d.low.to_numpy(float) * f


# ── 환경 ───────────────────────────────────────────────────────────────────
class DirectionGym:
    """walk 의 순서(사다리 → 손절 → 만기 → 진입)를 그대로 따르되 진입을 행동으로 받는다.

    `run(policy)` 의 `policy(i) -> (action, view)`: 빈 봉 i 에서 행동과 «측면 견해»를 돌려준다.
    view 는 지평 선택에만 쓰인다. 에이전트 경로에서는 view = 행동의 측면(인과적). selftest 는
    walk 와 같게 view 를 따로 뽑아 넣는다.
    """

    def __init__(self, d: pd.DataFrame, sm: dict, lo: int, hi: int, *, cap_x: float = H.CAP_X,
                 prices: tuple | None = None, entry_bp: float = H.ENTRY_BP,
                 peg_exit_bp: float = H.PEG_EXIT_BP, taker_exit_bp: float = H.TAKER_EXIT_BP):
        self.d, self.sm, self.lo, self.hi, self.cap = d, sm, lo, hi, float(cap_x)
        self.c, self.h, self.l = prices if prices is not None else (
            d.close.to_numpy(float), d.high.to_numpy(float), d.low.to_numpy(float))
        self.entry_bp, self.peg_bp, self.taker_bp = entry_bp, peg_exit_bp, taker_exit_bp

    def hold_bars(self, i: int, side: str) -> int | None:
        return H.pick_hold_bars(self.sm, i, side, self.cap)

    def run(self, policy) -> dict:
        c, hi_, lo_, sm, cap = self.c, self.h, self.l, self.sm, self.cap
        eq = H.START_EQUITY; peak = eq; mdd = 0.0
        pos = None; eq_entry = eq
        trades, decisions = [], []      # decisions: (i, action, reward, next_decision_i)
        stops = expiries = ladder_cuts = ladder_full = blocked = 0
        expo_sum = 0.0; expo_bars = 0; ruin = False
        pending = None                  # 진입 결정의 decisions 인덱스(보상은 청산 때 채운다)
        for i in range(self.lo, self.hi):
            if pos is not None:
                s, entry, qty, stop_px, end_i, lev, hb = pos
                expo_sum += qty * c[i] / max(eq, 1e-9); expo_bars += 1
                adverse = (entry - lo_[i]) / entry if s > 0 else (hi_[i] - entry) / entry
                hit = adverse >= STOP_LOSS_PCT
                if not hit and i < end_i:
                    mark = c[i]
                    unreal = qty * entry * s * (mark / entry - 1)
                    eq_now = eq + unreal
                    notion_now = qty * mark
                    m_now = float(sm[(hb * 5, "LONG" if s > 0 else "SHORT")][i])
                    if eq_now > 0 and m_now > 0:
                        need = exit_fraction_required(eq_now, m_now, notion_now,
                                                      hard_cap=cap)["required_fraction"]
                        if need > 0.01:
                            cut = min(1.0, need); closed = qty * cut
                            pnl = closed * entry * (s * (mark / entry - 1) - self.peg_bp / 1e4)
                            eq += pnl; ladder_cuts += 1; qty -= closed
                            if eq <= 0:
                                ruin = True; eq = 0.0; break
                            peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
                            if qty <= 1e-9:
                                ladder_full += 1
                                self._close(decisions, pending, eq, eq_entry, i)
                                trades.append({"i": end_i - hb, "exit_i": i, "side": s, "hb": hb,
                                               "lev": lev, "stopped": False, "r": 0.0,
                                               "logret": np.log(eq / eq_entry)})
                                pos = None; pending = None
                            else:
                                pos = (s, entry, qty, stop_px, end_i, lev, hb)
                            continue
                if hit:
                    fill_move = min(-STOP_LOSS_PCT - H.STOP_SLIP_MED_BP / 1e4, s * (c[i] / entry - 1))
                    cost = self.entry_bp + self.taker_bp; stops += 1
                elif i >= end_i:
                    fill_move = s * (c[i] / entry - 1)
                    cost = self.entry_bp + self.peg_bp; expiries += 1
                else:
                    continue
                held_min = 5 * (i - (end_i - hb))
                cost += funding_cost_bp(held_min, "LONG" if s > 0 else "SHORT")
                pnl = qty * entry * (fill_move - cost / 1e4)
                eq += pnl
                if eq <= 0:
                    ruin = True; eq = 0.0; break
                peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
                self._close(decisions, pending, eq, eq_entry, i)
                trades.append({"i": end_i - hb, "exit_i": i, "side": s, "hb": hb, "lev": lev,
                               "stopped": hit, "r": fill_move - cost / 1e4, "cost_bp": cost,
                               "logret": np.log(eq / eq_entry)})
                pos = None; pending = None
                continue
            # ── 빈 슬롯: 결정 ───────────────────────────────────────────────
            # walk 는 rng 를 먼저 소비하고 ok 를 본다 -- 정책을 먼저 부르고 ok 가 아니면 버린다.
            action, view = policy(i)
            if not sm["ok"][i]:
                continue
            k = len(decisions)
            decisions.append([i, action, 0.0, -1])
            if action == HOLD:
                continue
            side = "LONG" if action == LONG else "SHORT"
            hb = self.hold_bars(i, view or side)
            if hb is None or i + hb >= self.hi:
                continue
            s = 1.0 if action == LONG else -1.0
            m = float(sm[(hb * 5, side)][i])
            if not (m > 0):
                continue
            L = min(policy_leverage(m)["leverage"], cap)
            notional = eq * L
            lv = leverage_setting(cap_notional=eq * cap, equity=eq, current_notional=0.0)
            if lv.get("available") and notional > lv["max_notional"] + 1e-6:
                blocked += 1; continue
            entry = c[i]; qty = notional / entry
            stop_px = entry * (1 - STOP_LOSS_PCT) if s > 0 else entry * (1 + STOP_LOSS_PCT)
            pos = (s, entry, qty, stop_px, i + hb, L, hb)
            eq_entry = eq; pending = k
        # 열린 채 끝난 포지션은 보상 0(다음 결정 없음). 결정 전이의 다음 인덱스를 채운다.
        for k in range(len(decisions) - 1):
            decisions[k][3] = decisions[k + 1][0]
        n = len(trades)
        rets = np.array([t["r"] for t in trades]) if n else np.zeros(1)
        return {"trades": n, "stops": stops, "expiries": expiries, "ladder_cuts": ladder_cuts,
                "blocked_margin": blocked, "ladder_full": ladder_full, "equity": eq, "mult": eq / H.START_EQUITY, "mdd": mdd,
                "ruin": ruin, "expo_time_x": expo_sum / max(self.hi - self.lo, 1),
                "mean_expo_x": expo_sum / expo_bars if expo_bars else 0.0,
                "net_bp_mean": float(1e4 * rets.mean()), "_trades": trades,
                "_decisions": np.array(decisions, dtype=np.float64).reshape(-1, 4)}

    @staticmethod
    def _close(decisions, pending, eq, eq_entry, i):
        if pending is not None:
            decisions[pending][2] = float(np.log(max(eq, 1e-12) / eq_entry))
            decisions[pending][3] = i + 1        # 다음 결정은 청산 다음 봉


# ── 자체점검: walk 와 동일해야 한다 ───────────────────────────────────────────
def _walk_like_policy(gym: DirectionGym, rng, acc: float, p_entry: float):
    """walk 와 **같은 순서로 rng 를 소비**하는 정책. 조건 분기의 순서까지 같아야 한다."""
    c, sm, hi = gym.c, gym.sm, gym.hi

    def policy(i):
        if rng.random() >= p_entry:
            return HOLD, None
        s_view = "LONG" if rng.random() < 0.5 else "SHORT"
        hb = gym.hold_bars(i, s_view)
        if hb is None or i + hb >= hi:
            return HOLD, None
        truth = 1.0 if c[min(i + hb, len(c) - 1)] >= c[i] else -1.0
        s = truth if rng.random() < acc else -truth
        return (LONG if s > 0 else SHORT), s_view
    return policy


def selftest() -> None:
    import pickle
    d = H.load()
    cache = ROOT / "data/research/eth_rl_gym_safe_mae_cache_20260914.pkl"
    if cache.exists():
        sm = pickle.load(open(cache, "rb"))
    else:
        sm = H.safe_mae_series(d); cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(sm, open(cache, "wb"))
    ts = d.timestamp.to_numpy()
    for w0, w1 in (("2026-01-01", "2026-03-31"), ("2025-09-01", "2025-12-31")):
        lo = int(np.searchsorted(ts, np.datetime64(w0))); hi = int(np.searchsorted(ts, np.datetime64(w1 + "T23:59:59")))
        for acc in (0.5, 0.6):
            ref = H.walk(d, sm, lo, hi, acc=acc, p_entry=0.02, use_stop=True, cap_x=H.CAP_X,
                         use_ladder=True, selector=True, rng=np.random.default_rng(H.SEED))
            g = DirectionGym(d, sm, lo, hi)
            out = g.run(_walk_like_policy(g, np.random.default_rng(H.SEED), acc, 0.02))
            for k in ("stops", "ladder_cuts", "blocked_margin"):
                assert out[k] == ref[k], (w0, acc, k, out[k], ref[k])
            # walk 의 trades 는 사다리 부분청산마다 1건씩 세고 전량청산은 따로 안 센다
            assert ref["trades"] == out["trades"] - out["ladder_full"] + out["ladder_cuts"], \
                (ref["trades"], out["trades"], out["ladder_full"], out["ladder_cuts"])
            assert abs(out["mult"] - ref["mult"]) < 1e-9, (w0, acc, out["mult"], ref["mult"])
            assert abs(out["mdd"] - ref["mdd"]) < 1e-9
            assert abs(out["expo_time_x"] - ref["expo_time_x"]) < 1e-9
            # 결정 전이의 보상 합 = 로그 계좌배수 (사다리 부분청산 포함)
            dec = out["_decisions"]
            assert abs(dec[:, 2].sum() - np.log(out["mult"])) < 1e-9, (dec[:, 2].sum(), np.log(out["mult"]))
            print(f"OK {w0} acc={acc} 거래 {out['trades']} 손절 {out['stops']} 사다리 {out['ladder_cuts']} "
                  f"배수 {out['mult']:.6f} = walk · 결정 {len(dec):,}")
    # 상태 빌더: 인과성 절단검사 -- 앞을 잘라도 뒤 행이 안 변한다
    f = load_frame()
    cols = state_columns()
    norm = fit_normalizer(f, cols, lo - 50_000, lo)
    a = build_state(f, norm)
    b = build_state(f.iloc[: hi].reset_index(drop=True), norm)
    assert np.array_equal(a[:hi], b), "상태가 미래 행에 의존한다"
    assert a.shape[1] == len(cols) and np.isfinite(a).all()
    print(f"OK 상태 {a.shape} · 절단 불변 · 군 {DEFAULT_FAMILIES}")


if __name__ == "__main__":
    selftest()
