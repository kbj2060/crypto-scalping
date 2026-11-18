#!/usr/bin/env python3
"""ETH 지지·저항 접근·돌파 «레벨 5분 기울기»(배포 대기 A19) 실시간 워커 (2026-11-18).

  python -u scripts/live_sr_lean_20261118.py                 # 서버 상시 실행(scripts/ops/supervisor_sr_lean.sh)
  python scripts/live_sr_lean_20261118.py --selftest         # 네트워크 없이: 1초 묶음 · 번호 빠짐 · 미래 채움 불변
  python scripts/live_sr_lean_20261118.py --export           # 얼린 모델 저장(학습 2024~25 재적합 — 시드 고정 · tmp/sr_intensity_20261118/ml_train.parquet 필요)
  python scripts/live_sr_lean_20261118.py --replay 2026-11-15T18:00 2026-11-16T00:00    # 연구 빌더와 대조(1초 묶음 일 파일 필요 · 실시간 경로 그대로)

무엇: 1h·4h·24h 굴림 고저의 접근(0.1σ 안)·돌파 5초 뒤에 «다음 5분 안에 어느 쪽 0.1σ 를 먼저 닿나»를 얼린 앙상블로 채점한다.
  연구·판정: docs/experiments/sr_ml_rl_20261118.md(2023 확인 통과 AUC .531 · 양끝 10% 56.4% 대 45.3%) · 전진 판정 12-17(docs/prereg/sr_ml_forward_20261118.md).
  표시 전용이다 — 매매 근거 아님(진입 밴딧 비용 뒤 −7bp). 주문·계정 호출 없음.
학습↔서빙 같은 코드: 사건(`candidates`)·피쳐(`one`)는 연구 스크립트 함수를 그대로 부른다. 여기서 새로 하는 것은 ① @aggTrade → 1초 묶음(Vision 일 파일과 같은 정의)
  ② 형성 중인 분에서 문턱을 넘은 초 찾기 ③ 결정 초(사건 + 5초)가 닫히면 채점 ④ 300초 뒤 결과 기록 뿐이다.
자료: 웹소켓 `/market/ws/ethusdt@aggTrade`(묶음 번호 a 가 1씩 는다 → 빠지면 REST aggTrades fromId 로 메움 · 분당 호출 상한) · 시작/구멍 뒤 1분봉은 REST klines 1회(가중치 10).
산출: data/live/sr_lean/ethusdt.jsonl(사건·결과 — 대시보드 /api/sr-lean 이 꼬리만 읽는다) · ethusdt_feat.jsonl(피쳐 87개 — Vision 재구성 대조용) · ethusdt_state.json(10초마다 생존 표시).
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_sr_ml_rl_20261118 as RL  # noqa: E402
R = RL.R

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "dashboard" / "sr_lean_model_20261118.joblib"
OUTD = ROOT / "data" / "live" / "sr_lean"
WS = "wss://fstream.binance.com/market/ws/ethusdt@aggTrade"          # /ws/ 경로는 aggTrade 를 안 보낸다(2026-09-21 · 11-18 재확인 0건)
REST = "https://fapi.binance.com/fapi/v1/"
KEEP_S, KEEP_M = 3 * 3600, 3000                                       # 초 버퍼(피쳐는 3,610초 앞까지) · 분봉(24시간 고저 + σ 1,440분)
HZ, DB = 300, 0.10                                                    # 라벨 y_fp10_300: 300초 · ±0.10σ1h
FILL_MAX = 30                                                         # 번호 메움 REST 호출 상한(분당 · 가중치 20 × 30 = 600 ≪ IP 한도 2,400)


class Live:
    """초 버퍼 + 분봉 + 사건 대기열. 네트워크를 모른다 — trade()·second() 만 받는다(재생·자체 점검이 같은 길을 탄다)."""

    def __init__(self, art: dict | None, sink=None):
        self.sec: dict[int, list] = {}        # 초 → [ph, pl, pc, bq, sq, nt, q2] (체결 있던 초만)
        self.t0 = None                        # 이 초 «다음»부터 빈틈 없음(시작·구멍 뒤 다시 잡는다)
        self.done = None                      # 마지막으로 닫은 초
        self.last_pc = np.nan
        self.M = pd.DataFrame(columns=["h", "l", "c", "v", "tb"], dtype=float)
        self.seen: set = set(); self.pend: list = []; self.wait: list = []; self.asecs: deque = deque()
        self.art, self.sink, self.need_klines = art, sink or (lambda kind, row: None), True

    # ── 체결 → 1초 묶음 (build_eth_aggtrades_1s_20261118.agg 와 같은 정의)
    def trade(self, t_ms: int, p: float, q: float, first: int, last: int, buyer_maker: bool) -> None:
        s = t_ms // 1000; r = self.sec.get(s)
        if r is None:
            r = self.sec[s] = [p, p, p, 0.0, 0.0, 0.0, 0.0]
        r[0], r[1], r[2] = max(r[0], p), min(r[1], p), p
        r[4 if buyer_maker else 3] += q; r[5] += last - first + 1; r[6] += q * q

    def window(self, a: int, b: int):
        """R.Tape.window 와 같은 꼴. 아직 안 온 초(> done)는 체결 없는 초로 채운다 — 피쳐는 결정 초까지만 보므로 값이 같다(자체 점검·재생 대조)."""
        if self.t0 is None or a <= self.t0:                           # a 직전 체결가가 있어야 한다(연구 Tape 와 같은 조건)
            return None
        n = b - a + 1; W = {k: np.zeros(n) for k in ("bq", "sq", "nt", "q2")}; pc = np.full(n, np.nan); ph, pl = pc.copy(), pc.copy()
        for s in range(a, min(b, self.done) + 1):
            r = self.sec.get(s)
            if r is not None:
                i = s - a; ph[i], pl[i], pc[i], W["bq"][i], W["sq"][i], W["nt"][i], W["q2"][i] = r
        if np.isnan(pc[0]):
            pc[0] = next((self.sec[s][2] for s in range(a - 1, a - 3600, -1) if s in self.sec), np.nan)
        traded = ~np.isnan(ph); pc = pd.Series(pc).ffill().to_numpy(); ph, pl = np.where(traded, ph, pc), np.where(traded, pl, pc)
        W.update(pc=pc, ph=ph, pl=pl, traded=traded); return W

    def hole(self, s: int) -> None:
        """초 s 근처 체결을 잃었다 — 그 뒤부터 다시 빈틈 없음으로 친다. 걸린 사건은 버린다(틀린 값을 내느니 안 낸다)."""
        self.t0, self.need_klines = s + 1, True; self.pend.clear(); self.wait.clear()

    def klines(self, rows: list, cur: int) -> None:
        """REST 1분봉 [[open_ms, o, h, l, c, v, …, taker_buy_base(9), …]] — 내 초 버퍼가 온전히 덮지 못한 분만 채운다."""
        k = pd.DataFrame([[int(r[0]) // 60000, float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[9])] for r in rows], columns=["m", "h", "l", "c", "v", "tb"]).set_index("m")
        k.loc[k.v <= 0, ["h", "l", "c"]] = np.nan                    # 체결 없는 분은 연구 분봉에서 NaN
        # cur = 거래소 시각으로 지금 형성 중인 분(뺀다) · t0 가 든 분까지가 klines 몫, 그 뒤는 내 초
        k = k[(k.index < cur) & (k.index <= self.t0 // 60)]
        self.M = pd.concat([self.M[~self.M.index.isin(k.index)], k]).sort_index().iloc[-KEEP_M:]
        self.need_klines = False
        for mi in range(self.t0 // 60 + 1, cur):                      # klines 를 기다리는 동안 닫힌 분은 내 초로 만든다(초는 온전히 받았다)
            self.M.loc[mi] = self._minute(mi, mi * 60 + 59)

    def _minute(self, mi: int, upto: int):
        rs = [self.sec[s] for s in range(mi * 60, upto + 1) if s in self.sec]
        return [max(r[0] for r in rs), min(r[1] for r in rs), rs[-1][2], sum(r[3] + r[4] for r in rs), sum(r[3] for r in rs)] if rs else [np.nan] * 5

    # ── 닫힌 초마다
    def second(self, s: int) -> None:
        self.done = s; mi = s // 60
        if s % 60 == 59:
            for x in [x for x in self.sec if x < s - KEEP_S]:
                del self.sec[x]
        if self.t0 is None or self.need_klines or len(self.M) == 0:
            return
        if mi * 60 > self.t0:                                         # 이 분을 처음부터 온전히 받았을 때만 내 초로 분봉을 만든다(아니면 다음 klines 가 채운다)
            self.M.loc[mi] = self._minute(mi, s)
        if mi in self.M.index:
            m = self.M.reindex(range(mi - KEEP_M + 1, mi + 1)).astype(float)
            if mi * 60 > self.t0:
                c = RL.candidates(m, pd.Timestamp(mi * 60, unit="s"), pd.Timestamp(mi * 60 + 60, unit="s"))
                for r in c.itertuples():
                    key = (mi, int(r.look), r.kind, int(r.side))
                    if key in self.seen:
                        continue
                    X = self.window(mi * 60, s); hi, lo = X["ph"], X["pl"]            # 체결 없는 초 = 직전 체결가(연구 Tape 와 같음)
                    seg = (hi > r.th if r.kind == "brk" else hi >= r.th) if r.side > 0 else (lo < r.th if r.kind == "brk" else lo <= r.th)      # RL.one 의 사건 초 규칙
                    if not seg.any():
                        continue
                    a = mi * 60 + int(seg.argmax()); self.seen.add(key); self.asecs.append(a)
                    self.pend.append(dict(key=key, a=a, seen_at=s, prev=sum(1 for x in self.asecs if x >= a - 3600) - 1, th=float(r.th), L=float(r.L), s1h=float(r.s1h)))
            due = [e for e in self.pend if s >= e["a"] + RL.W]
            if due:
                M_ = R.marr(m); self.pend = [e for e in self.pend if e not in due]
                short = int(m.c.iloc[-1443:-1].notna().sum()) < 1430      # 분봉 맥락은 1,441분 앞까지 본다 — 모자라면 NaN 피쳐로 채점하지 않는다(reindex 라 one() 의 검사는 안 걸린다)
                for e in due:
                    self._score(e, None if short else M_)
        while self.asecs and self.asecs[0] < s - 7200:
            self.asecs.popleft()
        if s % 60 == 59:
            self.seen = {k for k in self.seen if k[0] >= mi - 2}; self.M = self.M.iloc[-KEEP_M:]
        for e in [e for e in self.wait if s >= e["d"] + HZ]:
            self.wait.remove(e); self._resolve(e)

    def _score(self, e: dict, M_: dict) -> None:
        mi, look, kind, side = e["key"]; f = RL.one(self, M_, mi, side, kind, look, e["th"], e["L"], e["s1h"]) if M_ is not None else None
        if f is None:                                                 # 초 버퍼 1시간·분봉 1,441분이 아직 안 찼다
            self.sink("skip", dict(a=e["a"], look=look, kind=kind, side=side, why="warm")); return
        assert f["asec"] == e["a"], (f["asec"], e["a"])
        hod = (e["a"] % 86400) / 86400; f.update(m_hsin=np.sin(2 * np.pi * hod), m_hcos=np.cos(2 * np.pi * hod), m_wkend=float(pd.Timestamp(e["a"], unit="s").dayofweek >= 5), m_prev60=float(e["prev"]))
        x = pd.DataFrame([[f[c] for c in self.art["cols"]]], columns=self.art["cols"]) if self.art else None
        p = float(np.mean([m_.predict_proba(x)[:, 1] for m_ in self.art["models"]])) if self.art else float("nan")
        lo, hi = self.art["thr"] if self.art else (0, 1); d = e["a"] + RL.W; P = float(self.window(d, d)["pc"][0])
        row = dict(id=f"{e['a']}_{look}_{kind}_{side}", a=e["a"], d=d, lag=self.done - d, side=side, kind=kind, look=look, L=e["L"], s1h=e["s1h"], P=P, p=round(p, 6), grp="top" if p >= hi else "bot" if p <= lo else "mid")
        self.wait.append(row); self.sink("ev", row); self.sink("feat", dict(id=row["id"], **{c: f[c] for c in (self.art["cols"] if self.art else [])}))

    def _resolve(self, e: dict) -> None:
        X = self.window(e["d"] + 1, e["d"] + HZ)                      # 라벨은 결정 다음 초부터(연구와 같은 식)
        if X is None:
            return
        fp = R.first_pass(X["ph"], X["pl"], e["P"], DB * e["s1h"]); z = (fp if fp in (1.0, -1.0) else np.sign(np.log(X["pc"][-1] / e["P"]))) * e["side"]
        hit = (X["ph"] >= e["P"] * np.exp(DB * e["s1h"])) | (X["pl"] <= e["P"] * np.exp(-DB * e["s1h"]))
        self.sink("res", dict(id=e["id"], y=1 if z > 0 else 0 if z < 0 else None, hit_s=int(hit.argmax()) + 1 if hit.any() else None, r=round(float(e["side"] * np.log(X["pc"][-1] / e["P"]) * 1e4), 2)))


def load_art() -> dict:
    import joblib
    a = joblib.load(MODEL)
    for m_ in a["models"]:                                            # 한 줄 예측에 스레드 8개를 띄우지 않는다(라이브 서버)
        for st in getattr(m_, "steps", [(None, m_)]):
            if hasattr(st[1], "n_jobs"):
                st[1].n_jobs = 1
    return a


def export() -> None:
    import joblib, warnings; warnings.filterwarnings("ignore")
    tr, oo = pd.read_parquet(RL.OUT / "ml_train.parquet"), pd.read_parquet(RL.OUT / "ml_train_oof_ens.parquet"); X = RL.sets(tr)["전체"]
    art = dict(models=RL.fit_ens(tr, X, tr[RL.YCONF].to_numpy()), cols=X, thr=[float(v) for v in np.quantile(oo[f"{RL.YCONF}|전체"], [.1, .9])], label=RL.YCONF,
               train=[str(tr.ts.min()), str(tr.ts.max()), len(tr)], rate={"top": [989, 1752], "bot": [1269, 2804]})    # rate = 2023 확인 검정의 «사건 쪽 먼저» 건수(화면 확률의 출처)
    joblib.dump(art, MODEL, compress=3); te = pd.read_parquet(RL.OUT / "ml_test.parquet"); te = te[te.ts >= R.TEST0]
    print("저장", MODEL, MODEL.stat().st_size, "바이트 · 문턱", art["thr"], "· 2026 시험 AUC", RL.auc_(te[RL.YCONF].to_numpy(), RL.pred_ens(load_art()["models"], te, X)), "(기대 .5353182)")


def replay(t0: str, t1: str) -> None:
    """1초 묶음 일 파일을 실시간 경로(trade 없이 초 버퍼 직접 채움 → second)로 흘려 연구 빌더(build + pred_ens)와 사건·피쳐·점수·결과를 대조한다."""
    import warnings; warnings.filterwarnings("ignore")
    a0, a1 = int(pd.Timestamp(t0).timestamp()), int(pd.Timestamp(t1).timestamp()); art = load_art(); got: dict = {"ev": {}, "feat": {}, "res": {}, "skip": {}}
    L = Live(art, lambda k, r: got[k].__setitem__(r.get("id", len(got[k])), r)); m = R.minute_bars(cache=RL.OUT / "min_from_tape_fwd.parquet"); T = R.Tape()
    L.t0 = (a0 - 3800) // 60 * 60 + 59; L.M = m[m.index <= L.t0 // 60].iloc[-KEEP_M:].copy(); L.need_klines = False             # 분 경계에서 시작(그 분까지는 «klines» 몫)
    g = pd.concat([T.day(d) for d in pd.date_range(pd.Timestamp(L.t0 - 3600, unit="s").normalize(), pd.Timestamp(a1 + 400, unit="s").normalize(), freq="D").strftime("%Y-%m-%d")])
    g = g[(g.s > L.t0 - 3600) & (g.s <= a1 + 400)]; rows = dict(zip(g.s.to_numpy().tolist(), g[["ph", "pl", "pc", "bq", "sq", "nt", "q2"]].to_numpy().tolist())); t_ = time.time()
    L.sec = {s_: r for s_, r in rows.items() if s_ <= L.t0}
    for s in range(L.t0 + 1, a1 + 400):
        if s in rows:
            L.sec[s] = rows[s]
        L.second(s)
    # m_prev60 이 앞 1시간 사건을 세므로 연구 쪽도 1시간 앞에서 만든다
    ref = RL.build(m, str(pd.Timestamp(a0 - 3600, unit="s")), t1); ref = ref[ref.asec >= a0].reset_index(drop=True); X = art["cols"]; ref["p"] = RL.pred_ens(art["models"], ref, X); ref["id"] = [f"{r.asec}_{r.look}_{r.kind}_{r.side}" for r in ref.itertuples()]
    ev = {k: v for k, v in got["ev"].items() if a0 <= v["a"] < a1}; ids = set(ref.id); both = sorted(ids & set(ev)); rr = ref.set_index("id")
    fd = max((float(np.nanmax(np.abs(np.array([got["feat"][i][c] for c in X if c != "m_prev60"], float) - rr.loc[i, [c for c in X if c != "m_prev60"]].to_numpy(float)))) for i in both), default=np.nan)
    pv = sum(1 for i in both if got["feat"][i]["m_prev60"] != rr.loc[i, "m_prev60"]); pdif = max((abs(ev[i]["p"] - rr.loc[i, "p"]) for i in both if got["feat"][i]["m_prev60"] == rr.loc[i, "m_prev60"]), default=np.nan)
    yd = sum(1 for i in both if i in got["res"] and got["res"][i]["y"] != (None if np.isnan(rr.loc[i, RL.YCONF]) else int(rr.loc[i, RL.YCONF])))
    print(f"재생 {t0}~{t1} · {time.time() - t_:.0f}초 · 연구 {len(ids)}건 · 실시간 {len(ev)}건 · 겹침 {len(both)} · 연구만 {len(ids - set(ev))} · 실시간만 {len(set(ev) - ids)} · 건너뜀 {len(got['skip'])}")
    print(f"피쳐 최대 차(m_prev60 제외) {fd:.2e} · m_prev60 다른 사건 {pv} · 점수 최대 차(m_prev60 같은 사건) {pdif:.2e} · 결과 불일치 {yd}/{sum(1 for i in both if i in got['res'])}")


def selftest() -> None:
    import build_eth_aggtrades_1s_20261118 as B
    rng = np.random.default_rng(0); n = 5000; t = np.sort(rng.integers(1_000_000_000, 1_000_600_000, n)); pr = np.round(2000 + np.cumsum(rng.normal(0, .05, n)), 2); q = np.round(rng.exponential(1, n), 3)
    f = np.cumsum(rng.integers(1, 4, n)); l = f + rng.integers(0, 3, n); ibm = rng.random(n) < .5
    ref = B.agg(pd.DataFrame({"price": pr, "qty": q, "first": f, "last": l, "t": t, "ibm": ibm})).set_index("s"); L = Live(None)
    for i in range(n):
        L.trade(int(t[i]), float(pr[i]), float(q[i]), int(f[i]), int(l[i]), bool(ibm[i]))
    assert set(L.sec) == set(ref.index) and all(np.allclose(L.sec[s], ref.loc[s, ["ph", "pl", "pc", "bq", "sq", "nt", "q2"]].to_numpy(float)) for s in ref.index), "1초 묶음이 Vision 빌더와 다르다"
    s0, s1 = int(ref.index[0]), int(ref.index[-1]); L.t0 = s0 - 1; L.done = s1 - 100; a = L.window(s0 + 10, s1)      # 아직 안 온 초는 체결 없는 초 · 결정 초까지의 값은 그대로
    L.done = s1; b = L.window(s0 + 10, s1); k = s1 - 100 - (s0 + 10) + 1
    assert all(np.array_equal(a[c][:k], b[c][:k]) for c in a) and a["bq"][k:].sum() == 0 and np.all(a["pc"][k:] == a["pc"][k - 1]) and L.window(s0 - 1, s1) is None
    gaps = []; fill = Filler(lambda lo, hi: gaps.append((lo, hi)))                                                 # 번호 빠짐: 1씩 늘면 그대로 · 건너뛰면 메울 구간 · 옛 번호는 버림
    assert [fill.check(i) for i in (10, 11, 14, 12, 15)] == ["ok", "ok", "gap", "dup", "ok"] and gaps == [(12, 13)]
    L2 = Live(None); L2.t0 = 100; L2.pend.append(1); L2.hole(500); assert L2.t0 == 501 and not L2.pend and L2.need_klines
    L3 = Live(None); L3.t0 = 60 * 1000 + 20; kl = [[mi * 60000, 0, 10 + mi % 7, 5, 8, (mi != 995) * 3.0, 0, 0, 0, 1.0, 0, 0] for mi in range(400, 1003)]      # 1분봉: t0 가 든 분(1000)까지 klines · 1001 은 내 초 · 1002(형성 중)는 뺀다
    for s_ in range(60 * 1001, 60 * 1002):
        L3.trade(s_ * 1000, 7.0 + (s_ % 3), 1.0, 1, 1, False)
    L3.klines(kl, 1002); assert not L3.need_klines and list(L3.M.index) == list(range(400, 1002)) and np.isnan(L3.M.loc[995, "c"]) and L3.M.loc[1000, "h"] == 10 + 1000 % 7, (L3.need_klines, len(L3.M))
    assert L3.M.loc[1001].tolist() == [9.0, 7.0, 7.0 + (60 * 1002 - 1) % 3, 60.0, 60.0]
    print("selftest ok")


class Filler:
    """묶음 번호 a 의 연속성. gap 이면 on_gap(빠진 첫 번호, 빠진 끝 번호)을 부른다."""
    def __init__(self, on_gap):
        self.last, self.on_gap = None, on_gap

    def check(self, a: int) -> str:
        if self.last is not None and a <= self.last:
            return "dup"
        gap = self.last is not None and a > self.last + 1
        if gap:
            self.on_gap(self.last + 1, a - 1)
        self.last = a; return "gap" if gap else "ok"


async def run() -> None:
    import aiohttp
    OUTD.mkdir(parents=True, exist_ok=True); files = {"ev": OUTD / "ethusdt.jsonl", "res": OUTD / "ethusdt.jsonl", "skip": OUTD / "ethusdt.jsonl", "feat": OUTD / "ethusdt_feat.jsonl"}

    def sink(kind: str, row: dict) -> None:
        with files[kind].open("a") as fh:
            fh.write(json.dumps(dict(k=kind, **row), ensure_ascii=False) + "\n")
        print(time.strftime("%H:%M:%S"), kind, {k: row[k] for k in ("id", "p", "grp", "y", "hit_s", "why") if k in row}, flush=True)

    art = load_art(); L = Live(art, sink); calls: deque = deque(); pending_gaps: list = []; fill = Filler(lambda lo, hi: pending_gaps.append((lo, hi)))
    print(f"시작 · 모델 {MODEL.name} · 문턱 {art['thr']} · 재료 {len(art['cols'])}개", flush=True)
    async with aiohttp.ClientSession() as http:
        async def get(path: str, **params):
            async with http.get(REST + path, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
                r.raise_for_status(); return await r.json()

        async def refill(lo: int, hi: int) -> bool:
            """빠진 묶음 번호 lo..hi 를 REST 로 메운다. 분당 FILL_MAX 번을 넘기면 포기(구멍 처리)."""
            miss = 0
            while lo <= hi:
                now = time.time()
                while calls and calls[0] < now - 60:
                    calls.popleft()
                if len(calls) >= FILL_MAX:
                    return False
                calls.append(now); raw = await get("aggTrades", symbol="ETHUSDT", fromId=lo, limit=1000)
                got = [r for r in raw if lo <= r["a"] <= hi]
                for r in got:
                    L.trade(r["T"], float(r["p"]), float(r["q"]), r["f"], r["l"], r["m"])
                if raw and raw[-1]["a"] >= hi:
                    return True                                       # REST 가 hi 까지 봤다 — 그 사이에 안 온 번호는 원래 없는 번호
                if got:
                    lo = got[-1]["a"] + 1
                else:                                                 # REST 가 아직 못 따라왔다(재접속 직후) — 1초씩 세 번까지 기다리고 안 되면 구멍
                    miss += 1
                    if miss > 3:
                        return False
                    await asyncio.sleep(1)
            return True

        last_state = kl_next = 0.0; own_chk = None
        while True:
            try:
                async with http.ws_connect(WS, heartbeat=20) as ws:
                    print("웹소켓 연결", flush=True)
                    while True:
                        try:
                            msg = await asyncio.wait_for(ws.receive(), 2)
                        except asyncio.TimeoutError:
                            msg = None
                        if msg is not None and msg.type != aiohttp.WSMsgType.TEXT:
                            raise ConnectionError(f"웹소켓 닫힘 {msg.type}")
                        top = None                               # 초는 «그 뒤 초의 체결이 왔을 때»만 닫는다(벽시계로 닫으면 끊긴 연결의 빈 초로 채점한다)
                        if msg is not None:
                            d = json.loads(msg.data); st = fill.check(d["a"])
                            for lo, hi in pending_gaps:               # 빠진 번호를 지금 체결보다 먼저 넣는다(그 초의 마지막 체결가 순서)
                                try:
                                    ok = await refill(lo, hi)
                                except Exception as e:
                                    ok = False; print("번호 메움 실패", type(e).__name__, e, flush=True)
                                print(f"묶음 번호 빠짐 {lo}~{hi}({hi - lo + 1}건) → {'메움' if ok else '구멍 처리(1시간 뒤부터 다시 채점)'}", flush=True)
                                if not ok:
                                    L.hole(d["T"] // 1000)
                            pending_gaps.clear()
                            if st != "dup":
                                L.trade(d["T"], float(d["p"]), float(d["q"]), d["f"], d["l"], d["m"]); top = d["T"] // 1000 - 1
                                if L.t0 is None:
                                    L.t0 = L.done = d["T"] // 1000; print("첫 체결", L.t0, flush=True)
                        now = time.time()
                        if L.need_klines and L.done is not None and now >= kl_next and L.done % 60 >= 3 and L.done // 60 * 60 > L.t0:      # 거래소 시각으로 t0 가 든 분이 닫힌 뒤에
                            kl_next = now + 60                        # 성공·실패와 무관하게 1분에 한 번만(가중치 10)
                            try:
                                rows = await get("klines", symbol="ETHUSDT", interval="1m", limit=1500)
                            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                                rows = None; print("1분봉 실패(1분 뒤 다시)", type(e).__name__, e, flush=True)
                            if rows:
                                L.klines(rows, L.done // 60); own_chk = L.done // 60 + 1; print("1분봉 받음", len(L.M), flush=True)
                        if own_chk is not None and L.done is not None and L.done // 60 > own_chk and L.done % 60 >= 3 and not L.need_klines:
                            # 한 번(받은 지 2분 뒤): ① 처음 받은 1분봉의 마지막 몇 분을 다시 받아 바꾼다 — REST 가 몇 초 늦어 직전 분이 덜 찬 값일 수 있다(11-18 실측: 분 경계 +3초 응답에 형성 중인 분이 없었다)
                            #   ② 내 초로 만든 분봉이 거래소 1분봉과 같은지 로그에 남긴다
                            try:
                                fixed = 0
                                for r in await get("klines", symbol="ETHUSDT", interval="1m", limit=10):
                                    mi = int(r[0]) // 60000; ref = [float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[9])]
                                    if mi <= L.t0 // 60 and mi in L.M.index and ref[3] > 0:
                                        fixed += not np.allclose(L.M.loc[mi].to_numpy(float), ref, rtol=1e-9); L.M.loc[mi] = ref
                                    elif mi == own_chk:
                                        mine = L.M.loc[mi].to_numpy(float).tolist(); # 고·저·종은 같아야 한다. 거래량은 분 경계 체결의 귀속이 달라 조금 어긋날 수 있다(Vision 묶음 vs Vision 1분봉도 하루 157~507분이 다르고 하루 합은 같다 — 11-18 대조)
                                        print("분봉 대조(내 초 vs 거래소)", "고·저·종 일치" if np.allclose(mine[:3], ref[:3], rtol=1e-9) else "🔴고·저·종 불일치", f"· 거래량 차 {mine[3] / ref[3] - 1:+.3%}", mine, ref, flush=True)
                                print("처음 받은 1분봉 끝 몇 분 다시 확인 · 바뀐 분", int(fixed), flush=True)
                            except (aiohttp.ClientError, asyncio.TimeoutError) as e:      # 다음 분에 다시(1분에 한 번)
                                print("분봉 대조 실패", type(e).__name__, e, flush=True); own_chk += 1; continue
                            own_chk = None
                        if top is not None and L.done is not None:
                            for s in range(L.done + 1, top + 1):
                                L.second(s)
                        if time.time() - last_state > 10:
                            last_state = time.time(); warm = int(last_state) + 3670 if L.t0 is None or L.need_klines else L.t0 + RL.PRE + RL.PAD + 60      # 1분봉을 못 받은 동안은 계속 «준비 중»
                            tmp = OUTD / "ethusdt_state.tmp"; tmp.write_text(json.dumps(dict(ts=int(last_state), done=L.done, t0=L.t0, warm_until=warm, minutes=len(L.M), need_klines=L.need_klines, thr=art["thr"], rate=art["rate"]))); tmp.replace(OUTD / "ethusdt_state.json")
            except (aiohttp.ClientError, ConnectionError, asyncio.TimeoutError) as e:   # 그 밖의 오류는 죽어서 감시 스크립트가 다시 띄운다(연속이면 5분 쉼)
                print("연결 끊김 — 5초 뒤 다시(빠진 묶음은 번호로 메운다)", type(e).__name__, e, flush=True); await asyncio.sleep(5)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--selftest"]:
        selftest()
    elif a[:1] == ["--export"]:
        export()
    elif a[:1] == ["--replay"]:
        replay(a[1], a[2])
    else:
        asyncio.run(run())
