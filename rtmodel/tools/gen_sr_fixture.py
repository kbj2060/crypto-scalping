"""레벨 5분(scripts/live_sr_lean_20261118.py) «체결·1분봉 → 사건 → 피쳐 87 → 확률 → jsonl 줄» 정답지 (2026-11-20 · R5 셋째 조각 · tests/sr_lean.rs).

정답은 원본 워커의 `Live`(상태 기계)·`load_art`(얼린 모델) 를 그대로 불러 만든다. 안전(소스 확인): 워커 모듈은 import 때 numpy·pandas·
연구 모듈 둘(research_sr_ml_rl · research_sr_intensity -- 둘 다 import 때 경로 상수뿐)만 부르고, 네트워크(aiohttp)는 run() 안에서만 import 한다.
여기서는 run() 을 부르지 않는다 -- run() 의 «메시지 하나 처리» 순서(trade → 첫 체결이면 t0 → 1분봉 받기 → 2분 뒤 다시 확인 → 닫힌 초 처리)를
아래 drive() 가 같은 순서로 흉내 내고, REST 응답(1분봉)은 합성 체결로 만든 거래소식 분봉 문자열로 준다. .env·텔레그램·거래소 접속 없음.

입력은 합성: 고정 시드로 초마다 체결(시각 ms·가격 0.01 단위·수량 0.001 단위·묶음 번호·매수자 메이커 여부)을 만들고, 앞 1,500분 이상은
1분봉(REST klines 꼴 문자열)으로만, 그 뒤는 체결 메시지로 흘린다. 기록 줄은 run() 의 sink 와 같은 json.dumps(dict(k=kind, **row), ensure_ascii=False).

    NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL" python rtmodel/tools/gen_sr_fixture.py            # 서버(AVX-512 없음)와 같은 계산 → tests/fixtures/sr_fixture.json
    python rtmodel/tools/gen_sr_fixture.py sr_fixture_avx512.json                                       # 참고: 개발기 AVX-512 numpy
    python rtmodel/tools/gen_sr_fixture.py --only normal                                                # 한 벌만(시간 재기)
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import live_sr_lean_20261118 as LS  # noqa: E402

OUT = ROOT / "rtmodel" / "tests" / "fixtures"
HIST_MIN = 3060            # 체결 메시지 시작 전 분봉 자료(REST limit 1500 · 꽉 찬 프레임 벌은 3,000분)


def tape(rng, t0, n, px0, sig, rate, drift=(), jumps=(), vol_x=(), quiet=(), wave=None):
    """초 t0..t0+n-1 의 체결. drift = [(시작 초, 끝 초, 총 로그수익)] · jumps = [(초, 로그 크기)] · vol_x = [(시작, 끝, σ 배수)] · quiet = [(시작, 끝, 체결률 배수)] ·
    wave = (주기 초, 로그 폭) -- 박스 장(주기가 60분보다 길어 1h 고저 접근·돌파가 주기마다 다시 «첫 사건»이 된다)."""
    sd = np.full(n, sig); r = np.full(n, rate)
    for a, b, k in vol_x:
        sd[a:b] *= k
    for a, b, k in quiet:
        r[a:b] *= k
    inc = rng.normal(0, 1, n) * sd
    for a, b, tot in drift:
        inc[a:b] += tot / (b - a)
    for a, k in jumps:
        inc[a] += k
    lp = np.log(px0) + np.cumsum(inc)
    if wave:
        lp += wave[1] * np.sin(2 * np.pi * np.arange(n) / wave[0])
    cnt = rng.poisson(r)
    sec = np.repeat(np.arange(n), cnt); N = len(sec)
    ms = rng.integers(0, 1000, N); o = np.lexsort((ms, sec)); sec, ms = sec[o], ms[o]
    p = np.round(np.exp(lp[sec] + rng.normal(0, 2e-5, N)), 2)
    q = np.maximum(np.round(rng.exponential(0.8, N), 3), 0.001)
    span = rng.integers(0, 3, N); first = 10_000_000 + np.cumsum(span + 1) - span - 1; last = first + span   # 묶음 하나에 체결 1~3건
    bm = rng.random(N) < 0.5 + 0.1 * np.sign(np.r_[0, np.diff(p)])
    return pd.DataFrame({"t": (t0 + sec) * 1000 + ms, "p": p, "q": q, "f": first, "l": last, "m": bm})


def kl_rows(tr: pd.DataFrame, m_from: int, m_to: int, upto_ms: int, partial: dict | None = None):
    """거래소 1분봉 [open_ms, o, h, l, c, v, close_ms, qv, n, tb, tbq, '0'] 문자열(분 m_from..m_to · 체결 시각 < upto_ms 만).
    체결 없는 분 = 직전 종가로 o=h=l=c · v=0(거래소 꼴). partial = {분: 그 분에서 넣을 마지막 초(REST 가 덜 찬 분을 준 경우)}."""
    g = tr[(tr.t >= m_from * 60000) & (tr.t < min((m_to + 1) * 60000, upto_ms))]
    if partial:
        for mi, upto_s in partial.items():
            g = g[~((g.t // 60000 == mi) & (g.t // 1000 > upto_s))]
    mm = g.t // 60000
    agg = g.assign(tb=np.where(g.m, 0.0, g.q)).groupby(mm).agg(o=("p", "first"), h=("p", "max"), l=("p", "min"), c=("p", "last"), v=("q", "sum"), tb=("tb", "sum"), n=("p", "size"))
    prev = tr[tr.t < m_from * 60000].p
    last_c = float(prev.iloc[-1]) if len(prev) else float(g.p.iloc[0])
    rows = []
    for mi in range(m_from, m_to + 1):
        if mi in agg.index:
            r = agg.loc[mi]; last_c = float(r.c)
            rows.append([mi * 60000, f"{r.o:.2f}", f"{r.h:.2f}", f"{r.l:.2f}", f"{r.c:.2f}", f"{r.v:.3f}", mi * 60000 + 59999, "0", int(r.n), f"{r.tb:.3f}", "0", "0"])
        else:
            c = f"{last_c:.2f}"; rows.append([mi * 60000, c, c, c, c, "0.000", mi * 60000 + 59999, "0", 0, "0.000", "0", "0"])
    return rows


def scenarios(rng):
    """(이름, 합성 설정). 시각은 초. 모든 벌: 앞 HIST_MIN 분은 분봉으로만, 그 뒤 live 초 동안 체결 메시지(앞 1시간 = 준비)."""
    base = lambda: int(pd.Timestamp("2026-03-02", tz="UTC").timestamp()) + int(rng.integers(0, 200)) * 86400 + int(rng.integers(0, 1440)) * 60 + int(rng.integers(0, 60))  # noqa: E731
    wave = lambda: (float(rng.uniform(65, 110)) * 60, float(rng.uniform(0.004, 0.012)))  # noqa: E731
    H = HIST_MIN * 60
    S = []
    for i in range(3):     # 보통 걸음(1h·4h 고저 접근·돌파가 저절로 생긴다)
        S.append((f"normal{i}", dict(t0=base(), live=10800, sig=1.0e-4, rate=2.0)))
    for i in range(3):     # 박스 장: 1h 고저에 주기마다 다가갔다 물러남·뚫음
        S.append((f"range{i}", dict(t0=base(), live=10800, sig=5e-5, rate=2.0, wave=wave())))
    S.append(("touch_retreat", dict(t0=base(), live=10800, sig=4e-5, rate=2.0, wave=(4000.0, 0.006),       # 연달아 닿음(60분 안 같은 사건 억제)
                                    drift=[(H + 3700 + k * 600, H + 3700 + k * 600 + 300, (-1) ** k * 0.003) for k in range(10)])))
    S.append(("pass_through", dict(t0=base(), live=10800, sig=8e-5, rate=3.0, drift=[(H + 3800, H + 5600, 0.06), (H + 7600, H + 9000, -0.07)])))   # 1h·4h·24h 고저를 한꺼번에 뚫고 지나감
    S.append(("both_sides", dict(t0=base(), live=10800, sig=1.2e-4, rate=4.0, wave=wave(), vol_x=[(H + 3650, H + 10800, 5.0)])))     # 판정 창 안에 양쪽 0.1σ 를 다 지남
    S.append(("spike", dict(t0=base(), live=10800, sig=1e-4, rate=2.0, wave=wave(), jumps=[(H + 3900, 0.02), (H + 4500, -0.03), (H + 6200, 0.015), (H + 8000, -0.02)])))
    S.append(("sparse", dict(t0=base(), live=10800, sig=1e-4, rate=0.08, wave=wave())))       # 체결 없는 초가 대부분(직전가 채움 · lag > 0 접근)
    S.append(("quiet_gaps", dict(t0=base(), live=10800, sig=1e-4, rate=2.0, wave=wave(), quiet=[(H + 3700, H + 3900, 0.0), (H + 4600, H + 4700, 0.0), (H + 5000, H + 8700, 0.02)])))
    S.append(("warm_only", dict(t0=base(), live=3500, sig=1.5e-4, rate=2.0, wave=wave())))   # 준비 시간(1시간) 안의 사건만 -- 전부 «warm»
    # 1,442분 맥락에 빈 분 15개(값 있는 분 1,427 < 1,430) → 시작 뒤 157분까지 사건은 «warm», 그 뒤 풀림(문턱 1,430 의 양쪽)
    S.append(("short_minutes", dict(t0=base(), live=10800, sig=1e-4, rate=2.0, wave=wave(), quiet=[(H - 1300 * 60, H - 1285 * 60, 0.0)], drift=[(H + 3700, H + 4000, 0.012)])))
    # 58분 동안 체결 없음 → 그 안에 창 시작이 걸린 사건(직전 체결가를 3,000~3,600초 앞에서 찾는다) · 끝난 뒤 53분에 돌파를 밀어 올림
    S.append(("long_silence", dict(t0=base(), live=12600, sig=1e-4, rate=2.0, quiet=[(H + 4000, H + 7500, 0.0)], drift=[(H + 10650, H + 10950, 0.012), (H + 10960, H + 11200, -0.02)])))
    S.append(("hole", dict(t0=base(), live=14400, sig=1e-4, rate=2.0, wave=wave(), hole=(H + 4000, H + 4030))))   # 번호 메움 실패 → 구멍 → 1분봉 다시 → 1시간 다시 준비
    S.append(("rest_partial", dict(t0=base(), live=10800, sig=1e-4, rate=2.0, wave=wave(), partial=True)))   # 처음 받은 1분봉의 마지막 분이 덜 참 → 2분 뒤 다시 확인이 고침
    for i in range(2):     # 주기 정확히 60분 박스: 앞 사건과 59·60·61분 떨어진 사건(중복 억제 60분 경계)
        S.append((f"period60_{i}", dict(t0=base(), live=14400, sig=3e-5, rate=2.0, wave=(3600.0, 0.006 + 0.002 * i))))
    for i in range(3):     # 분봉 3,000행이 꽉 찬 프레임 = 서버가 25시간 넘게 돈 상태(REST 1,500개 + 내 분봉 1,500개). σ 굴림 분산이 프레임 첫 행부터
        #   누적하는 몫이 여기서만 갈린다(프레임 앞이 NaN 이면 안 갈림 -- 양성 대조 «프레임 2,999행»이 1,500행 벌에선 못 잡았다). 1분봉 3,000개를 한 번에 줘서 그 상태로 시작
        S.append((f"full_frame{i}", dict(t0=base(), live=10800, sig=1e-4, rate=2.0, wave=wave() if i else None, kl_n=3000)))
    for nm, at in (("fri_sat", "2026-03-06 23:30:00"), ("sun_mon", "2026-03-08 23:30:00")):     # 사건이 요일·자정 경계에 걸리게(준비 1시간 뒤 = 경계 앞 30분)
        S.append((nm, dict(t0=int(pd.Timestamp(at, tz="UTC").timestamp()) - H - 3600 + int(rng.integers(0, 60)), live=10800, sig=1.2e-4, rate=2.0, wave=wave())))
    return S


def drive(art, tr: pd.DataFrame, t_live: int, cfg: dict):
    """run() 의 메시지 처리 순서 그대로 Live 를 돌리며 (연산 목록, 본 기록 줄, 피쳐 줄, 상태 줄) 을 모은다."""
    main, feat = [], []

    def sink(kind, row):
        (feat if kind == "feat" else main).append(json.dumps(dict(k=kind, **row), ensure_ascii=False))

    L = LS.Live(art, sink); ops = []; own_chk = None; state = []; next_state = None
    hole = cfg.get("hole"); holed = False
    live = tr[tr.t >= t_live * 1000]
    if hole:
        live = live[~((live.t >= (cfg["t0"] + hole[0]) * 1000) & (live.t < (cfg["t0"] + hole[1]) * 1000))]
    sec_ns = []
    for t, p, q, f, l, m in zip(live.t.tolist(), live.p.tolist(), live.q.tolist(), live.f.tolist(), live.l.tolist(), live.m.tolist()):
        if hole and not holed and t >= (cfg["t0"] + hole[1]) * 1000:
            holed = True; L.hole(t // 1000); ops.append(["hole", t // 1000])          # 번호 메움 실패 → 지금 체결 초 기준 구멍(run(): L.hole(d["T"] // 1000))
        L.trade(t, p, q, f, l, m); ops.append(["trade", t, p, q, f, l, bool(m)]); top = t // 1000 - 1
        if L.t0 is None:
            L.t0 = L.done = t // 1000; ops.append(["init", t // 1000])
        if L.need_klines and L.done is not None and L.done % 60 >= 3 and L.done // 60 * 60 > L.t0:
            cur = L.done // 60
            part = {cur - 1: (cur - 1) * 60 + 40} if cfg.get("partial") and own_chk is None and not ops_has(ops, "klines") else None
            rows = kl_rows(tr, cur - cfg.get("kl_n", 1500) + 1, cur, (L.done + 1) * 1000, part)
            L.klines(rows, cur); ops.append(["klines", rows, cur]); own_chk = cur + 1
        if own_chk is not None and L.done is not None and L.done // 60 > own_chk and L.done % 60 >= 3 and not L.need_klines:
            rows = kl_rows(tr, L.done // 60 - 9, L.done // 60, (L.done + 1) * 1000)
            for r in rows:                                                    # run() 의 «처음 받은 1분봉 끝 몇 분 다시 확인» 그대로
                mi = int(r[0]) // 60000; ref = [float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[9])]
                if mi <= L.t0 // 60 and mi in L.M.index and ref[3] > 0:
                    L.M.loc[mi] = ref
            ops.append(["recheck", rows]); own_chk = None
        if top is not None and L.done is not None and top > L.done:
            for s in range(L.done + 1, top + 1):
                t_ = time.perf_counter_ns(); L.second(s); sec_ns.append(time.perf_counter_ns() - t_)
            ops.append(["close", top])
            if next_state is None or top >= next_state:                       # 생존 표시(run() 은 10초마다 · 여기선 600초마다 같은 식)
                now = top + 2; warm = now + 3670 if L.t0 is None or L.need_klines else L.t0 + 3600 + 10 + 60
                state.append([now, json.dumps(dict(ts=now, done=L.done, t0=L.t0, warm_until=warm, minutes=len(L.M), need_klines=L.need_klines, thr=art["thr"], rate=art["rate"]))])
                next_state = top + 600
    return ops, main, feat, state, sec_ns


def ops_has(ops, k):
    return any(o[0] == k for o in ops)


def one_set(nm_cfg):
    nm, cfg = nm_cfg
    warnings.filterwarnings("ignore")
    with threadpool_limits(1):
        art = LS.load_art()
        srng = np.random.default_rng(int.from_bytes(__import__("hashlib").sha256(nm.encode()).digest()[:8], "little"))   # 벌마다 고정 시드(--only 로 하나만 돌려도 같은 입력)
        H = HIST_MIN * 60; n = H + cfg["live"] + 400
        tr = tape(srng, cfg["t0"], n, 3000.0 * np.exp(srng.normal(0, .3)), cfg["sig"], cfg["rate"], cfg.get("drift", ()), cfg.get("jumps", ()), cfg.get("vol_x", ()), cfg.get("quiet", ()), cfg.get("wave"))
        t_ = time.perf_counter(); ops, mn, ft, st, sec_ns = drive(art, tr, cfg["t0"] + H, cfg); el = time.perf_counter() - t_
        # 확률(반올림 전): _score 와 같은 식을 피쳐 줄 값으로 한 번 더(json 은 repr 왕복이라 비트 그대로)
        P, tim = [], []
        for ln in ft:
            r = json.loads(ln); x = pd.DataFrame([[r[c] for c in art["cols"]]], columns=art["cols"])
            each = [float(m_.predict_proba(x)[:, 1][0]) for m_ in art["models"]]
            t1 = time.perf_counter_ns(); p = float(np.mean([m_.predict_proba(x)[:, 1] for m_ in art["models"]])); tim.append(time.perf_counter_ns() - t1)
            P.append([int(np.float64(v).view(np.uint64)) for v in [p] + each])
    ev = [json.loads(x) for x in mn]
    kinds = {k: sum(1 for e in ev if e["k"] == k) for k in ("ev", "skip", "res")}
    sec_ns = np.array(sec_ns) / 1e6
    msg = (f"{nm:<14} 연산 {len(ops):>6} · 사건 {kinds['ev']:>3} · 건너뜀 {kinds['skip']:>3} · 결과 {kinds['res']:>3}"
           f" · top {sum(1 for e in ev if e.get('grp') == 'top')} bot {sum(1 for e in ev if e.get('grp') == 'bot')} · lag>0 {sum(1 for e in ev if e.get('lag', 0) > 0)}"
           f" · second() 중앙 {np.median(sec_ns):.1f}ms 상위1% {np.quantile(sec_ns, .99):.1f}ms · 총 {el:.0f}s · 앙상블 중앙 {np.median(tim) / 1e6 if tim else float('nan'):.1f}ms")
    return {"name": nm, "ops": ops, "main": mn, "feat": ft, "p": P, "state": st}, msg


def main():
    args = sys.argv[1:]
    only = args[args.index("--only") + 1] if "--only" in args else None
    name = next((a for a in args if a.endswith(".json")), "sr_fixture.json")
    S = [x for x in scenarios(np.random.default_rng(20261121)) if not only or x[0] == only]
    fx = {"npy_disable": os.environ.get("NPY_DISABLE_CPU_FEATURES", ""), "sets": []}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(os.environ.get("SR_PROCS", "6"))) as ex:     # 벌 하나 약 270MB · 초 하나 약 20ms(파이썬)
        for d, msg in ex.map(one_set, S):
            print(msg, flush=True); fx["sets"].append(d)
    path = OUT / name
    if only and path.exists():     # 한 벌만 다시 만들면 그 파일의 같은 이름 벌만 바꿔 넣는다
        old = json.loads(path.read_text()); assert old["npy_disable"] == fx["npy_disable"], "AVX 설정이 다른 정답지에 섞지 않는다"
        old["sets"] = [fx["sets"][0] if x["name"] == only else x for x in old["sets"]]; fx = old
    path.write_text(json.dumps(fx))
    print(f"{path.name} {path.stat().st_size:,} 바이트 · 벌 {len(fx['sets'])} · NPY_DISABLE_CPU_FEATURES={fx['npy_disable']!r}")


if __name__ == "__main__":
    main()
