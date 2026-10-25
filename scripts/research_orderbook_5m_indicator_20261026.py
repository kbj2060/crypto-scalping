"""5분 호가 지표 연구 (2026-10-26) -- 정보 표시용(예측 아님), 나중에 모델 입력 피쳐 후보.

사용자: «호가에 대해서 5분마다 호가 정보를 축약해서 하나의 지표로 … 호가벽이 있다던가 가짜 호가벽을 세워놓고
있다가 갑자기 급등락을 만들었다던가 … 과거 5분봉에 대해 지표로».
원천(서버 백업 미러, 읽기 전용): ~/backups/crypto-scalping-server/data/
  live/orderflow/depthdiff/<SYM>/<UTC시>.jsonl(.gz)   전체북 첫 줄 REST 스냅샷 + @depth@100ms
  lake/binance/tape/coin=<C>/date=*/                  1초 x 가격칸 테이커 체결
벽 추적·철회/먹힘 판정은 research_eth_wall_pull_jump_20261004 의 replay/fill_share 를 그대로 쓴다.
재생은 하루 단위 사슬: 전날 22시 파일부터 diff 를 끊김 없이 이어 간다(시간 파일의 중간 스냅샷은 replay 가 건너뜀).
  REST 스냅샷 1000호가는 ±50bp 를 못 덮는다(ETH 40~49bp · BTC 14~17bp) -> 깊은 칸은 diff 로 차오른 뒤 쓴다(WARM_MS).
시점 경계: 봉 b 의 값은 b+5분 이전 거래소 T 만 쓴다(철회 전후 이동도 봉 끝에서 자른다)
  -> 봉 b 행은 봉 b 종가 시각에 확정된다. 모델에 넣으면 라벨은 b+1 부터.
실행:
  python scripts/research_orderbook_5m_indicator_20261026.py --selftest
  python scripts/research_orderbook_5m_indicator_20261026.py --build ETH   -> tmp/ob5m_20261026/{sec,events,controls,cover}_ETH.*
  python scripts/research_orderbook_5m_indicator_20261026.py --bars ETH    -> bars_ETH.parquet
  python scripts/research_orderbook_5m_indicator_20261026.py --analyze ETH -> result_ETH.json · figs/
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_eth_wall_pull_jump_20261004 as W  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "backups/crypto-scalping-server/data"
OUT = ROOT / "tmp/ob5m_20261026"
BAR = 300_000
# 큰 벽 = 같은 쪽 ±50bp 깊이의 5% 이상 한 칸. 2% 는 하루 1만 번 철회되는 마켓메이커 일상이라(10-26 실측) 5% 에서 하루 수십 건으로 꺾인다.
WALL_SHARE = 0.05
# 코인별 (벽 몫, 추적 문턱 KS). SOL·XRP 는 칸이 굵어(1틱 0.7~0.9bp, ±50bp 55~70칸) 5% 가 늘 참이고 «중앙값 20배» 칸이 거의 없다.
#  10-26 결과 보기 전 규칙으로 보정: 벽 몫 = ETH «5% 벽 시간 비율»(공통 구간 쪽-초 33.6%)과 같은 분위,
#  추적 문턱 = 벽일 때 최대칸/중앙값 비의 1% 분위 × 0.8 내림 (SOL 7.0 -> 5, XRP 10.5 -> 8). ETH·BTC 는 그대로.
COIN_CFG = {"SOL": (0.098, (5, 12, 25)), "XRP": (0.097, (8, 20, 40))}
cfg = lambda coin: COIN_CFG.get(coin, (WALL_SHARE, (20, 50, 100)))  # noqa: E731
BIG_AGE_MS = 30_000    # 사건 벽은 30초 이상 서 있던 것만
NEAR_BP = 5.0          # 현재가 5bp 안에서 체결 없이 사라짐 = 가짜벽 의심 (멀리서 사라진 벽은 전후 이동 0)
JUMP_WIN = 5_000       # 철회 전후 ±5초 이동
OVERLAP_H = 2          # 하루 사슬을 전날 22시부터 시작
WARM_MS = 1_800_000    # 사슬 (재)시작 뒤 30분은 깊은 칸이 덜 찼다 -> 북 값 NaN
TAPE_GAP_S = 15        # 테이프 15초+ 빈 구간 = 수집 끊김(급변 때 분 단위로 빠진다). 그 ±60초 사건은 체결 몫 판정 불가
TAPE_PAD_S = 60
WALL_BACK_MS = 10_000  # 다른 칸이 10초 이하 잠깐 최대였다 돌아오면 같은 벽으로 이어 센다
SEC_COLS = ["t", "c0", "bb", "ba", "b10", "a10", "b50", "a50", "wb_i", "wb_q", "wa_i", "wa_q"]


def _chain(paths, cur):
    """시간 파일들을 한 줄 흐름으로. cur = [지금 파일 번호, 그 파일 안 줄 번호]."""
    for k, p in enumerate(paths):
        cur[0], cur[1] = k, 0
        with W._open(p) as f:
            for line in f:
                cur[1] += 1
                yield line


def tape_bad(sec, x):
    """x(초)가 테이프 빈 구간(연속 두 테이프 초 간격 > TAPE_GAP_S) ±TAPE_PAD_S 안이거나 테이프 범위 밖이면 True."""
    s = np.unique(sec)
    if not len(s):
        return np.ones(len(x), bool)
    out = (x < s[0]) | (x > s[-1])
    g = np.flatnonzero(np.diff(s) > TAPE_GAP_S)
    if not len(g):
        return out
    lo, hi = s[g] - TAPE_PAD_S, s[g + 1] + TAPE_PAD_S
    j = np.searchsorted(lo, x, "right") - 1
    return out | ((j >= 0) & (x <= hi[np.clip(j, 0, None)]))


def _tape(coin, days, cols):
    fs = [p for d in days for p in glob.glob(str(DATA / f"lake/binance/tape/coin={coin}/date={d}/*.parquet"))]
    return pd.concat([pd.read_parquet(p, columns=cols) for p in fs]) if fs else pd.DataFrame(columns=cols)


def day_job(args):
    coin, paths, keep_from, keep_to = args
    wall_share, W.KS = cfg(coin)   # ponytail: 재생기 모듈 상수를 작업자 프로세스 안에서 바꾼다 -- 인자로 빼면 replay·_row 를 다 고쳐야 함
    rows, evs, cts, covs = [], [], [], []
    c0 = [None]

    def on_sec(t, bid, ask, bb, ba):
        if c0[0] is None:
            c0[0] = t
        m = (bb + ba) / 2
        w50, w10 = int(m * 50e-4), int(m * 10e-4)
        b50, a50 = bid[bb - w50:bb + 1], ask[ba:ba + w50 + 1]
        jb, ja = int(b50.argmax()), int(a50.argmax())
        rows.append((t - 1, c0[0], bb, ba, bid[bb - w10:bb + 1].sum(), ask[ba:ba + w10 + 1].sum(), b50.sum(), a50.sum(),
                     bb - w50 + jb, b50[jb], ba + ja, a50[ja]))

    i = 0
    while i < len(paths):
        cur, c0[0] = [0, 0], None
        gen = _chain(paths[i:], cur)
        ev, ct, cov = W.replay(gen, W.TICKS[coin], ctrl_every_s=30, on_sec=on_sec, min_share=0.01)
        gen.close()
        evs += ev; cts += ct
        k = i + cur[0]
        covs.append(dict(cov, start=Path(paths[i]).name[:13], stop=Path(paths[k]).name[:13]))
        if cov["status"] == "ok":
            break
        i = k if (cur[1] <= 2 and k > i) else k + 1   # 파일 경계 끊김이면 그 파일 스냅샷에서, 아니면 다음 파일에서 다시

    keep = lambda D: D[(D.t >= keep_from) & (D.t < keep_to)].reset_index(drop=True) if len(D) else D  # noqa: E731
    S = keep(pd.DataFrame(rows, columns=SEC_COLS))
    E = keep(pd.DataFrame(evs))
    C = keep(pd.DataFrame(cts))
    if len(C):
        C = C[C.share >= wall_share].reset_index(drop=True).assign(age_trk=lambda D: D[f"age{W.KS[0]}"])
    if len(E):
        E = E[E.share >= 0.01].reset_index(drop=True).assign(age_trk=lambda D: D[f"age{W.KS[0]}"])   # 추적 문턱 이상으로 서 있던 시간
        d = pd.Timestamp(keep_from, unit="ms")
        days = [(d + pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in (-1, 0, 1)]
        tape = _tape(coin, days, ["ts_sec", "price_bin", "buy_qty", "sell_qty"])
        bad = tape_bad(tape.ts_sec.to_numpy(), E.t.to_numpy() // 1000)
        tape = tape[(tape.ts_sec >= keep_from // 1000 - 2) & (tape.ts_sec < keep_to // 1000 + 2)]
        tape = tape.groupby(["ts_sec", "price_bin"], as_index=False).sum() if len(tape) else pd.DataFrame()
        E = W.fill_share(E, tape, W.BUCKETS[coin])
        E.loc[bad, "fill"] = np.nan
    return S, E, C, dict(day=pd.Timestamp(keep_from, unit="ms").strftime("%Y-%m-%d"), chains=covs, n_sec=len(S), n_ev=len(E))


def build(coin):
    files = sorted(glob.glob(str(DATA / f"live/orderflow/depthdiff/{coin}USDT/*.jsonl*")))
    files = [f for f in files if not (f.endswith(".jsonl") and f + ".gz" in files)]   # 백업 중 복사된 미완성 사본 = 완성 .gz 의 앞부분
    hour = lambda f: pd.Timestamp(Path(f).name[:13] + ":00Z").value // 1_000_000  # noqa: E731
    days = sorted({Path(f).name[:10] for f in files})
    jobs = []
    for d in days:
        lo = pd.Timestamp(d + "T00:00Z").value // 1_000_000
        hi = lo + 86_400_000
        jobs.append((coin, [f for f in files if lo - OVERLAP_H * 3_600_000 <= hour(f) < hi], lo, hi))
    OUT.mkdir(parents=True, exist_ok=True)
    Ss, Es, Cs, covs = [], [], [], []
    with ProcessPoolExecutor(max_workers=9) as ex:
        for S, E, C, cov in ex.map(day_job, jobs):
            Ss.append(S); Es.append(E); Cs.append(C); covs.append(cov)
            print(cov, flush=True)
    pd.concat(Ss).sort_values("t").to_parquet(OUT / f"sec_{coin}.parquet")
    pd.concat([e for e in Es if len(e)]).sort_values("t").to_parquet(OUT / f"events_{coin}.parquet")
    pd.concat([c for c in Cs if len(c)]).sort_values("t").to_parquet(OUT / f"controls_{coin}.parquet")
    (OUT / f"cover_{coin}.json").write_text(json.dumps(covs, indent=0))


def _mid_at(ts, mid, x):
    """x 이하 마지막 1초 mid (3초 넘게 비면 NaN)."""
    i = np.searchsorted(ts, x, "right") - 1
    ok = (i >= 0) & (x - ts[np.clip(i, 0, None)] <= 3000)
    return np.where(ok, mid[np.clip(i, 0, len(mid) - 1)], np.nan)


def wall_age(t, on, idx):
    """같은 가격칸 벽이 서 있던 시간(ms, 인과). 그 칸이 WALL_BACK_MS 넘게 최대 벽이 아니었으면 새로 센다."""
    start, seen = {}, {}
    out = np.zeros(len(t))
    for k in np.flatnonzero(on):
        i, tk = idx[k], t[k]
        if tk - seen.get(i, -10**18) > WALL_BACK_MS:
            start[i] = tk
        seen[i] = tk
        out[k] = tk - start[i]
    return out


def toward(E, t, mid, dts):
    """사건마다 벽 쪽(+) 이동 bp. dts = {이름: (시작 오프셋, 끝 오프셋)} ms. 봉 경계로 자르지 않는다(사건 진단용)."""
    sg = np.where(E.side.to_numpy() == 1, 1, -1)
    f = lambda dt: _mid_at(t, mid, E.t.to_numpy() + dt)  # noqa: E731
    return {k: np.log(f(b) / f(a)) * 1e4 * sg for k, (a, b) in dts.items()}


def load(kind, coin):
    """events / controls parquet. 10-26 코인별 보정 전 산출물(ETH·BTC)은 age_trk 가 없고 age20 이 같은 뜻."""
    D = pd.read_parquet(OUT / f"{kind}_{coin}.parquet")
    D = D.drop_duplicates(["t", "side", "px"]) if kind == "events" else D
    return D if "age_trk" in D else D.assign(age_trk=D.age20)


def classify(E, wall_share=WALL_SHARE):
    """큰 벽(5%+, 30초+, ±50bp 안) 사건 -> eat(체결 80%+) · fake(체결 20% 미만 & 5bp 안) · pull(체결 20% 미만 & 밖) · mixed · unk(테이프 없음)."""
    big = E[(E.share >= wall_share) & (E.age_trk >= BIG_AGE_MS) & (E.dist_bp <= 50)].copy()
    big["kind"] = np.select([big.fill.isna(), big.fill >= .8, (big.fill < .2) & (big.dist_bp <= NEAR_BP), big.fill < .2],
                            ["unk", "eat", "fake", "pull"], "mixed")
    big["usd"] = (big.qref - big.q) * big.px
    return big


def bars_from(S: pd.DataFrame, E: pd.DataFrame, tick: int, wall_share: float = WALL_SHARE) -> pd.DataFrame:
    S = S.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    E = E.drop_duplicates(["t", "side", "px"])
    t = S.t.to_numpy()
    mid = (S.bb.to_numpy() + S.ba.to_numpy()) / 2 / tick
    S["bar"] = t // BAR
    S["mid"] = mid
    S["warm"] = (S.t - S.c0) < WARM_MS
    S["imb10"] = (S.b10 - S.a10) / (S.b10 + S.a10)
    S["imb50"] = (S.b50 - S.a50) / (S.b50 + S.a50)
    S["dep10"] = (S.b10 + S.a10) * mid
    S["dep50"] = (S.b50 + S.a50) * mid
    for s, q, i, d in (("b", "wb_q", "wb_i", "b50"), ("a", "wa_q", "wa_i", "a50")):
        sh = (S[q] / S[d]).to_numpy()
        on = sh >= wall_share
        S[f"sh_{s}"] = sh
        S[f"on_{s}"] = on
        S[f"age_{s}"] = wall_age(t, on, S[i].to_numpy())
        S[f"age_pick_{s}"] = np.where(on, S[f"age_{s}"], -1.0)   # 벽 아닌 초는 뽑히지 않게
        S[f"dist_{s}"] = np.abs(S[i] / tick - mid) / mid * 1e4
        S[f"usd_{s}"] = S[q] * mid
    g = S.groupby("bar")
    B = pd.DataFrame({
        "n_sec": g.size(), "warm": g.warm.any(), "o": g.mid.first(), "h": g.mid.max(), "l": g.mid.min(), "c": g.mid.last(),
        "imb10": g.imb10.mean(), "imb50": g.imb50.mean(), "imb50_last": g.imb50.last(),
        "dep10": g.dep10.mean(), "dep50": g.dep50.mean(), "dep50_min": g.dep50.min(),
    })
    for s in "ba":
        B[f"wall_frac_{s}"] = g[f"on_{s}"].mean()
        B[f"wall_age_{s}"] = g[f"age_{s}"].max() / 1000
        B[f"sh_max_{s}"] = g[f"sh_{s}"].max()
        k = S.loc[g[f"age_pick_{s}"].idxmax()]          # 봉 안에서 가장 오래 선 벽의 위치·크기
        on = k[f"on_{s}"].to_numpy()
        B[f"wall_px_{s}"] = np.where(on, (k["wb_i" if s == "b" else "wa_i"] / tick).to_numpy(), np.nan)
        B[f"wall_dist_{s}"] = np.where(on, k[f"dist_{s}"].to_numpy(), np.nan)
        B[f"wall_usd_{s}"] = np.where(on, k[f"usd_{s}"].to_numpy(), np.nan)
    book_cols = [c for c in B.columns if c not in ("n_sec", "warm", "o", "h", "l", "c")]
    B.loc[B.warm, book_cols] = np.nan
    B["ret"] = np.log(B.c / B.o) * 1e4
    B["rng"] = np.log(B.h / B.l) * 1e4

    big = classify(E, wall_share)
    big["bar"] = big.t // BAR
    pre = _mid_at(t, mid, big.t.to_numpy() - JUMP_WIN)
    post = _mid_at(t, mid, np.minimum(big.t.to_numpy() + JUMP_WIN, (big.bar.to_numpy() + 1) * BAR - 1))   # 봉 끝에서 자른다
    big["jump"] = np.log(post / pre) * 1e4 * np.where(big.side == 1, 1, -1)
    ev_cols = []
    for kind in ("eat", "fake", "pull"):
        X = big[big.kind == kind]
        for s, sv in (("b", 0), ("a", 1)):
            G = X[X.side == sv].groupby("bar")
            B[f"{kind}_n_{s}"] = G.size().reindex(B.index).fillna(0)
            B[f"{kind}_usd_{s}"] = G.usd.sum().reindex(B.index).fillna(0)
            B[f"{kind}_jump_{s}"] = G.jump.max().reindex(B.index)
            ev_cols += [f"{kind}_n_{s}", f"{kind}_usd_{s}", f"{kind}_jump_{s}"]
    unk = big[big.kind == "unk"].groupby("bar").size().reindex(B.index).fillna(0) > 0
    B.loc[unk | B.warm, ev_cols] = np.nan        # 판정 못 한 사건이 있는 봉 · 덜 찬 북 봉 = 0 이 아니라 모름
    return B


def bars(coin):
    S = pd.read_parquet(OUT / f"sec_{coin}.parquet")
    B = bars_from(S, load("events", coin), W.TICKS[coin], cfg(coin)[0])
    tf = sorted(glob.glob(str(DATA / f"lake/binance/tape/coin={coin}/date=*/*.parquet")))
    tp = pd.concat([pd.read_parquet(p, columns=["ts_sec", "buy_qty", "sell_qty"]) for p in tf])
    tbad = pd.Series(tape_bad(tp.ts_sec.to_numpy(), tp.ts_sec.unique()), index=tp.ts_sec.unique())
    tp = tp.groupby(tp.ts_sec * 1000 // BAR)[["buy_qty", "sell_qty"]].sum()
    tp = tp[~tbad.groupby(tbad.index * 1000 // BAR).any().reindex(tp.index).fillna(True)]   # 테이프 빈 구간 봉은 체결량 모름
    B = B.join(tp.rename(columns={"buy_qty": "tbuy", "sell_qty": "tsell"}))
    B.index = pd.to_datetime(B.index * BAR, unit="ms", utc=True)
    B.index.name = "bar"
    B.to_parquet(OUT / f"bars_{coin}.parquet")
    print(B.describe().T.to_string())


def analyze(coin):
    """레포트 숫자·그림 (설명용 통계 · 판정 아님)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    B = pd.read_parquet(OUT / f"bars_{coin}.parquet").asfreq("5min")   # 시간 격자 -> rolling·shift 가 실제 시간
    S = pd.read_parquet(OUT / f"sec_{coin}.parquet").drop_duplicates("t").sort_values("t")
    E, Cn = load("events", coin), load("controls", coin)
    ws = cfg(coin)[0]
    t, mid = S.t.to_numpy(), ((S.bb + S.ba) / 2 / W.TICKS[coin]).to_numpy()
    warm = ((S.t - S.c0) < WARM_MS).to_numpy()
    days = (~warm).sum() / 86_400        # 실제로 덮은(덜 찬 구간 뺀) 초
    ok_t = lambda x: ~warm[np.clip(np.searchsorted(t, x, "right") - 1, 0, None)]  # noqa: E731
    E = E[ok_t(E.t.to_numpy())]
    Cn = Cn[ok_t(Cn.t.to_numpy())]
    R = dict(coin=coin, days=round(days, 2), span=[str(B.index[0]), str(B.index[-1])],
             chains=pd.Series([c["status"] for d in json.loads((OUT / f"cover_{coin}.json").read_text()) for c in d["chains"]]).value_counts().to_dict())

    # 1) 벽 크기 기준: 몫별 하루 사라짐·먹힘 건수 (테이프 판정 가능한 것만)
    E["usd"] = (E.qref - E.q) * E.px
    R["unk_share"] = round(float(E.fill.isna().mean()), 3)
    R["share_sweep"] = [dict(share=sh, pull_per_day=round(((E.share >= sh) & (E.age_trk >= 3000) & (E.fill < .2)).sum() / days, 1),
                             eat_per_day=round(((E.share >= sh) & (E.age_trk >= 3000) & (E.fill >= .8)).sum() / days, 1),
                             usd_med_m=round(E.usd[E.share >= sh].median() / 1e6, 2)) for sh in (0.01, 0.02, 0.05, 0.10, 0.20)]
    # 2) 큰 벽 사건 전후 이동 (벽 쪽 +) vs 대조 = 같은 기준(5%+, 30초+)으로 서 있고 사라지지 않은 벽
    dts = dict(pre5=(-5000, 0), post5=(0, 5000), post60=(0, 60000))
    big = classify(E, ws)
    big = big.assign(**toward(big, t, mid, dts))
    se = lambda x: x.std() / np.sqrt(x.count())  # noqa: E731
    R["events"] = {k: dict(per_day=round(len(G) / days, 1), usd_med_m=round(G.usd.median() / 1e6, 1), dist_med=round(G.dist_bp.median(), 2),
                           crossed=round(G.crossed.mean(), 2), **{c: round(G[c].mean(), 2) for c in dts}, post60_se=round(se(G.post60), 2))
                   for k, G in big.groupby("kind")}
    Cn = Cn[(Cn.share >= ws) & (Cn.age_trk >= BIG_AGE_MS) & (Cn.dist_bp <= 50)].copy()
    Cn = Cn.assign(**toward(Cn, t, mid, dts), near=Cn.dist_bp <= NEAR_BP)
    R["controls"] = {("near" if k else "far"): dict(n=len(G), **{c: round(G[c].mean(), 2) for c in dts}, post60_se=round(se(G.post60), 2))
                     for k, G in Cn.groupby("near")}

    # 3) 봉 지표 서로·가격과의 관계 (n_sec 240 미만·덜 찬 봉 제외)
    B["liq"] = np.log(B.dep10 / B.dep10.rolling(288, min_periods=100).median().shift(1))
    B["wallbal"] = B.wall_frac_b - B.wall_frac_a
    for k in ("eat", "fake", "pull"):
        B[f"{k}_bal"] = B[f"{k}_n_a"] - B[f"{k}_n_b"]
    B["timb"] = (B.tbuy - B.tsell) / (B.tbuy + B.tsell)
    B["ret_prev1h"] = B.ret.rolling(12, min_periods=12).sum().shift(1)
    B["ret_next"] = B.ret.shift(-1)
    B = B[(B.n_sec >= 240) & ~B.warm.fillna(True).astype(bool)]
    R["bars"] = len(B)
    R["bars_event_unknown"] = round(float(B.eat_n_a.isna().mean()), 3)
    F = ["imb50", "imb10", "wallbal", "liq", "eat_bal", "fake_bal", "pull_bal"]
    C = ["ret", "ret_prev1h", "rng", "timb", "ret_next"]
    R["corr"] = B[F + C].corr("spearman").loc[F, F + C].round(2).to_dict("index")
    R["autocorr"] = {k: [round(B[k].corr(B[k].shift(1)), 2), round(B[k].corr(B[k].shift(12)), 2)]
                     for k in F}   # 격자에서 빠진 봉은 NaN 이라 짝에서 빠진다
    Z = B[F].dropna()
    w = np.linalg.eigvalsh(np.corrcoef(((Z - Z.mean()) / Z.std()).T.values))[::-1]
    R["pca_share"] = (w / w.sum()).round(3).tolist()
    q = pd.qcut(B.imb50, 5, labels=False)
    R["tilt_quintile"] = B.groupby(q).agg(imb50=("imb50", "mean"), ret_prev1h=("ret_prev1h", "mean"), ret=("ret", "mean"),
                                          ret_next=("ret_next", "mean")).round(3).reset_index(drop=True).to_dict("records")
    Bk = B.dropna(subset=["eat_n_a"])
    R["event_bars"] = {k: dict(bars_per_day=round((Bk[k] > 0).mean() * 288, 1), ret=round(Bk.ret[Bk[k] > 0].mean(), 1),
                               rng=round(Bk.rng[Bk[k] > 0].mean(), 1), ret_next=round(Bk.ret_next[Bk[k] > 0].mean(), 2))
                       for k in ("eat_n_a", "eat_n_b", "fake_n_a", "fake_n_b", "pull_n_a", "pull_n_b")}
    R["rng_all"] = round(B.rng.mean(), 1)
    R["imb50_sd"] = round(B.imb50.std() * 100, 1)
    if coin != "ETH":   # 사례 그림은 ETH 만
        (OUT / f"result_{coin}.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float))
        print(json.dumps(R, indent=1, ensure_ascii=False, default=float))
        return

    # 그림 (투명 배경 · 중간 회색 글자 -> 밝은/어두운 화면 둘 다)
    FG, GB, GA, AM = "#8b909c", "#2f9e62", "#d65a4c", "#c98500"
    from matplotlib import font_manager
    for fp in ("/mnt/c/Windows/Fonts/malgun.ttf",):   # WSL 개발 PC 의 한글 글꼴 (없으면 DejaVu -> 한글 깨짐)
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
    plt.rcParams.update({"text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": "#8b909c55",
                         "font.size": 10, "font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False})
    fig_dir = OUT / "figs"
    fig_dir.mkdir(exist_ok=True)

    def day_fig(day):
        D = B[(B.index >= pd.Timestamp(day + "T00:00Z")) & (B.index < pd.Timestamp(day + "T00:00Z") + pd.Timedelta(days=1))]
        x = ((D.index - pd.Timestamp(day + "T00:00Z")) / pd.Timedelta(minutes=5)).to_numpy()
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3], hspace=0.06))
        up = D.c >= D.o
        a1.vlines(x, D.l, D.h, color=np.where(up, GB, GA), lw=0.8)
        a1.bar(x, (D.c - D.o).abs().clip(lower=0.3), bottom=np.minimum(D.o, D.c), color=np.where(up, GB, GA), width=0.7)
        a1.scatter(x, D.wall_px_b, s=4, color=GB, alpha=0.45, lw=0, label="가장 오래 선 매수벽(5%+)")
        a1.scatter(x, D.wall_px_a, s=4, color=GA, alpha=0.45, lw=0, label="가장 오래 선 매도벽(5%+)")
        for col, ys, mk, c, up_ in (("eat_n_a", D.h, "^", GB, 1), ("eat_n_b", D.l, "v", GA, -1), ("fake_n_a", D.h, "D", AM, 1), ("fake_n_b", D.l, "D", AM, -1)):
            m = (D[col] > 0).to_numpy()
            a1.scatter(x[m], ys[m] + up_ * (D.h - D.l).mean() * 0.9, marker=mk, s=28, color=c, edgecolor="none", zorder=5)
        a1.legend(loc="upper left", frameon=False, fontsize=8, markerscale=3)
        a1.set_ylabel("ETH")
        a1.grid(axis="y", color=FG, alpha=0.15, lw=0.6)
        v = D.imb50 * 100
        a2.bar(x, v, color=np.where(v >= 0, GB, GA), width=0.8)
        a2.axhline(0, color="#8b909c66", lw=0.6)
        a2.set_ylabel("기울기 %")
        a2.grid(axis="y", color=FG, alpha=0.15, lw=0.6)
        ticks = np.arange(0, 288, 36)
        a2.set_xticks(ticks, [(pd.Timestamp(day + "T00:00Z") + pd.Timedelta(minutes=5 * int(i)) + pd.Timedelta(hours=9)).strftime("%H:%M") for i in ticks])
        a2.set_xlabel(f"{day} 09:00 ~ 다음 날 09:00 KST (5분봉)")
        for a in (a1, a2):
            a.spines[["top", "right"]].set_visible(False)
        fig.savefig(fig_dir / f"day_{day}.png", dpi=150, bbox_inches="tight", transparent=True)
        plt.close(fig)

    def case_fig(name, ev_t, side_label):
        lo, hi = ev_t - 600_000, ev_t + 600_000
        Q = S[(S.t >= lo) & (S.t < hi)]
        tm = (Q.t.to_numpy() - ev_t) / 60_000
        m_ = (Q.bb + Q.ba) / 2 / W.TICKS[coin]
        fig, ax = plt.subplots(figsize=(10, 3.6))
        for i, q, d, c in (("wb_i", "wb_q", "b50", GB), ("wa_i", "wa_q", "a50", GA)):
            sh = Q[q] / Q[d]
            on = sh >= WALL_SHARE
            ax.scatter(tm[on], Q[i][on] / W.TICKS[coin], s=(sh[on] * 60).clip(1, 30), color=c, alpha=0.35, lw=0)
        ax.plot(tm, m_, color=FG, lw=1.1)
        for b in np.arange(np.ceil(lo / BAR) * BAR, hi, BAR):
            ax.axvline((b - ev_t) / 60_000, color=FG, alpha=0.2, lw=0.6)
        ax.axvline(0, color=AM, lw=1.2, ls="--")
        ax.set_xlabel(f"사건 기준 분 ({side_label}) · 세로 선 = 5분봉 경계 · 점 크기 = 벽 몫")
        ax.set_ylabel("ETH")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=FG, alpha=0.15, lw=0.6)
        fig.savefig(fig_dir / f"case_{name}.png", dpi=150, bbox_inches="tight", transparent=True)
        plt.close(fig)

    day_fig("2026-10-02")
    R["cases"] = {}
    for name, kind, side in (("eat_bid", "eat", 0), ("fake_ask", "fake", 1), ("fake_bid", "fake", 0)):
        X = big[(big.kind == kind) & (big.side == side) & ~big.crossed].sort_values("usd")   # 가장 큰 벽 (가격이 지나간 것은 제외)
        e = X.iloc[-1]
        case_fig(name, int(e.t), f"{'매수' if side == 0 else '매도'}벽 ${e.usd / 1e6:.0f}M · {e.dist_bp:.1f}bp")
        R["cases"][name] = dict(kst=str(pd.Timestamp(int(e.t), unit="ms") + pd.Timedelta(hours=9)), px=e.px, usd_m=round(e.usd / 1e6, 1),
                                dist=round(e.dist_bp, 2), fill=round(e.fill, 2), age_s=round(e.age_trk / 1000), post60=round(e.post60, 1),
                                post600=round(float(toward(X.iloc[[-1]], t, mid, dict(p=(0, 600_000)))["p"][0]), 1))
    (OUT / f"result_{coin}.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(R, indent=1, ensure_ascii=False, default=float))


def selftest():
    import tempfile
    tick = 100
    # 1초 패널 합성: 봉 0 은 299.999초까지, 300.000초부터 봉 1. c0 = 사슬 시작(충분히 과거 -> 덜 찬 구간 아님)
    t = np.arange(0, 600_000, 1000) + 1
    S = pd.DataFrame({"t": t, "c0": -10**9, "bb": 9999, "ba": 10001, "b10": 3.0, "a10": 1.0, "b50": 100.0, "a50": 100.0,
                      "wb_i": 9990, "wb_q": 1.0, "wa_i": 10010, "wa_q": 5.0})
    S.loc[t >= 300_000, "ba"] = 10021                        # 봉 1 에서 mid +10틱
    S.loc[(t >= 100_000) & (t < 200_000), "wa_i"] = 10012    # 100초 동안 다른 칸 -> 원래 칸 나이 다시 시작
    E = pd.DataFrame({"t": [299_999, 301_000], "side": [1, 1], "px": [100.10, 100.10], "qref": [5.0, 5.0], "q": [0.0, 0.0],
                      "share": [0.05, 0.05], "age_trk": [40_000, 40_000], "crossed": [False, False], "fill": [0.0, 0.0],
                      "dist_bp": [3.0, 3.0]})
    B = bars_from(S, E, tick)
    assert list(B.index) == [0, 1] and (B.n_sec == 300).all()
    assert abs(B.imb10.iloc[0] - 0.5) < 1e-12 and abs(B.imb50.iloc[0]) < 1e-12
    assert B.wall_frac_a.iloc[0] == 1 and B.wall_frac_b.iloc[0] == 0          # 매도 5% 벽, 매수 1%
    assert abs(B.wall_age_a.iloc[0] - 99) < 1e-9                              # 200초에 칸 복귀 -> 200~299초 = 99초
    assert abs(B.wall_age_a.iloc[1] - 399) < 1e-9 and np.isnan(B.wall_px_b.iloc[0])
    assert B.fake_n_a.tolist() == [1, 1] and B.fake_usd_a.iloc[0] > 0 and B.pull_n_a.sum() == 0          # 경계 사건이 제 봉으로
    E2 = E.copy(); E2["t"] = [296_000, 302_000]
    J = bars_from(S, E2, tick).fake_jump_a
    assert abs(J.iloc[0]) < 1e-9, "봉 0 사건이 봉 1 mid 를 봤다"                # +5초 = 301초 이지만 봉 끝에서 잘림
    assert J.iloc[1] > 9                                                       # 297초(봉 0) -> 307초: +10틱 ≈ +10bp
    # 잠깐(5초) 다른 칸이 최대였다 돌아오면 나이 이어감 · 벽 아닌 초가 «가장 오래 선 벽»으로 뽑히지 않음
    S2 = S.copy(); S2["wa_i"] = 10010
    S2.loc[(t >= 50_000) & (t < 55_000), "wa_i"] = 10012
    S2.loc[t < 10_000, "wa_q"] = 1.0
    B2 = bars_from(S2, E.iloc[:0], tick)
    assert B2.wall_age_a.iloc[0] > 280 and B2.wall_px_a.iloc[0] == 100.10
    assert bars_from(S, E, tick, cfg("SOL")[0]).wall_frac_a.iloc[0] == 0 and cfg("BTC") == (0.05, (20, 50, 100))
    # 테이프 판정 불가 사건이 있는 봉 = 사건 열 NaN (0 아님)
    E3 = E.copy(); E3["fill"] = [0.0, np.nan]
    B3 = bars_from(S, E3, tick)
    assert B3.fake_n_a.iloc[0] == 1 and np.isnan(B3.fake_n_a.iloc[1]) and np.isnan(B3.eat_n_b.iloc[1])
    # 사슬 시작 30분 안 = 북 값 NaN
    S4 = S.copy(); S4["c0"] = 0
    assert bars_from(S4, E, tick).imb50.isna().all()
    # 테이프 빈 구간: 0~100초, 200~300초만 있음 -> 100~200 빈 구간 ±60초는 판정 불가, 범위 밖도
    sec = np.r_[np.arange(0, 101), np.arange(200, 301)]
    assert tape_bad(sec, np.array([10, 39, 41, 150, 259, 261, 500])).tolist() == [False, False, True, True, True, False, True]
    assert tape_bad(np.arange(0, 101), np.array([-1, 50, 101])).tolist() == [True, False, True]   # 빈 구간 없는 날

    # 하루 사슬: 시간 파일 둘을 diff 가 이어지면 한 사슬, 끊기면 둘째 파일 스냅샷에서 다시
    def hour_file(d, name, snap_u, us, pu0, t0):
        snap = {"lastUpdateId": snap_u, "bids": [[f"{100 - j * 0.01:.2f}", "1"] for j in range(60)],
                "asks": [[f"{100.01 + j * 0.01:.2f}", "1"] for j in range(60)]}
        lines = [json.dumps({"_snapshot": snap})]
        pu = pu0
        for n, u in enumerate(us):
            lines.append(json.dumps({"e": "depthUpdate", "T": t0 + n * 100, "U": u, "u": u, "pu": pu, "b": [["99.99", str(1 + n % 2)]], "a": []}))
            pu = u
        (Path(d) / name).write_text("\n".join(lines) + "\n")
        return str(Path(d) / name)

    with tempfile.TemporaryDirectory() as d:
        a = hour_file(d, "2026-01-01T00.jsonl", 10, range(10, 60), 9, 1_000_000)          # 5초
        b = hour_file(d, "2026-01-01T01.jsonl", 59, range(60, 110), 59, 1_005_000)        # 이어짐
        S5, _, _, cov = day_job(("ETH", [a, b], 0, 10**13))
        assert len(cov["chains"]) == 1 and cov["chains"][0]["status"] == "ok" and S5.c0.nunique() == 1 and S5.t.max() >= 1_008_000
        b2 = hour_file(d, "2026-01-01T02.jsonl", 200, range(200, 250), 199, 1_005_000)    # 끊김(pu 불연속)
        S6, _, _, cov = day_job(("ETH", [a, b2], 0, 10**13))
        assert [c["status"] for c in cov["chains"]] == ["gap", "ok"] and S6.c0.nunique() == 2, cov
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build")
    ap.add_argument("--bars")
    ap.add_argument("--analyze")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.build:
        build(a.build)
    elif a.bars:
        bars(a.bars)
    elif a.analyze:
        analyze(a.analyze)
    else:
        ap.print_help(); sys.exit(1)
