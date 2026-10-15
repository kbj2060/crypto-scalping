"""칸별 DOM 재현 → 그림 그대로의 흡수·아이스버그 검정 (2026-10-16, 사용자 «검증 잘못하는 거 아니냐 → 1번(칸별 DOM)으로 진행»).

앞 판(research_eth_dom_playbook_20261016.py)의 약점 = ±10bp 합계로 흡수를 봤다. 이번엔 가격 칸($0.1, 테이프 price_bin 과 같은 반올림 칸)마다.
결과 전 고정(사전등록) -- ETH depthdiff(백업 미러) 재생 2026-09-16 ~ 10-13 19:00 UTC, 30초 창.
- 재생: 10-07 order_battle replay_file 과 같은 동기화(U/u/pu)·끊김 규칙. 창마다 창 시작 mid 칸 기준 ±30칸(±$3) 매수·매도 잔량의 시작·최소·최대·끝,
  창 안 최저 최우선 매수·최고 최우선 매도.
- 매수 흡수(롱 쪽; 매도 흡수 = 거울) 칸 P, 창 w:
  ① P 가 창 시작 최우선 매수에서 2칸 안 ② 그 칸 테이커 매도량 ≥ 직전 24h 창별 최대 칸 매도량 95분위
  ③ 버팀: 창 안 최우선 매수 최저 ≥ P 칸 아래끝(P−0.05) ④ 다시 채움: 체결량 ≥ 시작 잔량 & 끝 잔량 ≥ 시작 잔량의 50%
  아이스버그 = 흡수 & 체결량 ≥ 창 안 최대 표시 잔량 × 2.  대조 «뚫린 큰 체결» = ①② 같고 ③ 실패.
- 사건 검정(그림 주장 «흡수 = 매수가 받아냈다 → 오른다»): 창 끝에서 1·5·15분 mid 수익(쪽 부호) · 15분 안 칸 아래끝 −$0.1 뚫림 비율. 일 블록 CI.
- 매매(그림 규칙): 핵심 레벨(앞 판 정의) ±5bp·반대편에서 옴 + 칸 흡수 + 불균형(최우선 5칸 매수 잔량 ≥ 매도 5칸 × 2) + 30초 확인(앞 판 정의)
  → 시장가(4bp) / 지정가. 손절 = P 칸 아래끝 − 3bp · 1R 절반 · 본전 · 다음 레벨/3R · 4h. 대조 = 레벨 없이 칸 흡수 + 불균형 + 확인.
- 판정: 사건 검정 흡수 − 뚫린 큰 체결 5분 차 CI 하한 > 0 · 매매 건당 순 bp CI 하한 > 0.
🆕전진 사전등록(2026-10-16 사용자 «전진 사전등록으로 고정»): 아이스버그 정의 그대로 · 2026-10-16 00:00 UTC 이후 사건만 ·
  결과 = 30초 창 시작 mid(연구 1초판과 상관 1.0·평균 7.61=7.61 확인) · 판정 = 아이스버그 ≥150건: r15 평균 > 1.41bp 이고 일 블록 CI 하한 > 0
  (보조: 15분 뚫림 < 뚫린 큰 체결). 판정 전엔 --status(건수만). 동결 docs/prereg/eth_dom_iceberg_forward_frozen_20261016.json.
사용: python scripts/research_eth_dom_levels_absorption_20261016.py --replay | (인자 없이) 검정 | --selftest | --status | --judge
"""
import gzip, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_eth_dom_playbook_20261016 as P  # noqa: E402

DD = Path.home() / "backups/crypto-scalping-server/data/live/orderflow/depthdiff/ETHUSDT"
OUT = Path("tmp/dom_levels_absorption_20261016")
TICK, IDX_MAX, K, WIN = 100, 10_000_000, 30, 30
T0, T1 = P.T0, P.T1


def _open(p):
    return gzip.open(p, "rb") if str(p).endswith(".gz") else open(p, "rb")


def buckets(arr, base):
    """반올림 칸 b = round(가격×10) = 눈금 [10b−5, 10b+4] → base−K..base+K 칸 합."""
    lo = (base - K) * 10 - 5
    return arr[lo:lo + (2 * K + 1) * 10].reshape(2 * K + 1, 10).sum(1)


def replay_levels(path):
    bid = np.zeros(IDX_MAX); ask = np.zeros(IDX_MAX); out = []
    with _open(path) as f:
        first = json.loads(f.readline()); snap = first.get("_snapshot") if isinstance(first, dict) else None
        if not snap:
            return None
        for p, q in snap["bids"]:
            bid[int(round(float(p) * TICK))] = float(q)
        for p, q in snap["asks"]:
            ask[int(round(float(p) * TICK))] = float(q)
        last_u, synced, valid, cur = snap["lastUpdateId"], False, 1, None
        bb, ba = int(np.flatnonzero(bid)[-1]), int(np.flatnonzero(ask)[0])
        W = None

        def best():
            nonlocal bb, ba
            w = max(int(bb * 0.02), 100)
            nz = np.flatnonzero(bid[max(bb - w, 0):bb + w + 1]); bb = max(bb - w, 0) + nz[-1] if len(nz) else bb
            nz = np.flatnonzero(ask[max(ba - w, 0):ba + w + 1]); ba = max(ba - w, 0) + nz[0] if len(nz) else ba

        def tick_sec(sec):
            nonlocal W
            w = sec // WIN
            if W is not None and W["w"] != w:
                out.append(W); W = None
            if W is None:
                base = int(round((bb + ba) / 2 / 10))
                b0, a0 = buckets(bid, base), buckets(ask, base)
                W = {"w": w, "base": base, "valid": valid, "bS": b0, "bMin": b0.copy(), "bMax": b0.copy(), "bE": b0,
                     "aS": a0, "aMin": a0.copy(), "aMax": a0.copy(), "aE": a0, "bbMin": bb, "baMax": ba, "bb0": bb, "ba0": ba}
            b1, a1 = buckets(bid, W["base"]), buckets(ask, W["base"])
            np.minimum(W["bMin"], b1, out=W["bMin"]); np.maximum(W["bMax"], b1, out=W["bMax"])
            np.minimum(W["aMin"], a1, out=W["aMin"]); np.maximum(W["aMax"], a1, out=W["aMax"])
            W["bE"], W["aE"] = b1, a1
            W["bbMin"] = min(W["bbMin"], bb); W["baMax"] = max(W["baMax"], ba); W["valid"] = min(W["valid"], valid)

        for line in f:
            try:
                m = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                valid = 0; continue
            if m.get("e") != "depthUpdate":
                continue
            if not synced:
                if m["u"] < last_u:
                    continue
                if not (m["U"] <= last_u <= m["u"]):
                    valid = 0
                synced = True
            elif m["pu"] != last_u:
                valid = 0
            last_u = m["u"]; sec = m["E"] // 1000
            if cur is None:
                cur = sec
            while sec > cur:
                tick_sec(cur); cur += 1
            for p, q in m["b"]:
                i = int(round(float(p) * TICK))
                if i < IDX_MAX:
                    bid[i] = float(q)
            for p, q in m["a"]:
                i = int(round(float(p) * TICK))
                if i < IDX_MAX:
                    ask[i] = float(q)
            best()
        if cur is not None:
            tick_sec(cur)
        if W is not None:
            out.append(W)
    if not out:
        return None
    S = lambda k: np.stack([o[k] for o in out]).astype(np.float32)
    A = lambda k, d=1.0: np.array([o[k] for o in out]) / d
    return {"w": A("w").astype(np.int64), "base": A("base").astype(np.int64), "valid": A("valid"), "bb0": A("bb0", TICK), "ba0": A("ba0", TICK),
            "bbMin": A("bbMin", TICK), "baMax": A("baMax", TICK), **{k: S(k) for k in ("bS", "bMin", "bMax", "bE", "aS", "aMin", "aMax", "aE")}}


def run_replay():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in DD.glob("*.jsonl*") if "2026-09-14T23" <= p.name[:13] <= "2026-10-13T19")
    parts = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, d in enumerate(ex.map(replay_levels, files)):
            if d is not None:
                parts.append(d)
            if i % 50 == 0:
                print(i, len(files), flush=True)
    R = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(R["w"], kind="stable"); R = {k: v[order] for k, v in R.items()}
    keep = np.r_[R["w"][1:] != R["w"][:-1], True]                          # 같은 창이 두 파일에 걸치면 뒤쪽(끝까지 본 쪽)
    np.savez_compressed(OUT / "levels30.npz", **{k: v[keep] for k, v in R.items()})
    print("창", int(keep.sum()), flush=True)


def tape_windows(R):
    rp = f"read_parquet('{P.LK}/binance/tape/coin=ETH/*/*.parquet', hive_partitioning=false)"
    t = duckdb.sql(f"select ts_sec // {WIN} w, price_bin b, sum(buy_qty) buy, sum(sell_qty) sell from {rp} "
                   f"where symbol='ethusdt' and ts_sec >= {int(R['w'][0]) * WIN} and ts_sec < {T1} group by 1, 2").df()
    wi = pd.Series(np.arange(len(R["w"])), index=R["w"])
    t = t[t.w.isin(wi.index)]
    rows = wi.loc[t.w.to_numpy()].to_numpy(); col = (t.b.to_numpy() - R["base"][rows] + K).astype(int)
    ok = (col >= 0) & (col <= 2 * K)
    SB = np.zeros((len(R["w"]), 2 * K + 1), np.float32); SS = np.zeros_like(SB)
    np.add.at(SB, (rows[ok], col[ok]), t.buy.to_numpy()[ok]); np.add.at(SS, (rows[ok], col[ok]), t.sell.to_numpy()[ok])
    return SB, SS


def events(R, SB, SS):
    """창마다 롱 쪽(매수 흡수)·숏 쪽(매도 흡수) 최우선 근처 최대 체결 칸 하나씩 → 사건 표."""
    n = len(R["w"]); rows = []
    bbk = np.round(R["bb0"] * 10).astype(int) - R["base"] + K; bak = np.round(R["ba0"] * 10).astype(int) - R["base"] + K
    qs = pd.Series(SS.max(1)).rolling(2880, min_periods=1000).quantile(.95).shift(1).to_numpy()
    qb = pd.Series(SB.max(1)).rolling(2880, min_periods=1000).quantile(.95).shift(1).to_numpy()
    for i in range(n):
        if R["valid"][i] != 1:
            continue
        for side in (1, -1):
            k0 = bbk[i] if side == 1 else bak[i]
            ks = [k for k in (range(k0 - 2, k0 + 1) if side == 1 else range(k0, k0 + 3)) if 0 <= k <= 2 * K]
            if not ks:
                continue
            ex = SS[i] if side == 1 else SB[i]
            k = max(ks, key=lambda c: ex[c]); vol = float(ex[k]); thr = qs[i] if side == 1 else qb[i]
            if not (np.isfinite(thr) and thr > 0 and vol >= thr):
                continue
            px = (R["base"][i] + k - K) * 0.1
            start = float((R["bS"] if side == 1 else R["aS"])[i, k]); end = float((R["bE"] if side == 1 else R["aE"])[i, k])
            mx = float((R["bMax"] if side == 1 else R["aMax"])[i, k])
            held = bool(R["bbMin"][i] >= px - 0.05) if side == 1 else bool(R["baMax"][i] <= px + 0.05)
            refill = start > 0 and vol >= start and end >= 0.5 * start
            bsum = R["bE"][i, max(0, bbk[i] - 4):bbk[i] + 1].sum(); asum = R["aE"][i, bak[i]:bak[i] + 5].sum()
            rows.append({"i": i, "t_end": int((R["w"][i] + 1) * WIN), "side": side, "px": px, "vol": vol, "start": start, "end": end, "max": mx,
                         "held": held, "absorb": held and refill, "iceberg": held and refill and vol >= 2 * mx,
                         "imb5": float(bsum / max(asum, 1e-9)) if side == 1 else float(asum / max(bsum, 1e-9))})
    return pd.DataFrame(rows)


def forward(E, D):
    idx0 = int(D["idx"][0]); mid = D["mid"]
    at = lambda t: mid[t - idx0] if 0 <= t - idx0 < len(mid) else np.nan
    for h in (60, 300, 900):
        E[f"r{h // 60}"] = [s * (at(t + h) / at(t) - 1) * 1e4 for t, s in zip(E.t_end, E.side)]
    br = []
    with np.errstate(all="ignore"):
        for t, s, px in zip(E.t_end, E.side, E.px):
            seg = mid[t - idx0:t - idx0 + 900]
            br.append(bool(np.nanmin(seg) < px - 0.15) if s == 1 else bool(np.nanmax(seg) > px + 0.15))
    E["broke15"] = br
    return E


def block(E, col, mask_a, mask_b=None, nb=1000):
    """평균(col | a) [− 평균(col | b)] · 일 블록 부트스트랩."""
    day = (E.t_end // 86400).to_numpy(); a = E[col].to_numpy(float)
    ix = {u: np.flatnonzero(day == u) for u in np.unique(day)}; ud = np.array(list(ix)); rng = np.random.default_rng(1)
    st = lambda s: np.nanmean(a[s][mask_a[s]]) - (np.nanmean(a[s][mask_b[s]]) if mask_b is not None else 0.0)
    bs = [st(np.concatenate([ix[u] for u in rng.choice(ud, len(ud))])) for _ in range(nb)]
    return round(float(st(np.arange(len(a)))), 3), [round(float(np.nanquantile(bs, q)), 3) for q in (.025, .975)]


def trades(E, D, use_level):
    n = len(D["mid"]); idx0 = int(D["idx"][0]); out = []; busy = -1
    for e in E[E.absorb & (E.imb5 >= 2)].sort_values("t_end").itertuples():
        i = int(e.t_end) - idx0
        if i <= busy or i >= n - 60 or (use_level and D["near"][i] != e.side):
            continue
        j = P.confirm(D, i, int(e.side))
        if j < 0:
            continue
        sl = (e.px - 0.05) * (1 - 3e-4) if e.side == 1 else (e.px + 0.05) * (1 + 3e-4)
        r_t, r_m = P.manage(D, j, int(e.side), sl, False), P.manage(D, j, int(e.side), sl, True)
        if r_t is not None:
            out.append((False, D["idx"][j], int(e.side), r_t[0], r_t[1]))
        if r_m is not None:
            out.append((True, D["idx"][j], int(e.side), r_m[0], r_m[1]))
        busy = r_t[2] if r_t else j
    return out


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    R = dict(np.load(OUT / "levels30.npz"))
    SB, SS = tape_windows(R)
    E = events(R, SB, SS)
    bk, vp = P.load(); D = P.signals(bk, P.levels(bk, vp))
    E = forward(E[(E.t_end >= T0) & (E.t_end < T1 - 900)].reset_index(drop=True), D)
    res = {"창": int(len(R["w"])), "큰 체결 사건": int(len(E)), "버팀": int(E.held.sum()), "흡수": int(E.absorb.sum()), "아이스버그": int(E.iceberg.sum())}
    ab, ice, br_ = E.absorb.to_numpy(), E.iceberg.to_numpy(), (~E.held).to_numpy()
    for h in ("r1", "r5", "r15"):
        res[f"{h} 흡수"] = block(E, h, ab); res[f"{h} 아이스버그"] = block(E, h, ice)
        res[f"{h} 뚫린 큰 체결"] = block(E, h, br_); res[f"{h} 흡수−뚫림"] = block(E, h, ab, br_)
    res["15분 안 칸 뚫림(흡수 / 아이스버그 / 뚫린 큰 체결)"] = [round(float(E.broke15[m].mean()), 3) for m in (ab, ice, br_)]
    for lab, lv in (("그림 규칙(레벨+칸 흡수+불균형+확인)", True), ("레벨 없이 칸 흡수+불균형+확인", False)):
        tr = trades(E, D, lv)
        for maker in (False, True):
            res[f"{lab} {'지정가' if maker else '시장가'}"] = P.summarize([(t, s, b, h) for mk, t, s, b, h in tr if mk == maker])
    for k, v in res.items():
        print(k, json.dumps(v, ensure_ascii=False), flush=True)
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    E.to_parquet(OUT / "events.parquet")


FWD0 = int(pd.Timestamp("2026-10-16", tz="UTC").timestamp())
FWD = Path("tmp/dom_iceberg_forward")
PREREG = Path("docs/prereg/eth_dom_iceberg_forward_frozen_20261016.json")
SRV = "llewyn@192.168.1.89"; SRV_ROOT = "/home/llewyn/crypto-scalping/data"


def sync():
    """서버에서 FWD0 하루 전부터의 depthdiff·봉인된 테이프만 읽어 온다(서버 쓰기 없음 · 공유 백업 미러 안 건드림)."""
    import subprocess
    day0 = pd.Timestamp(FWD0 - 86400, unit="s").strftime("%Y-%m-%d")
    for sub, remote, pat in (("depthdiff", f"{SRV_ROOT}/live/orderflow/depthdiff/ETHUSDT/", lambda n: n[:10] >= day0),
                             ("tape", f"{SRV_ROOT}/lake/binance/tape/coin=ETH/", lambda n: n.startswith("date=") and n[5:15] >= day0)):
        names = subprocess.run(["ssh", "-p", "2222", "-o", "BatchMode=yes", SRV, f"ls {remote}"], capture_output=True, text=True, check=True).stdout.split()
        names = [n for n in names if pat(n)]
        (FWD / sub).mkdir(parents=True, exist_ok=True)
        lst = FWD / f"{sub}.list"; lst.write_text("\n".join(names) + "\n")
        subprocess.run(["rsync", "-a", "-r", "-e", "ssh -p 2222 -o BatchMode=yes", f"--files-from={lst}", f"{SRV}:{remote}", str(FWD / sub)], check=True)


def forward_events():
    """전진 사건표: 연구와 같은 events() · 결과 = 30초 창 시작 mid(연구 1초판과 상관 1.0 확인) · 테이프가 봉인된 날까지만."""
    tape_days = sorted(p.name[5:15] for p in (FWD / "tape").glob("date=*"))
    if not tape_days:                                                       # 서버 lake 봉인이 아직 FWD0 전날에 못 미침
        print("봉인된 테이프 없음(서버 lake 가 하루 늦게 봉인) -- 사건 0"); return pd.DataFrame(columns=["t_end", "iceberg", "absorb", "held"])
    files = sorted(str(p) for p in (FWD / "depthdiff").glob("*.jsonl*"))
    last = int(pd.Timestamp(tape_days[-1], tz="UTC").timestamp()) + 86400
    files = [f for f in files if Path(f).name[:13] < pd.Timestamp(last, unit="s").strftime("%Y-%m-%dT%H")]
    with ProcessPoolExecutor(max_workers=6) as ex:
        parts = [d for d in ex.map(replay_levels, files) if d is not None]
    R = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    o = np.argsort(R["w"], kind="stable"); R = {k: v[o] for k, v in R.items()}
    keep = np.r_[R["w"][1:] != R["w"][:-1], True]; R = {k: v[keep] for k, v in R.items()}
    rp = f"read_parquet('{FWD}/tape/*/*.parquet', hive_partitioning=false)"
    t = duckdb.sql(f"select ts_sec // {WIN} w, price_bin b, sum(buy_qty) buy, sum(sell_qty) sell from {rp} where symbol='ethusdt' group by 1, 2").df()
    wi = pd.Series(np.arange(len(R["w"])), index=R["w"]); t = t[t.w.isin(wi.index)]
    rows = wi.loc[t.w.to_numpy()].to_numpy(); col = (t.b.to_numpy() - R["base"][rows] + K).astype(int); ok = (col >= 0) & (col <= 2 * K)
    SB = np.zeros((len(R["w"]), 2 * K + 1), np.float32); SS = np.zeros_like(SB)
    np.add.at(SB, (rows[ok], col[ok]), t.buy.to_numpy()[ok]); np.add.at(SS, (rows[ok], col[ok]), t.sell.to_numpy()[ok])
    E = events(R, SB, SS)
    mid0 = (R["bb0"] + R["ba0"]) / 2; wpos = pd.Series(np.arange(len(R["w"])), index=R["w"])
    def r(t_, s, h):
        a, b = t_ // WIN, (t_ + h) // WIN
        return s * (mid0[wpos[b]] / mid0[wpos[a]] - 1) * 1e4 if a in wpos.index and b in wpos.index else np.nan
    E = E[(E.t_end >= FWD0) & (E.t_end + 900 <= R["w"][-1] * WIN)].reset_index(drop=True)
    for h in (60, 300, 900):
        E[f"r{h // 60}"] = [r(t_, s, h) for t_, s in zip(E.t_end, E.side)]
    br = []
    for t_, s, px in zip(E.t_end, E.side, E.px):
        seg = slice(wpos[t_ // WIN], wpos.get((t_ + 900) // WIN, len(R["w"])))
        br.append(bool(R["bbMin"][seg].min() < px - 0.15) if s == 1 else bool(R["baMax"][seg].max() > px + 0.15))
    E["broke15"] = br
    E.to_parquet(FWD / "forward_events.parquet")
    return E


def forward(mode):
    sync(); E = forward_events()
    looks = FWD / "looks.json"; L = json.loads(looks.read_text()) if looks.exists() else []
    n_ice = int(E.iceberg.sum())
    print(json.dumps({"사건 끝": str(pd.Timestamp(int(E.t_end.max()) if len(E) else FWD0, unit="s")), "아이스버그": n_ice,
                      "흡수": int(E.absorb.sum()), "큰 체결": int(len(E)), "판정 문턱": 150}, ensure_ascii=False))
    if mode == "judge":
        if n_ice < 150 and "--force" not in sys.argv:
            print("아직 150건 미만 -- 결과를 보지 않는다(--force 는 사전등록 위반으로 기록)"); return
        L.append({"at": pd.Timestamp.utcnow().isoformat(), "n_iceberg": n_ice, "forced": n_ice < 150}); looks.write_text(json.dumps(L))
        ice, br_ = E.iceberg.to_numpy(), (~E.held).to_numpy()
        out = {"r15 아이스버그": block(E, "r15", ice, nb=2000), "r15 아이스버그 − 1.41": round(float(np.nanmean(E.r15[ice])) - 1.41, 3),
               "15분 뚫림 아이스버그 / 뚫린 큰 체결": [round(float(E.broke15[ice].mean()), 3), round(float(E.broke15[br_].mean()), 3)],
               "r1·r5 아이스버그": [block(E, "r1", ice), block(E, "r5", ice)], "r15 흡수(보조)": block(E, "r15", E.absorb.to_numpy())}
        print(json.dumps(out, ensure_ascii=False, indent=1))


def selftest():
    a = np.zeros(10_000); a[995:1005] = 1.0                                 # 칸 100 = 눈금 995..1004
    b = buckets(a, 100)
    assert b[K] == 10 and b.sum() == 10
    E = pd.DataFrame({"t_end": [0, 0, 86400, 86400], "x": [1., 3., 1., 3.]})
    est, ci = block(E, "x", np.array([True, False, True, False]), np.array([False, True, False, True]), nb=50)
    assert est == -2.0 and ci == [-2.0, -2.0]
    print("selftest ok")


if __name__ == "__main__":
    if "--status" in sys.argv or "--judge" in sys.argv:
        forward("judge" if "--judge" in sys.argv else "status")
    else:
        selftest() if "--selftest" in sys.argv else run_replay() if "--replay" in sys.argv else run()
