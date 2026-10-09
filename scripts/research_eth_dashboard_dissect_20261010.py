"""대시보드 해부 — 화면 원소 하나하나를 실시간 원천에서 다시 만들어 단기 방향(5·15·30·60분)을 원소별·조합별로 (2026-10-10).
사용자 «지금 대시보드에 데이터와 지표가 이렇게나 많은데 단기 방향을 찾을 수 있는 조합이 하나도 없다는게 말이 안돼.
좀 더 대시보드를 하나하나 해부해서 각각 원소를 쫙 펴보고 하나씩 분석해서 조합해줘».

원소 목록 = 10-10 대시보드 전수 인벤토리(app.js·server.py·market_ctx·micro_ref·flow_read·trend_rule·LM). 원천 = 서버 백업 lake·orderflow(로컬 사본).
제외: 계좌·알림(시장 아님) · 뉴스(10-05~) · 칼시(저장 안 됨) · 국채(10-06~) · 전환예고 HGB(아티팩트, 09-30 검증서 «봉 폭»보다 못함).

격자: 1분 끝 t(UTC ms). 피쳐는 t 이전에 «알려진» 값만 — 원천별 앎 시각:
  bookticker ts_ms<t · tape ts_sec∈[t−60s,t) · oi/mark/liq ts≤t · panel_1s sec 행 = sec+1 에 앎 · LS 비율 ts+5분 · Deribit recorded_at ≤ t ·
  5분 봉 지표 = t 이전에 닫힌 5분 봉 · 청산맵 = t 이전에 닫힌 1시간 봉 24개.
목표: 호가 중간가(bookticker) M[t+1초] → M[t+1초+h], h = 5·15·30·60분, bp. (마크가는 중간가보다 늦어 목표로 쓰면 가짜 예측력 — 10-10 확인)

사전등록(결과 전 고정). 창 09-20~10-07, DEV 09-20~09-29 · HO 09-30~10-07(주 판정).
 1 원소별: IC(DEV·HO) · 전체 IC 1시간 블록 CI · 동행 IC(과거 같은 길이 수익) · 부분 IC(과거 5·15·60분 수익 순위 통제).
   «안정» = DEV·HO 같은 부호 · 전체 CI 0 배제 · 부분 IC 같은 부호.
 2 묶음별 리지(DEV 적합 → HO IC) · 전체 리지·HGB(→ HO IC·상하위 10% bp·손익분기 IC).
 3 쌍 조합: 지평마다 DEV |IC| 상위 20 원소 → 190 쌍. 규칙 = 두 원소가 각자 DEV 방향으로 상·하위 20% 에 같이 있을 때 그쪽.
   DEV t 상위 5 쌍만 HO 채점(선택은 DEV 에서만).
 판정: HO 비용 후(테이커 8bp·메이커 1.4bp) 평균 1시간 블록 CI 하한 > 0 이면 «매매 가능» 후보 — 18일이라 그다음 전진 사전등록 필수.
플래그: fresh_forward_bar_by_bar=true · trade_ledgers_used_as_input=false · future_rows_used_for_entry=false.
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_store as DS  # noqa: E402
import live_liquidation_map_20260824 as LM  # noqa: E402
import research_eth_liqmap_5m_direction_20261010 as LQ  # noqa: E402

BK = Path.home() / "backups/crypto-scalping-server/data"
LAKE = BK / "lake"
PANEL = "/home/kbj20/crypto-scalping/.claude/worktrees/realtime-ledger-analysis-bdd373/tmp/order_battle_20261007/panel_1s.parquet"
OUT = "tmp/dash_dissect_20261010"
T0, T1 = pd.Timestamp("2026-09-20", tz="UTC"), pd.Timestamp("2026-10-08", tz="UTC")
SPLIT = pd.Timestamp("2026-09-30", tz="UTC")
HS = (5, 15, 30, 60)
FAM = list("ABCDEFGHIJKL")


def MS(ts):
    return int(ts.value // 10**6)


def lake(path, cols=None):
    fs = sorted(glob.glob(str(LAKE / path / "date=*/part.parquet")))
    return pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True) if fs else pd.DataFrame(columns=cols)


def mid_seconds():
    """M[s] = 순간 s(초) 의 호가 상태 = ts_ms < s·1000 인 마지막 갱신. 반환 (s0, mid, qi, spread_bp)."""
    parts = []
    for p in sorted((BK / "live/orderflow/bookticker/ETHUSDT").glob("*.bt*")):
        a = DS.read_bt(p)
        if len(a):
            d = pd.DataFrame({"s": a["ts_ms"] // 1000 + 1, "b": a["bid_px"], "a": a["ask_px"], "bq": a["bid_qty"], "aq": a["ask_qty"]})
            parts.append(d.groupby("s").last())
    D = pd.concat(parts); D = D[~D.index.duplicated(keep="last")].sort_index()
    s0 = int(D.index[0])
    D = D.reindex(np.arange(s0, int(D.index[-1]) + 1)).ffill(limit=30)
    return s0, ((D.b + D.a) / 2).to_numpy(), ((D.bq - D.aq) / (D.bq + D.aq)).to_numpy(), ((D.a - D.b) / ((D.a + D.b) / 2) * 1e4).to_numpy()


def per_min(ts_ms, val, t_grid, how="sum", closed_right=True):
    """이벤트(ts_ms, val) → 각 t 의 (t−60s, t] 합(closed_right=False 면 [t−60s, t)) 또는 t 이하 마지막 값."""
    ts_ms = np.asarray(ts_ms, dtype=np.int64); val = np.asarray(val, dtype=float)
    o = np.argsort(ts_ms, kind="stable"); ts_ms, val = ts_ms[o], val[o]
    if how == "last":
        i = np.searchsorted(ts_ms, t_grid, "right") - 1
        return np.where(i >= 0, val[np.maximum(i, 0)], np.nan)
    c = np.r_[0.0, np.cumsum(np.nan_to_num(val))]                              # 초기 파일에 없는 열(NaN)이 누적합 전체를 지우지 않게
    side = "right" if closed_right else "left"
    return c[np.searchsorted(ts_ms, t_grid, side)] - c[np.searchsorted(ts_ms, t_grid - 60_000, side)]


def tape_min(path, t_grid, price_scale=0.1):
    """체결 테이프 초 행 → 분 합(t−60s ≤ ts_sec < t). 원천 구간 밖은 NaN."""
    T = lake(path)
    keys = ("buy", "sell", "wb", "ws", "rb", "rs", "ntl", "n", "last")
    if not len(T):
        return {k: np.full(len(t_grid), np.nan) for k in keys}
    ts = T.ts_sec.to_numpy() * 1000
    out = {k: per_min(ts, T[c], t_grid, closed_right=False) for k, c in
           (("buy", "buy_qty"), ("sell", "sell_qty"), ("wb", "whale_buy_qty"), ("ws", "whale_sell_qty"), ("rb", "retail_buy_qty"), ("rs", "retail_sell_qty"))}
    out["ntl"] = per_min(ts, (T.buy_qty + T.sell_qty) * T.price_bin * price_scale, t_grid, closed_right=False)
    out["n"] = per_min(ts, T.buy_n + T.sell_n, t_grid, closed_right=False)
    inside = (t_grid >= ts.min() + 60_000) & (t_grid <= ts.max() + 1000)
    out = {k: np.where(inside, v, np.nan) for k, v in out.items()}
    out["last"] = per_min(ts + 999, T.price_bin * price_scale, t_grid, how="last")
    return out


def rsum(x, k):
    c = np.r_[0.0, np.nancumsum(np.nan_to_num(x))]
    o = np.full(len(x), np.nan); o[k - 1:] = c[k:] - c[:-k]
    return o


def roll_z(x, w=10080, mp=1440):
    """과거 w 분(현재 제외) 대비 견고 z = (x − 중앙)/(IQR/1.349). IQR 0 이면 NaN."""
    s = pd.Series(x).shift(1)
    q1, q2, q3 = (s.rolling(w, min_periods=mp).quantile(q) for q in (.25, .5, .75))
    return ((pd.Series(x) - q2) / ((q3 - q1) / 1.349).replace(0, np.nan)).to_numpy()


def roll_pct(x, w=10080, mp=1440):
    return pd.Series(x).rolling(w, min_periods=mp).rank(pct=True).to_numpy()


def _rss(tag):
    """단계별 메모리(10-10 빌드가 15GB 먹고 WSL 을 죽인 원인 추적용)."""
    st = open("/proc/self/status").read()
    print(tag, "RSS", st.split("VmRSS:")[1].split("\n")[0].strip(), "· 최고", st.split("VmHWM:")[1].split("\n")[0].strip(), flush=True)


def lagged(x, m):
    return np.r_[np.full(m, np.nan), x[:-m]]


def build():
    os.makedirs(OUT, exist_ok=True)
    s0, M, QI, SPR = mid_seconds()
    t = np.arange(MS(T0), MS(T1), 60_000, dtype=np.int64)
    si = t // 1000 - s0
    F = pd.DataFrame({"t": t})
    lm = np.log(M); mid_t = M[si]
    for h in HS:                                                                   # 목표: 중간가 t+1초 → t+1초+h
        e = si + 1 + 60 * h
        F[f"y{h}"] = np.where(e < len(M), (lm[np.minimum(e, len(M) - 1)] - lm[si + 1]) * 1e4, np.nan)
    _rss("A 시작")
    # ── A 가격·기술 ──
    for m in (1, 5, 15, 30, 60, 240):
        F[f"A_r{m}"] = (lm[si] - lm[si - 60 * m]) * 1e4
    sec_all = np.arange(len(M))
    b5 = pd.DataFrame({"g": (s0 + sec_all) // 300, "m": M}).groupby("g").m.agg(["max", "min", "last"])
    b5.index = (b5.index + 1) * 300_000                                           # 5분 봉 마감 시각(ms)
    c5 = b5["last"]; d5 = c5.diff()
    rsi = 100 - 100 / (1 + d5.clip(lower=0).ewm(alpha=1 / 14).mean() / (-d5.clip(upper=0)).ewm(alpha=1 / 14).mean())
    ma20, sd20 = c5.rolling(20).mean(), c5.rolling(20).std()
    tr = pd.concat([b5["max"] - b5["min"], (b5["max"] - c5.shift()).abs(), (b5["min"] - c5.shift()).abs()], axis=1).max(axis=1)
    r5 = np.log(c5).diff()
    B5 = pd.DataFrame({"A_rsi14": rsi, "A_pctb": (c5 - ma20) / (2 * sd20), "A_veto_z": (c5 - c5.rolling(144).mean()) / tr.ewm(alpha=1 / 144).mean(),
                       "A_rgamma": r5.rolling(144).apply(lambda v: np.corrcoef(v[:-1] - v.mean(), v[1:] - v.mean())[0, 1], raw=True),
                       "A_sig24": np.sqrt((r5 * 1e4).pow(2).rolling(288).sum() / 6)})
    i5 = np.searchsorted(B5.index.to_numpy(), t, "right") - 1                      # t 이전에 닫힌 5분 봉
    for c in B5:
        F[c] = B5[c].to_numpy()[i5]
    hi24 = pd.Series(M).rolling(86400, min_periods=3600).max().to_numpy()[si]; lo24 = pd.Series(M).rolling(86400, min_periods=3600).min().to_numpy()[si]
    F["A_pos24"] = (mid_t - lo24) / (hi24 - lo24)
    k1 = LQ.load_1m()                                                              # 일봉 추세 5표: 일 종가 = 1분 klines(~09-14) + 중간가
    dk = pd.Series(k1.c.to_numpy(), index=(k1.t.to_numpy() + 60_000 - 1) // 86_400_000).groupby(level=0).last()
    dm = pd.Series(M, index=((s0 + sec_all) * 1000 - 1) // 86_400_000).groupby(level=0).last()
    dc = pd.concat([dk[dk.index < dm.index[0] + 1], dm.iloc[1:]]).sort_index()
    known = (dc.index.to_numpy() + 1) * 86_400_000                                 # 그날 종가는 다음 날 00:00 에 앎
    di = np.searchsorted(known, t, "right") - 1; dv = dc.to_numpy()
    F["A_trend"] = np.mean([np.sign(dv[di] - dv[di - L]) for L in (7, 14, 28, 56, 90)], axis=0)
    _rss("B 시작")
    # ── B 풋프린트·CVD (바이낸스+OKX 선물) ──
    bn, ok = tape_min("binance/tape/coin=ETH", t), tape_min("okx/tape/coin=ETH", t)
    print("OKX/바이낸스 분 거래량 중앙 비:", round(float(np.nanmedian((ok["buy"] + ok["sell"]) / (bn["buy"] + bn["sell"]))), 3))
    buy = bn["buy"] + np.nan_to_num(ok["buy"]); sell = bn["sell"] + np.nan_to_num(ok["sell"])
    dlt, vol = buy - sell, buy + sell
    for m in (1, 5, 15, 30, 60):
        F[f"B_imb{m}"] = rsum(dlt, m) / rsum(vol, m)
    d30 = rsum(dlt, 30)
    F["B_cvd30_z"] = d30 / pd.Series(d30).shift(1).rolling(1440, min_periods=720).std().to_numpy()
    F["B_taker_ratio5"] = np.log(rsum(buy, 5) / rsum(sell, 5))
    F["B_okx_imb30"] = rsum(ok["buy"] - ok["sell"], 30) / rsum(ok["buy"] + ok["sell"], 30)
    T = lake("binance/tape/coin=ETH", ["ts_sec", "price_bin", "buy_qty", "sell_qty"])           # 5분 봉 × $0.5 칸
    T = T[(T.ts_sec >= MS(T0) // 1000 - 600) & (T.ts_sec < MS(T1) // 1000)]
    T["bar"] = (T.ts_sec // 300 + 1) * 300_000; T["cell"] = T.price_bin // 5
    C = T.groupby(["bar", "cell"])[["buy_qty", "sell_qty"]].sum().reset_index()
    C["ib"] = (C.buy_qty > 3 * C.sell_qty) & (C.buy_qty > 0); C["is_"] = (C.sell_qty > 3 * C.buy_qty) & (C.sell_qty > 0)
    C = C.sort_values(["bar", "cell"])

    def max_run(v):
        best = run = 0
        for x in v:
            run = run + 1 if x else 0; best = max(best, run)
        return best
    g = C.groupby("bar")
    FB = pd.DataFrame({"imbcnt": g.ib.sum() - g.is_.sum(),
                       "stk": g.ib.apply(lambda v: max_run(v.to_numpy())) - g.is_.apply(lambda v: max_run(v.to_numpy())),
                       "poc": C.loc[(C.buy_qty + C.sell_qty).groupby(C.bar).idxmax()].set_index("bar").cell * 0.5 + 0.25})
    j = np.searchsorted(FB.index.to_numpy(), t, "right") - 1
    F["B_fp_imbcnt"] = FB.imbcnt.to_numpy()[j]; F["B_fp_stack"] = FB["stk"].to_numpy()[j]
    F["B_poc_dist"] = (mid_t / FB.poc.to_numpy()[j] - 1) * 1e4
    _rss("C 시작")
    # ── C 고래·중형·리테일 ──
    wn = bn["wb"] - bn["ws"] + np.nan_to_num(ok["wb"] - ok["ws"]); rn = bn["rb"] - bn["rs"] + np.nan_to_num(ok["rb"] - ok["rs"])
    for nm, x in (("whale", wn), ("mid", dlt - wn - rn), ("retail", rn)):
        F[f"C_{nm}60"] = rsum(x, 60) / rsum(vol, 60)
        F[f"C_{nm}60_z"] = roll_z(rsum(x, 60))
    zw, zr = F.C_whale60_z.to_numpy(), F.C_retail60_z.to_numpy()
    F["C_duel"] = np.where((np.sign(zw) != np.sign(zr)) & (np.abs(zw) >= .5) & (np.abs(zr) >= .5), np.sign(zw), 0)
    _rss("D 시작")
    # ── D OI·사분면 ──
    O = lake("binance/oi_1s/coin=ETH", ["ts_ms", "open_interest"])
    oi_b = per_min(O.ts_ms, O.open_interest, t, "last")
    OK = lake("okx/oi/coin=ETH", ["ts_ms", "oi_base"])
    HC = lake("hl/asset_ctx/coin=ETH", ["recv_ms", "open_interest", "funding", "premium", "mid_px"])
    venues = [oi_b, per_min(OK.ts_ms, OK.oi_base, t, "last"), per_min(HC.recv_ms, HC.open_interest, t, "last")]
    for m in (5, 60):                                                              # 거래소별 변화량 합 ÷ 합계 — 합류 순간 계단 없음
        d = sum(np.nan_to_num(v - lagged(v, m)) for v in venues)
        F[f"D_doi{m}"] = d / sum(np.nan_to_num(v) for v in venues) * 1e4
    F["D_doi60_bn"] = (np.log(oi_b) - np.log(lagged(oi_b, 60))) * 1e4
    F["D_oi_z1h"] = roll_z(F.D_doi60.to_numpy())
    sd5, so5 = np.sign(rsum(dlt, 5)), np.sign(F.D_doi5.to_numpy())
    for nm, a, b in (("newlong", 1, 1), ("shortcover", 1, -1), ("newshort", -1, 1), ("longexit", -1, -1)):
        F[f"D_quad_{nm}"] = ((sd5 == a) & (so5 == b)).astype(float)
    mv60 = F.A_r60.to_numpy(); p75 = pd.Series(np.abs(mv60)).shift(1).rolling(1500, min_periods=300).quantile(.75).to_numpy()
    F["D_poi1h"] = np.where(np.abs(mv60) >= p75, np.sign(mv60) * np.sign(F.D_doi60.to_numpy()), 0)   # +1 = 가격·OI 같은 방향
    _rss("E 시작")
    # ── E 청산(바이낸스+OKX+HL), 숏 청산 = 강제 매수(+) ──
    L = lake("binance/liquidations/coin=ETH", ["ts_ms", "side", "usd"])
    bl, bs = per_min(L.ts_ms, np.where(L.side == "long", L.usd, 0), t), per_min(L.ts_ms, np.where(L.side == "short", L.usd, 0), t)
    lq_l, lq_s = bl.copy(), bs.copy()
    OL = lake("okx/liquidations/coin=ETH", ["ts_ms", "pos_side", "sz_base", "bk_px"])
    if len(OL):
        u = OL.sz_base * OL.bk_px
        lq_l += per_min(OL.ts_ms, np.where(OL.pos_side == "long", u, 0), t); lq_s += per_min(OL.ts_ms, np.where(OL.pos_side == "short", u, 0), t)
    HL = lake("hl/liquidations/coin=ETH", ["fill_ms", "dir", "px", "sz"])
    if len(HL):
        u = HL.px * HL.sz
        lq_l += per_min(HL.fill_ms, np.where(HL["dir"] == "Close Long", u, 0), t); lq_s += per_min(HL.fill_ms, np.where(HL["dir"] == "Close Short", u, 0), t)
    for m in (5, 30):
        a, b = rsum(lq_s, m), rsum(lq_l, m)
        F[f"E_liqnet{m}"] = (a - b) / np.maximum(a + b, 1.0)
        F[f"E_liqtot{m}"] = np.log1p(a + b)
    F["E_burst"] = (bs > np.expm1(13.21)).astype(float) - (bl > np.expm1(12.732)).astype(float)
    _rss("F 시작")
    # ── F 레버리지 ──
    MK = lake("binance/mark_1s/coin=ETH", ["ts_ms", "mark", "index_px", "funding_rate"])
    mk, ix = per_min(MK.ts_ms, MK.mark, t, "last"), per_min(MK.ts_ms, MK.index_px, t, "last")
    F["F_funding"] = per_min(MK.ts_ms, MK.funding_rate, t, "last") * 1e4
    F["F_basis"] = (mk / ix - 1) * 1e4
    F["F_basis_pct"] = roll_pct(F.F_basis.to_numpy())
    F["F_dbasis30"] = F.F_basis - F.F_basis.shift(30)
    OF = lake("okx/funding/coin=ETH", ["ts_ms", "funding_rate"])
    F["F_okx_funding"] = per_min(OF.ts_ms, OF.funding_rate, t, "last") * 1e4
    F["F_hl_funding8"] = per_min(HC.recv_ms, HC.funding, t, "last") * 8e4
    F["F_hl_premium"] = per_min(HC.recv_ms, HC.premium, t, "last") * 1e4
    LS = lake("binance/oi_lsratio_5m/coin=ETH", ["ts", "global_ls_ratio", "top_pos_ls_ratio", "top_acct_ls_ratio", "taker_ls_ratio"])
    lst = pd.to_datetime(LS.ts, utc=True).astype("int64").to_numpy() // 10**6 + 300_000      # 🔴 5분 구간 → 앎 = ts+5분
    for c in ("global_ls_ratio", "top_pos_ls_ratio", "top_acct_ls_ratio", "taker_ls_ratio"):
        v = per_min(lst, np.log(LS[c].astype(float)), t, "last")
        F[f"F_{c}"] = v; F[f"F_d60_{c}"] = v - lagged(v, 60)
    F["F_lev_state"] = np.where((F.F_basis_pct >= .9) & (F.D_oi_z1h >= 1), 1, np.where((F.F_basis_pct <= .1) & (F.D_oi_z1h >= 1), -1, 0))
    _rss("G 시작")
    # ── G 호가 ──
    P = pd.read_parquet(PANEL).reset_index()
    pk = P.sec.to_numpy().astype(np.int64) + 1                                     # sec 행 = sec+1 에 앎
    pi = np.searchsorted(pk, t // 1000, "right") - 1
    okp = (pi >= 0) & (t // 1000 - pk[np.maximum(pi, 0)] <= 5)
    for nm, a, b in (("G_imb10", "bid10", "ask10"), ("G_imb50", "bid50", "ask50")):
        F[nm] = np.where(okp, ((P[a] - P[b]) / (P[a] + P[b])).to_numpy()[np.maximum(pi, 0)], np.nan)
    cn = np.r_[0, np.nancumsum(((P.add_b - P.rem_b) - (P.add_a - P.rem_a)).to_numpy())]
    ct = np.r_[0, np.nancumsum((P.add_b + P.rem_b + P.add_a + P.rem_a).to_numpy())]
    lo60 = np.searchsorted(pk, t // 1000 - 60, "right")
    F["G_ofi60"] = np.where(okp, (cn[pi + 1] - cn[lo60]) / np.maximum(ct[pi + 1] - ct[lo60], 1e-9), np.nan)
    i50 = pd.Series(F.G_imb50.to_numpy()[::5])
    thr = np.repeat(i50.abs().shift(1).rolling(288, min_periods=100).quantile(.8).to_numpy(), 5)[:len(F)]
    F["G_wall"] = np.where(np.abs(F.G_imb50) > thr, np.sign(F.G_imb50), 0)
    F["G_qi"] = QI[si]; F["G_spread"] = SPR[si]
    _rss("H 시작")
    # ── H 교차시장 ──
    OM = lake("okx/mark/coin=ETH", ["ts_ms", "mark_px"])
    F["H_okx_gap"] = (per_min(OM.ts_ms, OM.mark_px, t, "last") / mk - 1) * 1e4
    F["H_hl_gap"] = (per_min(HC.recv_ms, HC.mid_px, t, "last") / mid_t - 1) * 1e4
    lb = np.log(pd.Series(tape_min("binance/tape/coin=BTC", t)["last"]).ffill().to_numpy())
    for m in (5, 30):
        F[f"H_btc_r{m}"] = (lb - lagged(lb, m)) * 1e4
    F["H_eth_btc_rel30"] = F.A_r30 - F.H_btc_r30
    sp = tape_min("binance/spot_tape/coin=ETH", t)
    F["H_spot_imb30"] = rsum(sp["buy"] - sp["sell"], 30) / rsum(sp["buy"] + sp["sell"], 30)
    F["H_spot_minus_perp30"] = F.H_spot_imb30 - F.B_imb30
    _rss("I 시작")
    # ── I 옵션(Deribit 10분 요약) ──
    rows = []
    for _, r in lake("deribit/option_summary/coin=ETH").iterrows():
        try:
            pl = json.loads(r.payload); cm = pl.get("cm", {}); ex0 = (pl.get("expiries") or [{}])[0]
            rows.append({"k": int(pd.Timestamp(r.recorded_at_utc).value // 10**6), "dvol": r.dvol, "ivrv": r.dvol - r.rv7,
                         "rr7": cm.get("7", {}).get("rr"), "bf7": cm.get("7", {}).get("bf"),
                         "term": (cm.get("7", {}).get("atm") or np.nan) - (cm.get("30", {}).get("atm") or np.nan), "pain": ex0.get("pain"), "pc": ex0.get("pc")})
        except (TypeError, ValueError):
            continue
    Q = pd.DataFrame(rows)
    for c in ("dvol", "ivrv", "rr7", "bf7", "term", "pc"):
        F[f"I_{c}"] = per_min(Q.k, Q[c].astype(float), t, "last")
    F["I_pain_dist"] = (per_min(Q.k, Q.pain.astype(float), t, "last") / mid_t - 1) * 1e4
    _rss("J 시작")
    # ── J 청산맵(추정, 직전 닫힌 1시간 봉 24개) · HL 고래 실제 청산가 ──
    vol_s = np.zeros(len(M))
    TT = lake("binance/tape/coin=ETH", ["ts_sec", "buy_qty", "sell_qty"])
    np.add.at(vol_s, np.clip(TT.ts_sec.to_numpy() + 1 - s0, 0, len(M) - 1), (TT.buy_qty + TT.sell_qty).to_numpy())
    hb = pd.DataFrame({"g": (s0 + sec_all - 1) // 3600, "m": M, "v": vol_s}).groupby("g").agg(high=("m", "max"), low=("m", "min"), close=("m", "last"), volume=("v", "sum"))
    hb["timestamp"] = pd.to_datetime(hb.index * 3600, unit="s", utc=True)
    rows = []
    for k in range(25, len(hb)):
        w = hb.iloc[k - 24:k].reset_index(drop=True)
        if w.volume.sum() <= 0:
            continue
        cp = float(w.close.iloc[-1])
        tp, sl = LM.compute_tier_profile(w, cp), LM.compute_spliced_levels(w, cp)
        if not tp:
            continue
        f = LQ.features(tp, cp)
        s1, r1 = sl.get("support_levels") or [], sl.get("resistance_levels") or []
        rows.append({"k": int(hb.index[k]) * 3_600_000, "lr_1": f["lr_1"], "lr_3": f["lr_3"], "lr_mag": f["lr_mag"],
                     "s1": s1[0]["price"] if s1 else np.nan, "r1": r1[0]["price"] if r1 else np.nan})
    J = pd.DataFrame(rows)
    for c in ("lr_1", "lr_3", "lr_mag"):
        F[f"J_{c}"] = per_min(J.k, J[c], t, "last")
    F["J_s1_dist"] = (mid_t / per_min(J.k, J.s1, t, "last") - 1) * 1e4
    F["J_r1_dist"] = (per_min(J.k, J.r1, t, "last") / mid_t - 1) * 1e4
    PS = lake("hl/positions/coin=ETH", ["ex_ms", "user", "szi", "liq_px"])
    PS = PS[np.isfinite(PS.liq_px) & (PS.liq_px > 0)].sort_values("ex_ms")
    ex = PS.ex_ms.to_numpy(); hr = []
    for tt in t[::5]:
        a, b = np.searchsorted(ex, tt - 240_000, "right"), np.searchsorted(ex, tt, "right")
        sn = PS.iloc[a:b].groupby("user").tail(1)
        if len(sn) < 20:
            continue
        d = (sn.liq_px / M[tt // 1000 - s0] - 1) * 1e4; u = sn.szi.abs() * sn.liq_px
        up, dn = (sn.szi < 0) & (d > 0) & (d < 1000), (sn.szi > 0) & (d < 0) & (d > -1000)
        hr.append({"k": tt, "up2": u[up & (d < 200)].sum(), "dn2": u[dn & (d > -200)].sum(),
                   "near_up": d[up].min() if up.any() else 1000.0, "near_dn": -d[dn].max() if dn.any() else 1000.0})
    HP = pd.DataFrame(hr)
    F["J_hl_lr2"] = np.log((per_min(HP.k, HP.up2, t, "last") + 1e5) / (per_min(HP.k, HP.dn2, t, "last") + 1e5))
    F["J_hl_near_up"] = per_min(HP.k, HP.near_up, t, "last"); F["J_hl_near_dn"] = per_min(HP.k, HP.near_dn, t, "last")
    _rss("K 시작")
    # ── K 탐지기(거래대금·체결 수 z288 ≥ 과거 2016 의 90분위)·RVOL ──
    q5, n5 = pd.Series(rsum(bn["ntl"], 5)[4::5]), pd.Series(rsum(bn["n"], 5)[4::5])
    zz = lambda s: (s - s.shift(1).rolling(288, min_periods=100).mean()) / s.shift(1).rolling(288, min_periods=100).std()
    zq, zn = zz(q5), zz(n5)
    det = (zq >= zq.shift(1).rolling(2016, min_periods=500).quantile(.9)) & (zn >= zn.shift(1).rolling(2016, min_periods=500).quantile(.9))
    rep = lambda s: np.r_[np.full(4, np.nan), np.repeat(s.to_numpy(), 5)][:len(F)]
    F["K_detect"] = rep(det.astype(float)); F["K_vol_z"] = rep(zq)
    day = t // 86_400_000; slot = (t % 86_400_000) // 60_000
    cum = pd.Series(np.nan_to_num(bn["ntl"])).groupby(day).cumsum().to_numpy()
    ref = pd.DataFrame({"d": day, "s": slot, "c": cum}).pivot(index="d", columns="s", values="c")
    med = ref.shift(1).rolling(7, min_periods=3).median().to_numpy()
    F["K_rvol"] = cum / med[np.searchsorted(ref.index.to_numpy(), day), slot]
    _rss("L 시작")
    # ── L 시간 ──
    F["L_hour"] = (t // 3_600_000) % 24; F["L_weekend"] = (pd.to_datetime(t, unit="ms").dayofweek >= 5).astype(float)
    F["split"] = np.where(t < MS(SPLIT), "DEV", "HO"); F["blk"] = t // 3_600_000
    F.replace([np.inf, -np.inf], np.nan, inplace=True)
    _rss("저장 직전")
    F.to_parquet(f"{OUT}/panel.parquet")
    FC = feat_cols(F)
    cov = F[FC].notna().mean().round(2)
    print(len(F), "rows ·", len(FC), "features · 목표 결측", round(float(F.y60.isna().mean()), 3))
    print("커버리지 < 0.9:", cov[cov < .9].to_dict())


def feat_cols(F):
    return [c for c in F.columns if len(c) > 2 and c[1] == "_" and c[0] in FAM]


def ic(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 200 or np.nanstd(x[m]) == 0:
        return np.nan
    return float(pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank()))


def boot(blk, fn, n=300, seed=0):
    rng = np.random.default_rng(seed)
    ub = np.unique(blk); g = {u: np.flatnonzero(blk == u) for u in ub}
    v = [fn(np.concatenate([g[u] for u in rng.choice(ub, len(ub))])) for _ in range(n)]
    return [round(float(x), 4) for x in np.nanquantile(v, [.025, .975])]


def ridge(X, y, lam=1.0):
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-12
    Z = np.clip(np.nan_to_num((X - mu) / sd), -5, 5)
    b = np.linalg.solve(Z.T @ Z + lam * len(Z) * 1e-2 * np.eye(Z.shape[1]), Z.T @ y)
    return lambda X2: np.clip(np.nan_to_num((X2 - mu) / sd), -5, 5) @ b


def test():
    from sklearn.ensemble import HistGradientBoostingRegressor
    F = pd.read_parquet(f"{OUT}/panel.parquet")
    FC = feat_cols(F)
    dev, ho = (F.split == "DEV").to_numpy(), (F.split == "HO").to_numpy()
    blk = F.blk.to_numpy()
    rk = lambda v: pd.Series(v).rank().to_numpy()
    out = {"n": {"DEV": int(dev.sum()), "HO": int(ho.sum())}, "features": FC, "uni": {}, "family": {}, "all": {}, "pairs": {},
           "flags": {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "future_rows_used_for_entry": False}}
    ctrl = F[["A_r5", "A_r15", "A_r60"]].to_numpy()
    for h in HS:
        y = F[f"y{h}"].to_numpy()
        cm = np.isfinite(ctrl).all(1) & np.isfinite(y)
        Z = np.c_[np.ones(cm.sum()), np.column_stack([rk(c) for c in ctrl[cm].T])]
        res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
        ry = res(rk(y[cm]))
        U = {}
        for f in FC:
            x = F[f].to_numpy()
            r = {"dev": round(ic(x[dev], y[dev]), 4), "ho": round(ic(x[ho], y[ho]), 4), "all": round(ic(x, y), 4),
                 "coinc": round(ic(x, F[f"A_r{h}"].to_numpy()), 4)}
            xm = x[cm]; okx = np.isfinite(xm)
            r["partial"] = round(float(np.corrcoef(res(rk(np.where(okx, xm, np.nanmedian(xm))))[okx], ry[okx])[0, 1]), 4) \
                if okx.sum() > 200 and np.nanstd(xm) > 0 else np.nan
            mm = np.isfinite(x) & np.isfinite(y)
            r["ci"] = boot(blk[mm], lambda ix, xx=x[mm], yy=y[mm]: ic(xx[ix], yy[ix]), n=200) if mm.sum() > 500 else [np.nan, np.nan]
            r["stable"] = bool(np.sign(r["dev"]) == np.sign(r["ho"]) and r["ci"][0] * r["ci"][1] > 0 and np.sign(r["partial"]) == np.sign(r["all"]))
            U[f] = r
        out["uni"][h] = U
        print(h, "안정:", {f: (U[f]["all"], U[f]["partial"]) for f in U if U[f]["stable"]}, flush=True)
        okr, te = dev & np.isfinite(y), ho & np.isfinite(y)
        sig = float(np.nanstd(y[te]))
        out["family"][h] = {fm: round(ic(ridge(F[cs].to_numpy()[okr], y[okr])(F[cs].to_numpy())[te], y[te]), 4)
                            for fm in FAM if (cs := [c for c in FC if c[0] == fm])}
        X = F[FC].to_numpy(); ra = {}
        for nm in ("ridge", "hgb"):
            p = ridge(X[okr], y[okr])(X) if nm == "ridge" else HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(X[okr], y[okr]).predict(X)
            pt, yt, bt = p[te], y[te], blk[te]
            lo, hi = np.quantile(pt, [.1, .9]); s = np.where(pt >= hi, 1, np.where(pt <= lo, -1, 0)); sel = s != 0
            g_ = (s * yt)[sel]
            ra[nm] = {"ic": round(ic(pt, yt), 4), "ic_ci": boot(bt, lambda ix: ic(pt[ix], yt[ix])),
                      "top10_bp": round(float(g_.mean()), 2), "top10_ci": boot(bt[sel], lambda ix: g_[ix].mean())}
        ra.update({"sigma": round(sig, 2), "breakeven_ic_taker": round(8 / (sig * 1.755), 3), "breakeven_ic_maker": round(1.4 / (sig * 1.755), 3)})
        out["all"][h] = ra
        print(h, "묶음", out["family"][h], "| 전체", ra, flush=True)
        top = sorted([f for f in FC if np.isfinite(U[f]["dev"]) and F[f].nunique() > 3], key=lambda f: -abs(U[f]["dev"]))[:20]
        sf = {}
        for f in top:
            x = F[f].to_numpy(); q20, q80 = np.nanquantile(x[dev], [.2, .8]); sd = np.sign(U[f]["dev"])
            sf[f] = np.where(x >= q80, sd, np.where(x <= q20, -sd, 0))
        pr = []
        for a_ in range(len(top)):
            for b_ in range(a_ + 1, len(top)):
                s = np.where((sf[top[a_]] == sf[top[b_]]) & (sf[top[a_]] != 0), sf[top[a_]], 0)
                m = dev & (s != 0) & np.isfinite(y)
                if m.sum() < 100:
                    continue
                bm = pd.Series(s[m] * y[m]).groupby(blk[m]).mean()
                pr.append((top[a_], top[b_], float((s[m] * y[m]).mean()), float(bm.mean() / (bm.std() / np.sqrt(len(bm)) + 1e-9)), int(m.sum())))
        pr.sort(key=lambda r: -r[3])
        best = []
        for fa, fb, dmean, dt, dn in pr[:5]:
            s = np.where((sf[fa] == sf[fb]) & (sf[fa] != 0), sf[fa], 0)
            m = ho & (s != 0) & np.isfinite(y); v = s[m] * y[m]
            best.append({"pair": [fa, fb], "dev_bp": round(dmean, 2), "dev_t": round(dt, 2), "dev_n": dn, "ho_n": int(m.sum()),
                         "ho_bp": round(float(v.mean()), 2) if m.any() else None, "ho_ci": boot(blk[m], lambda ix: v[ix].mean()) if m.sum() > 50 else None})
        out["pairs"][h] = {"n_pairs": len(pr), "top5": best}
        print(h, "쌍", json.dumps(best, ensure_ascii=False), flush=True)
    json.dump(out, open(f"{OUT}/report.json", "w"), ensure_ascii=False, indent=1, default=str)


def selftest():
    tg = np.array([60_000, 120_000], dtype=np.int64)
    assert list(per_min([0, 59_999, 60_000, 61_000], [1, 2, 4, 8], tg)) == [6.0, 8.0]                     # (t−60s, t]
    assert list(per_min([0, 59_999, 60_000, 61_000], [1, 2, 4, 8], tg, closed_right=False)) == [3.0, 12.0]  # [t−60s, t)
    assert list(per_min([10, 70_000], [5, 6], tg, "last")) == [5.0, 6.0]
    assert list(per_min([0, 59_999, 61_000], [np.nan, 2, 8], tg)) == [2.0, 8.0]                          # NaN 이 누적합을 지우지 않음
    assert np.allclose(rsum(np.array([1., 2, 3]), 2)[1:], [3, 5])
    rng = np.random.default_rng(0); x = np.r_[rng.normal(size=3000), 10.0]
    assert roll_z(x, w=2000, mp=100)[-1] > 5                                       # 과거 분포 대비 이상치
    assert np.isnan(lagged(np.array([1., 2.]), 1)[0])
    print("selftest OK")


if __name__ == "__main__":
    {"--build": build, "--test": test, "--selftest": selftest}[sys.argv[1]]()
