"""사전등록 전진 검정: 대시보드 4분할 기계의 «OI 역행 쌓임 정리»(X4) 청산 (동결 2026-10-08 · 판정 데이터 2026-10-09 00:00 UTC~).

설계 원문: docs/experiments/eth_oi_adverse_exit_fwd_prereg_20261008.md (이 스크립트와 같은 커밋 = 동결).
  --status  판정 창 거래 수·일수만(손익 안 봄)        --look  점검일(180·365일)에 도달했을 때만 판정
  --parity  과거(2025-01~2026-09-12)에서 연구 숫자 재현 + 일별 vision 재다운로드가 저장 패널과 같은지        --selftest
발견 연구: scripts/research_eth_flow_ladder_entry_exit_20261008.py (X4 · 핑퐁 후보 MAIN +6.30 vs 보유시간 섞기 −4.64).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_binance_vision_panel_20260915 import day as vision_day  # noqa: E402
from prereg_eth_pingpong_vol_sizing_fwd_20261007 import candidates, load_all, ms, size_mult, to_5m  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/kbj20/crypto-scalping/data/binance_vision")
CACHE = DATA / "panel_daily"
LOOKS = ROOT / "docs/experiments/eth_oi_adverse_exit_fwd_looks.json"
FREEZE = ms("2026-10-09")
D, K, HOR, FEE, NW = 44.0, 3.0, 1440, 5.0, 4               # 대시보드 기계: 44bp 균등 4분할 · 첫 진입가 ∓3σ24 · 24h · 테이커 5bp/단위
ZWIN = 2016                                                 # 강건 z 창(7일 5분봉), 그 봉 제외
LOOK_DAYS, Z_CRIT = (180, 365), (2.80, 1.98)                # O'Brien-Fleming 2회, 단측 α=.025
SEED, N_PERM, N_BOOT = 20261008, 20, 5000


@numba.njit(cache=False)
def simx(h, l, c, ei, side, S, H, exl, exs):
    """균등 4분할(지정가 0bp) · 첫 진입가 ∓S 손절 · 익절 없음 · H 분 시간 청산 · 신호 exl/exs[j](분 j 마감에 앎) → c[j+1] 테이커.
    손익 = 4분할 전체 명목 대비 bp(청산 FEE·체결 단위). 종류 2 시간 3 손절 4 신호."""
    n = len(ei)
    pnl = np.zeros(n); xi = np.zeros(n, np.int64); kind = np.zeros(n, np.int64)
    for k in range(n):
        i = ei[k]; s = side[k]; p0 = c[i]
        q = 1.0; cost = p0; nadd = 0
        end = min(i + H[k], len(c) - 2)
        exit_px = c[end]; kind[k] = 2; xi[k] = end
        for j in range(i + 1, end + 1):
            while nadd < NW - 1:
                lvl = p0 * (1 - s * D * (nadd + 1) / 1e4)
                if (s > 0 and l[j] <= lvl) or (s < 0 and h[j] >= lvl):
                    nadd += 1; q += 1.0; cost += lvl
                else:
                    break
            stop = p0 * (1 - s * S[k] / 1e4)
            if (s > 0 and l[j] <= stop) or (s < 0 and h[j] >= stop):
                gap = (s > 0 and h[j] < stop) or (s < 0 and l[j] > stop)
                exit_px = c[j] if gap else stop; kind[k] = 3; xi[k] = j
                break
            if exl[j] if s > 0 else exs[j]:
                exit_px = c[j + 1]; kind[k] = 4; xi[k] = j + 1
                break
        pnl[k] = (s * (exit_px * q - cost) / p0 * 1e4 - FEE * q) / NW
    return pnl, xi, kind


def rz(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    q = s.rolling(ZWIN, min_periods=ZWIN // 2).quantile
    q25, q50, q75 = q(.25).shift(1), q(.5).shift(1), q(.75).shift(1)
    return ((s - q50) / ((q75 - q25) / 1.349).replace(0, np.nan)).to_numpy()


def panel() -> pd.DataFrame:
    """저장 5분 패널 + 그 뒤 날짜는 data.binance.vision 일별(같은 빌더 함수) 캐시."""
    p = pd.read_parquet(DATA / "panel/ETHUSDT.parquet")
    d = p.timestamp.max().normalize() + pd.Timedelta(days=1)
    extra = []
    while d < pd.Timestamp.now().normalize():
        f = CACHE / f"ETHUSDT-5m-{d:%Y-%m-%d}.parquet"
        if f.exists():
            extra.append(pd.read_parquet(f))
        else:
            x = vision_day("ETHUSDT", f"{d:%Y-%m-%d}")
            if x is not None and x.sum_open_interest.notna().any():
                CACHE.mkdir(parents=True, exist_ok=True); x.to_parquet(f); extra.append(x)
        d += pd.Timedelta(days=1)
    return pd.concat([p] + extra).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def x4_signals(p: pd.DataFrame, t1: np.ndarray):
    """X4: 15분 역행 ≥ σ15 & zOI15 ≥ +2 (5분 마감에 앎 → 그 봉 마지막 분에 표시). 롱 정리 / 숏 정리."""
    t5 = p.timestamp.astype("int64").to_numpy() // 10**6
    c = p.close.to_numpy()
    r5 = np.diff(np.log(c), prepend=np.nan) * 1e4
    s15 = np.sqrt(pd.Series(r5 ** 2).rolling(288, min_periods=280).sum().to_numpy()) / np.sqrt(96)
    lc = pd.Series(np.log(c)); ret15 = ((lc - lc.shift(3)) * 1e4).to_numpy()
    loi = np.log(p.sum_open_interest.shift(1))                                  # OI 스탬프 규약 보수 1봉
    zoi15 = rz((loi - loi.shift(3)).to_numpy())
    out = []
    for m in ((ret15 <= -s15) & (zoi15 >= 2), (ret15 >= s15) & (zoi15 >= 2)):
        z = np.zeros(len(t1), np.bool_)
        want = t5[np.nan_to_num(m).astype(bool)] + 240_000
        j = np.searchsorted(t1, want); ok = j < len(t1); j, want = j[ok], want[ok]
        z[j[t1[j] == want]] = True
        out.append(z)
    return out[0], out[1]


def trades(k: pd.DataFrame, C: pd.DataFrame, exl, exs, lo: int, hi: int) -> pd.DataFrame:
    """후보(페이드 = −d) → X4 팔 · 무신호 24h 팔 · 보유시간 섞기 N_PERM 팔(같은 거래들 안에서). lo ≤ 결정 < hi 인 거래만."""
    b = to_5m(k); t5 = b.t.to_numpy()
    sig = np.sqrt(pd.Series((np.diff(np.log(b.c.to_numpy()), prepend=np.nan) * 1e4) ** 2).rolling(288, min_periods=280).sum().to_numpy())
    t1, h1, l1, c1 = (k[x].to_numpy() for x in ("t", "h", "l", "c"))
    tdec = C.t.to_numpy("int64") + 60_000; ei = np.searchsorted(t1, tdec)
    bi = np.searchsorted(t5, tdec // 300_000 * 300_000 - 300_000)
    ok = (tdec >= lo) & (tdec < hi) & (ei + HOR + 2 < len(t1)) & (bi < len(t5)) & np.isfinite(sig[np.minimum(bi, len(t5) - 1)])
    C, ei, bi, tdec = C[ok].reset_index(drop=True), ei[ok], bi[ok], tdec[ok]
    side = -C.d.to_numpy(np.int64); sg = sig[bi]; S = K * sg
    H = np.full(len(ei), HOR, np.int64); z = np.zeros(len(c1), np.bool_)
    px4, xi4, k4 = simx(h1, l1, c1, ei, side, S, H, exl, exs)
    pxh, _, _ = simx(h1, l1, c1, ei, side, S, H, z, z)
    rng = np.random.default_rng(SEED)
    pp = np.mean([simx(h1, l1, c1, ei, side, S, rng.permutation(xi4 - ei).astype(np.int64), z, z)[0] for _ in range(N_PERM)], axis=0)
    return pd.DataFrame({"t": tdec, "day": tdec // 86_400_000, "w": size_mult(sg / np.sqrt(6)), "x4": px4, "hold24": pxh, "perm": pp,
                         "kind": k4, "hold_min": xi4 - ei})


def verdict(T: pd.DataFrame, look: int) -> dict:
    rng = np.random.default_rng(SEED)
    d = (T.w * (T.x4 - T.perm)).groupby(T.day).agg(["sum", "size"])
    S = d.to_numpy(); idx = rng.integers(0, len(S), (N_BOOT, len(S)))
    bs = S[idx, 0].sum(1) / S[idx, 1].sum(1)                                    # 일 재표집 → 건당 평균
    est = S[:, 0].sum() / S[:, 1].sum(); z = est / bs.std()
    sec1 = float((T.w * (T.x4 - T.hold24)).mean())
    pas = bool(z >= Z_CRIT[look] and sec1 > 0)
    return dict(look=look + 1, days=int(T.day.nunique()), n=int(len(T)), x4_minus_perm_bp_w=float(est), z=float(z), z_crit=Z_CRIT[look],
                sec1_x4_minus_hold24_bp_w=sec1, x4_bp_w=float((T.w * T.x4).mean()), perm_bp_w=float((T.w * T.perm).mean()),
                hold24_bp_w=float((T.w * T.hold24).mean()), signal_exit=float((T.kind == 4).mean()), stop=float((T.kind == 3).mean()),
                result="확정" if pas else ("기각" if look == len(LOOK_DAYS) - 1 or z < 0 else "계속"))


def build(lo: int, hi: int, warm_days: int = 30) -> pd.DataFrame:
    k = load_all()
    k = k[k.t >= lo - warm_days * 86_400_000].reset_index(drop=True)
    exl, exs = x4_signals(panel(), k.t.to_numpy())
    return trades(k, candidates(k), exl, exs, lo, hi)


def main() -> None:
    T = build(FREEZE, ms("2100-01-01"))
    days = (int(T.day.max()) - FREEZE // 86_400_000 + 1) if len(T) else 0
    if "--status" in sys.argv:
        nxt = [d for d in LOOK_DAYS if d > days]
        print(f"판정 창 거래 {len(T)} · 경과 {days}일 · 다음 점검 {nxt[0] if nxt else '없음'}일")
        return
    looks = json.loads(LOOKS.read_text()) if LOOKS.exists() else []
    if len(looks) >= len(LOOK_DAYS) or any(r["result"] in ("확정", "기각") for r in looks):
        print("판정 완료:", looks[-1]); return
    li = len(looks)
    if days < LOOK_DAYS[li]:
        print(f"점검 {li + 1} 은 {LOOK_DAYS[li]}일째에만 — 지금 {days}일. 결과를 보지 않는다."); return
    T = T[T.day < FREEZE // 86_400_000 + LOOK_DAYS[li]]
    r = verdict(T, li); looks.append(r)
    LOOKS.write_text(json.dumps(looks, ensure_ascii=False, indent=1))
    print(json.dumps(r, ensure_ascii=False, indent=1))


def parity() -> None:
    """① 과거 재현: 연구(핑퐁 후보 · X4 MAIN) 크기가중 +6.30 / 보유시간 섞기 −4.64 (섞기는 시드·횟수가 달라 근사).
    ② vision 일별 재다운로드 = 저장 패널(2026-09-10~12)."""
    T = build(ms("2025-01-01"), ms("2026-09-12"))
    x4, pm = float((T.w * T.x4).mean()), float((T.w * T.perm).mean())
    print(f"재현: n {len(T)} (연구 13,688) · X4 {x4:+.2f} (연구 +6.30) · 섞기 {pm:+.2f} (연구 −4.64) · 신호 청산 {(T.kind == 4).mean():.3f} (연구 0.84)")
    assert abs(len(T) / 13688 - 1) < 0.02 and abs(x4 - 6.30) < 1.0
    p = pd.read_parquet(DATA / "panel/ETHUSDT.parquet").set_index("timestamp")
    for d in ("2026-09-10", "2026-09-11", "2026-09-12"):
        x = vision_day("ETHUSDT", d).set_index("timestamp")
        a = p.loc[x.index, ["close", "volume", "sum_open_interest"]]
        assert np.allclose(a.to_numpy(float), x[["close", "volume", "sum_open_interest"]].to_numpy(float), equal_nan=True), d
    print("vision 일별 = 저장 패널 (3일) OK")


def selftest() -> None:
    c = np.array([100.0, 99.5, 99.0, 98.6, 99.2, 100.0, 100.0]); h = c + 0.01; l = c - 0.01
    z = np.zeros(len(c), np.bool_); sx = z.copy(); sx[4] = True
    p, xi, kd = simx(h, l, c, np.array([0]), np.array([1]), np.array([500.0]), np.array([5]), sx, z)
    assert kd[0] == 4 and xi[0] == 5                                           # 분 4 신호 → 분 5 종가 정리
    q, cost = 4, 100 + 99.56 + 99.12 + 98.68                                   # 44·88·132bp 분할 셋 다 체결
    assert abs(p[0] - ((100.0 * q - cost) / 100 * 1e4 - FEE * q) / 4) < 1e-6, p
    _, _, k2 = simx(h, l, c, np.array([0]), np.array([1]), np.array([100.0]), np.array([5]), z, z)
    assert k2[0] == 3                                                           # 100bp 손절이 먼저
    assert np.allclose(size_mult([157.3941696083474 * 0.7657638410664205]), [1.0])
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else parity() if "--parity" in sys.argv else main()
