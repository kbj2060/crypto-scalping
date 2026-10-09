"""포지션 프로파일 — 최근 1·7·30일에 열려 아직 남은 계약의 가격 분포(추정, 롱=숏) (2026-10-10).

사용자 «풋프린트 체결 기둥 위 [체결 | 포지션] 토글 · A + C 미니맵». 연구: scripts/research_eth_position_profile_trading_tests_20261009.py,
HL 정답지 채점 tmp/hl_position_truth_20261009 (7일 평균 진입가 오차 79 → 27bp).
모델(5분봉마다 한 걸음): 닫을 양 = max(−ΔOI, 0) + K·거래량(회전) — 나이 가중 exp(−나이/τ)+ρ 로 새 묶음부터 닫는다(옛 포지션 가중 ρ)
  · 넣을 양 = max(ΔOI, 0) + K·거래량을 그 봉에 · 합 = OI 로 맞춤. 30일 넘은 묶음은 «옛 포지션»으로 접는다.
🔴추정이지 실측 포지션이 아니다. 덩어리 저항·재방문·고통 지수는 매매 검정 불통과(= VWAP 평균회귀 재표현) — 지도로만.
"""
from __future__ import annotations

import numpy as np

K, TAU, RHO = 0.057, 576.0, 0.02     # 회전(바이낸스 거래량/OI 로 환산) · 나이 τ = 2일(봉) · 옛 가중
RING = 8640                          # 30일(5분봉)
WINDOWS = (1, 7, 30)


class Book:
    """닫힌 5분봉을 차례로 밀어 넣는 상태. 봉마다 [시각(초), 고, 저, 남은 양]."""

    def __init__(self) -> None:
        self.t = np.zeros(0, np.int64); self.h = np.zeros(0); self.l = np.zeros(0); self.x = np.zeros(0)
        self.old = 0.0; self.oi = None

    @staticmethod
    def _step(x: np.ndarray, old: float, oi_prev: float, oi: float, v: float) -> tuple[np.ndarray, float, float]:
        """한 걸음(복사본) → (기존 묶음 x', old', 새 봉 양)."""
        d = oi - oi_prev; rem = max(-d, 0.0) + K * v
        x = x.copy()
        if rem > 0:
            w = np.exp(-np.arange(len(x), 0, -1) / TAU) + RHO            # 맨 뒤 = 나이 1
            wx = x * w; tw = wx.sum() + old * RHO
            if tw > 0:
                f = min(rem / tw, 1.0 / max(w.max() if len(w) else 0.0, RHO))
                x -= np.minimum(x, wx * f); old = max(old - old * RHO * f, 0.0)
        new = max(d, 0.0) + K * v
        tot = x.sum() + new + old
        if tot > 0:
            r = oi / tot; x *= r; old *= r; new *= r
        return x, old, new

    def push(self, t: int, h: float, l: float, v: float, oi: float) -> None:
        if not (np.isfinite(oi) and oi > 0):
            return
        if self.oi is None:                                               # 첫 봉: 전부 옛 포지션
            self.oi = oi; self.old = oi; return
        x, self.old, new = self._step(self.x, self.old, self.oi, oi, max(v, 0.0))
        self.t = np.append(self.t, t); self.h = np.append(self.h, h); self.l = np.append(self.l, l); self.x = np.append(x, new)
        self.oi = oi
        if len(self.x) > RING:                                            # 30일 넘은 묶음은 옛 포지션으로
            k = len(self.x) - RING; self.old += float(self.x[:k].sum())
            self.t, self.h, self.l, self.x = self.t[k:], self.h[k:], self.l[k:], self.x[k:]

    def snapshot(self, forming: tuple[int, float, float, float, float] | None = None):
        """(t, h, l, x) — forming = 형성 중인 봉(t, 고, 저, 지금까지 거래량, 지금 OI)을 임시 한 걸음으로 얹는다(상태는 안 바뀜)."""
        if forming is None or self.oi is None:
            return self.t, self.h, self.l, self.x
        ft, fh, fl, fv, foi = forming
        x, _, new = self._step(self.x, self.old, self.oi, foi, max(fv, 0.0))
        return np.append(self.t, ft), np.append(self.h, fh), np.append(self.l, fl), np.append(x, new)


def bins(h: np.ndarray, l: np.ndarray, x: np.ndarray, bw: float = 1.0) -> tuple[int, np.ndarray]:
    """봉마다 남은 양을 고가~저가 사이 $bw 칸에 고르게(칸 오차 연구: 고저 균등이면 $10 칸에서 0.3%). → (첫 칸 번호, 칸 값)."""
    m = x > 0
    if not m.any():
        return 0, np.zeros(0)
    h, l, x = h[m], l[m], x[m]
    hi = np.maximum(h, l + 1e-9); k0 = np.floor(l / bw).astype(np.int64); k1 = np.floor(hi / bw).astype(np.int64)
    n = k1 - k0 + 1; lo = int(k0.min()); out = np.zeros(int(k1.max()) - lo + 1)
    bar = np.repeat(np.arange(len(x)), n); k = k0[bar] + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    ov = np.minimum(hi[bar], (k + 1) * bw) - np.maximum(l[bar], k * bw)
    ov = np.clip(ov, 0, None) / (hi[bar] - l[bar])
    np.add.at(out, k - lo, x[bar] * ov)
    return lo, out


def oi_hist_map(rows) -> dict[int, float]:
    """openInterestHist 응답 → {스탬프(초): OI}. 스탬프 T = T 시점 스냅샷(서버 1초 OI 와 5분 변화 상관 .998, 10-10 실측) → 봉 b 끝 = b+300."""
    return {int(r["timestamp"]) // 1000: float(r["sumOpenInterest"]) for r in rows or []}


def bars_from_klines(rows, oi: dict[int, float], end_s: int) -> list[tuple[int, float, float, float, float]]:
    """klines 응답 + OI 이력 → 닫힌 5분봉 (시각, 고, 저, 거래량, 봉 끝 OI). 끝 스냅샷이 없는 봉은 뺀다(다음 증분에서 다시)."""
    out = []
    for r in rows or []:
        b = int(r[0]) // 1000
        o = oi.get(b + 300)
        if b < end_s and o is not None:
            out.append((b, float(r[2]), float(r[3]), float(r[5]), o))
    return out


def forming_from_cells(bars: dict, since_s: int, bucket: float) -> tuple[float, float, float] | None:
    """흐름 엔진 풋프린트 칸 {봉 시작: {가격 버킷: [매수, 매도, …]}} 에서 since_s 이후 (고, 저, 거래량). 비면 None."""
    ks, v = [], 0.0
    for b, cells in list(bars.items()):
        if b >= since_s:
            for k, c in list(cells.items()):
                ks.append(k); v += c[0] + c[1]
    if not ks:
        return None
    return (max(ks) + 0.5) * bucket, (min(ks) - 0.5) * bucket, v


def live_at(oi_1s: dict[int, float], t: int, tol: int = 30) -> float | None:
    """1초 OI {초: 값} 에서 t 이하 가장 가까운 값(tol 초 안). 없으면 None."""
    ks = [k for k in list(oi_1s) if t - tol <= k <= t]
    return oi_1s[max(ks)] if ks else None


def payload(book: Book, now_s: int, price: float, forming=None, bw: float = 1.0, windows=WINDOWS) -> dict:
    t, h, l, x = book.snapshot(forming)
    win = {}
    for d in windows:
        m = t >= now_s - d * 86400
        lo, vals = bins(h[m], l[m], x[m], bw)
        tot = float(vals.sum())
        win[str(d)] = {"lo": lo, "vals": np.round(vals, 1).tolist(), "tot": round(tot, 1),
                       "avg": round(float(((np.arange(len(vals)) + lo + 0.5) * bw * vals).sum() / tot), 2) if tot > 0 else None,
                       "hours": round(float(m.sum()) / 12, 1)}                     # 실제로 덮은 시간(30일이 덜 찼으면 작다)
    return {"asof": int(now_s), "price": price, "oi": book.oi, "bw": bw, "win": win,
            "basis": "추정(5분 OI·거래량 · 회전+나이 가중 닫기 · 롱=숏) -- 실측 포지션 아님, 지도로만"}


if __name__ == "__main__":
    # 자체점검: ① 양 합 = OI ② 칸 합 = 묶음 합 ③ 형성 봉 임시 걸음이 상태를 안 바꿈 ④ 30일 넘으면 접힘 ⑤ 연구 루프와 같은 값
    rng = np.random.default_rng(1)
    n = 9000; oi = 1e6 + np.cumsum(rng.normal(0, 2e3, n)); v = rng.uniform(1e3, 5e3, n); px = 2500 + np.cumsum(rng.normal(0, 2, n))
    b = Book()
    for i in range(n):
        b.push(i * 300, px[i] + 2, px[i] - 2, v[i], oi[i])
    assert abs(b.x.sum() + b.old - oi[-1]) < 1e-3 and len(b.x) == RING
    lo, vals = bins(b.h, b.l, b.x); assert abs(vals.sum() - b.x.sum()) < 1e-6 * b.x.sum()
    x0 = b.x.copy(); b.snapshot((n * 300, px[-1] + 1, px[-1] - 1, 2e3, oi[-1] + 5e3)); assert np.array_equal(x0, b.x)
    m = 600; L = np.zeros(m); old = oi[0]                                   # ⑤ 연구판(estimators.run_age 와 같은 식)
    for i in range(1, m):
        d = oi[i] - oi[i - 1]; w = np.exp(-(i - np.arange(i)) / TAU) + RHO; rem = max(-d, 0.0) + K * v[i]
        wx = L[:i] * w; tw = wx.sum() + old * RHO
        if rem > 0 and tw > 0:
            f = min(rem / tw, 1 / max(w.max(), RHO)); L[:i] -= np.minimum(L[:i], wx * f); old = max(old - old * RHO * f, 0)
        L[i] += max(d, 0.0) + K * v[i]; tot = L[:i + 1].sum() + old
        L[:i + 1] *= oi[i] / tot; old *= oi[i] / tot
    c = Book()
    for i in range(m):
        c.push(i * 300, px[i] + 2, px[i] - 2, v[i], oi[i])
    assert np.allclose(c.x, L[1:], rtol=1e-9, atol=1e-6), "연구 루프와 다름"
    p = payload(b, n * 300, float(px[-1])); assert set(p["win"]) == {"1", "7", "30"} and p["win"]["30"]["hours"] == 720.0
    assert set(payload(b, n * 300, float(px[-1]), windows=(7,))["win"]) == {"7"}
    # 서버 조각: OI 스탬프 = 봉 끝 · 끝 스냅샷 없는 봉 빠짐 · 형성 칸 고저·거래량 · 1초 OI 허용 오차
    oi = oi_hist_map([{"timestamp": 600_000, "sumOpenInterest": "10"}, {"timestamp": 900_000, "sumOpenInterest": "12"}])
    assert oi == {600: 10.0, 900: 12.0}
    kl = [[300_000, "0", "101", "99", "100", "5"], [600_000, "0", "102", "98", "100", "7"], [900_000, "0", "1", "1", "1", "1"]]
    assert bars_from_klines(kl, oi, 900) == [(300, 101.0, 99.0, 5.0, 10.0), (600, 102.0, 98.0, 7.0, 12.0)]
    assert forming_from_cells({0: {200: [1, 2]}, 300: {198: [3, 0], 205: [0, 4]}}, 300, 0.5) == (102.75, 98.75, 7.0)
    assert forming_from_cells({0: {200: [1, 2]}}, 300, 0.5) is None
    assert live_at({100: 1.0, 125: 2.0, 140: 3.0}, 130) == 2.0 and live_at({100: 1.0}, 200) is None
    print("selftest OK -- 합 = OI · 칸 합 보존 · 형성 봉 무해 · 30일 접힘 · 연구 루프 재현 · 창 고르기 · 서버 조각(OI 스탬프·봉·형성 칸·1초 OI)")
