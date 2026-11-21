#!/usr/bin/env python3
"""/api/flow/heatmap 정답지(2026-11-22 · 여섯째 조각): 합성 래스터 파일 + 원본 함수(read_window·row_stats·vol_scores·approach_ratio)를 그대로 불러 뜬 값.

🔴서버 판 numpy(2.2.5)로 돌린다(rtmodel/tools/server_env.sh · f32 합 차례가 판마다 다를 수 있다):
    source <(bash rtmodel/tools/server_env.sh <venv> --env) && $PY rtserve/tools/gen_heatmap_fixture.py    # 저장소 루트에서
출력(커밋 안 함 · 다시 만들 수 있는 생성물): rtserve/tests/fixtures/gen/raster/<SYM>/*.f32 + heatmap.json(배열은 f32 바이트 sha256 + 앞 몇 칸).
래스터 = 원본 _RasterWriter 로 쓴다(빈 초는 원본처럼 무효 행으로 메움) + 손으로 만든 갈래: 시 중간에 시작한 파일(앞이 0 = 구멍) · 잘린 마지막 행 ·
마법 글자 틀린 파일 · 칸 수가 다른 파일 · mid NaN 행 · bin_lo 가 움직이는 행.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import live_orderflow_raster_collector_20260914 as rc  # noqa: E402

OUT = ROOT / "rtserve" / "tests" / "fixtures" / "gen" / "raster"
H = 3_600_000
T_HOUR = 1_795_000_000_000 // H * H          # 2026-11-17 무렵 · 시 경계
SYM = "ethusdt"


def h(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_tree() -> int:
    import shutil
    shutil.rmtree(OUT, ignore_errors=True)
    rng = np.random.default_rng(7)
    w = rc._RasterWriter(SYM, root=OUT)
    mid = 3100.0
    end = T_HOUR + 5 * H - 1234_000          # 마지막 시는 중간까지
    t = T_HOUR + 3 * H // 4                  # 첫 시는 45분부터(앞 = 무효 행 메움)
    while t < end:
        mid += float(rng.normal(0, 0.35))
        lo = int(math.floor(mid / 0.5)) - rc.N_BINS // 2
        q = rng.gamma(0.6, 40.0, rc.N_BINS).astype(np.float32)
        q[: rc.N_BINS // 2] *= 1.0
        q[rc.N_BINS // 2:] *= -1.0
        q[rng.random(rc.N_BINS) < 0.15] = 0.0
        if rng.random() < 0.01:
            q[rng.integers(0, rc.N_BINS)] = float(rng.choice([2500.0, -3100.0]))   # 벽
        m = float("nan") if rng.random() < 0.02 else mid
        w.write(t, lo, m, None if rng.random() < 0.005 else q)
        t += 1000 if rng.random() > 0.003 else 1000 * int(rng.integers(2, 40))      # 수집 공백 → 무효 행
    w.close()
    # 손 갈래 ① 시 중간 구멍: 다음 시 파일을 헤더만 쓰고 600초 뒤부터 seek 로 써서 앞이 0(ts 0 · mid 0.0)
    p = rc._hour_path(SYM, T_HOUR + 5 * H, OUT)
    with open(p, "wb") as fh:
        fh.write(rc.HEADER.pack(rc.MAGIC, rc.VERSION, rc.N_BINS, rc.BIN_SIZE, 0, T_HOUR + 5 * H, 1000, 0))
        for k in range(600, 700):
            fh.seek(rc.HEADER.size + k * rc.row_bytes())
            q = rng.gamma(0.6, 40.0, rc.N_BINS).astype("<f4")
            q[:6] = (7.0, 0.0, -7.0)[k % 3]          # |값| 같고 부호 다른 칸(주기 3) -- agg 접기의 «첫 최대» 갈래
            fh.write(rc.ROW_HEAD.pack(T_HOUR + 5 * H + k * 1000, int(mid / 0.5) - 100, mid) + q.tobytes())
        fh.write(b"\x01\x02\x03")             # ② 잘린 마지막 행
    # ③ 마법 글자 틀림 · ④ 칸 수 다름 -- 둘 다 건너뛴다
    with open(rc._hour_path(SYM, T_HOUR - H, OUT), "wb") as fh:
        fh.write(b"XXXX" + b"\x00" * 28 + b"\x00" * rc.row_bytes() * 10)
    p2 = rc._hour_path(SYM, T_HOUR - 2 * H, OUT)
    with open(p2, "wb") as fh:
        fh.write(rc.HEADER.pack(rc.MAGIC, rc.VERSION, 100, 0.5, 0, T_HOUR - 2 * H, 1000, 0))
        for k in range(3600):
            fh.write(rc.ROW_HEAD.pack(T_HOUR - 2 * H + k * 1000, 6000, 3000.0) + np.full(100, 5.0, "<f4").tobytes())
    return end


def win_out(wd: dict) -> dict:
    return {k: (h(v.astype("<f4")) if isinstance(v, np.ndarray) else v) for k, v in wd.items()} | {
        "qty_shape": list(wd["qty"].shape), "mid_head": [None if not math.isfinite(x) else float(x) for x in wd["mid"][:5].tolist()]}


# ── server.py 옮겨 적음(Read 로 읽은 줄 · 서버 모듈은 import 하지 않는다) ──
def heatmap_obi(last_qty, price, center: float, band_pct: float = 0.5) -> float:          # 1136~1147
    band = np.abs((np.asarray(price, float) - center) / center * 100.0) <= band_pct
    q = np.asarray(last_qty, float)
    b = float(np.clip(q, 0, None)[band].sum()); k = float(np.clip(-q, 0, None)[band].sum())
    return round((b - k) / max(b + k, 1e-9), 3)


def _heatmap_rows(w, qty, mid, fp):                                                     # 4523~4627 (flows → fp: None = 흐름 엔진 없음)
    import base64
    ok = np.isfinite(mid)
    if not ok.any():
        return {"rows": None, "summary": None}
    a = np.abs(qty[ok])
    st = rc.row_stats(a, int(w["dt_s"]))
    inst, pers = st["inst"], st["pers"]
    spot = float(np.nanmedian(mid[ok]))
    price = (w["bin_lo"] + np.arange(w["n_bins"])) * w["bin_size"]
    m0, m1 = float(mid[ok][0]), float(mid[ok][-1])
    cov_half = 0.4 * rc.N_BINS * w["bin_size"]
    cov = (np.abs(price - m0) < cov_half) & (np.abs(price - m1) < cov_half)
    same_bid = cov & (price < m0) & (price < m1)
    same_ask = cov & (price > m0) & (price > m1)
    bid_now, ask_now = np.clip(qty[ok], 0, None), np.clip(-qty[ok], 0, None)
    bid_drop = np.clip(bid_now[0] - bid_now[-1], 0, None)
    ask_drop = np.clip(ask_now[0] - ask_now[-1], 0, None)
    fill_bid = np.zeros(w["n_bins"]); fill_ask = np.zeros(w["n_bins"])
    fill_source = "footprint"
    try:
        t_start = w["t0_ms"] / 1000.0
        t_end = t_start + w["cols"] * w["dt_s"]
        if fp is None:
            raise LookupError("symbol")
        for bar, cells in fp["bars"].items():
            if bar + 300 < t_start or bar > t_end:
                continue
            for key, v in cells.items():
                i = int(round(key * fp["bucket"] / w["bin_size"])) - w["bin_lo"]
                if 0 <= i < w["n_bins"]:
                    fill_ask[i] += float(v[0]); fill_bid[i] += float(v[1])
    except Exception:  # noqa: BLE001
        fill_source = None
    drop = float(bid_drop[same_bid].sum() + ask_drop[same_ask].sum())
    filled = float(np.minimum(bid_drop, fill_bid)[same_bid].sum() + np.minimum(ask_drop, fill_ask)[same_ask].sum())
    offtouch_leave_share = (round(1.0 - filled / drop, 3) if (fill_source and drop > 0) else None)
    return {"rows": {"bin_lo": int(w["bin_lo"]), "bin_size": float(w["bin_size"]),
                     **{f"{k}_f4": base64.b64encode(np.ascontiguousarray(v, "<f4")).decode() for k, v in st.items()}},
            "summary": {"spot": round(spot, 6), "persist_share": round(float(pers.sum() / max(inst.sum(), 1e-9)), 3),
                        "offtouch_leave_share": offtouch_leave_share, "fill_source": fill_source,
                        "offtouch_bins": int(same_bid.sum() + same_ask.sum()),
                        "obi": heatmap_obi(qty[ok][-1], price, m1), "obi_band_pct": 0.5, "window_s": int(w["cols"] * w["dt_s"])}}


def _heatmap_read(now_ms, cols, agg, rows_only, fp):                                    # 4482~4521
    import base64
    w = rc.read_window(SYM, now_ms, cols, agg, OUT)
    qty, mid = w.pop("qty"), w.pop("mid")
    nz = np.abs(qty[qty != 0])
    scale = float(np.percentile(nz, 97)) if nz.size else 1.0
    if not (scale > 0):
        scale = 1.0
    out = {**w, "scale": scale, **_heatmap_rows(w, qty, mid, fp)}
    if rows_only:
        try:
            b = rc.read_window(SYM, now_ms, 3, 1, OUT)
            live = np.nonzero(np.isfinite(b["mid"]))[0]
            if b["n_bins"] and live.size:
                i = int(live[-1])
                out["book"] = {"t_ms": int(b["t0_ms"]) + i * 1000, "bin_lo": int(b["bin_lo"]), "bin_size": float(b["bin_size"]), "mid": float(b["mid"][i]),
                               "q_f4": base64.b64encode(np.ascontiguousarray(b["qty"][i], dtype="<f4")).decode()}
        except Exception:  # noqa: BLE001
            pass
        return out
    q8 = np.rint(np.clip(qty / scale, -1.0, 1.0) * 127.0)
    return {**out, "qty_i8": base64.b64encode(np.ascontiguousarray(q8, dtype=np.int8)).decode(),
            "mid_b64": base64.b64encode(np.ascontiguousarray(mid, dtype="<f4")).decode()}


def _approach_read(now_ms):                                                             # 4672~4690
    import base64
    w = rc.read_window(SYM, now_ms, 2880, 5, OUT)
    a = rc.approach_ratio(w)
    return {"approach_bin_lo": a["bin_lo"], "approach_bin_size": a["bin_size"], "approach_qualifying": a["qualifying"], "approach_window_s": 4 * 3600,
            "approach_f4": base64.b64encode(np.ascontiguousarray(a["ratio"], "<f4")).decode()}


def _volsig_ref(now_ms):                                                                # 4629~4647
    w = rc.read_window(SYM, now_ms, 4 * 3600, 1, OUT)
    past, refill = rc.vol_scores(w, range(600, 4 * 3600, 60))
    if past.size < 60:
        return {}
    sp, sr = np.sort(past), np.sort(refill)
    sc = np.sort(np.searchsorted(sp, past) / sp.size + np.searchsorted(sr, refill) / sr.size)
    return {"past": sp, "refill": sr, "score": sc}


def _volsig_read(now_ms, ref):                                                          # 4649~4670
    if not ref:
        return {}
    w = rc.read_window(SYM, now_ms, 660, 1, OUT)
    past, refill = rc.vol_scores(w)
    if past.size == 0:
        return {}
    pp = float(np.searchsorted(ref["past"], past[0]) / ref["past"].size)
    rp = float(np.searchsorted(ref["refill"], refill[0]) / ref["refill"].size)
    sc = float(np.searchsorted(ref["score"], pp + rp) / ref["score"].size)
    return {"vol_pct": round(100 * sc), "vol_past_pct": round(100 * pp), "vol_refill_pct": round(100 * rp), "vol_ref_n": int(ref["score"].size)}


def fake_fp(end_ms: int) -> dict:
    rng = np.random.default_rng(3)
    bars = {}
    for b in range((end_ms // 1000 - 6 * 3600) // 300 * 300, end_ms // 1000 + 1, 300):
        c = {}
        for _ in range(60):
            key = int(rng.integers(6080, 6280))
            v = c.setdefault(key, [0.0, 0.0])
            v[0] += round(float(rng.gamma(0.5, 3.0)), 3); v[1] += round(float(rng.gamma(0.5, 3.0)), 3)
        bars[b] = c
    return {"bucket": 0.5, "bars": bars}


def main() -> None:
    end = write_tree()
    T = T_HOUR + 5 * H + 650_000
    cases = [(end - 1, 300, 3), (end - 1, 600, 1), (end - 1, 3, 1), (end - 1, 2880, 5), (end - 1, 660, 1), (T, 900, 1), (T, 300, 3),
             (T_HOUR - 1, 300, 1), (T_HOUR - 2 * H + 1800_000, 120, 2), (T_HOUR + 3 * H // 4 + 30_000, 300, 1), (end + 3 * H, 60, 1)]
    out = []
    for to, cols, agg in cases:
        wd = rc.read_window(SYM, to, cols, agg, OUT)
        out.append({"to_ms": to, "cols": cols, "agg": agg, "window": win_out(wd)})
    fp = fake_fp(end)
    hm = []
    for to, cols, agg, ro, use_fp in [(end - 1, 300, 3, True, True), (end - 1, 300, 3, False, True), (end - 1, 900, 1, True, False), (end - 1, 60, 1, False, True),
                                      (T, 300, 3, True, True), (T_HOUR - 1, 300, 1, True, True), (end - 1, 120, 60, True, True), (end + 3 * H, 300, 3, False, True)]:
        f = fp if use_fp else None
        try:
            body = json.dumps(_heatmap_read(to, cols, agg, ro, f))
        except Exception as e:  # noqa: BLE001
            body = "ERR " + type(e).__name__
        hm.append({"to_ms": to, "cols": cols, "agg": agg, "rows_only": ro, "fp": use_fp, "body": body})
    ap = [{"to_ms": to, "body": json.dumps(_approach_read(to))} for to in (end - 1, T, T_HOUR - 1)]
    vs = []
    for to in (end - 1, end - 700_000, T):
        ref = _volsig_ref(to)
        vs.append({"to_ms": to, "ref_n": int(ref["score"].size) if ref else 0,
                   "ref": {k: h(v.astype("<f4") if v.dtype == np.float32 else v.astype("<f8")) for k, v in ref.items()} if ref else {},
                   "read": json.dumps(_volsig_read(to, ref))})
    # api_flow_heatmap(4692~4732) 조립: rows 모드 = 읽기 + 접근행동(rows 에) + 변동 신호(summary 에) · 흐름 엔진 풋프린트는 빈 봉(허브 없음과 같게)
    api = []
    for to, q in [(end - 1, "?mode=rows"), (end - 1, "?cols=120&agg=1"), (T, "?mode=rows&symbol=ETHUSDT")]:
        rows_only = "mode=rows" in q
        cols, agg = (120, 1) if "cols=120" in q else (300, 3)
        pay = _heatmap_read(to, cols, agg, rows_only, {"bucket": 0.5, "bars": {}})
        if rows_only and pay.get("rows"):
            pay["rows"].update(_approach_read(to))
            ref = _volsig_ref(to)
            if pay.get("summary"):
                pay["summary"].update(_volsig_read(to, ref))
        api.append({"to_ms": to, "q": q, "body": json.dumps(pay)})
    fpj = {str(b): [[k * fp["bucket"], v[0], v[1]] for k, v in c.items()] for b, c in fp["bars"].items()}
    (OUT / "heatmap.json").write_text(json.dumps({"t_hour": T_HOUR, "read_window": out, "heatmap": hm, "approach": ap, "volsig": vs, "fp": fpj, "api": api}, indent=0))
    print(f"read_window {len(out)} · heatmap {len(hm)} · approach {len(ap)} · volsig {len(vs)} · numpy {np.__version__}")


if __name__ == "__main__":
    main()
