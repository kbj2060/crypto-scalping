"""4h 상대 강도(xsec) 기록 정답지 (2026-11-21 · tests/dirrest.rs xsec_parity). 서버 pred_record_loop 끝의 tick(dashboard/server.py:2857-2859)을
원본 함수로 흉내 낸다: 9코인 `d4._klines(c, 60, now_ms)` → `pred_record.xsec_record`. `_get` 은 가짜(합성 5분봉 · 요청을 적음) -- 네트워크 없음.
회차: 격자(UTC 시 %4==3 · 55분 봉)를 지나는 시각들 · 같은 격자 다시(새 줄 0) · 한 코인 봉 빠짐(240분 검사 실패) · 한 코인 빈 응답.
    python rtmodel/tools/gen_xsec_fixture.py      # → tests/fixtures/xsec_fixture.json (커밋 안 함)
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "dashboard"))
import live_eth_dir4h_20261015 as d4  # noqa: E402
import pred_record as PR  # noqa: E402

BAR = 300_000


def main():
    t0 = 1_795_000_000_000 // 14_400_000 * 14_400_000
    px = {c: 10 ** (k / 3 - 1) * np.exp(np.cumsum(np.random.default_rng(k).normal(0, 2e-3, 400))) for k, c in enumerate(PR.X_COINS)}
    nows = [t0 + 3 * 3_600_000 + h * 3_600_000 + 20_000 for h in range(0, 9)]          # 격자 03·07·11시(UTC) 55분 봉이 닫힌 뒤 포함
    nows += [nows[4], nows[-1] + 4 * 3_600_000, nows[-1] + 8 * 3_600_000]             # 같은 시각 다시(새 줄 0) · 봉 빠짐 · 빈 응답
    tmp = Path(tempfile.mkdtemp())
    runs = []
    for k, now in enumerate(nows):
        reqs = []

        def get(path, params, now=now, reqs=reqs, k=k):
            reqs.append([path, [[a, str(b)] for a, b in params.items()], []])
            c = params["symbol"][:-4]
            if k == len(nows) - 1 and c == "DOGE":
                return []
            last = min(int(params.get("endTime", now)), now) // BAR * BAR
            out = []
            for t in range(last - (int(params["limit"]) - 1) * BAR, last + 1, BAR):
                i = (t - t0) // BAR
                if k == len(nows) - 2 and c == "LINK" and t == last - 20 * BAR:
                    continue                                                           # 봉 하나 빠짐
                v = px[c][i] * (1.0004 if t + BAR - 1 >= now else 1.0)
                out.append([t, "1", f"{v * 1.001:.6g}", f"{v * 0.999:.6g}", f"{v:.6g}", "10", t + BAR - 1, "0", 5, "3", "0", "0"])
            reqs[-1][2] = out                                                          # 응답도 적는다(Rust 시험이 그대로 돌려줌)
            return out
        d4._get = get
        PR._xs_latest.clear()
        n = PR.xsec_record({c: d4._klines(c, 60, now).set_index("timestamp").close for c in PR.X_COINS}, d=tmp)
        f = tmp / "xsec.jsonl"
        runs.append({"now": now, "n": n, "requests": reqs, "file": f.read_text() if f.exists() else None, "latest": json.dumps(PR._xs_latest, ensure_ascii=False) if PR._xs_latest else None})
        print(now, n, len(reqs), PR._xs_latest.get("t"))
    p = ROOT / "rtmodel/tests/fixtures/xsec_fixture.json"
    p.write_text(json.dumps({"runs": runs})); print(p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
