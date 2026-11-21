"""참고: AVX-512 를 끈 정답지(서버와 같은 계산) 대 켠 정답지(개발기 기본 numpy)의 차이. 같은 입력이라 다른 곳은 np.log·np.exp 의 SIMD 끝자리뿐이다.
    python rtmodel/tools/sr_avx_diff.py   (tests/fixtures/sr_fixture.json · sr_fixture_avx512.json 둘 다 먼저)"""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
a, b = (json.loads((D / f).read_text()) for f in ("sr_fixture.json", "sr_fixture_avx512.json"))
assert a["npy_disable"] and not b["npy_disable"], (a["npy_disable"], b["npy_disable"])
col, nl, nd, grp, ev, pm = {}, 0, 0, 0, 0, 0.0
for x, y in zip(a["sets"], b["sets"]):
    key = lambda L: [(r["k"], r.get("a"), r.get("look"), r.get("kind"), r.get("side")) for r in map(json.loads, L) if r["k"] != "res"]  # noqa: E731
    ev += key(x["main"]) != key(y["main"])
    for l1, l2 in zip(x["main"] + x["feat"], y["main"] + y["feat"]):
        nl += 1; nd += l1 != l2
        r1, r2 = json.loads(l1), json.loads(l2)
        grp += r1.get("grp") != r2.get("grp")
        if r1["k"] == "feat":
            for c in r1:
                if c not in ("k", "id") and not (r1[c] == r2[c] or (np.isnan(r1[c]) and np.isnan(r2[c]))):
                    e = col.setdefault(c, [0, 0.0]); e[0] += 1; e[1] = max(e[1], abs(r1[c] - r2[c]) / max(abs(r2[c]), 1e-300))
    for p1, p2 in zip(x["p"], y["p"]):
        pm = max(pm, abs(np.uint64(p1[0]).view(np.float64) - np.uint64(p2[0]).view(np.float64)))
print(f"줄 {nl} 중 글자 다른 줄 {nd} · 사건 열이 다른 벌 {ev} · 구분이 다른 줄 {grp} · 반올림 전 확률 최대 차 {pm:.2e}")
print("피쳐 열별(다른 사건 수 · 최대 상대 차):", {c: f"{n} · {r:.1e}" for c, (n, r) in sorted(col.items(), key=lambda kv: -kv[1][0])})
