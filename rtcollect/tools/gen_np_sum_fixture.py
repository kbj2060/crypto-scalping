"""np.sum 정답지: 서버 판 numpy(2.2.5 · rtmodel/tools/server_env.sh venv)로 경계 길이 배열의 합 비트.
x_i = ((i·2654435761) mod 1000003) / 997 − 500.3 (파이썬 float 연산 = Rust f64 와 같다).
  $PY rtcollect/tools/gen_np_sum_fixture.py
"""
import json
import sys
from pathlib import Path

import numpy as np

N = [1, 7, 8, 9, 127, 128, 129, 1000, 8191, 8192, 8193, 9000, 16384, 16385, 20000, 60000, 200001]
out = {}
for n in N:
    x = np.array([((i * 2654435761) % 1000003) / 997.0 - 500.3 for i in range(n)], dtype=np.float64)
    out[str(n)] = float(np.sum(x)).hex()
    assert float(x.sum()).hex() == out[str(n)]
(Path(__file__).resolve().parents[1] / "tests/fixtures/np_sum_golden.json").write_text(json.dumps({"numpy": np.__version__, "sums": out}, indent=0))
print(np.__version__, out)
