"""15분·1시간 방향 65열판 모델 + 표본 밖 분위 씨앗 → rt_dirpred 가 읽는 JSON (2026-11-21). 서버에서(모델·씨앗이 서버에 있다) 실행한다.
    python rtmodel/tools/export_dirshort.py [출력 디렉터리 · 기본 data/live/rt_dirpred]
쓰는 것: ds_<코인>.json(load_art 그대로 · 부스터 텍스트 + 척도) · oos_<코인>.json(정렬된 표본 밖 예측 · f64 비트). ETH 는 75열판이라 아직 안 쓴다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_dir_short_20261028 as ds  # noqa: E402  -- import 때 REST·sqlite 접속 없음(함수 안)


def bits(a):
    return np.ascontiguousarray(np.asarray(a, np.float64)).view(np.uint64).tolist()


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for coin in ["ETH", *ds.TABLE]:                       # ETH = 75열판(호가 깊이 10열 · coin·src 칸 없는 아티팩트 → src null)
        if not all(p.exists() for p in ds.art_paths(coin)) or not ds.OOS_SEEDS[coin].exists():
            print(coin, "모델 또는 분위 씨앗 없음 -- 건너뜀"); continue
        art = ds.load_art(coin)
        d = {"coin": coin, "month": art["month"], "src": art.get("src"), "sources": [p.name for p in ds.art_paths(coin)],   # 원본 joblib(낡음 검사 rtmodel::model_stale)
             "models": {str(h): [[{"txt": m.booster_.model_to_string(), "scale": bits([s])[0]} for m, s in ms] for ms in seeds] for h, seeds in art["models"].items()}}
        (out / f"ds_{coin.lower()}.json").write_text(json.dumps(d))
        S = pd.read_parquet(ds.OOS_SEEDS[coin])
        (out / f"oos_{coin.lower()}.json").write_text(json.dumps({nm: bits(np.sort(S[f"p{nm}"].to_numpy(float))) for nm in ds.NAMES.values()}))
        print(coin, "→", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "live" / "rt_dirpred")
