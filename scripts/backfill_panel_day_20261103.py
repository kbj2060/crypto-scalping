"""vision 일별 klines 가 404 인 하루(2026-10-28)를 패널에 채운다 -- klines = fapi REST(하루 1요청/코인), metrics 5열 = vision.
서버에서 실행(로컬 바이낸스 REST 는 IP 한도 공유라 금지):  python backfill_panel_day_20261103.py DAY PANEL_DIR [--apply]
--apply 없으면 읽기 전용 점검. 적용 시 PANEL_DIR 옆 *_bak_<DAY> 에 원본을 복사하고, 임시 파일에 쓴 뒤 os.replace 로 교체한다.
"""
import json, os, shutil, sys, time, urllib.request
import io
from pathlib import Path
import pandas as pd

sys.path.insert(0, "scripts")
import build_binance_vision_panel_20260915 as b

DAY, PANEL, APPLY = sys.argv[1], Path(sys.argv[2]), "--apply" in sys.argv
BAK = PANEL.parent / f"{PANEL.name}_bak_{DAY.replace('-', '')}"


def rest_day(sym: str, old: pd.DataFrame) -> pd.DataFrame:
    t0 = int(pd.Timestamp(DAY).timestamp() * 1000)
    u = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=5m&startTime={t0}&endTime={t0 + 86_399_999}&limit=1500"
    r = json.load(urllib.request.urlopen(u, timeout=30))
    k = pd.DataFrame(r, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "count", "taker_buy_volume", "tbq", "ig"])[b.KCOLS]
    k = k.astype({c: float for c in b.KCOLS[1:]})
    k["timestamp"] = pd.to_datetime(k.pop("open_time"), unit="ms")
    k = k.rename(columns={"count": "trades", "taker_buy_volume": "taker_buy_base"})
    m = b.fetch(f"{b.BASE}/metrics/{sym}/{sym}-metrics-{DAY}.zip")
    assert m is not None, f"{sym} metrics {DAY} 없음"
    md = pd.read_csv(io.StringIO(m)); md["timestamp"] = pd.to_datetime(md["create_time"])
    k = k.merge(md[["timestamp"] + b.MCOLS], on="timestamp", how="left")
    return k[old.columns].astype(old.dtypes.to_dict())


for f in sorted(PANEL.glob("*USDT.parquet")):   # 11-07: 4코인 고정 -> 패널 전 코인(학습 문이 전 코인 구멍을 본다)
    sym = f.stem[:-4]
    old = pd.read_parquet(f)
    day = old.timestamp.dt.strftime("%Y-%m-%d") == DAY
    if day.any():
        print(sym, "이미", int(day.sum()), "행 있음 -- 건너뜀"); continue
    new = rest_day(f"{sym}USDT", old)
    assert len(new) == 288 and new.timestamp.is_unique, (sym, len(new))
    out = pd.concat([old, new]).sort_values("timestamp").reset_index(drop=True)
    ok5 = bool((out.timestamp.diff().dropna() == pd.Timedelta("5min")).all())
    c = out.set_index("timestamp").close
    t = pd.Timestamp(DAY)
    jin = (c[t] / c[t - pd.Timedelta("5min")] - 1) * 1e4
    jout = (c[t + pd.Timedelta("1D")] / c[t + pd.Timedelta("1D") - pd.Timedelta("5min")] - 1) * 1e4
    r5 = c.pct_change().abs() * 1e4
    print(f"{sym} {len(old):,}->{len(out):,}행 diff==5min:{ok5} 경계점프 진입 {jin:+.1f}bp 이탈 {jout:+.1f}bp "
          f"채운날 최대5분 {r5[new.timestamp].max():.1f}bp(패널 전체 99.9분위 {r5.quantile(.999):.1f}) OI결측 {int(new.sum_open_interest.isna().sum())}", flush=True)
    if APPLY and ok5:
        BAK.mkdir(exist_ok=True)
        if not (BAK / f.name).exists():
            shutil.copy2(f, BAK / f.name)
        tmp = f.with_suffix(".tmp"); out.to_parquet(tmp); os.replace(tmp, f)
    time.sleep(1)
