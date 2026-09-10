#!/usr/bin/env python3
"""무기한 아카이브 **완결성 검증 및 복구** (2026-09-10).

## 왜 필요한가 — 2026-09-10 에 실제로 터진 두 결함
① **2021-12 klines 가 124종 전부에서 조용히 누락.** 바이낸스는 **2022-01 부터** klines 아카이브에
   헤더 행을 넣었다. 그 이전 파일은 헤더가 없어 첫 데이터 행이 컬럼명이 되고, 개명이 안 먹어
   KeyError → `except Exception: pass` 가 삼켰다. 무작위가 아니라 계통적이라 **전 종목 같은 달**이 빠졌다.
② **요청 대비 확보를 대조하지 않았다.** 그래서 ①이 드러나지 않았다.

⇒ 교훈: **다운로드 스크립트는 "무엇을 요청했고 무엇을 얻었는가"를 스스로 대조해야 한다.**
   조용한 부분 실패는 결과를 틀리게 만들지 않고 **조용히 좁게** 만든다.

③ **병렬 실행 경쟁.** 같은 목록을 도는 프로세스를 둘 띄우면 같은 CSV 를 동시에 쓴다.
   2026-09-10 에 뒤쪽 절반을 2번 프로세스에 줬는데 1번은 **전체 목록**을 돌고 있어 충돌했고,
   1번이 쓰기 도중 파일을 읽어 `KeyError: timestamp` 로 죽었다(영구 손상은 없었다).
   ⇒ 목록을 나눌 거면 **양쪽 다** 자기 몫만 돌게 해야 한다.

## 무엇을 하는가
결손 항목마다 실제로 받아 본다. **404 = 정당한 결손**(상장 전·폐지 후), **200 = 진짜 실패**라 복구한다.
판정을 `manifest.json` 에 남겨 다음 실행이 정당한 결손을 다시 두드리지 않게 한다.

사용법 `python3 scripts/verify_repair_perp_archive_20260910.py [--repair]`
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
KDIR = ROOT / "binance_data/klines"
MDIR = ROOT / "binance_data/metrics"
MANIFEST = ROOT / "binance_data/RAW_COMPLETENESS_MANIFEST.json"
BASE = "https://data.binance.vision/data/futures/um"
START, END = pd.Timestamp("2021-12-01"), pd.Timestamp("2026-08-28")
UA = {"User-Agent": "research/1.0"}
WORKERS = 16
ARCH_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
             "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
RENAME = {"open_time": "timestamp", "count": "trades",
          "taker_buy_volume": "taker_buy_base", "taker_buy_quote_volume": "taker_buy_quote"}
COLS = ["timestamp", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def log(m):
    print(f"[verify {time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch(url, tries=3):
    """(bytes, 상태) — 상태: ok / gone(404, 정당한 결손) / fail(오류 소진)."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read(), "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, "gone"
            time.sleep(1.0 * (i + 1))
        except Exception:
            time.sleep(1.0 * (i + 1))
    return None, "fail"


def parse_klines(raw: bytes) -> pd.DataFrame:
    """⭐헤더 유무를 **판별**해서 읽는다 — 2021-12 이전은 헤더가 없다."""
    headerless = not raw[:9].startswith(b"open_time")
    return (pd.read_csv(io.BytesIO(raw), header=None, names=ARCH_COLS) if headerless
            else pd.read_csv(io.BytesIO(raw)))


def main(argv) -> int:
    repair = "--repair" in argv
    months = pd.date_range(START, END, freq="MS").strftime("%Y-%m").tolist()
    days = pd.date_range(START, END, freq="D").strftime("%Y-%m-%d").tolist()
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    syms = sorted(p.name for p in KDIR.iterdir() if p.is_dir())
    log(f"종목 {len(syms)} · 요구 {len(months)}개월 klines · {len(days)}일 metrics · "
        f"모드 {'복구' if repair else '검증만'}")

    tot = {"k_gone": 0, "k_fail": 0, "k_fixed": 0, "m_gone": 0, "m_fail": 0, "m_fixed": 0}
    for si, sym in enumerate(syms, 1):
        e = man.setdefault(sym, {"klines_gone": [], "metrics_gone": []})
        f = KDIR / sym / f"{sym}-5m-api.csv"
        try:                                   # ③ 경쟁으로 반쯤 쓰인 파일을 만나도 죽지 않는다
            prev = pd.read_csv(f, parse_dates=["timestamp"]) if f.exists() else None
            if prev is not None and "timestamp" not in prev.columns:
                raise ValueError("timestamp 컬럼 없음")
        except Exception as ex0:
            log(f"  ⚠️{sym} 기존 CSV 읽기 실패({type(ex0).__name__}) — 전체 재수집 대상으로 전환")
            prev = None
        have_m = set(prev["timestamp"].dropna().dt.strftime("%Y-%m")) if prev is not None else set()
        need_m = [m for m in months if m not in have_m and m not in e["klines_gone"]]
        got = {}
        if need_m:
            with ThreadPoolExecutor(WORKERS) as ex:
                fs = {ex.submit(fetch, f"{BASE}/monthly/klines/{sym}/5m/{sym}-5m-{m}.zip"): m
                      for m in need_m}
                for fu in as_completed(fs):
                    b, st = fu.result(); m = fs[fu]
                    if st == "gone":
                        e["klines_gone"].append(m); tot["k_gone"] += 1
                    elif st == "fail":
                        tot["k_fail"] += 1
                    else:
                        try:
                            z = zipfile.ZipFile(io.BytesIO(b))
                            got[m] = parse_klines(z.read(z.namelist()[0]))
                        except Exception as ex2:
                            log(f"    ⚠️{sym} {m} 파싱 실패 {type(ex2).__name__}"); tot["k_fail"] += 1
        if got and repair:
            d = pd.concat(list(got.values()), ignore_index=True).rename(columns=RENAME)
            d["timestamp"] = pd.to_datetime(d["timestamp"], unit="ms", errors="coerce")
            for c in COLS:
                if c not in d.columns:
                    d[c] = 0
            if prev is not None:
                d = pd.concat([prev[COLS], d[COLS]], ignore_index=True)
            d = d.dropna(subset=["timestamp"]).sort_values("timestamp") \
                 .drop_duplicates("timestamp", keep="last")
            f.parent.mkdir(parents=True, exist_ok=True)
            d[COLS].to_csv(f, index=False)
            tot["k_fixed"] += len(got)
            log(f"  🔧 {sym} klines {len(got)}개월 복구 → {d.timestamp.min().date()} ~ {d.timestamp.max().date()}")
        elif got:
            log(f"  🔴 {sym} klines {sorted(got)} 는 받을 수 있는데 없다 (--repair 필요)")

        have_d = set(x.name[len(sym) + 9:-4] for x in MDIR.glob(f"{sym}-metrics-*.zip"))
        need_d = [x for x in days if x not in have_d and x not in e["metrics_gone"]]
        if need_d:
            with ThreadPoolExecutor(WORKERS) as ex:
                fs = {ex.submit(fetch, f"{BASE}/daily/metrics/{sym}/{sym}-metrics-{x}.zip"): x
                      for x in need_d}
                for fu in as_completed(fs):
                    b, st = fu.result(); x = fs[fu]
                    if st == "gone":
                        e["metrics_gone"].append(x); tot["m_gone"] += 1
                    elif st == "fail":
                        tot["m_fail"] += 1
                    elif repair:
                        (MDIR / f"{sym}-metrics-{x}.zip").write_bytes(b); tot["m_fixed"] += 1
                    else:
                        tot["m_fixed"] += 1
        for k in ("klines_gone", "metrics_gone"):
            e[k] = sorted(set(e[k]))
        if si % 20 == 0 or si == len(syms):
            MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=0))
            log(f"  [{si}/{len(syms)}] klines 정당결손 {tot['k_gone']} 복구 {tot['k_fixed']} 실패 {tot['k_fail']} · "
                f"metrics 정당결손 {tot['m_gone']} 복구 {tot['m_fixed']} 실패 {tot['m_fail']}")
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=0))
    log("=" * 96)
    log(f"klines  정당결손 {tot['k_gone']:,} · **복구 {tot['k_fixed']:,}** · 🔴미해결 실패 {tot['k_fail']:,}")
    log(f"metrics 정당결손 {tot['m_gone']:,} · **복구 {tot['m_fixed']:,}** · 🔴미해결 실패 {tot['m_fail']:,}")
    log(f"매니페스트 {MANIFEST}")
    if tot["k_fail"] or tot["m_fail"]:
        log("🔴미해결 실패가 있다 — 다시 돌려라. 이 상태로 패널을 지으면 조용히 좁은 표본이 된다.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
