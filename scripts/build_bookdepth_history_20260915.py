"""data.binance.vision 의 **과거 bookDepth** 전 구간 수집 — 30초마다 ±1~5% 밴드 깊이.
이 저장소의 방향 연구는 전부 5분 집계를 썼다. 호가창 «모양」은 한 번도 안 본 정보다."""
import io,zipfile,urllib.request,urllib.error,pandas as pd,pathlib,sys,concurrent.futures as cf
# 인자: [시작 끝 출력폴더] -- 기본값 = 원래 수집(2024-01-01~2026-08-22, data/binance_vision)
OUT=pathlib.Path(sys.argv[3] if len(sys.argv)>3 else "data/binance_vision"); OUT.mkdir(parents=True,exist_ok=True)
SYMS=["ETHUSDT","BTCUSDT","SOLUSDT"]
days=pd.date_range(*(sys.argv[1:3] if len(sys.argv)>2 else ("2024-01-01","2026-08-22")),freq="D")
def one(args):
    sym,d=args
    ds=d.strftime("%Y-%m-%d")
    p=OUT/f"{sym}_bookDepth_{ds}.parquet"
    if p.exists(): return 0
    u=f"https://data.binance.vision/data/futures/um/daily/bookDepth/{sym}/{sym}-bookDepth-{ds}.zip"
    try:
        raw=urllib.request.urlopen(u,timeout=40).read()
    except urllib.error.HTTPError as e:
        return -1 if e.code==404 else -2
    except Exception:
        return -2
    z=zipfile.ZipFile(io.BytesIO(raw))
    df=pd.read_csv(z.open(z.namelist()[0]))
    df["timestamp"]=pd.to_datetime(df["timestamp"])
    # 🔴±0 밴드가 있어 int(-0.0)==int(0.0)==0 으로 컬럼이 충돌한다. ±1~5 만 쓴다.
    df=df[df["percentage"].abs().between(0.5,5.5)]
    # 밴드별 깊이를 넓은 형태로 — 행당 timestamp 하나
    w=df.pivot_table(index="timestamp",columns="percentage",values="notional",aggfunc="first")
    w.columns=[f"n{int(c)}" for c in w.columns]
    w.to_parquet(p)
    return 1
jobs=[(s,d) for s in SYMS for d in days]
ok=miss=err=0
with cf.ThreadPoolExecutor(max_workers=12) as ex:
    for i,r in enumerate(ex.map(one,jobs)):
        ok+= r==1; miss+= r==-1; err+= r==-2
        if (i+1)%300==0: print(f"{i+1}/{len(jobs)} · 받음 {ok} · 없음 {miss} · 실패 {err}",flush=True)
print(f"완료 {len(jobs)} · 받음 {ok} · 없음 {miss} · 실패 {err}")
