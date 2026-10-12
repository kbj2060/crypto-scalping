"""**틱 체결(aggTrades) → 5분 주문흐름 피쳐** — 저장 없이 스트리밍.

이 저장소의 주문흐름 피쳐는 전부 **5분 집계 `taker_buy_base`** 였다. 틱 단위면 처음으로
만들 수 있는 것들이 있다(전부 «봉 집계로는 원리적으로 불가능»):
  · **대형/소형 체결 분리 불균형** — 고래와 잔챙이를 가른다(봉은 합쳐버린다)
  · **체결 건수 불균형** ≠ 거래량 불균형
  · **공격자 부호 자기상관/연속길이** — 정보성 거래의 군집(Cont 의 persistent flow)
  · **VPIN** — 거래량 버킷 기준 정보기반거래 확률
  · **Kyle λ 대리** — |Δ가격| / 거래량 = 단위 거래량당 충격(=유동성)
  · **체결 간격 강도** — 도착률
🔴19GB 를 디스크에 두지 않는다. 하루 받아 피쳐만 남기고 원본은 버린다(일별 parquet ~수십 KB).
"""
import io,zipfile,urllib.request,urllib.error,numpy as np,pandas as pd,pathlib,sys,concurrent.futures as cf
# 인자: SYM [시작 끝 출력폴더] -- 기본값 = 원래 수집(2024-01-01~2026-08-22, data/binance_vision/aggfeat)
SYM=sys.argv[1] if len(sys.argv)>1 else "ETHUSDT"
days=pd.date_range(*(sys.argv[2:4] if len(sys.argv)>3 else ("2024-01-01","2026-08-22")),freq="D")
OUT=pathlib.Path(sys.argv[4] if len(sys.argv)>4 else "data/binance_vision/aggfeat"); OUT.mkdir(parents=True,exist_ok=True)

def feats(df: pd.DataFrame) -> pd.DataFrame:
    df=df.rename(columns={"transact_time":"ts","is_buyer_maker":"bm"})
    df["ts"]=pd.to_datetime(df["ts"],unit="ms")
    # is_buyer_maker=True 면 **매수자가 메이커** ⇒ 공격자는 **매도자**. 부호는 공격자 방향.
    s=np.where(df["bm"].astype(str).str.lower().isin(["true","1"]),-1.0,1.0)
    q=df["quantity"].to_numpy(float); p=df["price"].to_numpy(float)
    notional=q*p
    big=notional>=np.quantile(notional,0.99)          # 상위 1% = 대형
    sml=notional<=np.quantile(notional,0.50)
    g=df.assign(sv=s*notional, av=notional, sgn=s,
                bigsv=np.where(big,s*notional,0.0), bigav=np.where(big,notional,0.0),
                smlsv=np.where(sml,s*notional,0.0), smlav=np.where(sml,notional,0.0),
                cnt=1.0, scnt=s).set_index("ts")
    r=g.resample("5min")
    out=pd.DataFrame({
        "tf_cvd":      r["sv"].sum()/r["av"].sum().replace(0,np.nan),        # 거래량 가중 불균형
        "tf_bigimb":   r["bigsv"].sum()/r["bigav"].sum().replace(0,np.nan),  # ⭐대형 체결만
        "tf_smlimb":   r["smlsv"].sum()/r["smlav"].sum().replace(0,np.nan),  # ⭐소형 체결만
        "tf_cntimb":   r["scnt"].sum()/r["cnt"].sum().replace(0,np.nan),     # ⭐건수 불균형
        "tf_ntrade":   r["cnt"].sum(),
        "tf_bigshare": r["bigav"].sum()/r["av"].sum().replace(0,np.nan),     # 대형 비중
        "tf_meansz":   r["av"].sum()/r["cnt"].sum().replace(0,np.nan),
    })
    # ⭐부호 자기상관·연속길이를 **벡터화**한다(.apply 는 파이썬 루프라 10배 느리다).
    #   s∈{−1,+1} 이므로 E[s]=m, Var=1−m² ⇒ ρ1 = (E[s_t·s_{t+1}] − m²)/(1−m²).
    b = g.index.floor("5min")
    prod = np.r_[np.nan, s[1:] * s[:-1]]
    chg = np.r_[np.nan, (s[1:] != s[:-1]).astype(float)]
    t1 = pd.DataFrame({"b": b, "prod": prod, "chg": chg, "s": s})
    gb = t1.groupby("b")
    m1 = gb["s"].mean(); e2 = gb["prod"].mean(); n1 = gb["s"].size()
    out["tf_signac"] = ((e2 - m1 * m1) / (1 - m1 * m1).replace(0, np.nan)).reindex(out.index)
    out["tf_runlen"] = (n1 / (gb["chg"].sum() + 1.0)).reindex(out.index)
    hi = g["price"].resample("5min").max(); lo = g["price"].resample("5min").min()
    op = g["price"].resample("5min").first(); cl = g["price"].resample("5min").last()
    out["tf_kyle"] = (np.abs(np.log(cl / op)) * 1e4) / np.maximum(r["av"].sum() / 1e6, 1e-9)
    out["tf_range"] = np.log(hi / lo) * 1e4
    tot = r["av"].sum()
    out["tf_vpin"] = r["sv"].sum().abs() / tot.replace(0, np.nan)
    return out

def one(d):
    ds=d.strftime("%Y-%m-%d"); p=OUT/f"{SYM}_{ds}.parquet"
    if p.exists(): return 0
    u=f"https://data.binance.vision/data/futures/um/daily/aggTrades/{SYM}/{SYM}-aggTrades-{ds}.zip"
    try: raw=urllib.request.urlopen(u,timeout=120).read()
    except urllib.error.HTTPError as e: return -1 if e.code==404 else -2
    except Exception: return -2
    try:
        z=zipfile.ZipFile(io.BytesIO(raw))
        df=pd.read_csv(z.open(z.namelist()[0]),
                       usecols=["price","quantity","transact_time","is_buyer_maker"])
        feats(df).to_parquet(p)
    except Exception as e:
        print("처리실패",ds,type(e).__name__,str(e)[:80],flush=True); return -2
    return 1
ok=miss=err=0
with cf.ThreadPoolExecutor(max_workers=10) as ex:
    for i,r in enumerate(ex.map(one,days)):
        ok+= r==1; miss+= r==-1; err+= r==-2
        if (i+1)%50==0: print(f"{i+1}/{len(days)} · 받음 {ok} · 없음 {miss} · 실패 {err}",flush=True)
print(f"완료 {SYM} {len(days)} · 받음 {ok} · 없음 {miss} · 실패 {err}")
