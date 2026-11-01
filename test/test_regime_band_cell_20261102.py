"""⑤ 예측 줄 «횡보·추세» 칸 검증: 서버 모듈 셀프테스트 + app.js 의 칸 블록을 Node 로 상태별 실행.   python test/test_regime_band_cell_20261102.py (node 필요)"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = r"""
const fs=require("fs"); const src=fs.readFileSync(process.argv[1],"utf8");
const a=src.indexOf("  const rb = latestRegimeBand"), b=src.indexOf("  // 2026-10-17 펜 메모 «타이틀 제거");
if(a<0||b<0) throw new Error("칸 블록을 못 찾음");
const block=src.slice(a,b), now=Date.now();
const run=(latest,asset)=>new Function("latestRegimeBand","activeSnapshotAsset","cell","kst","DIRSHORT_ASSETS",block+"\nreturn rbCell;")(latest,asset,(hz,c,m,s,t)=>[hz,c,m,s,t.includes("방향은 말하지 않는다")],()=>"00:00",new Set(["eth","btc","sol","xrp"]));
const mk=(p,conf,age)=>({ok:true,coins:{eth:{p_trend:p,confidence:conf,decided_at_ms:now-age*60e3,support:2322.3,resistance:2420.14}}});
console.log(JSON.stringify({none:run(null,"eth"),empty:run({ok:true,coins:{btc:{}}},"eth"),other:run({ok:true,coins:{}},"hype"),range:run(mk(0.12,0.76,20),"eth"),trend:run(mk(0.91,0.82,20),"eth"),vague:run(mk(0.47,0.06,20),"eth"),old:run(mk(0.9,0.8,120),"eth")}));
"""


def main():
    subprocess.run([sys.executable, "-I", str(ROOT / "dashboard" / "regime_band.py"), "--selftest"], check=True)
    out = json.loads(subprocess.run(["node", "-e", JS, str(ROOT / "dashboard" / "live" / "app.js")], check=True, capture_output=True, text=True).stdout)
    assert out["none"][1:4] == ["flat", "-", "불러오는 중"], out["none"]
    assert out["empty"][3] == "기록 대기" and out["other"][3] == "ETH·BTC·SOL·XRP 전용"
    assert out["range"][1:4] == ["flat", "횡보", "88% · 확신 높음"], out["range"]
    assert out["trend"][1:4] == ["trd", "추세", "91% · 확신 높음"], out["trend"]                 # 추세만 주황(trd), 상승/하락 색(up/dn)은 절대 쓰지 않는다
    assert out["vague"][1:4] == ["flat", "애매", "추세 47%"], out["vague"]
    assert out["old"][1:4] == ["flat", "-", "지연"], out["old"]
    assert all(v[4] for v in out.values()), "툴팁에 «방향은 말하지 않는다» 안내가 있어야 한다"
    assert not any(v[1] in ("up", "dn") for v in out.values())
    print("OK")


if __name__ == "__main__":
    main()
