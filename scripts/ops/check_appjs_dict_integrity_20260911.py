"""app.js 최상위 객체 리터럴의 **키 목록**을 뽑아 이전 버전과 대조한다.
2026-09-11 사고 유형 전용: 문자열 편집이 사전 경계를 넘어 항목을 조용히 지워도
중괄호 균형·HTTP200·자산 서빙 검사는 전부 통과한다. 키 목록만이 그걸 잡는다."""
import re, subprocess, sys

def dicts(src):
    out={}
    lines=src.split('\n')
    for i,l in enumerate(lines):
        m=re.match(r'^(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*\{\s*$', l)
        if not m: continue
        name=m.group(1); depth=1; keys=[]
        for j in range(i+1,len(lines)):
            t=lines[j]
            if depth==1:
                k=re.match(r'^\s{2}(?:"([^"]+)"|([A-Za-z_$][\w$]*))\s*:', t)
                if k: keys.append(k.group(1) or k.group(2))
            depth+=t.count('{')-t.count('}')
            if depth<=0: break
        out[name]=keys
    return out

cur=dicts(open('dashboard/live/app.js').read())
base=dicts(subprocess.run(['git','show','HEAD:dashboard/live/app.js'],
                          capture_output=True,text=True).stdout)
bad=False
allk=sorted(set(base)|set(cur))
print(f"{'사전':34s} {'이전':>5} {'현재':>5}  변화")
for n in allk:
    a,b=base.get(n),cur.get(n)
    if a is None: print(f"{n:34s} {'-':>5} {len(b):>5}  신규"); continue
    if b is None: print(f"{n:34s} {len(a):>5} {'-':>5}  🔴사전 자체 소실"); bad=True; continue
    lost=sorted(set(a)-set(b)); add=sorted(set(b)-set(a))
    mark="✅" if not lost and not add else ("🔴 소실 "+str(lost) if lost else "＋"+str(add))
    if lost and not (set(lost) <= {"liq_direction"}): bad=True
    print(f"{n:34s} {len(a):>5} {len(b):>5}  {mark}")
print()
print("🔴 의도치 않은 소실 있음" if bad else "✅ 의도한 변경(liq_direction) 외 소실 없음")
sys.exit(1 if bad else 0)
