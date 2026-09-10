"""app.js 최상위 객체 리터럴의 **키 목록**을 뽑아 이전 버전과 대조한다.
2026-09-11 사고 유형 전용: 문자열 편집이 사전 경계를 넘어 항목을 조용히 지워도
중괄호 균형·HTTP200·자산 서빙 검사는 전부 통과한다. 키 목록만이 그걸 잡는다."""
import re, subprocess, sys

# 이번 변경에서 **의도적으로** 제거한 키. 인자로 넘긴다:
#   python3 check_appjs_dict_integrity_20260911.py liq_direction liq_size
# 비워두면 어떤 소실도 실패로 본다.
EXPECTED_REMOVED = set(sys.argv[1:])

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
    if lost and not (set(lost) <= EXPECTED_REMOVED): bad=True
    print(f"{n:34s} {len(a):>5} {len(b):>5}  {mark}")
print()
# ⭐호출되는데 정의가 없는 최상위 함수 잡기 (2026-09-11 실장애: liqRiskIndicatorItem 이
# 호출만 남고 정의가 빠져 대시보드가 죽었다. 여러 단계 편집 중 한 단계가 파일에 안 써졌는데
# 그대로 진행한 게 원인 -- 중괄호 균형·사전 검사·HTTP200 전부 통과했다).
src=open('dashboard/live/app.js').read()
defined=set(re.findall(r'^(?:async )?function ([A-Za-z_$][\w$]*)', src, re.M))
defined |= set(re.findall(r'^(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(', src, re.M))
defined |= set(re.findall(r'(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=', src))
called=set(re.findall(r'\b([A-Za-z_$][\w$]*)\s*\(', src))
# 내장·DOM·라이브러리는 제외: 정의부가 우리 파일에 있는 이름만 본다
suspects=sorted(n for n in called if n.endswith(('IndicatorItem','SubText','Tone','Html','Item'))
                and n not in defined)
if suspects:
    print(f"🔴 호출되는데 정의 없음: {suspects}"); bad=True
else:
    print("✅ 호출-정의 대조: 이상 없음")

# ⭐가장 싼 진짜 검사: 파일 전체를 실제로 파싱한다 (2026-09-11 실장애 2건째 --
# 사전에 쉼표가 둘(`},,`) 들어가 app.js 전체가 SyntaxError 였다. 정규식 검사기는
# 키 목록도 중괄호 균형도 전부 통과시켰다). esprima 는 ES2020 미지원이라 문법 검사에
# 영향 없는 최신 표기만 등가 치환한다. ponytail: 브라우저 실행(playwright)이 이 머신에
# 시스템 라이브러리 부족으로 안 뜬다 -- 뜨면 콘솔 오류 확인으로 올리는 게 맞다.
import esprima
_s = re.sub(r"\?\.(?=[(\[])", "", src)
_s = re.sub(r"\?\.", ".", _s).replace("??=", "||=").replace("??", "||")
try:
    esprima.parseScript(_s)
    print("✅ 문법 파싱: 통과")
except Exception as e:
    print(f"🔴 문법 오류 -- 이 파일은 브라우저에서 아예 로드되지 않는다: {e}"); bad=True

print("🔴 의도치 않은 소실 있음" if bad else
      f"✅ 의도한 제거({', '.join(sorted(EXPECTED_REMOVED)) or '없음'}) 외 소실 없음")
sys.exit(1 if bad else 0)
