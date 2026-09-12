import json, sys, urllib.request
d = json.load(urllib.request.urlopen("http://127.0.0.1:8787/api/ops-status", timeout=20))
rows = []
def walk(o):
    if isinstance(o, dict):
        if "component" in o or ("status" in o and ("detail" in o or "message" in o)):
            rows.append(o); return
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(d)
bad = [r for r in rows if str(r.get("status", "")).upper() not in ("OK", "PASS", "GREEN", "")]
print(f"총 점검 {len(rows)}개 · 정상 아님 {len(bad)}개\n")
print("=== ⚠️ 정상이 아닌 항목 ===")
for r in bad:
    comp = r.get("component", "?")
    print(f"  [{r.get('status')}] {comp}: {str(r.get('detail') or r.get('message'))[:90]}")
    ex = {k: v for k, v in r.items() if k not in ("component", "status", "detail", "message")}
    if ex:
        print(f"       {json.dumps(ex, ensure_ascii=False, default=str)[:220]}")
print("\n=== deribit / gex 관련 (상태 무관) ===")
for r in rows:
    if "gex" in json.dumps(r, default=str).lower() or "deribit" in json.dumps(r, default=str).lower():
        print(f"  [{r.get('status')}] {r.get('component')}  {json.dumps(r, ensure_ascii=False, default=str)[:300]}")
