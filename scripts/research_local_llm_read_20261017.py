#!/usr/bin/env python3
"""**서버 로컬 LLM 이 해석 입력을 맞게 읽나** — Haiku 시험 세트 그대로(흐름 24 · 청산맵/풋프린트 16) 시점마다 1 질의, 같은 채점기 (2026-10-17).

왜: API 키가 없어 Haiku 운영 불가 → 사용자 «우리가 만든 입력 텍스트로 … 로컬 모델 조사, server 에서 돌아갈 작은 모델» → «A(GPU)로 진행».
후보: qwen3:8b · qwen3.5:4b(없으면 qwen3:4b) · EXAONE-4.0-1.2B Q4_K_M — 서버 시험 전용 Ollama(127.0.0.1:11436, bwrap CUDA 가리기),
  뉴스 판정 워커 SIGSTOP + jevk5 내림 동안만. Ollama format=json · think=false · temperature 0.2 · num_ctx 16384.
기준선(같은 채점): Haiku 흐름 C 부호 100%·엄격 92.7%(주장 218) · 청산맵/풋프린트 277/277.
  python scripts/research_local_llm_read_20261017.py prompts      # 질의 파일 만들기(tmp/local_llm_20261017/prompts)
  python scripts/research_local_llm_read_20261017.py eval          # 서버 결과(tmp/local_llm_20261017/out) 채점
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "tmp/local_llm_20261017"
SRC = {"window": [ROOT / f"tmp/haiku_window_20261017_inputs/C_{i}.txt" for i in (0, 1)],
       "liqfp": [ROOT / f"tmp/haiku_liqfp_20261017_inputs/batch_{i}.txt" for i in (0, 1)]}
OBJ = {"window": 'Output ONLY one JSON object: {"id": "<the snapshot id>", "interpretation": "<Korean, at most 6 sentences>", "claims": [...]}',
       "liqfp": 'Output ONLY one JSON object: {"id": "<the snapshot id>", "interpretation": "<Korean, at most 6 sentences>", "facts": [...]}'}

RUNNER = r'''
import json, sys, time, urllib.request, os
models, root = sys.argv[1].split(","), sys.argv[2]
def call(body):
    req = urllib.request.Request("http://127.0.0.1:11436/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)
for m in models:
    od = os.path.join(root, "out", m.replace("/", "_").replace(":", "_")); os.makedirs(od, exist_ok=True)
    for fn in sorted(os.listdir(os.path.join(root, "prompts"))):
        dst = os.path.join(od, fn.replace(".txt", ".json"))
        if os.path.exists(dst):
            continue
        prompt = open(os.path.join(root, "prompts", fn)).read()
        body = {"model": m, "messages": [{"role": "user", "content": prompt}], "stream": False, "format": "json", "think": False,
                "options": {"num_ctx": 16384, "temperature": 0.2, "num_predict": 1500}, "keep_alive": "10m"}
        t = time.time()
        try:
            res = call(body)
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if "think" in msg:
                body.pop("think"); res = call(body)
            else:
                res = {"error": msg}
        res["_wall_s"] = round(time.time() - t, 2)
        json.dump(res, open(dst, "w"), ensure_ascii=False)
        print(m, fn, res.get("_wall_s"), res.get("eval_count"), "err" if "error" in res else "", flush=True)
print("RUN_DONE", flush=True)
'''


def prompts() -> None:
    pd = OUT / "prompts"; pd.mkdir(parents=True, exist_ok=True)
    n = 0
    for task, files in SRC.items():
        for f in files:
            head, body = f.read_text().split("Snapshots:\n", 1)
            head = re.sub(r"Output ONLY one JSON array:.*", OBJ[task], head)
            head = head.replace("at several independent snapshots (different days, shuffled)", "at one snapshot").replace("For EACH snapshot", "For the snapshot")
            for line in body.splitlines():
                if line.strip():
                    sid = json.loads(line)["id"]
                    (pd / f"{task}_{sid}.txt").write_text(head + "Snapshot:\n" + line)
                    n += 1
    (OUT / "runner.py").write_text(RUNNER)
    print(f"{n} 질의 → {pd}")


def parse(res: dict) -> dict | None:
    try:
        txt = res["message"]["content"]
        j = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
        return j if isinstance(j, dict) else None
    except Exception:  # noqa: BLE001
        return None


def evaluate() -> None:
    import research_llm_haiku_window_read_20261017 as W
    import research_llm_haiku_liqfp_read_20261017 as F
    tw = json.loads((W.OUT / "truth.json").read_text())
    tf = json.loads((F.OUT / "truth.json").read_text())
    print(f"{'모델':40s} {'JSON':>6s} | 흐름: 주장 부호정확 엄격정확 | 청산맵/풋프린트: 사실 정확 | 한글비율 초/질의 토큰/초")
    for md in sorted((OUT / "out").iterdir()):
        ok_json, n_all, wclaims, wsign_bad, wstrict_ok, wstrict_bad, ffacts, fok, hang, secs, tps = 0, 0, 0, 0, 0, 0, 0, 0, [], [], []
        for f in sorted(md.glob("*.json")):
            res = json.loads(f.read_text()); n_all += 1
            secs.append(res.get("_wall_s", np.nan))
            if res.get("eval_count") and res.get("eval_duration"):
                tps.append(res["eval_count"] / (res["eval_duration"] / 1e9))
            j = parse(res)
            if not j:
                continue
            ok_json += 1
            txt = str(j.get("interpretation", ""))
            hang.append(len(re.findall(r"[가-힣]", txt)) / max(1, len(re.sub(r"\s", "", txt))))
            task, sid = f.stem.split("_", 1)
            if task == "window":
                for c in j.get("claims") or []:
                    if not isinstance(c, dict):
                        continue
                    r = W.check(c, tw[sid]); wclaims += 1
                    wsign_bad += W.sign_wrong(c, tw[sid])
                    wstrict_ok += r is True; wstrict_bad += r is False
            else:
                for c in j.get("facts") or []:
                    if not isinstance(c, dict):
                        continue
                    r = F.check(c, tf[sid])
                    if r is not None:                      # 형식 밖(정해진 항목 아님)은 따로 -- 오답으로 안 셈
                        ffacts += 1; fok += r is True
        sign = 1 - wsign_bad / max(1, wstrict_ok + wstrict_bad)
        strict = wstrict_ok / max(1, wstrict_ok + wstrict_bad)
        print(f"{md.name:40s} {ok_json:>3d}/{n_all:<3d}| {wclaims:4d} {sign:7.1%} {strict:7.1%} | {ffacts:4d} {fok / max(1, ffacts):7.1%} | "
              f"{np.mean(hang) if hang else 0:5.0%} {np.nanmedian(secs):6.1f} {np.median(tps) if tps else 0:6.1f}")
    print("기준선 Haiku: 흐름 주장 218 부호 100% 엄격 92.7% | 사실 277 정확 100%")


if __name__ == "__main__":
    {"prompts": prompts, "eval": evaluate}[sys.argv[1]]()
