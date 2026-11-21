// 원본 «전체 흐름» 정답지(tools/gen_flow_fixture.py) → 그림자 기록 JSONL(rtorder::shadow 형식). flow_fixture·shadow_replay 시험이 같이 쓴다.
// 정답지는 가짜 시계(T0 + t)라 시각이 글자까지 같다 -- 시험은 그대로(exact) 또는 «실제 기록처럼» 시계 부분을 흔든 뒤(normalize) 재생한다.
use serde_json::{json, Value};

pub fn iso(t: f64) -> String {
    rtorder::watch::iso_utc(t)
}

/// 정답지 한 벌 → 작업(chain)별 기록 줄들.
pub fn flow_to_log(doc: &Value) -> Vec<Value> {
    let t0 = doc["t0"].as_f64().unwrap();
    let mut out = vec![];
    for (si, sc) in doc["scenarios"].as_array().unwrap().iter().enumerate() {
        let books: Vec<(f64, Value)> = sc["books"].as_array().unwrap().iter().map(|b| (b[0].as_f64().unwrap(), b[1].clone())).collect();
        let (mut slot, mut file) = (json!({"phase": "submitting"}), sc["file0"].clone());
        let mut pos = json!({"LONG": 0.0, "SHORT": 0.0});
        for (i, st) in sc["steps"].as_array().unwrap().iter().enumerate() {
            let id = format!("s{si}.{i}");
            let slot0 = if st["slot0"].is_object() { st["slot0"].clone() } else { slot.clone() }; // 작업 직전 칸(청산 접수는 접수 문이 칸을 바꾼 뒤)
            out.push(json!({"rec": "start", "chain": id, "op": st["op"], "t": t0 + st["t"].as_f64().unwrap(), "plan": st["plan"], "limit": st["limit"],
                            "rearm": st["rearm"], "slot": slot0, "file": file, "name": sc["name"]}));
            for (q, r) in st["requests"].as_array().unwrap().iter().zip(st["responses"].as_array().unwrap()) {
                out.push(json!({"rec": "wire", "chain": id, "method": q["method"], "target": q["target"], "status": r["status"], "body": r["body"]}));
            }
            let mut xp = st["exit_plans"].as_array().unwrap().iter();
            for c in st["calls"].as_array().unwrap() {
                let (t, what) = (t0 + c[0].as_f64().unwrap(), c[1].as_str().unwrap());
                let args: Vec<Value> = c.as_array().unwrap()[2..].to_vec();
                let value = match what {
                    "account" => {
                        let p: Vec<Value> = ["LONG", "SHORT"].iter().filter(|s| pos[**s].as_f64().unwrap() != 0.0).map(|s| json!({"symbol": "ETHUSDC", "side": s, "qty": pos[*s]})).collect();
                        json!({"ok": true, "generated_at": iso(t), "positions": p})
                    }
                    "bars" => json!([]),
                    "sec" => st["sec"].clone(),
                    "exit_plan" => json!({"plan": xp.next().unwrap()}),
                    "book" => books.iter().filter(|(bt, _)| t - t0 >= *bt).last().unwrap().1[args[0].as_str().unwrap()].clone(),
                    "filters" => json!({"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 5.0}),
                    x => panic!("모르는 입력 {x}"),
                };
                out.push(json!({"rec": "input", "chain": id, "t": t, "what": what, "args": args, "value": value}));
            }
            out.push(json!({"rec": "end", "chain": id, "t": t0 + st["t_end"].as_f64().unwrap(), "slot": st["entry"], "file": st["file"], "limits": st["limits"],
                            "logs": st["logs"], "raised": st["raised"]}));
            slot = st["entry"].clone();
            file = st["file"].clone();
            pos = st["pos"].clone();
        }
    }
    out
}

pub fn to_jsonl(rows: &[Value]) -> String {
    rows.iter().map(|r| serde_json::to_string(r).unwrap()).collect::<Vec<_>>().join("\n")
}
