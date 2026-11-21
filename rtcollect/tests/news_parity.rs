//! 뉴스 RSS 일치: tools/gen_news_fixture.py 가 원본 run() 을 가짜 urlopen·가짜 시계(5초 × 90바퀴)로 돌린 정답지와
//!   items·polls 행 · 스키마 글자 · 요청 순서(URL)·머리글(UA·조건부 GET) 을 대조한다. 응답은 원본에 준 것을 같은 차례로.
use serde_json::Value;
use std::collections::HashMap;

#[test]
fn news_matches_python() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/news_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/news_golden.json")).unwrap();
    let d = std::env::temp_dir().join(format!("rtc_news_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let db = d.join("news.sqlite");
    {
        let c = rtcollect::news::connect(&db).unwrap();
        for r in inp["pre_items"].as_array().unwrap() {
            c.execute("INSERT INTO items VALUES(?,?,?,?,?,?,?,?)", rusqlite::params![r[0].as_str(), r[1].as_str(), r[2].as_str(), r[3].as_str(), r[4].as_str(), r[5].as_i64(), r[6].as_i64(), r[7].as_i64()]).unwrap();
        }
        for r in inp["pre_polls"].as_array().unwrap() {
            c.execute("INSERT INTO polls VALUES(?,?,?,?,?,?)", rusqlite::params![r[0].as_i64(), r[1].as_str(), r[2].as_i64(), r[3].as_i64(), r[4].as_i64(), r[5].as_i64()]).unwrap();
        }
    }
    let reqs = inp["reqs"].as_array().unwrap();
    let mut k = 0;
    let t = std::cell::Cell::new(inp["t0"].as_f64().unwrap());
    let (ticks, mut n) = (inp["ticks"].as_u64().unwrap(), 0);
    let mut log = vec![];
    rtcollect::news::run(&db, &rtcollect::news::FEEDS, &mut || t.get(), &mut |s| {
        n += 1;
        if n >= ticks {
            return false;
        }
        t.set(t.get() + s);
        true
    }, &mut |u: &str, h: &[(&str, String)]| {
        let r = &reqs[k];
        k += 1;
        assert_eq!(u, r["url"].as_str().unwrap(), "요청 {k} 주소");
        let mut got: Vec<(String, String)> = h.iter().map(|(a, b)| (a.to_lowercase(), b.clone())).collect();
        got.sort();
        let mut want: Vec<(String, String)> = r["headers"].as_array().unwrap().iter().map(|x| (x[0].as_str().unwrap().to_lowercase(), x[1].as_str().unwrap().to_string())).collect();
        want.sort();
        assert_eq!(got, want, "요청 {k} 머리글");
        assert_eq!(r["timeout"], 20);
        let sp = &r["resp"];
        match sp["kind"].as_str().unwrap() {
            "ok" => Ok((sp["status"].as_u64().unwrap() as u16, sp["body"].as_str().unwrap().as_bytes().to_vec(),
                        sp["headers"].as_object().unwrap().iter().map(|(a, b)| (a.to_lowercase(), b.as_str().unwrap().to_string())).collect::<HashMap<_, _>>())),
            "http" => Ok((sp["status"].as_u64().unwrap() as u16, vec![], HashMap::new())),
            _ => Err("URLError(refused)".into()),
        }
    }, 5.0, &mut |l| log.push(l)).unwrap();
    assert_eq!(k, reqs.len(), "요청 수");
    let c = rusqlite::Connection::open(&db).unwrap();
    let dump = |sql: &str| -> Value {
        let mut st = c.prepare(sql).unwrap();
        let n = st.column_count();
        let rows: Vec<Value> = st.query_map([], |r| Ok(Value::Array((0..n).map(|i| match r.get_ref(i).unwrap() {
            rusqlite::types::ValueRef::Null => Value::Null,
            rusqlite::types::ValueRef::Integer(x) => Value::from(x),
            rusqlite::types::ValueRef::Text(t) => Value::from(String::from_utf8_lossy(t).to_string()),
            v => panic!("{v:?}"),
        }).collect()))).unwrap().map(|x| x.unwrap()).collect();
        Value::Array(rows)
    };
    let items = dump("SELECT * FROM items ORDER BY rowid");
    for (i, (a, b)) in items.as_array().unwrap().iter().zip(gold["items"].as_array().unwrap()).enumerate() {
        assert_eq!(a, b, "items 행 {i}\n{log:#?}");
    }
    assert_eq!(items, gold["items"], "items");
    assert_eq!(dump("SELECT * FROM polls ORDER BY rowid"), gold["polls"], "polls");
    let schema: Vec<Value> = c.prepare("SELECT sql FROM sqlite_master ORDER BY name").unwrap().query_map([], |r| r.get::<_, Option<String>>(0)).unwrap()
        .map(|x| x.unwrap().map_or(Value::Null, Value::from)).collect();
    assert_eq!(Value::Array(schema), gold["schema"], "스키마");
    let _ = std::fs::remove_dir_all(&d);
    eprintln!("뉴스 RSS 일치: 요청 {k} · 기사 {} 행 · polls {} 행 · 바퀴 {ticks}", items.as_array().unwrap().len(), gold["polls"].as_array().unwrap().len());
}
