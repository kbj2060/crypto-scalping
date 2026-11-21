//! 래스터 14일 보관 일치: tools/gen_raster_archive_fixture.py 가 원본 `_purge_old` 를 돌린 정답지와
//!   남은 원본 파일 · 보관 파일 이름 · parquet 열 이름·형·행(값) · gzip 푼 바이트 · orderbook.duckdb 뷰 집계를 대조한다.
//!   양방향: (가) 원본(pyarrow)이 쓴 parquet 을 Rust duckdb 가 읽어 같은 행 (나) Rust 가 쓴 보관 폴더를 파이썬이 되읽기 --
//!   RTDUCK_KEEP=<폴더> 로 돌리면 그 폴더를 남긴다 → `python3 tools/gen_raster_archive_fixture.py --check <폴더>`.
use duckdb::Connection;
use serde_json::{json, Value};
use std::path::Path;

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

/// parquet 한 파일 → 열 [이름, pyarrow 형 글자] · 행(NaN = null).
fn read(p: &Path) -> Value {
    let c = Connection::open_in_memory().unwrap();
    let q = format!("SELECT * FROM read_parquet('{}')", p.display());
    let mut st = c.prepare(&q).unwrap();
    let mut rows = st.query([]).unwrap();
    let mut out = vec![];
    let schema: Vec<Value> = c.prepare(&format!("DESCRIBE {q}")).unwrap().query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))).unwrap().map(|x| x.unwrap())
        .map(|(n, t)| json!([n, match t.as_str() { "BIGINT" => "int64", "INTEGER" => "int32", "FLOAT" => "float", o => panic!("{o}") }])).collect();
    while let Some(r) = rows.next().unwrap() {
        out.push(Value::Array((0..schema.len()).map(|i| match r.get_ref(i).unwrap() {
            duckdb::types::ValueRef::BigInt(v) => json!(v),
            duckdb::types::ValueRef::Int(v) => json!(v),
            duckdb::types::ValueRef::Float(v) if v.is_nan() => Value::Null,
            duckdb::types::ValueRef::Float(v) => json!(v as f64),
            v => panic!("{v:?}"),
        }).collect()));
    }
    json!({"schema": schema, "rows": out})
}

fn tree(root: &Path) -> Vec<String> {
    let mut v = vec![];
    let mut st = vec![root.to_path_buf()];
    while let Some(d) = st.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() { st.push(p) } else { v.push(p.strip_prefix(root).unwrap().to_string_lossy().to_string()) }
        }
    }
    v.sort();
    v
}

#[test]
fn raster_archive_matches_python() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/raster_archive_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/raster_archive_golden.json")).unwrap();
    // (가) 원본이 쓴 parquet 을 Rust duckdb 로
    let d = std::env::var("RTDUCK_KEEP").map(std::path::PathBuf::from).unwrap_or_else(|_| std::env::temp_dir().join(format!("rtduck_rast_{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&d);
    let py = d.join("py");
    for (rel, h) in gold["python_parquet"].as_object().unwrap() {
        let p = py.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, hex(h.as_str().unwrap())).unwrap();
        assert_eq!(read(&p), gold["archive"][rel], "원본 parquet 을 Rust 가 읽은 값 {rel}");
    }
    let _ = std::fs::remove_dir_all(&py);
    // 원본 함수와 같은 입력으로 Rust 보관
    let (root, archive) = (d.join("raster"), d.join("archive"));
    for (sect, base) in [("files", &root), ("pre", &archive)] {
        for (rel, h) in inp[sect].as_object().unwrap() {
            let p = base.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, hex(h.as_str().unwrap())).unwrap();
        }
    }
    let mut done = false;
    let mut log = vec![];
    for s in ["ethusdt", "btcusdt"] {
        log.extend(rtduck::raster_archive::purge_old(&root, &archive, s, inp["now_s"].as_f64().unwrap(), 14, true, &mut done));
    }
    assert_eq!(json!(tree(&root)), gold["raster_left"], "남은 원본 {log:#?}");
    let files: Vec<String> = tree(&archive).into_iter().filter(|f| !f.ends_with(".duckdb") && !f.ends_with(".wal")).collect();
    let want: Vec<String> = gold["archive"].as_object().unwrap().keys().cloned().collect();
    assert_eq!(files, want, "보관 파일 이름");
    let mut rows = 0;
    for f in &files {
        let g = &gold["archive"][f];
        if f.ends_with(".gz") {
            assert_eq!(rtcollect::gunzip_file(&archive.join(f)).unwrap(), hex(g["gz"].as_str().unwrap()), "gz {f}");
        } else {
            let got = read(&archive.join(f));
            rows += got["rows"].as_array().unwrap().len();
            assert_eq!(got, *g, "parquet {f}");
        }
    }
    let c = Connection::open(archive.join("orderbook.duckdb")).unwrap();
    let q = |sql: &str, n: usize| -> Value {
        let mut st = c.prepare(sql).unwrap();
        let v: Vec<Value> = st.query_map([], |r| Ok(Value::Array((0..n).map(|i| match r.get_ref(i).unwrap() {
            duckdb::types::ValueRef::Text(t) => json!(String::from_utf8_lossy(t)),
            duckdb::types::ValueRef::BigInt(x) => json!(x),
            duckdb::types::ValueRef::HugeInt(x) => json!(x as i64),
            v => panic!("{v:?}"),
        }).collect()))).unwrap().map(|x| x.unwrap()).collect();
        Value::Array(v)
    };
    let view = json!({"book": q("select symbol, count(*), count(bin_size), sum(bin) from book group by 1 order by 1", 4),
                      "book_mid": q("select symbol, count(*), count(mid) from book_mid group by 1 order by 1", 3)});
    assert_eq!(view, gold["view"], "뷰");
    drop(c);
    if std::env::var("RTDUCK_KEEP").is_err() {
        let _ = std::fs::remove_dir_all(&d);
    }
    eprintln!("래스터 보관 일치: 원본이 쓴 parquet {}개를 Rust 가 읽음 · 보관 파일 {} · parquet 행 {rows} · 남은 원본 {} · 뷰 {view}",
              gold["python_parquet"].as_object().unwrap().len(), files.len(), gold["raster_left"].as_array().unwrap().len());
}
