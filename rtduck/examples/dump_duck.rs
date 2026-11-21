//! cargo run --release --example dump_duck -- <file.duckdb>  (파이썬이 쓴 파일을 Rust DuckDB 로 읽는 대조)
fn main() {
    let a: Vec<String> = std::env::args().collect();
    println!("{}", rtduck::duckdump::dump(std::path::Path::new(&a[1])).unwrap());
}
