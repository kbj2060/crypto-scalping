//! ob5m·bookdepth 정답지 입력(tests/fixtures/ob5m/ · ob5m_tape.json)으로 Rust 워커를 돌린다(시험·tools/gen_ob5m_fixture.py --check-readers 공용).
//!   cargo run --release --example replay_ob5m -- <fixtures 폴더> <out_dir>
fn main() {
    let a: Vec<String> = std::env::args().collect();
    rtcollect::ob5m::replay_fixture(std::path::Path::new(&a[1]), std::path::Path::new(&a[2]), None, None);
}
