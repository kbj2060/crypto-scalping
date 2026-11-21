#![recursion_limit = "256"]
//! DuckDB 로 쓰는 수집기(사용자 결정 2026-11-21 질문지 19: DuckDB 형식 그대로 · rtcollect 와 따로 된 실행 파일).
pub mod deribit;
pub mod duckdump;
pub mod hl_positions;
pub mod hl_trades;
pub mod liq_tab;
pub mod market;
pub mod micro;
pub mod oi;
pub mod raster_archive;
pub mod seal;
