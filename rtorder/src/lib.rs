//! rtorder -- 실주문 경로의 Rust 이전 첫 조각(2026-11-20). **어디에도 연결하지 않는다**: 거래소·키·네트워크를 모른다.
//! plan = 무엇을 보낼지(파이썬 `scripts/live_manual_peg_entry_20260912.py` 순수 함수) · exec = 지정가 집행 상태 기계(거래소는 trait 뒤).
//! 규칙 추적표: rtorder/RULES.md.

pub mod exec;
pub mod plan;
pub mod pyfmt;
