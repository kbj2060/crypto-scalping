//! rtorder -- 실주문 경로의 Rust 이전(2026-11-20 첫·둘째 조각). **어디에도 연결하지 않는다**(실행 파일·예제 없음 · 시험은 127.0.0.1 가짜 서버만).
//! plan = 무엇을 보낼지(파이썬 `scripts/live_manual_peg_entry_20260912.py` 순수 함수) · exec = 지정가 집행 상태 기계(거래소는 trait 뒤) ·
//! sign = 서명·쿼리·오류 문자열 · http = 바이낸스 REST 전송(`Exchange` 구현) · bracket = 손절·익절·레버리지·고아 정리 ·
//! watch = 손절·익절 감시·무장 저장본·주문 상태 칸·재시작 복구(orders.py 의 클로저 -- 바깥은 `WatchIo` 뒤).
//! 규칙 추적표: rtorder/RULES.md.

pub mod bracket;
pub mod exec;
pub mod http;
pub mod plan;
pub mod pyfmt;
pub mod sign;
pub mod watch;
