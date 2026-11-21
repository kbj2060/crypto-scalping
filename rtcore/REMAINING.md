# rtcore(R1) 남은 것 · 서버 교체 순서 (2026-11-21 · 2026-11-22 실제 파일로 다시 맞춤)

R1 = 거래소 웹소켓 → 메모리 → `/rt` 허브로 밀기(docs/rtcore_design_20261120.md §11). 코어는 **REST 를 쓰지 않는다**(2026-09-26 418 밴 · 결정 2026-11-21).

## 1. 옮김 (원본과 대조 시험이 있는 것)

| 무엇 | 원본 | 시험 |
|---|---|---|
| 체결·호가·마크·1분봉 수신 · 재연결 · 감시 | feed | fake_exchange · tls_supervise |
| 1초 수급(바이낸스·현물·OKX·청산 3거래소·OKX OI) | coin_flow supply_1s · server.py | supply1s_parity · supply1s_venues_parity |
| 풋프린트 봉·칸·dhl(**바이낸스 선물만** -- 원본 api_footprint 4000~4005 «11-02 바이낸스만» · venues [binance-perp]) | coin_flow footprint_add · api_footprint(3990~4056) · fp_dhl(3949~3988) | footprint_parity(ETH·XRP·BTC) |
| 재시작 복원(스냅숏 · 테이프 · OKX 1초·OI · 강제청산 · Bybit) | footprint_load·tape_restore·supply_1s_restore·liq_events_load | restore_parity · restore_venues_parity · sigterm_snapshot |
| OKX 풋프린트 봉 복원(15초 · 5분 × 8회) · OKX OI 5분봉 · OKX 청산 | okx_footprint_restore · okx_tape_footprint | okx_fp_restore(두 변형 × 8걸음) |
| 5분봉 청산 원 바이낸스·OKX(메모리 몫 = first_bar «뒤»)·Bybit 몫 | liq_5m_add · api_liquidation_5m_history · merge_bybit_liq | quote_parity · okx_fp_restore |
| 주문 경로 시세(bookTicker 꼴 · 1분봉 변동성 · 5분봉 종가 · 마지막 체결) | live_manual_peg_entry · orders.py | quote_parity |
| 30분 상황 근거(classify · 마크 링 · prev_dir) | _situation_inputs · situation.classify · deriv_from_ring | situation_parity |
| 미시 값(mid·spread·qi·vol60·liq60 · OKX OI 5분봉 · OKX 봉 합) | collect_micro_ref(1936~2008) · api_oi_5m(4363~4411) · _flow_read_ctx onet·okx30 재료(2332~2339 · 2374~2376) | micro_parity · okx_fp_restore(OKX 봉 합) |
| 허브 문 단속(경로·Origin·같은 호스트·연결·구독 상한·ping) | docs/rtcore_browser_route_20261120.md §5 | hub_gate |

## 2. 안 옮김 — 다른 꾸러미 몫(결정·근거는 SITUATION.md)

| 무엇 | 왜 코어 밖인가 | 몫 |
|---|---|---|
| veto(12h 추세) | 5분봉 720개 = REST klines | rtserve(«요청 때 REST» 묶음) → situation 과 합침 |
| 바이낸스 OI(/api/oi-5m 의 바이낸스 몫) | 웹소켓 없음 · REST 폴링 | rtcollect 가 쓰고 rtserve 가 hot SQLite 를 읽음 |
| 호가 obi·persist · agree·trigger·imb40·sweep | 래스터 수집기 파일 | rtcollect(쓰기) · rtserve(읽기 · agree 는 코어 qi 와 합침) |
| vol60_pct·act · flow_scales | 테이프 DB 시별 기준선 | rtserve(코어 vol60 을 받아 분위) |
| sr(청산맵 레벨) | 청산맵 | rtmodel · rtserve |
| HL 청산 | 코어가 안 받는 원천 | rtcollect · rtserve |
| 카드30 모델 · flow_read 융합 · 장부 · vol_mult | 모델 파일 · REST 5분봉 하루 | rtmodel · rtserve |
| okx30 · onet 계산 | 재료(Micro.okx_bars)만 코어 | rtserve |
| 청산 원 OKX 몫의 first_bar «앞» 봉 | 원본 merge_okx_liq(878~904)는 그 봉을 hot `okx_liquidations` 봉 합(okx_liq_hot_bars 856~875 · 기록 시작이 걸친 봉 «뒤»)으로 채운다 -- 2026-11-07 판. 코어 Liq5m.okx 는 메모리 몫(first_bar «뒤»)만 | rtserve(hot SQLite 를 읽는 쪽 · REST 아님) |

## 3. 안 옮김 — 코어 안에서 남은 것 · 알려진 차이
- vol60·OKX 봉 합·봉 델타 합: 마지막 비트가 다를 수 있다. 원본은 dict 를 넣은 차례로 더하고, 코어는 키 순서로 더한다(허용 오차 + 건수 상한 · 의도한 차이).
- 풋프린트 dhl 의 테이프 가지: 원본은 6분 뒤 테이프를 다시 읽고 기억한다. 코어는 링에서 얼린다(footprint.rs 머리 주석 ①~④).
- OKX OI 읽기가 행 도중 실패하면 원본은 모은 만큼 두고, 코어는 비운다(실서버에서 본 적 없음).
- Bybit 5분봉은 원본은 DB 를 요청마다 읽고, 코어는 실시간 사건이다 — DB 에 쓰이는 지연만큼 코어가 빠르다.
- 실서버 확인이 아직 안 된 것:
  - QUOTE.md 8개 항목(실서버 녹화와 대조).
  - cloudflared 가 Host 를 넘기는지(HUB.md).
  - Bybit·OKX 메시지 필드(녹화본 대조 전).
- 커밋 안 하는 정답지: `tests/fixtures/gen/` 아래(restore_venues · okx_fp · footprint xrp/btc). 시험 전에 생성기를 돌린다(각 시험 머리 주석).

## 4. 서버에서 새 실행 파일로 바꾸는 순서
1. **빌드·시험**: 서버에서 `cargo build --release -j 2`. 개발기에서 `cargo test` 전부 통과를 확인하고 정답지 생성기를 먼저 돌린다.
2. **배포 전 점검**: `bash scripts/ops/check_deploy_drift.sh`.
   - 새 실행 파일을 다른 이름으로 받아 두고 md5 를 확인한다.
   - `rtcore/target/release/rtcore` 를 원자적으로 갈아 끼운다(같은 파일에 덮어쓰기 금지 — 돌고 있는 파일).
3. **내리기**: rtcore pid 에 SIGTERM.
   - 코어가 스냅숏을 한 번 쓰고 종료코드 0 으로 끝난다(메우는 중이면 안 쓴다).
   - 감시자(`supervisor_rtcore.sh` → `_supervise.sh`)가 **15초 뒤** 새 실행 파일로 다시 띄운다.
4. **그동안 구독자에게 생기는 일**
   - 허브 연결이 끊긴다(rtui 밀기 · rtserve 허브 구독 · 화면 그림자 구독). 각자 다시 붙어 **새 스냅숏**을 받는다. seq 는 새 프로세스에서 1부터 다시 시작한다 — 받는 쪽은 스냅숏으로 갈아 끼우니 문제없다.
   - 약 15초 동안 밀기가 없다.
   - 붙은 직후:
     - 풋프린트·1초 수급은 스냅숏 + 테이프로 메운 값이다. 메우는 동안에는 `holes` 로 보인다.
     - 풋프린트는 바이낸스만이라 OKX first_bar 와 무관하다(스냅숏+테이프 복원 그대로). OKX first_bar 는 청산 원 OKX 몫·OI 5분봉 OKX 합산·시장 맥락 OKX 재료의 경계다 -- 첫 OKX 체결 때 진행 봉이 되고 15초 뒤 첫 복원 회차부터 앞으로 당겨진다(원본도 같다).
     - 청산 원 OKX 몫도 같은 경계를 따른다.
     - situation 의 prev_dir 는 0 에서 시작한다(원본 재시작과 같다).
     - 마크 링이 비어 basis_d 는 30분, 임계는 약 1시간 뒤에 생긴다.
     - Micro 는 첫 호가 뒤부터 나온다.
   - 주문 경로(rtorder 가 같은 프로세스로 rtcore 를 쓰면 그쪽 재시작 규칙 · 시세 나이 검사)는 rtorder 문서를 따른다.
5. **확인**
   - 로그 `logs/supervisor/rtcore_YYYYMMDD.log` 에서 «복원 … 스냅숏 · 테이프 …» 줄과 «OKX 풋프린트: 테이프에서 N봉 …» 줄을 본다.
   - `ss -ltn | grep 8788`.
   - 60초 수신→송신 통계 줄을 본다.
   - 그림자 대조기(`tools/shadow_footprint.py` · `shadow_supply.py`)로 원본과 대조한다.
6. **되돌리기**: 옛 실행 파일로 2~3번을 반복한다. 스냅숏 판 번호(v)가 같으면 서로 읽는다.
