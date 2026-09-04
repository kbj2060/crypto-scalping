# A4. 크로스심볼 동시노출 캡 — 설계 착수 (2026-08-31)

## 배경

[프로젝트 아이디어 지도](eth_project_wide_idea_map_20260824.md) A4 항목("설계 미착수",
"지금 유일하게 바로 착수 가능한 신규 빌드")에 따라 사용자 요청으로 설계 착수. ETH(실거래
경로)+BTC/SOL(섀도우)가 `trading_bot.py` 한 프로세스 안에서 독립적으로 사이징돼, 세
심볼이 동시에 같은 방향으로 몰리는 리스크에 대한 캡이 사실상 없는 상태를 다룬다.

## ⚠️ 핵심 발견: 이미 90%가 만들어져 있었다

설계를 처음부터 시작하기 전에 조사한 결과, **2026-07-12/13에 정확히 이 문제를 다루는
연구 프로토타입이 이미 존재**했다 — 08-24 아이디어 지도 작성 시점에 참조되지 않은 것으로
보인다(메모리 인덱스에도 이 라인에 대한 항목이 없음).

- **`scripts/replay_portfolio_concurrent_3asset_native_20260712.py`** — ETH/SOL/BTC
  각자 독립 포지션 슬롯(`positions: dict[str, Position | None]`)을 bar-by-bar로 동시
  추적하는 진짜 concurrent 포트폴리오 리플레이. 공유 현금풀(`cash: float`), causal
  fresh-forward(`fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false` —
  CLAUDE.md Fresh-Forward 규칙의 요구 플래그와 정확히 일치).
- **캡 모드 3종이 이미 구현·실측됨**(`_replay_concurrent()`):
  - `reject`(v2): 예산 초과시 후보 완전 폐기 — 거부된 신호가 나중에 다른(때로 더 나쁜)
    신호로 대체되는 문제로 기각.
  - `scale`(v3): 남은 예산만큼 notional 축소 — 자산 처리 순서(`eth,sol,btc` 고정)가
    먼저 처리되는 쪽에 유리해 **SOL이 항상 손해보는 순서-의존 기아(starvation)** 발견.
  - `prealloc`(v4, 최종 채택) — 자산마다 고정 지분(`total_notional_cap * asset_shares`)을
    비경쟁으로 배정, 순서 무관. **이미 production `trading_bot_modules/portfolio_risk.py`로
    이식돼 있음**(단, 활성화 게이트가 기본 꺼짐 — 아래 "활성화 단계" 참고).
  - `same_direction_notional_cap` 파라미터도 이미 구현돼 있으나, prealloc 모드에서는
    "세 자산이 전부 최대치로 동시 동방향이어도 지분 합이 이미 total_notional_cap을
    못 넘게 구조적으로 막혀 있어 불필요"라는 결론으로 미사용 처리됨 — **이 판단을
    그대로 승계할지 재검토할지가 이번 설계의 첫 결정 지점**(아래 "결정 필요" 참고).
- **동시노출 실측**(무캡 v1 베이스라인, `docs/model_contracts/portfolio_concurrent_3asset_native_20260712.md`):
  VAL/OOS 전 구간에서 봉의 87~88%가 2개 이상 자산 동시오픈, 64~69%는 3개 전부 동시오픈.
  합산 realized MDD -29%~-38%(개별 자산 단독보다 훨씬 나쁨).
- **CURRENT_BASELINE**(07-12, 최종 채택 참조 설정): `--duration-gate off
  --eth-notional-multiplier 1.5`, **캡 미적용**(uncapped) — ETH 사이징만 1.5배로 올린
  상태에서 VAL PnL +22.9%/MDD -36.71%, OOS(확장) PnL +292.19%/MDD -31.23%.
- **미해결로 명시된 캐비어트(원 저자가 직접 기록)**:
  1. **prealloc 캡 + ETH 1.5배 승수를 합쳐서 테스트한 적이 없다** — 이번 설계의 가장
     직접적인 다음 실행 대상.
  2. **이 체인 전체(v1→v2→v3→v4→duration-gate→ETH승수 스윕)가 같은 2026-01~06 OOS
     윈도우를 반복 조회**했다는 자기지적("heavily peeked window hazard") — 아래
     "타이밍 문제" 참고.
  3. 07-13 fresh-window 확인(07-01~07-12, 12일)은 방향은 맞았으나(+43.63%, MDD -1.12%)
     트레이드 7건뿐이라 **"통계적 확인 아님"**으로 스스로 결론.

## 지금 시점 크로스심볼 스태킹 실측 갱신 (신규, 안전 — OOS 조회 아님)

08-24 카운팅("겹침 25페어 중 동방향 17페어")이 스크립트로 남지 않아, 이번에 재사용 가능한
형태로 새로 작성해 서버에서 최신 원장을 pull한 뒤 재실행했다. **이건 이미 일어난 라이브/
섀도우 결정을 세는 것이라 09-30 규율과 무관**(review-only, no signal-selection).
코드: [scripts/research_eth_cross_symbol_exposure_concurrency_check_20260831.py](../scripts/research_eth_cross_symbol_exposure_concurrency_check_20260831.py).

`data/live/trade_journal.jsonl`(2026-07-07~08-27, trading_bot.py 인프로세스 단일슬롯
ETH+BTC+SOL 섀도우 — BTC 독립 3슬롯 실험은 별도 파일이라 미포함):

| 자산쌍 | 겹침 이벤트 | 동방향 | 동방향 비율 |
|---|---:|---:|---:|
| btc-eth | 10 | 5 | 50% |
| btc-sol | 10 | 10 | **100%** |
| eth-sol | 11 | 4 | 36% |
| **합계** | **31** | **19** | **61%** |

08-24 수치(겹침 25/동방향 17)보다 늘었다 — 계속 쌓이는 중. **BTC-SOL은 겹칠 때마다 항상
같은 방향**(10/10)이라는 게 새 발견 — 알트코인 베타 상관관계로 보이며, ETH는 BTC/SOL
어느 쪽과도 절반 정도만 같은 방향(상대적으로 독립적). 원장 마지막 시점(08-27 20:05) 기준
**세 자산이 동시에 오픈 중**이었다: BTC LONG(08-21~), ETH SHORT(08-25~), SOL LONG(08-27~) —
이 시점만 보면 ETH가 BTC/SOL과 반대 방향이라 동방향 스태킹은 아니지만, BTC-SOL 두 개는
같은 방향으로 겹쳐 있다.

## 결정 필요 — 이번 설계에서 사용자가 확인/선택할 지점

1. **same_direction_notional_cap을 되살릴지**: prealloc의 "구조적으로 불필요" 논리는
   세 자산이 **각자 자기 지분 안에서** 동시 동방향이어도 합이 total_notional_cap을 못
   넘는다는 뜻이지, "동방향이면 각자 지분보다 더 작게 가라"는 걸 의미하지 않는다. 위 실측
   (BTC-SOL 100% 동방향 겹침)을 볼 때, "동방향일 땐 지분 자체를 줄인다"는 **추가** 캡을
   원하는지, 아니면 prealloc 지분 상한만으로 충분하다고 볼지 — 이건 리스크 선호의 문제라
   숫자로만 결정할 수 없다.
2. **ETH 1.5배 승수를 그대로 승계할지**: 이건 "버그 수정"이 아니라 "더 높은 리스크를
   의도적으로 택한다"는 결정이라고 원 문서가 명시적으로 구분해뒀다(2.0배 이상부터 VAL이
   음전환). prealloc 캡과 합쳐 재검증 전까지는 잠정 유지를 권장하나 확정은 사용자 판단.
3. **활성화 범위**: 아래 "활성화 단계" 참고 — 섀도우 추적용으로만 켤지, BTC/SOL 실거래
   집행까지 염두에 둘지는 완전히 다른 결정이며 이번 설계 요청 범위에 없다고 판단해
   **어느 쪽도 제안하지 않았다.**

## 활성화 단계 (참고용 — 이번에 어느 것도 건드리지 않음)

이미 배선은 존재하나 기본 비활성:
- **섀도우/리스크추적 단계**(블라스트 레디어스 작음): ETH는
  `FINAL_GOVERNOR_OMEGA4_6_1_ETH_PORTFOLIO_CAP_ENABLE`(`trading_bot_modules/runtime_config.py:347`),
  BTC/SOL은 `ctx["executor"] is not None`일 때만 적용 경로 진입(`trading_bot.py:12552-12559`).
  둘 다 기본 `False`.
- **BTC/SOL 실거래 집행 단계**(블라스트 레디어스 큼, 별개 결정):
  `FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_REAL_EXECUTION_ENABLE`(기본 `False`,
  `trading_bot_modules/runtime_config.py:447-449`) — 이게 꺼져 있으면 포트폴리오 캡
  로직 자체가 사실상 no-op(주문 경로가 없으니 캡할 것도 없음). **이 플래그를 켜는 건
  "동시노출 캡 설계"와 무관한 별도의, 훨씬 큰 결정**(BTC/SOL을 섀도우에서 실거래로
  승격하는 것) — 이번 요청 범위 밖으로 간주하고 어떤 제안도 하지 않는다.
- 현재 `portfolio_risk.py`의 기본 지분(`asset_shares`: ETH 0.5/BTC 0.3/SOL 0.2,
  `runtime_config.py:497-516`)은 07-12 리서치의 `--eth-share 0.5 --btc-share 0.3
  --sol-share 0.2` 기본값과 **정확히 일치** — 이미 이 리서치 라인의 결론이 그대로
  이식된 상태다.

## ⚠️ 타이밍 문제 — 사용자 확인 필요

이 라인(v1~v4~ETH승수스윕~fresh-window확인)은 전부 **2026-01-01~06-30 구간을 반복
조회**하며 설계 선택을 내렸다(원 저자 자인). [[eth_recency_walkforward_data_split_
literature_review_20260820]]의 08-20 최종 결정으로 그 구간(구 VAL+OOS-Q1+Q2)은 이제
**TRAIN에 편입**돼 "신선"하지 않고, 새 single-touch OOS는 **2026-07-01~09-30**이며
프로젝트 규율상 09-30 전 조기 부분체크가 금지돼 있다(그 자체가 조회로 집계됨 —
`eth_project_wide_idea_map_20260824.md` "메타 패턴" 참고).

이게 두 가지를 뜻한다:
- prealloc+ETH1.5배 조합을 지금 당장 "확인"하려면 쓸 수 있는 **깨끗한(안 본) 데이터가
  없다** — 07-01~09-30은 예약된 단일노출 OOS라 지금 만지면 안 되고, 그 이전은 이미
  소진됐다.
- **이번 세션에서 한 일(위 표)은 조회가 아니다** — 백테스트 PnL이 아니라 이미 실현된
  라이브/섀도우 사실을 센 것뿐이라 이 규율과 무관하다.

**질문**: 아래 세 경로 중 어떻게 진행할지 결정이 필요하다(사용자만 판단 가능한 리스크
규율 문제라 임의로 고르지 않았다):

1. **09-30까지 순수 설계/준비만**: config를 지금 확정(prealloc + ETH1.5배 조합, 위
   "결정 필요" 답변 반영)하고 실행 코드까지 다듬어두되, 실제 확인 실행은 09-30 OOS가
   열린 뒤로 미룬다. 프로젝트의 기존 규율과 가장 일치.
2. **이미 소진된 과거 구간(~2026-06-30 이전)에서 조합만 재확인**: 결정에 다시 쓰지
   않는다는 전제로("이미 본 데이터에서 조합 자체가 안 깨지는지"만 기계적으로 확인)
   prealloc+ETH1.5배를 과거 구간에 한 번 더 돌려본다 — 새 정보는 아니지만 최소한 코드
   경로 자체가 정상 작동하는지 스모크테스트는 된다.
3. **다른 접근**: 사용자가 다르게 보는 부분이 있다면.

## 다음 단계 (사용자 답변 후)

위 "결정 필요" 3개 + "타이밍 문제" 질문에 대한 답이 정해지면:
1. `_replay_concurrent(cap_mode="prealloc", total_notional_cap=..., eth_notional_multiplier=1.5, ...)`
   조합 실행(경로 1이면 09-30 이후, 경로 2면 지금 과거구간에 스모크테스트로).
2. same_direction_notional_cap 부활 여부가 "예"라면 prealloc 모드 자체에 그 축을 다시
   넣는 코드 변경(현재는 prealloc 분기에서 무시하도록 하드코딩돼 있음, 스크립트 127-138행
   docstring 참고) — 사용자 결정 이후 착수.
3. 활성화는 이번 설계 범위 밖으로 유지(섀도우 추적조차 지금 제안하지 않음).

## ⚠️ 2026-08-31 사용자 결정 (기록)

두 질문 모두 답변 받음:
- **타이밍**: "지금 바로 07-01~08월 구간으로 확인" 선택 — **예약된 단일노출 OOS(2026-07-01~
  09-30)를 09-30 전에 의도적으로 사용**하기로 함. 이건 실수/누락이 아니라 사용자가 명시적으로
  택한 예외다. **이 노출은 이번 A4 확인 목적으로만 유효** — 다른 축(예: BTC regime-shift
  재개후보, 다른 판정보류 항목)이 "이미 07-01~08월을 봤으니 같이 보자"고 승계하면 안 된다.
  09-30 이후에도 이 구간 자체는 여전히 "이번 세션에서 이미 조회됨"으로 취급할 것.
- **동방향 캡**: "prealloc 지분 상한만으로 충분(기존 결론 유지)" — same_direction_notional_cap
  도입 안 함, 코드 변경 불필요.

**착수한 작업**: 확인해보니 이게 "빠른 체크"가 아니라 07-13 확인 세션이 했던 전체 파이프라인
확장(원시데이터+피처+레짐오버레이+라벨+ETH/SOL/BTC 3개 고정 parent 번들 재스코어링)을
2026-08-31까지 다시 해야 하는 작업임을 확인함(ETH 피처는 08-19, BTC는 08-01, SOL은 07-21까지만
확장돼 있고 SOL/BTC 고정예측은 07-13에 멈춰 있었음). GPU 재스코어링이 포함된 무거운 작업이라
`handoff.sh launch server`로 서버에 위임, 백그라운드 에이전트로 진행 — 완료.

## ✅ 최종 결과 (2026-08-31, 파이프라인 완료)

전체 결과: [docs/model_contracts/portfolio_concurrent_3asset_prealloc_eth15x_fresh_confirmation_20260831.md](model_contracts/portfolio_concurrent_3asset_prealloc_eth15x_fresh_confirmation_20260831.md).
ETH/SOL/BTC 전부 2026-08-30 23:55:00까지 균일 확장(원시 klines는 08-31 11:30까지 받았으나
metrics 아카이브가 전일까지만 발행돼 그 이상은 안전하게 못 감 — 조용히 갭내지 않고 그 경계에서
정확히 멈춤). config A(prealloc cap=3.0, 지분 0.5/0.3/0.2)와 config B(무캡, 기존
CURRENT_BASELINE) 둘 다 duration_gate off + ETH 1.5배 승수로 실행.

| split | ΔPnL(A-B) | ΔMDD realized | MDD(A)/MDD(B) |
|---|---:|---:|---:|
| validation(오염) | -16.67pp | +8.93pp | 75.7% |
| oos_extended(오염, 01-01~08-30) | -72.79pp | +11.24pp | 80.7% |
| **fresh_window(07-01~08-30, 이번에 새로 확보)** | **-0.25pp** | **+8.97pp** | **60.4%** |

**핵심 발견**: 오염된(이미 여러 번 조회된) 과거 구간에서는 캡이 PnL의 70~75%를 희생하고 MDD는
19~25%만 줄이는 나쁜 트레이드오프였는데, **이번에 처음 확보한 진짜 fresh 구간에서는 PnL 손실이
사실상 0(18.68% vs 18.93%, -0.25pp)이면서 MDD는 상대적으로 ~40% 줄어든다**(realized -13.71%
vs -22.69%, MTM -18.25% vs -29.13%). 24트레이드/2개월이라 통계적으로 확정된 건 아니지만,
"prealloc+ETH1.5배 조합"이라는 미해결 캐비어트에 대한 첫 fresh-forward 답이 방향적으로
긍정적이다. 동시노출도 fresh 구간 실제활성봉(n=17,567) 기준 2자산+ 99.17%/3자산 93.40%로
원 v1베이스라인(87~88%/64~69%, 01~06월)보다 오히려 더 심해짐 — 캡의 필요성이 약해진 게 아니라
강해졌다.

**부수 성과**: 파이프라인 진행 중 07-13 절차를 문자 그대로 재현했다면 2026-08-23에 발견·수정된
metrics 아카이브 미래참조 버그(원시 병합 경로 자체는 패치 안 됐던 문제)가 이번 신규 확장 구간에
그대로 재발했을 것을 에이전트가 사전에 포착해 08-23 수정 스크립트를 재적용 — Fresh-Forward
인과성 규칙을 실제로 지켰다. 이 과정에서 발견된 별개의 pandas 자정 타임스탬프 CSV 직렬화 버그
(마지막 행이 정확히 자정이면 시간정보 없이 날짜만 기록됨)는 컨디네이터(나) 직접진단+수정 1건과
에이전트의 컷오프 5분 이동 수정이 중복 적용됐으나 서로 호환 확인됨.

**활성화(재확인)**: 이 결과는 여전히 순수 리서치 리플레이다 — `trading_bot.py`/`portfolio_risk.py`
코드 변경 없음, 라이브/섀도우 계정 무관. 결과가 고무적이라 활성화를 다음으로 논의할 만하지만,
이번 세션에서 그 결정은 내리지 않았고 별도 확인이 필요하다. 07-01~08-30 구간은 이 A4 확인
목적으로 소진됐다 — 다른 축에서 "이미 본 신선한 데이터"로 재사용 금지.

## ✅ 2026-09-01 후속 — cap/asset_shares 재스윕(fresh window)

전문: [docs/model_contracts/portfolio_concurrent_3asset_prealloc_cap_shares_sweep_fresh_20260901.md](model_contracts/portfolio_concurrent_3asset_prealloc_cap_shares_sweep_fresh_20260901.md).
07-12 `gate_off_cap_sweep` 문서가 이미 스윕해뒀던 그리드(cap 7점, shares 4조합, 신규
자유파라미터 없음)를 fresh 구간(07-01~08-30, 위 확인에서 이미 소진된 바로 그 구간 재사용,
추가 신규노출 없음)에 재적용.

**핵심 결과**: 8/31에 채택한 cap=3.0/shares=50-30-20은 fresh 데이터 기준 최적점이 아니었다.
- cap 스윕: fresh 구간에서는 cap이 작을수록(1.5, 그리드 최솟값) PnL·MDD 둘 다 최선(PnL
  23.38%>18.68%, MDD -6.94%>-13.71%) — oos_extended(오염구간)의 "cap↑=PnL↑MDD↑" 정상
  트레이드오프와 정반대 방향(변동성 드래그로 해석, 확정 아님·표본24건).
- shares 스윕: 07-12 문서가 "최고"로 꼽았던 60/25/15가 fresh에서는 **최악**으로 재현 실패,
  대신 **균등가중 33/33/33이 최고**(PnL 27.54%, MDD -9.19%, 승률70.8%).
- 조합(cap=1.5+shares=33/33/33): 두 축 개선이 상쇄되지 않고 함께 살아남음 — **fresh PnL
  27.65%(8/31 원안 대비 +8.97pp), MDD -4.65%(8/31 원안 -13.71% 대비 상대개선 약66%), 승률
  87.5%(24건중21승)**.

여전히 24트레이드/2개월 표본, 활성화 미결정. 다음 논의 대상은 "이 조합을 다음 단계(섀도우
추적 활성화 또는 더 큰 캘린더 게이트인 09-30 이후 재확인)로 가져갈지"다.

## ⚠️ 2026-09-04 정정 — "활성화 게이트 기본 꺼짐 → 지금은 no-op"은 틀렸다

이 문서의 "활성화 단계" 절은 `runtime_config.py`의 **코드 기본값**(플래그 False, cap 3.0)만 읽고
"지금은 no-op"이라고 적었다. 서버 `.env`(gitignore, 로컬 체크아웃에 없음)를 직접 확인한 결과:
`FINAL_GOVERNOR_OMEGA4_6_1_ETH_PORTFOLIO_CAP_ENABLE=True`(07-14 체크포인트부터),
`FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP=3.0` + 지분 0.5/0.3/0.2, 그리고
`FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_REAL_EXECUTION_ENABLE=True`(executor 존재 → BTC/SOL 경로도 캡 적용,
단 `BINANCE_EXECUTION_ENABLED=False`라 dry-run). 저널이 이를 증명한다 — ETH notional 정확히 1.5(08-21,
08-22; 컴포넌트 cap 1.8·레버리지 cap 5.0으로는 안 나오는 값), SOL 정확히 0.6(=3.0×0.2, 7회).
즉 **prealloc 캡은 늦어도 2026-08-20부터 cap 3.0 / 50-30-20으로 라이브(페이퍼)에서 작동 중**이었고,
08-31·09-01 결과가 말하는 것은 "배선"이 아니라 "재구성(1.5 + 균등지분)"이다.
후속 전문: [experiments/eth_cross_symbol_cap_a4_activation_20260904.md](experiments/eth_cross_symbol_cap_a4_activation_20260904.md)
(서버 실태, 텔레메트리 코드, 실현 저널 재사이징 평가, 적용 절차).
