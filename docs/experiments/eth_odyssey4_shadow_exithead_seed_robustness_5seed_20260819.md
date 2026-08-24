# ETH Odyssey4 섀도우 고유축(h48qual liveATR-relabel exit_head) 시드 강건성 — N=5 정식검증 (2026-08-19)

**⚠️ 2026-08-20 정정**: 아래 "범위의 한계" 절("이 축만 격리된 결과, 라이브 dual 본체의 취약성을
해소하지 않는다")은 후속 원인규명(`eth_odyssey4_shadow_full_reseed_causal_isolation_20260820.md`)
에서 **기각됐다**. zig075를 이 실험 이후 진짜로 재시드해도(신규학습 없이 기존 N=5 번들 재조합)
6/6 부호일치가 유지됐고, 단일요인격리로 추적한 결과 "veto+guard 메커니즘 자체가 실질적 안정화
효과를 가진다"는 게 확정됐다(h48qual/zig075 어느 쪽이 변하든 veto+guard 엔진이면 0/6 플립,
없으면 5~6/6 플립). 문서 끝에 후속 절 추가.

## 배경 / 목표

같은 날 먼저 완료한 ETH/BTC/SOL 라이브 dual 자체의 시드 강건성 검증(`docs/experiments/
eth_live_promotion_seed_robustness_3seed_20260819.md`(N=5로 확장, 최종 CONFIRMED — 6창 중 4창
부호플립, VAL 자체도 플립), `btc_live_promotion_seed_robustness_5seed_20260819.md`,
`sol_live_promotion_seed_robustness_5seed_20260819.md`)에 이어, 라이브 h48qual+zig075 dual 위에
추가로 얹혀 서버에서 상시 가동 중인 **오디세이4 섀도우**(`live_eth_odyssey4_zig075_entry_veto_
shadow_cleanroom_20260816.py`, systemd `eth-odyssey4-shadow.service`)의 **고유 신규축**을 같은
방법론으로 검증한다.

오디세이4 섀도우는 라이브 dual 위에 두 계층을 얹는다(둘 다 결정론적 규칙, 자유변수 0개):
1. **zig075 SHORT 진입거부** — `SustainedUptrendDetector`(rolling 1주 `dual_momentum>0` 비율 >
   p90=0.8025793650793651, 2025-Q1+Q2만으로 캘리브레이션) 활성 시 zig075 SHORT 신호를 스킵.
2. **h48qual 레짐인지형 exit 가드** — 같은 감지기가 활성 상태에서 h48qual이 포지션을 보유 중이면,
   exit 판정을 liveATR-relabel exit_head(2026-08-13) 대신 h48qual **원본(재라벨 이전)** exit_head로
   전환.

이 두 계층 자체(감지기 공식·임계값·라우팅 규칙)는 신규 자유변수가 0개인 순수 규칙이라 시드축이
없다. 오디세이4 섀도우가 라이브 dual과 다른 **유일한 신규 "학습" 컴포넌트**는 h48qual liveATR-
relabel exit_head(`research_eth_omega461_exit_head_liveatr_relabel_20260813.py`)뿐이다 — 이
스크립트는 그 축 하나만 N=5 시드로 바꿔가며 평가한다. h48qual "guard" 번들(원본, 재라벨 이전)과
zig075는 오늘 세션 전체에서 지킨 원칙(ETH/BTC/SOL risk sidecar frozen과 동일 논리로 인코더
시드효과를 다른 축과 분리)에 따라 라이브 원본(260620) 하나로 고정한다 — 그 축은 이미 별도로 N=5
검증 완료(위 세 문서)됐으므로 이 실험에서 다시 섞지 않는다.

## 오디세이4 섀도우 구조 확인 (재조사, 코드/문서 원본 직접 대조)

`live_eth_odyssey4_zig075_entry_veto_shadow_cleanroom_20260816.py`(현재 서버 배포판)와
`docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md` /
`odyssey4_eth_full_stack_architecture_20260814.md`를 직접 읽어 확인:

- `H48QUAL_NEW_BUNDLE_PATH`(entry+기본 exit) = liveATR-relabel판, `H48QUAL_ORIGINAL_BUNDLE_PATH`
  (감지기 활성 시에만 exit_probability로 조회) = 라이브 h48qual 원본 — 사전 설명대로 확인됨.
- zig075는 감지기가 SHORT entry만 건드리고 direction/quality/TP-SL/사이징은 전부 무변경으로 확인됨.
- 로컬에 남아있던(2026-08-17 12:45 KST 기준, 이번 세션에서 새로 가져오지 않은 stale snapshot)
  섀도우 자체 `state.json`을 참고로 확인한 결과 `detector_bars_seen=843`(<2016, 즉 감지기가 아직
  풀윈도우도 못 채운 상태)·`h48qual_guard_active_bars=0`·`zig075_short_veto_bars=0` — 라이브
  섀도우는 배포 이후 지금까지 이 두 계층이 실제로 발동한 적이 **한 번도 없다**(계약문서의 "forward
  에서 진짜 지속 상승장을 겪은 적 없음" 정직한 한계와 일치). 즉 이 두 계층의 유일한 실증 근거는
  아래의 backtest뿐이다.

## 기존 백테스트 엔진 재사용 (재구현 없음)

`grep -rli "SustainedUptrendDetector\|odyssey4" scripts/*.py`로 탐색한 결과,
**`research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py::greedy_replay_
entry_veto`**가 정확히 이 두 계층(entry veto + exit guard)을 함께 구현하는 유일한 기존 엔진임을
확인했다 — Odyssey3의 h48qual 가드(`research_eth_omega461_regime_aware_exit_head_uptrend_guard_
20260814.py::greedy_replay_regime_aware_exit_guard`)를 renamed-copy한 뒤 zig075 SHORT veto 블록
하나만 추가한 구조로, 라이브 섀도우가 실제로 구현하는 로직과 정확히 일치한다. 그 외 후보로 검토한
`eval_eth_odyssey4_{posfix_canonicaldata,baseline_original,pinned102}_freshforward_20260818.py`
계열은 (docstring 직접 확인 결과) 순수 h48qual+zig075 dual 평가 엔진으로 veto/guard 로직이 아예
없음을 확인 — 오늘 ETH/BTC/SOL dual N=5 검증에 쓰인 것과 같은 계열이지 오디세이4 전용이 아니다.

이 세션에서 신규 추가한 것은 `scripts/eval_eth_odyssey4_shadow_exithead_seed_robustness_
20260819.py` 하나뿐이며, 그 안에서도 신규 로직은 `prepare_regime_aware_components_seeded`
함수 하나뿐이다 — `guard.prepare_regime_aware_components`의 renamed-copy로, h48qual "liveatr"
(default) 번들 경로만 파라미터화했다(guard 번들·zig075·우선순위·threshold·TP/SL·사이징은 전부
원본 함수와 동일 소스에서 무변경 재사용). `greedy_replay_entry_veto` 자체는 import해서 그대로
호출한다.

**G0 fidelity 검증**: seed=260813_original(=`portfolio.NEW_H48QUAL_BUNDLE`, 즉 파라미터화 이전
원본과 동일 경로)로 실행한 결과가 `docs/model_contracts/odyssey4_eth_entry_veto_baseline_
contract_20260814.md`의 G0 표(6창 no_gate+with_gate pnl/mdd/trades 전부)와 **완전히 일치**
(`g0_fidelity_seed260813_vs_odyssey4_contract.pass=True`, 6/6창 no_gate·with_gate 모두 매치) —
파라미터화 래퍼가 원본 함수 조합과 바이트 수준으로 동등함을 실증했다.

## 방법론

1. **시드 5개**: 260813(원본, 오늘 세션 시작 시점에 이미 학습됨) + 신규 4개(497101020/912177061/
   29403054/458139929, `eth_live_promotion_seed_robustness_odyssey4_exithead_seed_variant_
   20260819.py`로 이번 세션 중 학습 — encoder/direction/quality는 라이브 h48qual 번들에서
   frozen 복사, exit_head만 시드별로 재학습). 5개 번들 전부 `--max-candidates 1500`(원본
   아티팩트와 동일 규모), 다른 하이퍼파라미터는 스크립트 기본값 그대로.
2. **평가**: `greedy_replay_entry_veto`(fresh-forward, causal 단일 순방향 패스) + 6개 사전등록
   창(`eth_omega461_multiwindow_confirmation_gate_20260814.py::WINDOW_DEFS` — 2025q1/q2/q3=context,
   val=2025-10-01~12-31, oos_q1=2026-01-01~03-31, oos_q2=2026-04-01~06-30). `with_gate`
   metric(`mfe_width._duration_gated`)이 공식 승격 판정 지표.
3. **감지기**: `guard.build_detector()`로 재계산한 p90 threshold가 잠긴 값(0.8025793650793651)과
   1e-12 이내로 일치함을 assert(통과) — 감지기 자체는 시드와 무관한 상수.

**학습 소요/운영 메모**: 시드당 데이터셋 빌드 ~8분, h48qual exit_head 재학습 32~52분(서버 부하에
따라 변동), zig075 재학습(이 섀도우엔 불필요하지만 학습 스크립트 구조상 항상 같이 돎) ~22분.
912177061/29403054/458139929 세 개를 병렬로 띄웠을 때 29403054가 h48qual 재학습 도중 외부 요인
(Python traceback 없이 프로세스만 소멸 — 3개 병렬 실행에 따른 메모리 경합으로 추정, 서버는 실거래
봇+다른 섀도우와 공유 중)으로 죽어 **단독 재실행**으로 성공 완료했다(재시도 시 서버 가용메모리
23GiB 확인 후 진행, 이후 정상 완료).

**Fresh-Forward 준수**: `fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`,
`saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false` (평가 스크립트
report.json 필드로 명시, greedy_replay_entry_veto 자체가 causal 단일 순방향 패스).

## 결과 — with_gate PnL(%) / MDD(%) / trades, 6창 × 5시드

| window (tier) | 260813(원본) | 497101020 | 912177061 | 29403054 | 458139929 | 부호일치 |
|---|---:|---:|---:|---:|---:|---|
| 2025q1 (context) | +44.98 / -20.62 / 20 | +44.30 / -20.62 / 19 | +44.21 / -20.62 / 19 | +44.07 / -20.62 / 19 | +44.30 / -20.62 / 19 | ✅ YES |
| 2025q2 (context) | +5.62 / -23.59 / 19 | +5.27 / -23.73 / 19 | +1.45 / -26.14 / 20 | +4.75 / -23.73 / 19 | +4.75 / -24.11 / 20 | ✅ YES |
| 2025q3 (context) | +20.17 / -19.72 / 17 | +21.46 / -19.72 / 17 | +21.46 / -19.72 / 17 | +20.85 / -19.72 / 17 | +19.35 / -19.72 / 18 | ✅ YES |
| **val** | **+77.31 / -21.76 / 26** | +52.28 / -23.02 / 29 | +78.69 / -21.84 / 27 | +77.12 / -21.84 / 26 | +59.01 / -19.59 / 28 | ✅ **YES** |
| **oos_q1** | **+67.25 / -15.48 / 19** | +66.40 / -15.48 / 19 | +67.09 / -15.48 / 19 | +65.42 / -15.48 / 18 | **+17.30 / -29.43 / 17** | ✅ YES (부호는 일치, 크기 이탈 존재 — 아래 해석 참고) |
| **oos_q2** | -12.69 / -20.76 / 10 | -4.53 / -13.72 / 11 | -4.53 / -13.72 / 11 | -4.49 / -13.68 / 11 | -5.11 / -14.24 / 11 | ✅ YES |

**6창 전부 부호일치(all_same_sign=True)** — 5개 시드 예외 없이.

## entry veto / exit guard 발동 횟수 (bar 단위, 5시드 사실상 동일)

| window | veto_bars(zig075 SHORT 스킵) | h48qual guard_active_bars | guard_decision_differs_bars | 감지기 활성 비율 |
|---|---:|---:|---:|---:|
| 2025q1 | 0 | 0 | 0 | 7.6% |
| 2025q2 | 10 | 1340 | 0~1 (시드별) | 11.6% |
| 2025q3 | 19 | 8286 | 183~338 (시드별) | 43.0% |
| val | 12 | 887 | 0~3 (시드별) | 7.6% |
| oos_q1 | 0 | 1029(3시드) / 1172(2시드) | 0 | 5.4% |
| oos_q2 | 0 | 0 | 0 | 8.2% |

감지기 발동 자체(`veto_bars`, 감지기 활성 비율)는 5개 시드 전부 **완전히 동일**하다 — 감지기는
순수 규칙(자유변수 0개)이라 시드와 무관하기 때문이다. `guard_active_bars`(가드가 실제로 열린
h48qual 포지션과 겹친 bar 수)만 시드별로 소폭 다른데, 이는 감지기가 아니라 각 시드의 exit_head가
그 시점에 포지션을 보유하고 있었는지 여부(hold 타이밍)가 시드마다 다르기 때문이다. **판정 3창
(val/oos_q1/oos_q2) 중 공식 OOS 확인창 2개(oos_q1/oos_q2)는 모두 veto_bars=0** — 이 세션의
backtest 범위 안에서 zig075 entry veto는 OOS 구간에서 단 한 번도 실제로 개입하지 않았고, 오직
h48qual exit guard만 개입했다(2025q3 context 창이 유일하게 두 계층 다 강하게 개입하는 구간).

## 해석

**핵심 결론: 오디세이4 섀도우의 고유 신규축(h48qual liveATR-relabel exit_head) 자체는 6/6창 전부
시드에 걸쳐 부호가 일치한다** — 이는 같은 날 검증한 라이브 dual 자체(encoder/direction/quality
시드축, 4/6창 부호플립, VAL 자체도 플립)와 뚜렷이 대조된다. 감지기·라우팅 규칙(자유변수 0개)이
시드에 완전히 불변인 것은 물론, exit_head 자체를 5개 시드로 바꿔도 어느 창에서도 부호가 뒤집히지
않았다.

**단, 크기(magnitude) 차원에서는 완전히 균일하지 않다** — 특히 **oos_q1에서 시드458139929가
뚜렷한 이상치**다: 다른 4개 시드는 65.42~67.25%(스프레드 1.83pp)로 매우 조밀하게 모이는데
458139929만 17.30%로 약 50pp 낮고 MDD도 -15.48%→-29.43%로 거의 2배 악화된다(no_gate PnL도
91~94%대에서 35.46%로 급락, trades는 17~19로 큰 차이 없음 — 거래 수는 비슷한데 개별 거래 결과가
많이 다르다는 뜻). 같은 `guard_active_bars=1172`를 공유하는 29403054는 이 이상치를 보이지 않으므로
(29403054의 oos_q1 with_gate=65.42%, 정상 클러스터), 이 이탈은 감지기/가드 메커니즘 공유분이
아니라 458139929의 exit_head가 학습한 고유 행동(oos_q1 구간에서의 개별 거래 청산 타이밍)에서
비롯된 것으로 보인다. val도 스프레드가 상당하다(52.28~78.69%, ~26pp) — 부호는 안 갈리지만 원
승격 근거였던 260813의 77.31%가 시드 분포 상단에 가깝다는 점은 ETH/BTC dual N=5 검증에서 반복
관찰된 패턴("원 시드가 우연히 좋았을 가능성")과 유사한 결이다.

**범위의 한계를 분명히 해야 한다**: 이 실험은 "오디세이4 섀도우가 라이브 dual 위에 추가하는
신규축만" 격리해 검증한 것이지, 오디세이4 섀도우 전체의 승격 가능성을 판정한 것이 아니다. 이
섀도우가 얹혀 있는 라이브 dual 자체(encoder/direction/quality 시드축)는 이미 별도 검증에서
"seed-robust하지 않음, CONFIRMED"로 판정됐다(4/6창 부호플립) — 그 문제는 이 실험이 손대는 축과
무관하게 그대로 남아있다. 즉 "오디세이4 고유축은 강건하다"와 "오디세이4 섀도우 전체가 강건한
기반 위에 있다"는 서로 다른 명제이며, 후자는 이 실험만으로 성립하지 않는다.

## 한계 / 교란변수

- **h48qual guard 번들·zig075 둘 다 frozen(260620 단일)** — 의도된 격리(신규축만 순수하게 보기
  위함)지만, 만약 이 두 축의 시드효과가 h48qual liveATR-relabel exit_head 시드효과와 상호작용한다면
  이 실험은 그 상호작용을 볼 수 없다. 각 축은 이미 별도로(라이브 dual N=5 검증) 강건성이 낮다고
  확인됐으므로, 세 축을 모두 동시에 바꾸는 결합 시드 스윕은 아직 미실시.
- **감지기 자체가 forward에서 실제로 관찰된 적 없음** — 라이브 섀도우 배포(2026-08-14) 이후
  지금(2026-08-19)까지 감지기가 풀윈도우조차 못 채웠고(843/2016 bar, stale snapshot 기준)
  실발동 0회. 이 실험의 모든 veto/guard 발동은 backtest(2025 과거 데이터+2026 OOS)에서 나온
  것이며, 라이브 forward 관찰로 재확인된 바 없다 — Odyssey4 계약문서 자체가 이미 명시한 정직한
  한계와 동일.
- **N=5**: CLAUDE.md Seed-Diversity Ensemble Promotion Gate 문턱(N≥5 진짜 랜덤시드)은 충족하지만,
  oos_q1의 단일 이상치(458139929)가 진짜 이상치인지 시드분산의 정상범위인지는 추가 시드 없이는
  통계적으로 확정할 수 없다(BTC N=5 검증의 750703416 이상치와 동일한 구조의 한계).
- **거래빈도가 낮은 창**: oos_q1/oos_q2 모두 trades=10~19 수준 — 한두 건의 거래 결과가 전체 부호를
  좌우할 수 있는 표본 크기다.

## 결론 / 제안

**오디세이4 섀도우의 고유 신규축(entry veto + exit guard 메커니즘과 그 유일한 학습 컴포넌트인
h48qual liveATR-relabel exit_head)은 N=5 시드에 걸쳐 6/6창 전부 부호가 일치한다** — 이 축만
놓고 보면 CLAUDE.md Seed-Diversity Gate의 "OOS 부호일치" 요건을 충족한다. 다만:

1. 이 결과는 이 축에 한정된다 — 그 아래 깔린 라이브 dual(encoder/direction/quality) 자체의
   seed-robustness 문제(이미 CONFIRMED, 4/6창 부호플립)는 이 실험으로 해소되지 않으며, 오디세이4
   섀도우 전체의 실질적 강건성은 두 축을 결합해서 봐야 완전한 그림이 된다.
2. oos_q1의 458139929 이상치(부호는 유지하나 크기가 다른 4개 시드 대비 ~50pp 낮음)는 추가 조사
   또는 N≥8로의 확장 없이는 "정상 시드분산"인지 "구조적 취약점"인지 판정할 수 없다.
3. 감지기 자체가 라이브에서 한 번도 실발동한 적이 없으므로, 이 backtest 결과는 여전히 "관찰 대기"
   지위다 — 오디세이4 계약문서의 기존 정직한 한계가 이 시드 강건성 축에도 그대로 적용된다.

## 산출물

- 스크립트(신규): `scripts/eval_eth_odyssey4_shadow_exithead_seed_robustness_20260819.py`
- 스크립트(오늘 세션 중 기존 생성, 재사용): `scripts/eth_live_promotion_seed_robustness_odyssey4_
  {exithead_seed_variant,liveatr_snapshot}_20260819.py`
- 번들(서버, `/home/llewyn/crypto-scalping/` 기준): `tmp/causal_regen_20260516/
  eth_omega461_exit_head_liveatr_relabel_20260813_full1500{,_seedvariant_
  {497101020,912177061,29403054,458139929}}/h48qual/true_3head_tabm_bundle.pt`
- 평가 결과(로컬 pull 완료): `tmp/causal_regen_20260516/eth_odyssey4_shadow_exithead_seed_
  robustness_20260819/report.json`, 시드×창별 trade ledger CSV 30개(서버에만 있음, pull 안 함)

**fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_
used=false, future_rows_used_for_entry=false** (평가 스크립트 자체 report.json 필드로 명시).
