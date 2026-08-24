# 오디세이4 섀도우 6/6 부호일치의 원인 규명 — 단일요인격리 (2026-08-20)

## 배경 / 질문

전날(`eth_odyssey4_shadow_exithead_seed_robustness_5seed_20260819.md`) 오디세이4 섀도우의
고유축(h48qual liveATR-relabel exit_head)만 N=5 시드로 검증했더니 6/6창 전부 부호일치가
나왔다. 이때 zig075와 exit-guard 감지기는 라이브 원본으로 고정했었다.

사용자 질문: "exit_head가 진입/청산에 영향을 안 미쳐서 통과한 거 아니냐?" → report.json의
`source_component_counts` 직접확인 결과, exit_head는 h48qual 자기 트레이드의 64~80%를
청산시킬 만큼 실제로 영향을 주지만, **zig075가 공식 OOS창 트레이드의 82~83%를 차지하는데
그 zig075가 이 실험에서 애초에 시드 변수가 아니었다**는 게 안정성의 진짜 이유로 보였다(1차
가설). "그럼 zig075도 진짜로 바꾸면 플립이 재현될 것"이라는 예측이 나왔고, 사용자 지시로
검증에 착수했다.

## 1차 가설 검증 — 예상밖 결과

ETH 라이브 dual N=5 검증(`eth_live_promotion_seed_robustness_3seed_20260819.md`)에서 이미
학습된 zig075 인코더 5개(신규학습 없음, 그대로 재사용)를 오디세이4 h48qual exit_head N=5와
인덱스 페어링해서 오디세이4의 실제 veto+guard 엔진으로 재평가
(`scripts/eval_eth_odyssey4_shadow_full_reseed_20260820.py`):

| window | pair1(원본) | pair2 | pair3 | pair4 | pair5 | 부호 |
|---|---:|---:|---:|---:|---:|---|
| 2025q1 | +44.98 | +44.30 | +44.21 | +44.07 | +44.30 | ✅ |
| 2025q2 | +5.62 | +5.27 | +1.45 | +4.75 | +4.75 | ✅ |
| 2025q3 | +20.17 | +21.46 | +21.46 | +20.85 | +19.35 | ✅ |
| val | +77.31 | +52.28 | +78.69 | +77.12 | +59.01 | ✅ |
| oos_q1 | +67.25 | +58.35 | +59.00 | +57.42 | +11.62 | ✅ (진폭 요동) |
| oos_q2 | -12.69 | -4.53 | -4.53 | -4.49 | -5.11 | ✅ |

**6창 전부 여전히 부호일치.** zig075를 진짜로 섞어도 안 흔들렸다 — 1차 가설(zig075 고정이
원인) 기각.

## 2차 가설 — ETH dual의 원래 4/6 플립은 h48qual 인코더 때문?

ETH 라이브 dual N=5의 원래 4/6플립이 h48qual 인코더 변동 때문인지 zig075 인코더 변동 때문인지
단일요인격리(one-factor-at-a-time)로 분리(`scripts/eval_eth_dual_single_factor_isolation_
20260820.py`). 신규학습 없이 기존 두 N=5 번들셋 재사용, **veto/guard 없는 순수 dual 엔진**
(`eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818.py` — 원래 4/6플립을 만든
것과 동일 엔진).

**Test C(h48qual만 변경, zig075=260620 고정)**:

| window | 260620 | 94046540 | 524707103 | 312069414 | 44751167 | 부호 |
|---|---:|---:|---:|---:|---:|---|
| 2025q1 | +28.54 | +66.88 | +88.86 | +69.27 | +118.92 | ✅ |
| 2025q2 | +39.99 | -45.59 | -8.96 | -31.46 | -29.46 | ❌ 플립 |
| 2025q3 | -9.73 | -25.22 | +37.10 | -10.36 | -11.76 | ❌ 플립 |
| val | +54.88 | +35.87 | +25.66 | +56.47 | +31.72 | ✅ |
| oos_q1 | +28.17 | +18.83 | +29.48 | +26.46 | +15.96 | ✅ |
| oos_q2 | +9.85 | +55.94 | +35.38 | +27.13 | +50.28 | ✅ |

→ 6창 중 **2창만** 플립(2025q2, 2025q3).

**Test B-plain(zig075만 변경, h48qual=260620 고정, 순수 dual 엔진)**:

| window | 260620 | 94046540 | 524707103 | 312069414 | 44751167 | 부호 |
|---|---:|---:|---:|---:|---:|---|
| 2025q1 | +28.54 | -19.39 | +50.97 | +5.16 | +97.39 | ❌ 플립 |
| 2025q2 | +39.99 | -71.96 | -44.49 | +19.52 | +12.62 | ❌ 플립 |
| 2025q3 | -9.73 | -8.39 | -22.94 | -2.67 | -4.59 | ✅ |
| val | +54.88 | -18.09 | -28.11 | +57.57 | -1.79 | ❌ 플립 |
| oos_q1 | +28.17 | +33.79 | -30.36 | +88.78 | +17.41 | ❌ 플립 |
| oos_q2 | +9.85 | -40.19 | -21.32 | +36.76 | +58.75 | ❌ 플립 |

→ 6창 중 **5창** 플립 — h48qual 단독(2/6)보다 훨씬 불안정하고, **원래 둘 다 바꾼 Test D(4/6)
보다도 더 불안정**(부분 상쇄/상호작용 효과 시사) → 2차 가설도 방향이 반대(h48qual이 아니라
zig075가 주범).

## 교란변수 발견 및 최종 판별

오디세이4축(1차 가설 검증, veto+guard 엔진)은 h48qual 기준선이 liveATR-relabel판이었는데, Test
C/B-plain은 순수라이브 h48qual을 썼다 — **엔진(veto/guard 유무)과 h48qual 기준선 두 가지가
동시에 다름**, 어느 쪽이 원인인지 불명확.

**Test B-liveatr**로 해소: zig075 5시드 변경 + h48qual을 liveATR-relabel 번들로 고정(오디세이4와
동일 기준선) + **veto/guard 없는 순수 dual 엔진**:

| window | 260620 | 94046540 | 524707103 | 312069414 | 44751167 | 부호 |
|---|---:|---:|---:|---:|---:|---|
| 2025q1 | +44.50 | -17.83 | +41.22 | +3.22 | +72.14 | ❌ 플립 |
| 2025q2 | +31.49 | -63.67 | -48.33 | +15.94 | +7.41 | ❌ 플립 |
| 2025q3 | -18.87 | -5.55 | -30.67 | +29.33 | -5.47 | ❌ 플립 |
| val | +77.31 | -9.30 | +15.17 | +62.24 | +23.71 | ❌ 플립 |
| oos_q1 | +67.25 | +35.37 | -28.75 | +89.88 | +18.10 | ❌ 플립 |
| oos_q2 | -12.69 | -38.92 | -20.78 | +76.70 | +54.49 | ❌ 플립 |

→ **6창 전부 플립** — 순수라이브 h48qual 고정판(5/6)보다도 더 심하다.

## 최종 결론

| 변경축 | 고정축 | 엔진 | 플립 |
|---|---|---|---:|
| h48qual+zig075 둘다(Test D, 2026-08-19) | — | plain | 4/6 |
| h48qual만(Test C) | zig075 | plain | 2/6 |
| zig075만(Test B-plain) | h48qual(순수라이브) | plain | 5/6 |
| zig075만(Test B-liveatr) | h48qual(liveATR-relabel) | plain | **6/6** |
| h48qual exit_head만(전날 N=5) | zig075 | **veto+guard** | **0/6** |
| zig075 전체(1차 가설검증) | h48qual(liveATR-relabel) | **veto+guard** | **0/6** |

**h48qual 기준선이 무엇이든(순수라이브/liveATR-relabel) 상관없이, veto+guard 엔진 유무가
안정성을 가른다.** "격리축이라 우연히 안정적으로 보였다"가 아니라 **veto+guard 메커니즘
자체가 실질적 안정화 효과를 가진다**는 것이 6종 비교로 확정됐다.

**유력 메커니즘 가설** (트레이드 단위로 직접 확인하지는 않음): `SustainedUptrendDetector`가
"상승추세 중 zig075 SHORT 진입"을 결정론적(자유파라미터 0개)으로 걸러낸다. 이건 모델의
확신도/캘리브레이션이 시드마다 가장 크게 갈리는 유형의 트레이드(추세 역행)인데, 시드와
무관하게 이 변동성 원천 자체를 통째로 잘라내는 것으로 추정된다. h48qual의 레짐인지형 exit
가드는 실제 판정을 바꾸는 bar가 5시드 중 0~3개뿐(전날 문서 기록)이라 이 안정화 효과의 주된
기여자는 아닐 가능성이 높다 — 즉 안정화의 실질적 원천은 **entry veto** 쪽일 가능성이 exit
guard보다 크다(직접 검증 안 됨, 다음 절 참고).

## 한계 / 후속 필요

1. **트레이드 단위 메커니즘 가설은 여전히 미검증**: veto가 실제로 발동한 트레이드만 따로 뽑아
   시드간 분산을 비-veto 트레이드와 대조해야 확정할 수 있다 — 아래 08-20b는 이것과는 다른
   질문(veto/guard 중 어느 엔진 성분이 기여자인가)을 풀었을 뿐, "왜 상승추세중 SHORT를
   걸러내면 안정화되는가"의 트레이드 단위 검증 자체는 아직 하지 않았다.
2. **N=5 페어링은 인덱스순 임의조합 1세트뿐** — 정식 N≥5 게이트라기보다는 방향성 확인.
3. **다른 자산(BTC/SOL) 이식 가능성** — `btc_odyssey4_shadow_uptrend_short_entry_veto_20260820.md`
   에서 BTC entry-veto를 정식 이식+G0검증+N=5 인과재현까지 테스트했으나 부호일치 개선 0/3창으로
   기각됨(SOL은 저비용진단만으로 종결). 아래 후속에서 이 기각 원인도 함께 규명됐다.
4. 이 실험 전체가 "라이브 dual 자체가 seed-robust하지 않다"는 기존 CONFIRMED 판정
   (`eth_live_promotion_seed_robustness_3seed_20260819.md`)을 뒤집지는 않는다 — 오디세이4
   섀도우가 그 위에서 실질적으로 안정화 효과를 낸다는 것이 새로 확인된 사실이다.
5. ~~entry-veto와 exit-guard 중 무엇이 기여자인지 미분리~~ → **아래 "후속(08-20b)"에서 해소**
   (entry-veto 단독으로 충분, exit-guard는 매그니튜드만 부가 조정).

## 후속(08-20b) — entry-veto 단독 vs exit-guard 결합, 그리고 BTC 이식 실패 원인 규명

**배경**: 사용자 질문 "BTC엔 detector/veto가 왜 이식이 안 되냐"에 답하는 과정에서, 위 표의
"veto+guard 0/6" 결과가 전부 entry-veto와 exit-guard가 **동시에** 켜진 엔진
(`greedy_replay_entry_veto`)에서만 나왔다는 게 드러났다. BTC 이식은 exit-guard용 h48qual
대체 exit_head가 없어 entry-veto만 테스트했으므로, "ETH의 절반만 이식해서 실패한 것 아니냐"는
반론이 성립했다. 사용자가 이 분리검증을 요청("그렇게 해줘").

**방법**: 신규 리플레이 로직 0줄. `greedy_replay_entry_veto`의 exit-guard 분기는
`comp.get("sustained_uptrend_mask")`가 `None`이면 완전히 no-op이 되도록 이미 설계돼 있음을
소스에서 직접 확인(주석: "No mask attached -> byte-identical to the unmodified greedy_replay's
own behaviour"). `prepare_regime_aware_components_dual_seeded`의 renamed copy에서 h48qual에
guard_* 부착 5줄만 생략하고(`h48qual_liveatr`을 그대로 사용), zig075의 `short_entry_veto_mask`
부착은 유지. 같은 5페어(zig075 N=5 x h48qual liveATR-relabel N=5)×6창, 같은 엔진 함수 재사용.
`h48qual_guard_active_bars` 진단값이 전 구간에서 0임을 리포트로 직접 확인해 guard가 구조적으로
비활성화됐음을 검증(`guard_structurally_disabled_everywhere=True`).

**결과 — entry-veto 단독으로 6/6 전부 동일 부호**:

| 창 | veto-only 부호일치 | veto+guard 부호일치 | 비고 |
|---|---|---|---|
| 2025q1 | ✅ | ✅ | 5페어 수치까지 완전동일(guard 0회 발동) |
| 2025q2 | ✅ | ✅ | guard 1340회 발동(pair1)하나 부호불변, 매그니튜드만 일부 조정 |
| 2025q3 | ✅ | ✅ | guard 8286회 발동+338회 실제판정변경(pair1), pair5는 +86.91%→+19.35%로 매그니튜드 크게 축소되나 부호는 유지 |
| val | ✅ | ✅ | guard 887회 발동(pair1), 일부 페어 매그니튜드 차이 있으나 부호불변 |
| oos_q1 | ✅ | ✅ | 5페어 수치까지 완전동일(guard 0회 발동, pair1 기준) |
| oos_q2 | ✅ | ✅ | 5페어 수치까지 완전동일(guard 0회 발동) |

**결론**: entry-veto 단독으로 veto+guard 결합과 **동일한 6/6 부호일치**를 달성한다. exit-guard는
일부 창(2025q2/q3/val)에서 실제로 자주 발동하고 매그니튜드에 유의미한 영향을 주지만(2025q3
pair5는 판정 338회 변경으로 PnL이 86.91%→19.35%까지 줄어듦), **부호일치 여부라는 이 축의
핵심 지표에는 기여분이 0이다** — entry-veto가 안정화의 필요충분조건이고 exit-guard는 부가적
매그니튜드 조정 역할로 보인다.

**BTC 이식 실패 원인 재해석**: 이로써 "BTC는 exit-guard가 없어서 실패했다"는 가설은 기각된다
— ETH도 entry-veto 단독으로 충분했으므로, BTC의 entry-veto 단독 테스트는 ETH 메커니즘의
"절반짜리"가 아니라 **핵심 메커니즘 그 자체**였다. 그런데도 BTC에서 실패했다는 건, 실패 원인이
"어느 절반을 이식했는가"가 아니라 **BTC의 단일컴포넌트 구조 자체**에 있다는 뜻이다. 구체적으로:
ETH는 zig075(베토 대상)가 트레이드의 82~83%를 차지하면서도 h48qual과 서로 다른 시점에 신호를
내는 구조라, zig075의 SHORT신호를 베토하면 그 슬롯이 자주 비거나 h48qual이 대신 채운다. BTC는
컴포넌트가 하나뿐이라 베토 대상 신호(SHORT)와 포지션을 점유하는 것도 같은 모델이고, 상승추세가
지속되는 동안 그 모델은 이미 어떤 포지션에 들어가 있는 경우가 대부분이다(BTC 테스트의
`veto_bars`가 창당 최대 1331까지 나왔는데도 거래수가 그대로였던 게 바로 이 현상 — 후보 bar
대부분이 이미 포지션 보유중이라 베토가 실제로 작동할 flat 슬롯을 못 만남). 즉 BTC 이식 실패는
"메커니즘을 덜 이식해서"가 아니라 **베토가 작동하려면 필요한 "대체 신호원이 있거나 자주
flat인" 구조 자체가 BTC엔 없어서**로 보인다(직접 트레이드단위 재현은 아님, 정황증거).

**산출물**: 스크립트 `scripts/eval_eth_odyssey4_shadow_entry_veto_only_20260820.py`(신규,
`greedy_replay_entry_veto` 등 기존 함수 무수정). 결과
`tmp/causal_regen_20260516/eth_odyssey4_shadow_entry_veto_only_20260820/report.json`. 신규
학습 없음(기존 N=5 zig075/h48qual liveATR-relabel 번들 100% 재사용).
fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.

## 산출물

- 스크립트: `scripts/eval_eth_odyssey4_shadow_full_reseed_20260820.py`,
  `scripts/eval_eth_dual_single_factor_isolation_20260820.py`
- 결과: `tmp/causal_regen_20260516/eth_odyssey4_shadow_full_reseed_20260820/report.json`,
  `tmp/causal_regen_20260516/eth_dual_single_factor_isolation_20260820_summary.json`,
  `tmp/causal_regen_20260516/eth_dual_single_factor_isolation_20260820_testBliveatr_summary.json`

신규 학습 없음(기존 N=5 번들 재조합만) — 전체 과정이 순수 재평가.
**fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_
timestamps_used=false, future_rows_used_for_entry=false**.
