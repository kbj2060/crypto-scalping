# ETH Omega4.6.1 h48qual exit_head 라이브 ATR 재라벨 — NOT_ROBUST 메커니즘 진단 (2026-08-15)

## 배경

`docs/experiments/eth_omega461_exit_head_liveatr_relabel_walkforward_20260814.md`(이하 "워크포워드 실험")는 h48qual exit_head 라이브 ATR 재라벨(현재 `live_eth_exithead_asymmetric_shadow_20260813.py`로 섀도우 배포 중)이 학습구간을 바꿔 재학습하면 4개 폴드 중 3개(B/C/D)에서 "재라벨이 원본을 이긴다"는 패턴을 재현하지 못한다(`NOT_ROBUST`)는 것을 확인했다. 그 실험은 **재현 여부**를 확인하는 데 집중했고, **왜** 재현되지 않는지 메커니즘은 다루지 않았다.

이 문서는 워크포워드 실험이 이미 만들어 저장해 둔 산출물(각 폴드 `report.json`의 `component_h48qual` 집계 + 각 폴드 학습 `report.json`의 `dataset` 라벨 구성 통계)과 원본 레시피 코드를 재분석해서 원인을 규명한다. **신규 학습·신규 replay는 없다** — 순수 사후 진단이며, 이 문서의 결론은 워크포워드 실험의 `NOT_ROBUST` 판정 자체를 바꾸지 않는다(범위 밖).

## 방법

읽은 소스:
1. `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_walkforward_20260814/report.json` — `folds.{A,B,C,D}.component_h48qual`(baseline/new 각각의 pnl/mdd/trades/wr/avg_hold_bars/exit_reasons/max_trade_pnl/p95_trade_pnl)
2. `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814_fold{B,C,D}/report.json` — `dataset`(라벨 구성: positive_rate/continued_exit_reasons)
3. `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/report.json` — 폴드 A(원본, 섀도우 배포 번들)의 동일 `dataset` 통계
4. `scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py`의 `_build_exit_dataset_entry_label_live_atr_barrier`(라벨 구성 로직, 245~503줄) 정독

## 결과

### 1. exit_head 발동 행태 — 4개 폴드 전부에서 같은 방향으로 변한다

| 폴드 | 판정 | baseline 발동/거래수/승률/평균보유bar | relabel 발동/거래수/승률/평균보유bar |
|---|---|---|---|
| A(VAL) | 승리 | exit_head 0/29건, wr 41.4%, 670.3bar | exit_head 52/63건(82.5%), wr 30.2%, 210.8bar |
| B(2025Q3) | 패배 | exit_head 1/26건, wr 38.5%, 884.8bar | exit_head 147/154건(95.5%), wr 30.5%, 89.8bar |
| C(OOS-Q1) | 패배 | exit_head 0/14건, wr 50.0%, 781.9bar | exit_head 25/28건(89.3%), wr 39.3%, 128.1bar |
| D(OOS-Q2) | 패배 | exit_head 0/12건, wr 58.3%, 1502.2bar | exit_head 26/29건(89.7%), wr 27.6%, 255.9bar |

원본 exit_head는 4개 폴드 전부에서 **청산사유로 거의 등장하지 않는다**(0~1건 — Odyssey4가 zig075에서 확인한 것과 같은 종류의 구조적 무관여). 재라벨 exit_head는 정반대로 **4개 폴드 전부에서 거래의 82~96%를 스스로 청산**하고, 평균 보유시간을 3.2~9.9배 압축하며, **승률을 예외 없이 낮춘다**(-8.3pp~-30.7pp, 승리한 폴드 A도 포함). 즉 "재라벨이 이겼다/졌다"의 차이는 재라벨이 하는 행동 자체의 차이가 아니다 — **행동은 4개 폴드에서 사실상 동일하다**(거의 항상 빨리 청산). 차이는 그 행동의 대가뿐이다.

### 2. 그 대가는 "큰 승리 거래를 얼마나 깎아먹는가"로 정확히 갈린다

| 폴드 | 판정 | p95_trade_pnl: baseline→relabel | 변화율 |
|---|---|---|---|
| A | 승리 | 4.69%→4.22% | **-10.0%** |
| B | 패배 | 4.46%→0.97% | **-78.3%** |
| C | 패배 | 4.26%→2.90% | **-31.9%** |
| D | 패배 | 4.30%→1.03% | **-76.0%** |

승리한 A는 상위 승리거래 규모를 10%만 깎아먹었다. 패배한 B/C/D는 32~78%를 깎아먹었다 — 승률 하락(공통)과 결합하면 PnL이 버틸 수 없다. **이 지표 하나가 4개 폴드의 승패를 정확히 갈랐다.**

### 3. 라벨 구성이 4개 폴드 전부에서 거의 동일하다 — "재라벨이 늘 같은 것을 배운다"는 뜻

`_build_exit_dataset_entry_label_live_atr_barrier`(245~426줄)는 포지션-bar 단위로 3개 조건의 OR로 양성("청산해야 함") 라벨을 만든다:

```python
terminal = bars_to_barrier_end < tw          # 배리어 해소 3bar 이내 → 항상 양성
adverse = unreal <= adverse_unreal           # 미실현 손익 ≤ -1.0% → 양성 (진짜 손절 신호)
gave_back = mfe >= min_mfe_for_giveback and giveback >= giveback_min and unreal > 0.0
            # 한때 MFE ≥ +0.6%를 찍고 그 65% 이상을 반납, 단 아직 순양(+)
if terminal: label = 1
elif adverse: label = 1
elif gave_back: label = 1
else: label = 0  # "hold"
```

4개 폴드(A=원본 08-13, B/C/D=08-14 재학습)의 학습 데이터셋 라벨 구성:

| 폴드 | 학습구간 실제일수 | 양성률 | `mfe_giveback_exit` 비중 | `adverse_unreal_exit` 비중 |
|---|---|---|---|---|
| A | ~145일 | 19.90% | **75.65%** | 22.52% |
| B | ~53일 | 18.56% | **79.83%** | 17.60% |
| C | ~237일 | 19.51% | **76.25%** | 21.79% |
| D | ~296일 | 19.48% | **76.80%** | 21.40% |

양성률(18.6~19.9%)도, 양성 라벨 중 `mfe_giveback_exit`가 차지하는 비중(75.7~79.8%)도 학습구간 길이(53~296일, 5.6배 차이)와 무관하게 거의 일정하다. `min_mfe_for_giveback=0.006`(+0.6%)은 5분봉 크립토에서 노이즈만으로도 쉽게 도달하는 낮은 문턱이고, `giveback_min=0.65`는 그 작은 피크의 대부분만 반납하면 충족된다 — 즉 라벨의 압도적 다수(약 4분의 3)는 "포지션이 진짜로 실패하고 있다"(`adverse_unreal_exit`, 손실 -1% 이하)가 아니라 "작은 반등 뒤 일시적으로 눌렸다"는 국소적·단기 노이즈 패턴이다. 이 패턴은 어느 구간에서 학습하든 통계적으로 거의 동일하게 존재하므로, exit_head는 어느 폴드에서 학습되든 거의 같은 종류의 "빨리 청산" 정책을 배운다(위 1절 결과와 정합).

## 결론

**폴드 A의 "승리"는 exit_head가 뭔가 더 똑똑한 것을 배워서가 아니다.** 4개 폴드 모두 라벨 구성이 동일하고(§3), 그 결과 학습된 exit_head의 행동도 동일하다(§1: 82~96% 발동, 승률 하락, 보유시간 3~10배 압축) — 재라벨 레시피는 **어느 구간에서 학습하든 "작은 반등 뒤 눌리면 나가라"는 사실상 같은 정책 하나를 배운다.** 이 정책이 순이익으로 이어지는가는 순전히 **그 확인구간의 큰 승리거래들이 그 국소 되돌림 트리거 시점 이후에도 얼마나 더 뛰었는가**(§2)에 달려 있다 — VAL(폴드 A)에서는 우연히 적었고(승리거래 규모 -10%), Q3/OOS-Q1/OOS-Q2(폴드 B/C/D)에서는 훨씬 많았다(-32~-78%). 라벨 자체가 미래를 보지 못하는 국소·후향적(backward-looking) 휴리스틱이므로 이 차이를 사전에 구분할 방법이 없다.

**학습량 부족 가설은 기각된다**: 폴드 C(237일)·D(296일)는 폴드 A(145일)보다 각각 1.6배·2.0배 많은 데이터로 학습했지만 둘 다 패배했다 — 더 많은 데이터가 이 불안정성을 해결하지 못한다. 문제는 표본 크기가 아니라 **라벨 자체의 설계**(giveback 트리거의 낮은 문턱)다.

이 진단은 워크포워드 실험의 `NOT_ROBUST` 판정을 바꾸지 않는다(그럴 목적도 아니었다) — 왜 그런 판정이 나왔는지에 대한 메커니즘 설명을 더한 것이다.

## 라이브 관련 함의

이 exit_head는 오늘(2026-08-15) 사용자가 `install_and_cutover_odyssey4_shadow_20260814.sh`를 실행해 섀도우에서 은퇴하는 절차가 진행 중이다(관련: [[eth_odyssey3_zig075_short_entry_veto_uptrend_confirmed_20260815]]). 이 문서는 그 결정에 새로운 액션을 요구하지 않는다 — 다만 **이 레시피(라이브 ATR-adaptive 배리어 + MFE giveback 재라벨)를 향후 다시 시도할 경우**, 학습구간을 바꾸거나 데이터를 늘리는 것만으로는 안정화되지 않을 가능성이 높다는 것을 기록해 둔다. 재시도한다면 `mfe_giveback_exit` 트리거 자체(예: `min_mfe_for_giveback` 상향, `giveback_min` 상향, 또는 이 조건을 아예 제거하고 `adverse_unreal_exit`+`near_barrier_resolution_exit`만 사용)를 재설계하는 것이 먼저다.

## 산출물

신규 학습·신규 스크립트 없음 — 이미 저장된 아티팩트의 사후 재분석뿐:
- `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_walkforward_20260814/report.json`
- `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814_fold{B,C,D}/report.json`
- `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/report.json`

## 준수 확인

`trade_ledgers_used_as_input=false`(집계 통계·라벨 구성 통계만 사용, 개별 거래 원장을 의사결정에 조인하지 않음). `saved_parent_exit_timestamps_used=false`. `future_rows_used_for_entry=false`. 신규 bar-by-bar replay를 수행하지 않았으므로 `fresh_forward_bar_by_bar`는 N/A — 인용한 모든 수치는 워크포워드 실험(08-14)이 이미 그 원칙을 준수해 생성한 `report.json`에서 그대로 가져왔다(그 문서의 준수 확인 절 참고). 이 진단은 promotion·모델 선택 근거가 아니라 순수 진단이다(범위: 왜 `NOT_ROBUST`인지 설명, 섀도우 존속 여부는 다루지 않음).
