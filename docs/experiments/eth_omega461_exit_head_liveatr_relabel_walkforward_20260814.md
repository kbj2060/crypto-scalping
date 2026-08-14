# ETH Omega4.6.1 h48qual exit_head 라이브 ATR 재라벨 — 워크포워드 재학습 강건성 검증 (2026-08-14)

## 배경

`scripts/research_eth_omega461_exit_head_liveatr_relabel_20260813.py`(이하 "원본 스크립트")는 h48qual/zig075의 exit_head를 라이브 ATR-adaptive 배리어 라벨로 재학습하는 레시피다. 이 레시피의 h48qual 결과물은 이미 `scripts/live_eth_exithead_asymmetric_shadow_20260813.py`로 **섀도우 배포되어 있다**(zig075는 재라벨이 악화되어 원본 그대로 유지, h48qual만 교체). 지금까지의 검증은 전부 **딱 한 번 학습된 모델**을 여러 창에서 평가하는 것(`scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py`, VAL+OOS-Q1+OOS-Q2)이었다 — 학습 자체를 재현한 적은 없다.

2026-08-13 밤 JM 실험(N=5 시드)은 "같은 레시피, 시드만 바꿔도 1/5~2/5만 재현된다"는 것을 발견했다(레포 CLAUDE.md의 Seed-Diversity Ensemble Promotion Gate 정책 배경). 이번 실험은 같은 종류의 질문을 **시드축이 아니라 시간축**으로 묻는다: 학습구간을 바꿔 처음부터 다시 학습해도 "재라벨이 원본 exit_head를 이긴다"는 패턴이 재현되는가?

## 방법

### 레시피 재사용 확인

원본 스크립트(`research_eth_omega461_exit_head_liveatr_relabel_20260813.py`, 671줄)를 전체 읽고 다음을 확인했다:

- **라벨 구성**: 모든 `zigzag_action∈(1,2)` bar를 독립 후보로 삼아, 진입 시점부터 라이브 ATR-adaptive TP/SL 배리어(`atr_window=192, tp_mult=12.0, sl_mult=6.0, min_tp=0.075, min_sl=0.040, max_tp=0.22, max_sl=0.12` — `trading_bot_modules/omega4_6_1_live.py`의 `_ComponentConfig` 기본값과 동일)를 순방향 시뮬레이션해서 배리어 해소 bar를 찾고, `terminal_window=3`/`adverse_unreal=-0.010`/`min_mfe_for_giveback=0.006`/`giveback_min=0.65` 규칙으로 포지션-bar 단위 이진 라벨을 만든다.
- **후보 서브샘플링**: `--max-candidates`(기본값 2000)개를 `np.random.default_rng(seed)`로 비복원 추출, `--seed` 기본값 260813.
- **아키텍처/하이퍼파라미터**: encoder/direction_head/quality_head는 원본 라이브 번들에서 완전히 동결, exit_head만 `pricemove_retrain._fit_exit_head_only`로 재학습(`--epochs` 기본값 8).
- **체크포인트 게이트**: 학습 전 배리어 해상도 분포(중앙값 bars-to-resolution)가 30bar 이상인지 먼저 확인하고 통과해야 본학습 진행.

새 스크립트(`scripts/train_eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814.py`)는 원본을 **모듈로 import**해서 위 함수들(`_fast_timescale_checkpoint`, `_build_exit_dataset_entry_label_live_atr_barrier`, `_retrain_component_exit_head_liveatr`, `LIVE_ATR_CFG`, `TIMESCALE_GATE_MIN_MEDIAN_BARS`)을 **전혀 재구현하지 않고 그대로 호출**한다. 원본 파일은 한 줄도 수정하지 않았다(`git status` 무수정 확인). 새로 작성한 코드는 딱 하나: `_prepare_frames_walkforward(train_start, train_end)` — 원본이 하드코딩한 `train_all[timestamp < parent.SPLIT_TS]`(2025-10-01) 대신 임의의 `[train_start, train_end)` 구간을 받는, 데이터 준비 단계만의 포크다.

**레시피 동일성의 독립 검증**: 평가 스크립트(`eval_eth_omega461_exit_head_liveatr_relabel_walkforward_20260814.py`)의 G0 자체검증이 이미 확립된 폴드 A(원본, 미재학습) 번들 + VAL 창 조합에서 `docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md`가 보고한 수치(h48qual baseline PnL 5.4545%/MDD -11.6196%/29건 → 재라벨 9.2289%/-7.5940%/63건)를 소수점까지 정확히 재현했다(아래 결과 참고). 이는 평가 파이프라인 자체가 원본과 동일한 결과를 내는 것을 확인해, 이후 B/C/D 결과의 신뢰성을 뒷받침한다.

### 폴드 설계 (4개, 전부 사전 확정)

| 폴드 | 학습구간(명목) | 확인구간 | 재학습 여부 |
|---|---|---|---|
| A(기준점) | 2025-01-01~09-30(원본 그대로) | VAL(2025-10-01~12-31) | 아니오 — 기존 섀도우 배포 번들 재사용 |
| B | 2025-01-01~06-30(H1) | 2025-Q3(지속상승장) | 예 |
| C | 2025-01-01~12-31(2025 전체) | OOS-Q1(하락장) | 예 |
| D | 2025-01-01~2026-03-31(OOS-Q1까지 확장) | OOS-Q2(휩소) | 예 |

확인구간은 전부 `eth_omega461_multiwindow_confirmation_gate_20260814.py`의 `load_all_windows()`가 반환하는 6개 사전등록 창 중 하나를 그대로 재사용했다(새 로더를 만들지 않음).

### ⚠️ 실행 중 발견 — 학습구간 데이터 커버리지 한계 (사전에 알지 못했던 제약)

`_prepare_frames_walkforward`가 학습 데이터를 정렬하는 `omega.TABM_2025`/`omega.TABM_2026`(Omega1.2 라우터 OOF 예측 파일)과 `omega.EVAL_CSV`(2026 원천 피처 파일)의 **실제 날짜 커버리지가 명목 학습구간보다 좁다**는 것을 체크포인트 사전실행 중 발견했다:

- `TABM_2025`는 **2025-05-08부터** 시작한다(2025-01-01부터가 아님). `train_all`(원천 피처)은 2025-01-01부터 있지만, 정렬(`omega._align`) 단계에서 예측 커버리지가 없는 2025-01-01~05-07 구간이 전부 드롭된다.
- `omega.EVAL_CSV`(`trade_candidates_2026...csv`)와 `TABM_2026` 둘 다 **2026-02-28까지만** 존재한다(2026-03-31까지가 아님).

이 제약은 원본 스크립트(폴드 A)에도 이미 있던 것이고 내가 새로 만든 결함이 아니다 — 원본도 `train_all[timestamp < SPLIT_TS]`를 이 동일한 `TABM_2025`에 정렬하므로, "2025-01-01~09-30 학습"이라는 문서화된 표현은 실제로는 **2025-05-08~09-30**을 의미했다(실측 후보 모집단 37,245건, 원본 report.json에서 직접 확인). 이번 실험에서 4개 폴드 전부에 이 커버리지 한계가 그대로 적용된 결과, **실제(effective) 학습구간**은 다음과 같다:

| 폴드 | 명목 학습구간 | 실제 학습구간 | 비고 |
|---|---|---|---|
| A | 2025-01-01~09-30 | **2025-05-08~09-30**(~145일) | 원본 report.json 재확인 |
| B | 2025-01-01~06-30 | **2025-05-08~06-30**(~53일) | **폴드 A 구간의 부분집합** — 독립 구간 아님 |
| C | 2025-01-01~12-31 | **2025-05-08~12-31**(~237일) | 폴드 A를 뒤로 확장 |
| D | 2025-01-01~2026-03-31 | **2025-05-08~2026-02-28**(~296일) | 폴드 C를 더 확장하되 **2026년 3월 데이터 전무** |

**이 발견이 결과 해석에 미치는 영향**: 폴드 B는 명목상 "다른 독립 기간"이 아니라 폴드 A와 같은 시작점(2025-05-08)에서 끝만 짧게 자른 부분집합이다. 따라서 폴드 B 단독은 "완전히 다른 시간대에서도 재현되는가"에 대해 폴드 C/D보다 약한 증거다. 반대로 폴드 C/D는 폴드 A 구간을 순수하게 뒤로 확장한 것이라 시간축 강건성 질문에 더 적합하다. 사전에 확정된 4개 폴드 설계 자체는 결과를 본 뒤 바꾸지 않았다(그대로 실행) — 이 표는 사후에 발견한 데이터 제약을 투명하게 보고하는 것이지, 폴드 재설계가 아니다.

### 컴퓨팅 — dev/서버 분산 실행

dev에서 CPU로 먼저 시도했다. 폴드 B(가장 짧은 학습구간)가 데이터셋 빌드(741.7초) + h48qual 재학습(1337.5초) + zig075 재학습(1418.3초) = **약 57분**으로 이미 30분 기준을 초과함을 확인, 이후 폴드 C를 `scripts/ops/handoff.sh`로 서버(`llewyn@192.168.0.232`)에 오프로드해 dev의 폴드 D와 병렬 실행했다(서버는 3개 섀도우봇 + `trading_bot.py` 상시 실행 중, 실행 전 `free -h`로 23GB 여유·12코어·load average 0.31 확인 후 진행, 순차 1개 작업만 배정). 서버 결과(`true_3head_tabm_bundle.pt` 2개 + `report.json`)는 `handoff.sh pull`로 회수해 md5 일치를 직접 확인했다. 최종 소요: 폴드 B 57분(dev), 폴드 C 약 55분(서버, dev와 병렬), 폴드 D 약 96분(dev, 학습구간이 가장 길어 데이터셋 1.72M행으로 가장 큼 — 도중 평가 스크립트를 동시 실행해 CPU 경합이 생겨 일부 시간이 늘어남, 이후 평가는 학습 완료까지 중단).

## 결과

컴포넌트 레벨(h48qual 단독, direction/quality/encoder는 4개 폴드 전부 원본 그대로 동결) — `research_eth_omega461_exit_sweep_20260721.prep_component`/`.replay_exit_variant`를 그대로 재사용하되 창을 `load_all_windows()`에서 가져오도록 일반화한 `_evaluate_component_on_window`로 평가:

| 폴드 | 확인창 | 원본 exit_head (PnL/MDD/거래수) | 재라벨 exit_head (PnL/MDD/거래수) | PnL 개선? | MDD 개선? | **재라벨이 원본을 이기는가** |
|---|---|---|---|---|---|---|
| A(기준, 미재학습) | VAL | +5.45% / -11.62% / 29건 | **+9.23%** / **-7.59%** / 63건 | ✅ | ✅ | **✅ 승리** |
| B(재학습) | 2025-Q3 | -0.55% / -16.94% / 26건 | -5.74% / -15.24% / 154건 | ❌ | ✅ | ❌ **패배**(PnL 악화) |
| C(재학습) | OOS-Q1 | +9.49% / -6.54% / 14건 | +6.38% / -4.87% / 28건 | ❌ | ✅ | ❌ **패배**(PnL 악화) |
| D(재학습) | OOS-Q2 | +14.81% / -9.01% / 12건 | -3.05% / -10.72% / 29건 | ❌ | ❌ | ❌ **패배**(PnL 부호 반전, MDD도 악화) |

포트폴리오 레벨(h48qual=해당 폴드 exit_head, zig075=전 폴드에서 완전 원본 동결 — `gate.run_portfolio_variant`를 `bundle_override`만 바꿔 그대로 재사용, 4개 폴드 모두에서 baseline_both_original 대비):

| 폴드 | no_gate 베이스라인→재라벨 | with_gate 베이스라인→재라벨 | 승패(no_gate/with_gate) |
|---|---|---|---|
| A | +36.82%/-24.34% → **+46.59%/-21.70%** | +54.88%/-31.11% → **+77.31%/-21.76%** | ✅/✅ |
| B | -35.54%/-49.79% → -42.79%/-58.12% | -9.73%/-44.37% → -31.57%/-51.12% | ❌/❌ |
| C | +49.32%/-16.20% → +21.01%/-28.70% | +44.48%/-15.48% → +4.71%/-28.70% | ❌/❌ |
| D | +3.13%/-15.00% → -14.72%/-22.48% | +9.85%/-15.00% → -1.76%/-17.16% | ❌/❌ |

**요약: 4개 폴드 중 컴포넌트·포트폴리오·no_gate·with_gate 전 지표에서 "재라벨이 원본을 이긴다"는 패턴이 재현된 것은 폴드 A(재학습되지 않은 원본 그 자체) 하나뿐이다. 진짜로 독립 재학습된 폴드 B/C/D 3개는 전부, 예외 없이 재라벨이 원본보다 나빴다.**

부가 관찰: 재라벨 exit_head는 모든 재학습 폴드에서 거래수가 큰 폭으로 늘었다(B: 26→154건, C: 14→28건, D: 12→29건) — exit_head가 원본보다 훨씬 자주 발동한다는 뜻으로, 폴드 A(29→63건, 약 2.2배)보다도 증가율이 크다(B는 5.9배). 학습구간이 원본보다 짧거나(B) 구성이 다를 때 exit_head가 "더 자주 청산하자"로 과적합하는 경향이 있어 보인다.

체크포인트 게이트는 4개 폴드 전부 통과했다(중앙값 bars-to-resolution: A 622/595, B 550/459, C 579/612, D 613/609 — 전부 30bar 기준을 크게 상회).

## 결론

**레시피가 시간축에서 강건하지 않다.** 폴드 A의 "재라벨이 원본 exit_head를 이긴다"는 결과는 그 특정 (학습구간, 확인구간) 조합에 특유한 것으로 보이며, 학습구간을 바꿔 처음부터 재학습하면 3/3 폴드에서 재현되지 않았다 — 오히려 방향이 일관되게 반대(재라벨이 더 나쁨)였다.

JM N=5-시드 사례(레포 CLAUDE.md Seed-Diversity Ensemble Promotion Gate 정책의 배경)와 비교하면: JM은 "같은 레시피, 시드만 바꿔도 1~2/5만 재현"이었다. 이번 결과는 그보다도 더 명확하다 — **진짜 독립 재학습은 0/3 재현**이고, 유일하게 성공한 사례는 재학습이 아니라 원래 그 학습 실행 자체다. 즉 "학습구간을 조금만 바꿔도 패턴이 사라진다"는 것은, 애초에 그 패턴이 레시피의 일반적 속성이 아니라 그 특정 학습 실행(및 그것을 평가한 그 특정 VAL 창)의 우연일 가능성을 시사한다.

폴드 B가 폴드 A와 학습 시작점을 공유한다는(위 데이터 커버리지 절 참고) 한계를 감안해도, 폴드 C·D는 폴드 A 구간을 순수하게 뒤로 확장한 독립적 케이스이며 둘 다 명확히 패배했다는 점에서 이 결론은 그 한계로 설명되지 않는다.

**라이브 관련 함의**: h48qual의 이 재라벨 exit_head는 현재 `live_eth_exithead_asymmetric_shadow_20260813.py`로 섀도우 배포 중이다. 이 실험은 그 섀도우 자체를 중단시키거나 승격 여부를 결정하지 않는다(범위 밖) — 다만 그 섀도우가 근거로 삼은 VAL/OOS 검증이 "우연히 잘 맞은 특정 학습 실행"에 의존했을 가능성을 제기하므로, 향후 그 섀도우의 forward 관찰 결과를 해석할 때 이 강건성 부재를 함께 고려해야 한다.

## 산출물

1. `scripts/train_eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814.py` — 학습구간 파라미터화 스크립트(원본 미수정, 함수 재사용만).
2. `scripts/eval_eth_omega461_exit_head_liveatr_relabel_walkforward_20260814.py` — 평가 스크립트(G0 자체검증 + 4폴드 컴포넌트/포트폴리오 비교).
3. 학습 산출물(각 폴드 h48qual/zig075 `true_3head_tabm_bundle.pt` + `report.json`):
   - `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814_foldB/`(dev에서 학습)
   - `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814_foldC/`(서버에서 학습 후 dev로 pull, md5 일치 확인)
   - `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814_foldD/`(dev에서 학습)
4. 평가 산출물: `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_walkforward_20260814/report.json`(G0 자체검증 + 4폴드 전체 수치 + 판정).

## 준수 확인

`fresh_forward_bar_by_bar=true`(모든 컴포넌트/포트폴리오 평가는 `sweep.replay_exit_variant` 또는 `greedy.greedy_replay`의 단일 순방향 bar-by-bar 패스, 학습 데이터셋 빌드도 원본 스크립트의 미수정 순방향 배리어 시뮬레이션). `trade_ledgers_used_as_input=false`. `saved_parent_exit_timestamps_used=false`. `future_rows_used_for_entry=false`. 원본 스크립트(`research_eth_omega461_exit_head_liveatr_relabel_20260813.py`)와 그 의존 모듈(`research_eth_omega461_exit_head_h48cons_relabel_20260813.py`, `research_eth_omega461_exit_sweep_20260721.py`, `eth_omega461_multiwindow_confirmation_gate_20260814.py`, `research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py` 등)은 전부 import로 재사용만 하고 수정하지 않았다. `trading_bot.py`/`trading_bot_modules/omega4_6_1_live.py`/`trading_bot_modules/runtime_config.py`/`.env`는 dev·서버 양쪽에서 `git status --porcelain` 무변경(0줄) 확인. zig075는 4개 폴드 전부에서 포트폴리오 평가 시 완전히 원본 그대로(재학습 대상은 h48qual 비교를 위해 컴포넌트 레벨에서도 함께 재학습했지만 — 원본 스크립트가 항상 h48qual/zig075 둘 다 재학습하는 구조라 그대로 재사용한 부산물이며, 포트폴리오 비교의 zig075는 항상 `portfolio._component_cfg("zig075")`로 원본 고정). 재학습 시드는 4개 폴드(A는 미재학습) 전부 원본 기본값 260813으로 고정 — "학습구간 하나만 바꾼다"는 원칙을 지키기 위해 시드를 함께 바꾸지 않았다(시드축 강건성은 별도 질문, JM 실험이 이미 다룸).
