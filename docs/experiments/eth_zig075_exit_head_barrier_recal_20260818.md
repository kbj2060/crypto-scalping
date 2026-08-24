# zig075 exit_head 배리어 재보정(terminal/adverse/giveback) + encoder unfreeze 시도 (2026-08-18)

## 배경

[[eth_odyssey4_zig075_exit_head_threshold_review_20260817]]에서 zig075 exit_head 발동을
"작동시키는 것 자체"를 목표로 삼는 건 방향이 틀렸다고 결론 났었다(giveback_min=0.65가
발동시 평균 97.6% giveback 후에야 반응 — threshold를 아무리 조정해도 구제 안 됨). 오늘 밤
[[eth_odyssey4_exit_head_liveatr_barrier_and_label_reaudit_20260818]]에서 `adverse_unreal`/
`min_mfe_for_giveback`가 옛 고정배리어(2.6%/1.4%) 시절 값을 새 ATR floor배리어(7.5%/4.0%,
96~99% 바인딩 확인됨)로 한 번도 재보정 안 한 채 방치돼 있었다는 걸 발견 — 사용자가 이
세 파라미터(+giveback_min)를 새 배리어 기준으로 재보정해서 다시 시도해보자고 요청, 추가로
"encoder도 freeze하지 말고 exit_head와 함께 학습"도 테스트 요청.

## 설계

zig075 자체 부모(`omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_
01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629`, 즉 라이브 배포 인코더)를
고정하고 exit_head만 재학습. 재보정값(근거는 배리어폭 대비 원래 설계 비율 복원, 상세는 스크립트
docstring):

| 파라미터 | 기존(발견1a 수정후) | 새값 | 근거 |
|---|---:|---:|---|
| `adverse_unreal` | −1.0% | **−2.0%** | 새 SL(4.0%)의 50% (원래 옛배리어 대비 71%였던 설계의도에 근접, 다소 보수적) |
| `min_mfe_for_giveback` | +0.6% | **+1.5%** | 새 TP(7.5%)의 20% (원래 23%와 거의 동일 비율 복원) |
| `giveback_min` | 0.65 | **0.45** | 0.65(너무 늦음, 실측97.6%)와 0.25(eth_candidate 다른 인코더, 발동률0%) 사이 미탐색 중간값 |
| `terminal_window` | 3봉 | 3봉(불변) | 이번 발견들과 무관, 변수 하나로 줄이려 유지 |

시나리오 A(encoder unfreeze, exit loss만 backprop, direction/quality는 그대로 얼림 — 조인트
멀티태스크 재구성은 오늘 밤 범위 밖, 이전 세션에서 이미 이렇게 스코프 확정)도 같이 테스트.
재현: `scripts/train_eth_zig075_exit_head_barrier_recal_20260818.py --seed 101` (frozen 기본,
`--unfreeze-encoder` 옵션).

## ⚠️ unfrozen 실행 결과는 오염됨 — 동시세션 편집과 충돌

frozen 실행(03:19:28 시작)은 정상 완료(1500/1500 후보, 1,388,671행). 그런데 **unfrozen
실행(03:52:29 시작, frozen 뒤 순차 실행)이 도중에 동시세션이 공유 파일
(`research_eth_omega461_exit_head_liveatr_relabel_20260813.py`)에 적용한 발견1b 수정
(03:26:33 — `_risk_sizing_for_component` 신설, `risk_margin`/`risk_leverage` 필수 인자화)을
그대로 물고 가서, 후보 1500개 중 1383개가 `risk_sizing_nonpositive`로 스킵되고 117개(90,301행)
만으로 학습됐다.** frozen은 자기 프로세스가 03:19:28에 이미 구버전을 메모리에 로드해서 편집과
무관했지만, unfrozen은 새 프로세스라 그 시점의 (아직 완성 안 된 중간상태로 추정되는) 공유
파일을 그대로 읽었다. **unfrozen 결과는 "encoder freeze 여부"뿐 아니라 "학습 데이터 규모/구성"
까지 같이 바뀐 오염된 비교라 폐기한다.** 동시세션의 발견1b 작업이 안정화된 뒤 frozen과 함께
깨끗하게 재실행 필요.

## frozen 결과 — VAL 단독, N=1시드, exit_threshold 스윕

| exit_threshold | pnl | mdd | trades | exit_head 발동 |
|---:|---:|---:|---:|---:|
| baseline(라이브, exit_head 미사용) | +40.31% | −13.07% | 29 | 0 |
| 0.99 | +35.49% | −13.07% | 31 | 2 |
| **0.95(라이브 기본값)** | **+36.32%** | **−10.07%** | 38 | **11** |
| 0.90 | +10.15% | −14.63% | 64 | 45 |
| 0.80 | −7.37% | −14.40% | 128 | 119 |
| 0.70 | −2.29% | −14.84% | 171 | 165 |

## 해석

**좋은 신호**: 처음으로 발동률이 0%가 아니다(0.95에서 11건, 이전 eth_candidate 테스트의
giveback_min=0.25는 6창 전체 0건이었음). exit_threshold=0.95(라이브값 그대로)에서는 PnL이
소폭(+40.31%→+36.32%, −4pp) 감소하지만 **MDD가 오히려 개선**된다(−13.07%→−10.07%) — 수익을
약간 내주고 리스크를 낮추는 트레이드오프로 보인다.

**우려되는 신호**: threshold를 낮출수록(발동을 더 허용할수록) PnL이 단조적으로 나빠진다
(0.95:+36% → 0.90:+10% → 0.80:−7% → 0.70:−2%) — 이건 옛 giveback_min=0.65의 "발동될 때마다
해롭다"는 패턴과 질적으로 같다. 즉 재보정이 "발동은 하게 만들었지만" "발동이 유익하다"는
문제까지 해결했는지는 아직 불확실하다 — 0.95 근방의 소수 발동(11건)은 우호적일 수 있어도,
문턱을 넓히면 여전히 나쁜 발동이 섞여 들어온다는 뜻일 수 있다.

## 한계 — 결론 내리기엔 이르다

- **N=1시드**뿐(이 리포 규율은 N≥5). 단일 시드 결과는 노이즈일 수 있다.
- **VAL 1개 창**뿐(6개 표준창 중). 다른 창에서도 같은 패턴인지 확인 안 됨.
- unfrozen(encoder 공유학습) 비교는 이번엔 오염돼 사용 불가.

## 다음 단계 (미실행)

1. frozen 변형을 N≥5 시드로 확장 + 6개 창 전체 평가(fresh-forward greedy_replay).
2. exit_threshold=0.95 부근에서 실제 발동 거래들의 giveback 비율을 개별 확인 —
   "이번엔 진짜 일찍 발동하는지" 대 "여전히 늦게 발동하되 횟수만 늘었는지" 구분 필요.
3. 동시세션의 발견1b(risk_sizing) 작업이 안정화된 뒤, unfrozen(encoder 공유학습) 버전을
   같은 조건(risk sizing 방식 통일)으로 깨끗하게 재실행.

## 준수 확인

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false. live/섀도우 파일
무변경. 산출물: `tmp/causal_regen_20260516/eth_zig075_exit_head_barrier_recal_20260818_seed101_frozen/`,
`..._seed101_unfrozen/`(오염, 참고용).
