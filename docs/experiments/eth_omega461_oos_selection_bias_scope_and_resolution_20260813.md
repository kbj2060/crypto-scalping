# ETH Omega4.6.1 — OOS 선택편향 영향범위 확인 및 해결 (2026-08-13)

## 배경

`docs/experiments/eth_val_only_sizing_bias_quantification_20260813.md`의 "추가 조사" 절이
`quality_threshold` 선택 스크립트(`scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py:1173`)의
정렬 키가 `(oos_pnl, validation_pnl)` — **OOS pnl이 1순위** — 임을 발견했다. 이는 이 서브프로젝트가
오늘 밤 내내 우려해온 "VAL 과최적화"와는 질적으로 다른 문제다: VAL이 아니라 **OOS 성과 자체가
라이브 모델의 핵심 하이퍼파라미터를 고르는 직접 최적화 타겟으로 쓰였다**는 뜻이고, 만약 이
"OOS"가 오늘 밤 다른 실험들이 "한 번만 봐야 할 깨끗한 구간"으로 취급해온 2026-01~02월과
겹친다면, 오늘 밤 "OOS로 확인했다"는 모든 주장의 전제 자체가 흔들린다.

이 문서는 그 영향범위를 정확히 확인하고 해결 방향을 실행/제안한다. 재학습·GPU는 쓰지 않았다 —
코드 감사 + 기존 저장 예측/랭킹 CSV의 가벼운 재계산만 수행했다. **라이브 파일
(`trading_bot_modules/omega4_6_1_live.py`, `trading_bot.py`, `runtime_config.py`, 실제 배포된
sidecar/bundle)은 이 작업에서 전혀 수정하지 않았다** — `git status`로 확인.

인용 대상 선행 문서: `docs/experiments/eth_val_only_sizing_bias_quantification_20260813.md`,
`docs/experiments/eth_val_oos_regime_mismatch_investigation_20260813.md`,
`docs/model_contracts/eth_omega4_6_1_live_risk_assessment_20260812.md`.

## 방법론

1. `trading_bot_modules/omega4_6_1_live.py`에 하드코딩된 모든 하이퍼파라미터/threshold를
   전수 나열하고, 각각을 만든 스크립트를 찾아 코드를 직접 읽어 VAL/OOS 사용 방식을 확인했다
   (grep + git 히스토리 + 저장된 report.json/CSV 아티팩트 직접 대조).
2. `quality_threshold` 선택 스크립트가 실제로 로딩하는 "oos" CSV의 타임스탬프 범위를 pandas로
   직접 열어 확인했다.
3. 배포된 두 `quality_threshold_ranking.csv`(h48qual/zig075)를 직접 열어 VAL-최적 threshold와
   배포된(OOS-최적) threshold의 실제 수치 격차를 확인했다.
4. (영향이 크다고 판단됨에 따라) 이미 저장된, 재학습이 필요 없는 "frozen 번들 순방향 추론"
   예측 파일(`tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/`, 2026-07-06에 다른
   세션이 만든 것, 2026-01-01~07-12 커버)을 재사용해, `quality_threshold` 선택이 한 번도 보지
   않은 구간(2026-03-01~07-12)에서 threshold를 다시 스윕하는 진단 스크립트를 새로 작성해
   실행했다: `scripts/research_eth_omega461_quality_threshold_clean_reselection_20260813.py`.

---

## 1단계 — 라이브 파이프라인 하이퍼파라미터/threshold 선택 방식 전수조사

### 요약 표

| # | 파라미터 | 라이브 값 | 라이브 코드 위치 | 선택 스크립트/근거 | 분류 |
|---|---|---|---|---|---|
| 1 | `quality_threshold` (h48qual) | 0.50 | `trading_bot_modules/omega4_6_1_live.py:289` | `scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py:1173` | **OOS-selected (OOS 직접 최적화)** |
| 2 | `quality_threshold` (zig075) | 0.75 | `trading_bot_modules/omega4_6_1_live.py:290` | 동일 스크립트 | **OOS-selected (OOS 직접 최적화)** |
| 3 | `duration_threshold`(`ou_halflife` 게이트) | 0.005417 | `trading_bot_modules/omega4_6_1_live.py:63` | `scripts/select_duration_gate_threshold_val_20260706.py` | VAL-selected |
| 4 | 리스크 사이드카 margin/leverage 매핑(`selected_mapping`) | sidecar pkl 내 sigmoid map | `omega4_6_1_live.py:216-225` (적용부) | `scripts/train_eval_omega4_2_risk_sidecar_20260622.py:833,1327-1330,1428-1445` | VAL-selected (코드로 강제됨) |
| 5 | `SCALE_MAP`(L7 컴포넌트×사이드 배율) | h48qual_L=0.38 / h48qual_S=2.499 / zig075_L=2.446 / zig075_S=2.478 | `omega4_6_1_live.py:66` | `nested_router_scale_robust_family_oos_blind_20260630` 아티팩트(아래 상세) | VAL-selected, OOS-blind |
| 6 | `PRIORITY`(라우팅 순서 h48qual>zig075) | tuple | `omega4_6_1_live.py:67` | 위와 동일 검색의 일부(router_order 후보 비교 포함) | VAL-selected, OOS-blind |
| 7 | ATR TP/SL: `atr_window`/`tp_mult`/`sl_mult`/`min_tp`/`min_sl`/`max_tp`/`max_sl` | 192 / 12.0 / 6.0 / 0.075 / 0.040 / 0.22 / 0.12 | `omega4_6_1_live.py:91-97` | `scripts/eval_omega4_1_atr_safety_sltp_20260622.py` + `docs/model_contracts/omega4_2_atr192_tp12_sl6_floor_tp075_sl040_exit070_20260622_contract.md` | VAL-selected |
| 8 | `EXIT_THRESHOLD` | 0.95 | `omega4_6_1_live.py:68` | `scripts/test_omega4_6_1_exit_threshold_20260706.py` + `docs/model_contracts/omega4_6_1_upgrade_investigation_20260706.md` | VAL-tied/OOS로 확정(약한 OOS 관여, 아래 참고) |
| 9 | `LEVERAGE_CAP`/`NOTIONAL_CAP` | 5.0 / 1.8 | `omega4_6_1_live.py:64-65` | 검색의 고정 입력값(출력이 아님) | 고정값(선택 안 됨) |
| 10 | Regime3 라우팅(전문가 서브넷 선택) | argmax(bull,bear,chop) | `train_omega1_regime3_expert_direction_head_volpca_20260602.py:79-83` | 없음 — threshold 자체가 없는 구조적 규칙 | 해당 없음(임계값 미존재) |

### 항목별 상세 근거

**#1/#2 `quality_threshold` — OOS-selected (기존 발견 재확인)**

```python
# scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py:1173
rows.sort(key=lambda r: (float(r["oos_pnl"]), float(r["validation_pnl"])), reverse=True)
```

직접 재확인한 신규 사실: 배포된 두 아티팩트의 실제 `report.json`을 열어보면 저장된 랭킹 키가
`ranking_by_oos_pnl` **하나뿐**이다 — `ranking_by_validation_pnl` 같은 자매 키가 아예 없다
(`h48qual`/`zig075` report.json 둘 다 top-level key 목록: `[..., 'ranking_by_oos_pnl',
'artifacts']`). 이 스크립트의 다른 형제 스크립트들(#7 ATR 스윕 등)은 `ranking_by_validation`과
`ranking_by_oos`를 **둘 다** 저장해 사람이 비교할 수 있게 하는데, quality_threshold 스크립트는
OOS 정렬 결과만 저장한다 — VAL-정렬 뷰가 애초에 리포트에 존재하지 않는다.

**#3 `duration_threshold` — VAL-only (직접 재확인)**

`scripts/select_duration_gate_threshold_val_20260706.py` 전체를 읽어 확인: docstring
1~11행이 "Selection is VALIDATION-ONLY"를 명시하고, `main()`(40~120행)이 `load_val_frame()`
하나만 호출한다 — **이 스크립트는 OOS CSV를 어디에서도 로드하지 않는다**(파일 전체에
`EVAL_CSV`/`oos_raw` 참조가 없음). 저장된 산출물
`tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/duration_threshold_val_selection.csv`의
16개 후보 중 `threshold=0.005417`이 `duration_priority_score`(전부 VAL 렛저 기반) 최고점이며
라이브 값과 정확히 일치. **확정: VAL-only, OOS 자체를 로딩하지 않는 가장 엄격한 형태.**

**#4 리스크 사이드카 margin/leverage 매핑 — VAL-only, 코드로 강제됨 (직접 재확인)**

```python
# scripts/train_eval_omega4_2_risk_sidecar_20260622.py:833
ap.add_argument("--selection-scope", choices=["validation_only"], default="validation_only")
# :1327-1330
if str(args.selection_scope) != "validation_only":
    raise RuntimeError("risk sidecar promotion selection must be validation_only; ...")
# :1428-1445 (selection_objective="log_risk"일 때)
selected_full = max(full_eligible, key=lambda r: (
    float(r["validation"]["log_risk_utility"]),
    float(r["validation"]["mdd"]),
    float(r["validation"]["pnl"]),
))
```

CLI 자체가 `"validation_only"` 외 다른 선택지를 허용하지 않고(`choices=` 제약), 런타임에서도
다시 한번 강제한다. `key=` 람다는 `r["validation"][...]` 필드만 참조하며 `r["oos"][...]`는
selection에 전혀 등장하지 않는다 — `oos_raw`가 로딩되긴 하지만 리포트 표기용일 뿐(아래 2단계
참고). **`eth_val_only_sizing_bias_quantification_20260813.md`가 언급한
`selection_scope="validation_only"`(sidecar pkl 메타데이터) 주장을 소스 코드 레벨에서 재확인
— 일치.**

**#5/#6 `SCALE_MAP`/`PRIORITY` — VAL-only, OOS-blind (신규 확인)**

`omega4_6_1_full_architecture_blueprint_20260706.md`는 L7(SCALE_MAP)을 "base
`omega4_6_plus_t12_nohold_risk1_20260630` 모델 자체의 튜닝에서 상속"이라고만 적어 원 출처가
불명확했다. 역추적한 결과, 실제 선택 아티팩트를 찾았다:
`tmp/causal_regen_20260516/omega_creative_until_10am_20260630/nested_router_scale_robust_family_oos_blind_20260630/`.
최상위 `report.json`의 `selection_contract` 필드 원문: *"Robust label-family allowlist selected
before OOS readout. Router and scale use validation monthly folds only."* 최종 채택된
`selected_readouts/plus_t12_target_guard_03/report.json`의 `selection` 블록:

```json
"selection": {
  "scope": "validation_only_robust_family",
  "oos_used": false,
  "eligible_scales": 53986,
  "total_scales": 60041,
  "router_selection_oos_used": false,
  "scale_selection_oos_used": false,
  "oos_loaded_after_selection": true
}
```

`SCALE_MAP`(h48qual_L=0.38 등)과 `router_order`(h48qual>zig075, 대안 zig075>h48qual도 후보에
있었음)가 정확히 라이브 값과 일치. `oos_loaded_after_selection: true`는 선택이 끝난 **뒤에만**
OOS를 리포팅용으로 읽었다는 뜻 — 이 서브프로젝트가 지향해야 할 "OOS-blind" 선택의 모범 사례에
해당한다. (원 검색을 실행한 스크립트 파일 자체는 현재 저장소에 남아있지 않음 — 아티팩트
report.json의 명시적 필드가 근거.)

**#7 ATR TP/SL 배리어 — VAL-selected (신규 확인)**

`scripts/eval_omega4_1_atr_safety_sltp_20260622.py`도 `quality_threshold` 스크립트와 똑같이
`ranking_by_validation`/`ranking_by_oos` 둘 다 계산한다(251~252행) — **구조만 보면 OOS-primary
오염 가능성이 있어 실제 채택 결과를 직접 대조했다.** 라이브 값과 정확히 일치하는 그리드 실행은
`tmp/causal_regen_20260516/omega4_1_atr_safety_sltp_20260622_q070_exit070_wider_floor_grid3/`
(contract: `min_tp=0.075, min_sl=0.04, max_tp=0.22, max_sl=0.12`, 후보에 `atr192_tp12_sl6` 포함).
이 grid3의 `ranking_by_validation` 1위가 정확히 `atr192_tp12_sl6`(val_pnl=16.02%)이고,
`ranking_by_oos` 1위는 다른 후보(`atr192_tp16_sl8`, oos_pnl=18.76%)다. 실제 승격 문서
`docs/model_contracts/omega4_2_atr192_tp12_sl6_floor_tp075_sl040_exit070_20260622_contract.md:88,111-113`이
명시적으로 확인해준다: *"Selected variant: atr192_tp12_sl6, chosen by validation first... 
atr192_tp16_sl8 had higher OOS PnL, but materially weaker validation PnL and validation MDD, so
it is not promoted."* **VAL이 이겼고 더 높은 OOS를 주는 대안을 명시적으로 거절한, 이
서브프로젝트가 지향해야 할 정확한 패턴.** (다만 floor/cap 자체는 grid1→grid2→grid3에 걸쳐
사람이 점진적으로 넓혀가며 재실행한 것이라 이 자체가 하나의 공식 그리드 축은 아니었음 — 최종
조합의 채택 방식만 VAL-first임.)

**#8 `EXIT_THRESHOLD=0.95` — 약한 형태의 OOS 관여(질적으로 다름, 별도 표기)**

`scripts/test_omega4_6_1_exit_threshold_20260706.py` docstring: *"VAL = selection, OOS =
one-shot confirm. Same discipline as the h48qual test."* 0.95는 이 스윕 **이전부터 이미
"frozen"** 값이었다(스윕은 "이 값을 낮추면 좋아지는가?"를 검증한 것). 실측
(`docs/model_contracts/omega4_6_1_upgrade_investigation_20260706.md:48-55`):

| exit_th | VAL PnL/MDD | OOS PnL/MDD |
|---|---|---|
| 0.95(frozen) | +54.88 / -31.11 | **+145.34 / -10.13** |
| 0.90 | +54.88 / -31.11 | +89.22 / -14.90 |
| 0.80 | +54.88 / -31.11 | +59.62 / -16.23 |
| 0.70 | +38.99 / -31.11 | +21.66 / -25.02 |

**VAL은 0.95/0.90/0.80 세 후보에서 완전히 동일한 수치다** — 이 구간에서 exit head가 VAL에서는
아예 발동하지 않아 VAL 자체가 세 후보를 구분 못 한다. OOS만 세 후보를 구분하고, 그 결과
(0.95가 가장 좋음)가 원래 frozen 값과 우연히 일치해 "Keep 0.95"로 결론났다. **`quality_threshold`처럼
VAL에서 뚜렷이 이기는 후보를 OOS 성과 때문에 버린 사례는 아니지만, VAL이 무차별한 구간에서
OOS가 최종 타이브레이커로 쓰였다는 점에서 완전히 결백하지도 않다** — 더 약한 형태의 같은 패턴으로
별도 표기한다. 0.70 이하로 내려가면 VAL 자체가 뚜렷이 나빠지므로 그 구간의 결론(0.95 유지)은
VAL 단독으로도 이미 지지된다.

**#9 `LEVERAGE_CAP=5.0`/`NOTIONAL_CAP=1.8` — 고정값**

SCALE_MAP 검색의 `selection_contract` 원문에 "Raw scale floor 0.25, per-trade leverage cap 5"로
**검색의 입력 조건**으로 명시돼 있다 — 이 두 캡 자체가 여러 값 중에서 VAL/OOS로 비교돼 뽑힌
것이 아니라, 검색 이전에 고정된 안전 상한값이다. `build_omega_plus_t12_livepass_candidate_20260630.py:373-375`의
CLI 기본값(`--leverage-cap 5.0 --notional-cap 1.8`)도 동일하게 그대로 통과되는 상수임을 보여준다.

**#10 Regime3 라우팅 — 해당 임계값 없음**

```python
# scripts/train_omega1_regime3_expert_direction_head_volpca_20260602.py:79-83
def _route_id(frame: pd.DataFrame) -> np.ndarray:
    values = frame[ROUTE_COLS].to_numpy(dtype=np.float64)
    return np.argmax(values, axis=1).astype(np.int64)
```

순수 argmax다. 최소 확신도(confidence) 컷오프 같은 게 없다 — `trading_bot_modules/omega4_6_2_source_parent_live.py`의
`Regime3CurrentLiveFeatures` 클래스도 결측치 처리/유한성 검사만 하고 임계값을 두지 않는다.
**VAL/OOS로 선택할 대상 자체가 존재하지 않는다.**

---

## 2단계 — 윈도우 겹침 확인 (핵심)

`train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`의 "oos" 프레임은
`omega._load_omega_frames()`(`scripts/train_eval_omega1_2_tabm_diffusion_risk_20260603.py:198-214`)가
로딩하는 `EVAL_CSV`다:

```python
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
```

직접 pandas로 열어 확인한 타임스탬프 범위:

```
min 2026-01-01 00:00:00
max 2026-02-28 16:00:00
n   16897
```

**결론: 완전히 겹친다.** `quality_threshold`의 OOS-primary 정렬이 직접 최적화 타겟으로 쓴
구간은 **정확히 2026-01-01~02-28**이며, 이는 `docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`가
"h48qual/zig075 예측 CSV의 실제 export 범위"로 명시한 그 좁은 OOS 창과 **동일한 파일, 동일한
구간**이다. 오늘 밤 다른 문서들이 쓴 "OOS" 창(`eth_val_oos_regime_mismatch_investigation_20260813.md`의
2026-01-01~03-31 또는 ~06-30, `eth_val_only_sizing_bias_quantification_20260813.md`의
2026-01-01~03-31)은 전부 이 2개월을 부분집합으로 포함한다 — 즉 **오늘 밤 "OOS로 확인했다"고
표기된 모든 실험은, 최소한 그 OOS 구간의 앞 2개월 부분에 한해서는, quality_threshold 선택
시점에 이미 한 번 소비된 동일한 데이터를 다시 본 것이다.**

같은 `EVAL_CSV`(따라서 같은 2026-01-01~02-28 창)를 로딩하는 다른 스크립트도 확인됨:
- `scripts/eval_omega4_1_atr_safety_sltp_20260622.py`(`_load_frames()`가 동일한
  `omega._load_omega_frames()` 재사용, CLI에 `--eval-csv` 오버라이드 옵션 자체가 없음) — 다만
  이건 VAL-selected로 확인됐으므로(1단계 #7) 이 구간을 "선택 타겟"으로 쓰지는 않았다.
- `scripts/train_eval_omega4_2_risk_sidecar_20260622.py`도 기본 `omega.EVAL_CSV`를 리포팅용으로
  로딩하지만(`--eval-csv`로 오버라이드 가능, 실제 무엇이 쓰였는지는 이번 조사에서 확정하지
  않음), selection 자체는 VAL 필드만 참조하므로(1단계 #4) 이 구간이 선택 타겟은 아니었다.

**결론적으로 "선택 타겟"으로서 이 구간을 소비한 레이어는 quality_threshold 하나뿐이다.** 하지만
그 하나만으로도, 이 구간을 "quality_threshold 선택 이후 처음 보는 OOS"라고 전제한 오늘 밤의
모든 후속 실험(h48qual/zig075 baseline의 OOS 성과를 인용하는 모든 곳)의 전제가 깨진다 — 그
"baseline"이 이미 quality_threshold 자체가 이 구간에서 잘 보이도록 골라진 결과물이기 때문이다.

---

## 3단계 — 실질적 영향 측정

배포된 두 `quality_threshold_ranking.csv`를 직접 pandas로 열어 VAL-최적 threshold와 배포된
threshold를 대조했다(파일:
`tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/quality_threshold_ranking.csv`,
`tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/quality_threshold_ranking.csv`).

| 컴포넌트 | 배포 threshold | 배포 VAL pnl | 배포 OOS pnl(전체 1위) | VAL-최적 threshold | VAL-최적의 VAL pnl | VAL-최적의 OOS pnl |
|---|---:|---:|---:|---:|---:|---:|
| h48qual | **0.50** | +4.58% | **+10.65%**(1위) | 0.35 | **+22.47%** | +3.29% |
| zig075 | **0.75** | +11.09% | **+14.77%**(1위) | 0.55 | **+13.37%** | **-4.37%**(부호 반전) |

**threshold 값 자체의 격차**: h48qual 0.50 vs 0.35 = 0.15(43% 상대), zig075 0.75 vs 0.55 =
0.20(36% 상대) — 둘 다 작은 튜닝 디테일이 아니라 게이트의 permissive/strict 정도를 크게 바꾸는
차이다(h48qual VAL 거래수 29건→33건, zig075 36건→38건).

**성과 격차**: h48qual은 VAL-최적 후보가 배포 후보보다 VAL에서 **4.9배** 더 잘한다(22.47%
vs 4.58%) — 만약 진짜 VAL-only로 선택했다면 전혀 다른(그리고 VAL 기준으로 훨씬 우수한) threshold가
나왔을 것이다. zig075는 VAL 격차가 상대적으로 작지만(13.37% vs 11.09%, 20% 상대), OOS에서는
배포 후보가 +14.77%인 반면 VAL-최적 후보는 **-4.37%로 부호가 뒤집힌다** — 이게 바로
OOS-primary 정렬이 "골라낸" 차이다: OOS에서 잘 보이는 후보를 직접 고른 것이므로 당연히
OOS에서 좋아 보이고, VAL 기준 진짜 승자는 OOS에서 나쁘게 나온다.

**판정: 실질적 영향은 크다.** threshold 자체가 30~40% 상대적으로 다르고, 그 결과 VAL 성과가
최대 5배, OOS 성과는 부호까지 뒤집힐 정도로 다르다 — "존재하지만 무시할 수준"이 아니다.
4단계는 "실질적 영향이 크다" 분기를 따른다.

---

## 4단계 — 미접촉 데이터 기반 클린 재선택 진단 (추가 리서치, 진단 전용)

### 설계

quality_threshold의 오염 구간은 VAL(2025-10-01~12-31, 정상적인 VAL 용도)과 OOS(2026-01-01~02-28,
선택 타겟으로 소비됨) 둘뿐이다. "quality_threshold 선택에 한 번도 쓰이지 않은" 구간에서 다시
스윕해보면 배포된 threshold가 우연히 좋았던 것인지 실제로 안정적인지 힌트를 얻을 수 있다.

**데이터**: `tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/{h48qual,zig075}/oos_predictions_qXXX.csv`
— 2026-07-06에 다른 세션이 frozen 파일 순방향 추론(재학습 아님)으로 만든, 2026-01-01~07-12를
덮는 예측 파일. 이 파일은 `dir_action`/`quality_for_action`(threshold-무관 raw 값)을 그대로
담고 있어서, **모델을 다시 돌리지 않고도** 어떤 threshold든 `final_action = dir_action if
(dir_action != 0 and quality_for_action >= q) else 0` 규칙만 재적용하면 재현할 수 있다 — 이
규칙은 라이브 게이트(`omega4_6_1_live.py:178`)와 원 선택 스크립트(`train_omega1_regime3_routed_expert_direction_quality_20260602.py:239-240`)
둘 다와 동일하다.

**신선(fresh) 윈도우**: 2026-03-01~07-12. quality_threshold의 OOS-primary 정렬이 실제로 본
구간(01-01~02-28) 바로 다음부터, 이 예측 파일이 커버하는 마지막 날짜까지다.
**주의(정직하게 명시)**: 2026-03-01~06-30 구간은 오늘 밤 다른 실험(멀티슬롯 용량/MFE게이팅,
`docs/experiments/eth_val_oos_regime_mismatch_investigation_20260813.md` 참고)이 **포트폴리오
합계 수준으로는** 이미 들여다봤다 — 다만 quality_threshold 자체를 스윕하는 목적으로는 한 번도
쓰인 적 없다. 이 진단이 막으려는 오염(quality_threshold 선택에 재사용)만 놓고 보면 이 구간은
깨끗하다. 오늘 밤 그 무엇도 건드리지 않은 구간은 2026-07-01~07-12뿐이며(12일, 단독으로 쓰기엔
너무 짧음), 그 이후 데이터는 이번 조사에서 새로 만들지 않았다(순방향 추론이 필요해 이번
"가벼운 재계산" 범위를 넘어감).

스크립트: `scripts/research_eth_omega461_quality_threshold_clean_reselection_20260813.py`.
산출물: `tmp/research_20260813/omega461_quality_threshold_clean_reselection/{clean_reselection_grid.csv,clean_reselection_report.json}`,
로그 `tmp/research_20260813/omega461_quality_threshold_clean_reselection_run.log`.

Fresh-Forward 공시: `fresh_forward_bar_by_bar=true`(단일 순방향 bar-loop, `omega._metrics`
재사용), `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`. **다만 이 스크립트가 재사용하는 예측 파일 자체가 이미
"저장된 모델 예측 replay"라는 점에서, 원본 `quality_threshold_ranking.csv`와 동일한
research/diagnostic 등급이다 — live-promotion 등급이 아니다.**

### 중요한 부수 발견 — 자체 정합성(self-check) 오차

같은 2026-01-01~02-28 구간, 같은 threshold 값으로 원본 `quality_threshold_ranking.csv`의
OOS pnl과 이 스크립트의 재계산 값을 대조했다(자체 정합성 검증 목적):

| 컴포넌트 | 최대 절대오차(pp) |
|---|---:|
| h48qual | 13.26pp (q=0.35: 원본 +3.29% vs 재계산 -9.97%) |
| zig075 | 8.43pp (q=0.60: 원본 +2.95% vs 재계산 +11.38%) |

**원인은 이 스크립트의 버그가 아니라 이미 알려진 파이프라인 간극이다.** 이 예측 파일을 만든
`retest_omega4_6_1_extended_oos_20260706.py` 자신의 `report.json`이 이미 이 사실을 기록해뒀다:

```
"known_limitations": {"feature_drift": "ou_halflife/kel/evt_excess_z/btc_corr_60/
dual_momentum differ from original alpha6/7-lineage scoring; ou_halflife re-selected and
confirmed robust, others (parent inputs) unresolved without a full parent retrain"}
```

즉 원본 `quality_threshold_ranking.csv`가 쓴 "alpha6/7-lineage" 피쳐 파일과, 이 진단이 재사용한
"extended" 피쳐 파일(`data/splits/year_oos/training_features_2026_rebuilt.csv` 기반)은 같은
날짜라도 일부 파생 피쳐 값이 다르다 — 그 피쳐들이 **parent 모델(h48qual/zig075)의 실제 입력**에
포함되므로, 모델 예측 자체가 두 파이프라인 사이에서 갈린다. 2026-07-06 당시 자체 점검은
"장기 합산 PnL"(여러 threshold/컴포넌트가 섞인 최종 합계, +145.34% vs +145.46%, 0.12pp 오차)만
비교해 거의 일치한다고 결론냈는데, 이번 조사로 **개별 threshold 후보 단위로 쪼개면 오차가
8~13pp까지 커진다**는 게 새로 확인됐다 — 집계 수준에서는 서로 다른 방향의 오차가 상쇄돼
작아 보였을 뿐이다. **이 발견 자체를 이 문서의 결과로 기록한다** — `EXT_PRED_DIR` 예측 파일을
threshold별 세밀 비교에 쓸 때는 이 노이즈 바닥(8~13pp)을 감안해야 한다.

### Fresh 윈도우 결과 (노이즈 바�드 8~13pp 감안, 방향성 참고용)

| 컴포넌트 | threshold | 성격 | VAL pnl(원본) | Fresh(03-01~07-12) pnl | Fresh 거래수 |
|---|---:|---|---:|---:|---:|
| h48qual | 0.50 | 배포값 | +4.58% | **+15.09%** | 20 |
| h48qual | 0.35 | VAL-최적 | +22.47% | -6.52% | 35 |
| h48qual | (grid 전체 중 fresh 최고) | 0.50 그 자체 | — | +15.09% | 20 |
| zig075 | 0.75 | 배포값 | +11.09% | -10.12% | 36 |
| zig075 | 0.55 | VAL-최적 | +13.37% | -6.75% | 36 |
| zig075 | 0.60 | (grid 전체 중 fresh 최고) | +0.17% | **+15.31%** | 32 |

**h48qual**: 배포값(0.50)이 이 fresh 윈도우에서도 grid 전체 중 최고 성과를 낸다(+15.09%,
VAL-최적 대안의 -6.52%보다 21.6pp 높음 — 노이즈 바닥 13.26pp보다 커서 방향은 신뢰할 만하다).
즉 h48qual은 선택 **절차**는 잘못됐지만(OOS-primary), 그 절차가 우연히 골라낸 **결과값**이
진짜 미접촉 구간에서도 나쁘지 않다 — 절차의 오류가 결과의 오류로 이어졌다는 증거는 fresh
데이터에서 나오지 않는다(다만 절차 자체가 부적절했다는 사실은 변하지 않는다).

**zig075**: 배포값(0.75)과 VAL-최적 대안(0.55) **둘 다 fresh 윈도우에서 마이너스**다
(-10.12%, -6.75%, 격차 3.37pp는 노이즈 바닥 8.43pp보다 작아 두 값의 우열은 이 데이터로
가릴 수 없음). grid 전체에서 fresh 최고 성과를 내는 threshold는 이 둘 중 어느 것도 아닌
0.60이다. 이건 "OOS-primary가 아니라 VAL-only로 골랐으면 좋았을 것"이라는 단순한 이야기가
아니라 — **zig075의 quality_threshold는 어떤 방식으로 골랐어도(OOS-cherry-pick이든 VAL-only든)
2026-03월 이후로는 잘 일반화되지 않는다**는, 선택 절차 문제와는 별개의 추가 경고 신호다.

### 이 진단이 답하지 못하는 것

- 정밀한 수치 비교는 8~13pp 노이즈 바닥 때문에 신뢰구간이 넓다 — "배포값이 fresh에서 이겼다/졌다"는
  판정은 그 격차가 노이즈 바닥을 확실히 넘는 경우(h48qual)에만 방향성 있게 읽고, 넘지 못하는
  경우(zig075)는 "가릴 수 없음"으로 읽어야 한다.
- 이 진단은 승격/채택 근거가 아니다 — 순수 진단이며, 재선택된 threshold를 라이브에 반영하지
  않았다(요청대로 라이브 파일 불변경).
- 완전히 깨끗한 재선택(같은 alpha6/7-lineage 피쳐 파이프라인을 3월 이후로 연장해 순방향 추론)은
  이번 조사 범위를 넘어간다 — 향후 원칙(아래)에 남긴다.

---

## 종합 판단과 실행한/제안하는 해결 방향

**사용자가 제시한 판단 기준을 그대로 적용**: 지금 당장 실자본 리스크는 없다
(`data/live/dashboard_state_governor.json`의 `account.enabled=false, testnet=true`,
`docs/model_contracts/eth_omega4_6_1_live_risk_assessment_20260812.md` 4절 재확인) — 급하게
고쳐야 한다는 긴급성은 없고, 오염된 데이터로 성급하게 재선택하면 오히려 문제를 키울 수 있다는
경고도 유효하다.

3단계에서 실질적 영향이 크다고 확인됐으므로 **(b)+(c) 분기**를 따랐다:

1. **(a) 문서화**: 이 문서 자체가 그 문서화다 — 표, 겹침 확인, 영향 크기, 진단 결과를 전부
   기록했다.
2. **(b) 미접촉 데이터 기반 재선택 시뮬레이션**: `scripts/research_eth_omega461_quality_threshold_clean_reselection_20260813.py`를
   작성·실행해 fresh 윈도우 결과를 얻었다(4단계). 승격 제안이 아닌 진단 결과로만 취급한다.
3. **(c) 향후 원칙**: 아래에 명시.
4. **라이브 파일은 전혀 건드리지 않았다** — `trading_bot_modules/omega4_6_1_live.py`,
   `trading_bot.py`, `runtime_config.py`, 배포된 sidecar/bundle 파일 중 어느 것도 이 세션에서
   수정하지 않았음을 `git status`로 재확인.

**중요한 구분 — 이 발견은 Artifact Integrity 감사와 무관하다.** CLAUDE.md의 "Omega Artifact
Integrity Promotion Gate"(`scripts/audit_omega_artifact_integrity_20260630.py`)는 **아티팩트
정합성**(정확한 threshold의 예측 파일 존재, report.json/사이드카 태그 일치)을 검증하는 것이지
**threshold가 어떻게 선택됐는지는 검증하지 않는다**. h48qual/zig075 둘 다 2026-07-06에
`promotion_pass=true`를 통과한 상태는 이 발견으로 무효화되지 않는다 — 두 감사는 서로 다른
질문에 답한다(`eth_omega4_6_1_live_risk_assessment_20260812.md` 3절이 이미 같은 구분을
명시함).

**오늘 밤 다른 문서들에 대한 함의(재작업은 이 문서의 범위 밖)**: 2단계에서 확인했듯
2026-01-01~02-28은 quality_threshold 선택에 이미 소비된 구간이다. 오늘 밤 h48qual/zig075
baseline의 OOS 성과를 인용한 모든 문서(`eth_val_oos_regime_mismatch_investigation_20260813.md`,
`eth_val_only_sizing_bias_quantification_20260813.md` 등)는 이 사실을 몰랐던 채로 작성됐다 —
이 문서는 그 문서들을 재검증하지 않는다(범위 밖, 사용자 지시대로 이 작업은 진단 문서화에
한정). 다만 "OOS는 순수 readout"이라는 그 문서들의 전제가 quality_threshold 레이어에 한해
깨졌다는 사실 자체는 이 문서로 확정됐으므로, 향후 그 문서들의 결론을 재사용할 때는 이 한계를
함께 고려해야 한다.

---

## 향후 원칙 (권고 — 다음 재학습/재선택 시 지킬 것)

1. **어떤 threshold/하이퍼파라미터든 정렬 키에 `oos_*` 필드를 1순위로 두지 않는다.** VAL
   필드만으로 순위를 매기고, OOS는 선택이 끝난 뒤에만 읽는다(`nested_router_scale_robust_family_oos_blind_20260630`,
   `omega4_2_atr192_tp12_sl6_..._contract.md`가 이미 보여준 이 프로젝트 자체의 좋은 선례를
   표준으로 삼는다).
2. **리포트에 `ranking_by_oos_*` 키만 저장하지 않는다.** VAL-정렬 뷰가 리포트에 아예 없으면
   사람이 검토할 때도 OOS 우선으로 유도된다 — `ranking_by_validation`/`ranking_by_oos`를 항상
   같이 저장하고, "무엇을 기준으로 승격했는지"를 report.json에 `selected`/`selection_scope`
   필드로 명시한다(이번 조사에서 quality_threshold의 report.json에는 이 필드 자체가 없었다).
3. **nested/walk-forward 재선택을 기본으로 한다.** 하나의 VAL/OOS 창을 여러 레이어(사이징 →
   duration → quality_threshold → ...)가 반복 재사용하면, 각 레이어가 "VAL-only"라도 같은
   저표본 구간이 여러 번 다른 목적으로 소비된다 — 레이어마다 겹치지 않는 fold를 쓰거나, 최소한
   몇 개 레이어가 같은 구간을 몇 번 썼는지 report.json에 명시적으로 기록한다.
4. **`EXT_PRED_DIR`류 "확장 예측" 아티팩트를 threshold별로 세밀 비교할 때는 원본 파이프라인과의
   자체 정합성(self-check)을 집계 수준이 아니라 후보 단위로 반드시 확인한다.** 이번 조사가
   보여준 8~13pp 노이즈 바닥은 집계 합산(0.12pp 오차)만 봐서는 드러나지 않았다.
5. **`quality_threshold`를 실제로 재선택하게 된다면**, alpha6/7-lineage 피쳐 파이프라인을
   2026-03월 이후로 순방향 연장해(새 학습 아님, 기존 frozen 번들의 추론만 다시 실행) 이번
   진단의 "노이즈 바닥" 문제 자체를 없앤 상태로 다시 비교할 것을 권고한다 — 이번 조사 범위
   밖이라 실행하지 않았다.

---

## 스코프와 한계

1. 1단계 표는 라이브 파일(`omega4_6_1_live.py`)에 실제로 하드코딩된 값만 대상으로 했다 —
   SOL/BTC 자산의 동등 파라미터(`SOL_BASE_TEMPLATE` 등)는 조사하지 않았다.
2. `#5/#6 SCALE_MAP/PRIORITY`의 원 검색 스크립트 파일 자체는 저장소에서 찾지 못했다(아마
   2026-06-30 당시의 1회성 스크립트가 정리됨) — 판단 근거는 저장된 report.json의 명시적 필드다.
3. 4단계 진단은 fresh 윈도우가 짧고(2026-03-01~07-12, 원본 OOS의 2배 남짓) 피쳐 파이프라인
   drift로 노이즈 바닥이 크다 — "배포값이 옳았다/틀렸다"를 확정하는 근거가 아니라 방향성
   참고용이다.
4. `EXIT_THRESHOLD`의 원래 "frozen" 0.95 자체가 어디서 처음 정해졌는지(2026-07-06 스윕
   이전의 더 오래된 출처)는 추적하지 않았다 — 이번 조사에서 확인해야 할 우선순위가 낮다고
   판단(quality_threshold처럼 VAL 승자를 직접 버린 사례가 아니므로).
5. 오늘 밤 다른 문서들의 재검증(quality_threshold 오염을 반영한 재작성)은 이 문서의 범위
   밖이다 — 위 "종합 판단" 절에 함의만 명시했다.

## 산출물

- 신규 스크립트: `scripts/research_eth_omega461_quality_threshold_clean_reselection_20260813.py`
- 실행 로그: `tmp/research_20260813/omega461_quality_threshold_clean_reselection_run.log`
- 결과: `tmp/research_20260813/omega461_quality_threshold_clean_reselection/clean_reselection_grid.csv`,
  `tmp/research_20260813/omega461_quality_threshold_clean_reselection/clean_reselection_report.json`
- 라이브 파일 변경: **없음**(확인됨)
