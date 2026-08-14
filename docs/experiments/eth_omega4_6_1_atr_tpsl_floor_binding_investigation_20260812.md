# Omega4.6.1 라이브 ATR 적응형 TP/SL — floor 고정 문제 확인 (2026-08-12)

## 배경

사용자가 이전에 발견했던 문제("TP가 12%로 갑자기 뻥튀기") 재조사. `trading_bot_modules/
omega4_6_1_live.py`의 ATR 적응형 TP/SL 로직을 직접 검증.

## 코드

```python
# trading_bot_modules/omega4_6_1_live.py:86-97 (_ComponentConfig 기본값, h48qual/zig075 둘 다 오버라이드 없이 그대로 씀)
atr_window: int = 192
tp_mult: float = 12.0
sl_mult: float = 6.0
min_tp: float = 0.075
min_sl: float = 0.040
max_tp: float = 0.22
max_sl: float = 0.12

# :181-185
atr_pct = _atr_eval._atr_pct(frame, self.cfg.atr_window)[-1]
take_profit = clip(max(min_tp, atr_pct * tp_mult), 0, max_tp)
stop_loss = clip(max(min_sl, atr_pct * sl_mult), 0, max_sl)
```

`_atr_pct`(`scripts/eval_omega4_1_atr_safety_sltp_20260622.py:46-57`)는 표준 True Range를
192bar(16시간) 롤링 평균한 뒤 종가로 나눈 정상적인 퍼센트 ATR — 계산 자체엔 버그 없음.

## 실측 (2025 TRAIN 전체 + 2026 OOS 전체, atr_window=192)

| | 2025(TRAIN, n=105,064) | 2026(OOS, n=16,897) |
|---|---:|---:|
| atr_pct 중앙값 | - | 0.26% |
| atr_pct 최댓값 | 1.07% | 1.13% |
| tp_raw(=atr_pct×12)이 min_tp(7.5%) 초과 비율 | 2.5% | 5.0% |
| sl_raw(=atr_pct×6)이 min_sl(4.0%) 초과 비율 | 1.5% | 3.7% |
| tp_raw이 max_tp(22%) 초과 비율 | **0.00%** | **0.00%** |
| sl_raw이 max_sl(12%) 초과 비율 | **0.00%** | **0.00%** |
| tp_raw 관측 최댓값 | 12.82%(2025-02-03 08:35) | 13.51% |

## 결론

**`min_tp=0.075`/`min_sl=0.040`이 사실상 유일하게 바인딩되는 노브다** — 전체 시간의
95~98.5%에서 TP/SL이 정확히 이 floor 값에 고정된다. `tp_mult`/`sl_mult`(ATR 스케일링)는
ETH 5분봉의 실제 ATR%가 floor를 뚫을 만큼 크지 않아 거의 항상 무력화된다(1.5~5%만 floor
초과). `max_tp=0.22`/`max_sl=0.12`(상한 캡)는 2025~2026 전체 데이터에서 **단 한 번도
발동한 적이 없다** — 설계상으로만 존재하는 죽은 파라미터.

즉 "ATR 적응형 TP/SL"이라는 설계 의도와 달리, 실질적으로는 **거의 항상 고정 7.5%(TP)/
4.0%(SL) 타겟으로 도는 시스템**이다. 관측된 TP 최댓값(12.82%)이 사용자가 기억한 "12%
뻥튀기"의 실체 — ATR이 순간적으로 튀면서 floor 위로 벗어난 드문 경우(전체의 ~2.5~5%)다.

**26일 보유 포지션과의 연결**: 이전 조사(`eth_omega4_6_1_live_risk_assessment_20260812.md`)에서
확인한 26일 보유 SHORT 포지션의 TP/SL이 정확히 0.075/0.04(floor 값)였다. "설계상 최대보유
없음"이라는 이전 결론 자체는 여전히 맞지만, **왜 그렇게 오래 걸리는지에 대한 추가 설명**이
이걸로 생긴다 — 진짜 ATR 적응형이 아니라 고정된 7.5%/4% 타겟이라, 5분봉에서 그 정도 크기의
가격 움직임이 나올 때까지 구조적으로 오래 기다리게 된다.

## 미해결 / 다음 결정

이게 "버그"인지 "의도된 설계"인지는 별도 판단이 필요하다 — 가능성:
1. `tp_mult=12.0`/`sl_mult=6.0`이 애초에 이 자산/타임프레임의 실제 ATR% 규모를 고려하지
   않고 설정된 잘못된 캘리브레이션(다른 자산이나 다른 timeframe 기준값을 그대로 가져왔을
   가능성).
2. `min_tp`/`min_sl` 자체가 (ATR과 무관하게) "이 정도는 최소한 노려야 한다"는 별도 근거로
   의도적으로 설정된 값이고, ATR 스케일링은 "가끔 변동성이 크면 더 크게"라는 보조 역할만
   하도록 의도된 것일 수도 있음.

어느 쪽이든 현재 상태는 "ATR 적응형"이라는 이름이 사실과 다르다 — 재현·근본원인 확정은
완료했고, 수정 여부·방향은 사용자 판단 필요.
