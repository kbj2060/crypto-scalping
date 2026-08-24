# ETH 무기한선물 대안 venue 검토 — 수수료 구조가 닫힌 축을 다시 여는가 (2026-08-23)

## 배경과 결정 문제

[MM 수수료 산술 cheap-gate](experiments/eth_candidate_maker_mm_spread_capture_fee_cheap_gate_20260823.md)가
REJECTED_FEE_STRUCTURE로 닫히며 남긴 재개조건 (1) "maker 실효수수료 ≤ 0.007%/leg가 되는
수수료 현실"이 실제로 존재하는지, 존재하면 어디인지를 검토한다. 더 넓게는: 이 저장소에서
비용 벽으로 죽은 축들(단기 OFI/정보시간 gross ≤1.3bp, MM 스프레드 수확)의 산술이 venue
교체로 뒤집히는지가 결정 문제다. **이건 연구 라인이 아니라 운영 검토다** — OOS 소진 없음,
모델 없음. 계정 개설/자본 이동/관할 판단은 사용자 몫이고, 이 문서는 그 결정에 필요한
수치만 제공한다.

검토일: 2026-08-23 (07:00 UTC 전후). 수수료는 웹 공개자료 검증, 스프레드는 공개 API
실시간 샘플링 실측.

## venue별 실측·검증 결과

### 1. Binance USDT-M (현행 기준선)

- 수수료: VIP0 maker 2.0bp / taker 5.0bp (BNB 할인 1.8/4.5) — 이 계정의 실제 현실임이
  taker 실측 5.03bp/leg로 확인된 상태. VIP1 진입에만 월 선물 $15M, maker 0%는 VIP9($5B+).
- 스프레드: **0.053bp(1틱) 고정** — 파일럿 raw L2 53h 실측, 극단일 p90도 동일.
- 결론: cheap-gate 그대로 — MM 최대관용 net −2.61bp/RT, 단기축 RT 비용 ~10bp. 닫힘.

### 2. Hyperliquid (perp DEX, HYPE 체인)

- 수수료(웹 검증, 2026): **base maker 1.5bp / taker 4.5bp**, 14일 롤링 볼륨 7티어.
  HYPE 스테이킹 할인 5~40%(최대 시 maker ~0.9bp), 추천인 4%. **maker 리베이트는 플랫폼
  전체 메이커 볼륨의 0.5%+ 점유부터**(−0.001%~−0.003%) — 기관 MM 영역, 이 계정 도달 불가.
- 스프레드 실측(API l2Book, 3회 샘플): **0.418bp 고정**(2390.2/2390.3, 틱 0.1달러),
  touch 깊이 bid 342~444 ETH / ask 15~47 ETH — 유동성 실재.
- 펀딩: 1시간 주기(Binance 8h와 다름 — 캐리 동학 차이, 단 ETH 펀딩 자체가 시장 전반
  붕괴 상태라 실익 없음).
- **MM 산술**: 수익상한 = 스프레드 0.418 + 알파상한 ≤1.3(Binance 실측치, 이식성 미검증)
  ≈ 1.7bp/RT vs maker 수수료 3.0bp/RT(base) 또는 1.8bp/RT(최대 스테이킹). **base에서
  −1.3bp/RT, 최대 할인으로도 AS=0 가정에서 겨우 본전** → 실제 AS>0이므로 사실상 불성립.
- 결론: **MM 게이트 재개조건 (1) 미충족**(0.9~1.5bp > 0.7bp). taker 축도 RT ~9.4bp로
  Binance보다 오히려 나쁨. 기각.

### 3. Lighter (zk-rollup perp DEX, Ethereum) — 유일한 조건 충족 venue

- 수수료(웹 검증 + **API 레벨 재확인**): **Standard 계정 maker 0% / taker 0%**, 최소볼륨
  없음, 옵트인 불요. ETH 마켓 메타데이터의 `taker_fee`/`maker_fee` 필드가 실제로 0.0000.
  Premium(HFT용, 볼륨쿼터 해제+레이턴시 300ms→140ms)은 maker 0.4bp/taker 2.8bp(LIT
  스테이킹 시 0.28/1.96) — **standard 제로가 끝나도 폴백 요율이 Binance의 1/5~1/7**.
- ETH 마켓 실측(API, 4회 샘플): 스프레드 **0.50~0.92bp**(평균 ~0.70bp, Binance의 10~17배
  이나 절대값은 여전히 작음), touch 깊이 0.06~70 ETH로 얇고 변동 큼. 일거래량 **$330M**,
  OI 38,648 ETH(~$92M) — 우리 사이즈에는 충분한 유동성.
- 제약: standard 레이트리밋 60 req/min(5분 주기 봇에는 여유, 고밀도 MM 호가에는 빠듯),
  taker 레이턴시 300ms(우리 cadence 무관), USDC 담보 단일.
- **MM 산술**: net/RT = 스프레드(~0.70bp) [+ 알파 이식분] − 2×AS − 0. **AS <
  0.35bp/leg(스프레드만) ~ 1.0bp/leg(알파 이식 가정)이면 양수** — 수수료 항이 사라져
  cheap-gate 재개조건 (1)이 **오늘 기준 충족**. 단 조건 (2) resting AS 실측이 그대로
  본질 관문으로 남고, 얇은 touch 깊이 때문에 Binance보다 AS가 클 개연성도 있다.
- **단기축(taker) 산술**: RT 비용 = 스프레드 ~0.7bp + 임팩트 vs Binance ~10bp — **14배
  절감**. Binance에서 실측된 gross 상한 1.3bp가 Lighter에서도 존재한다면(미검증, 신호
  이식성 별개 문제) 산술상 처음으로 비용 위에 올라선다.

### 4. CEX MM 프로그램 (Bybit/OKX 등)

- Bybit MM 인센티브: 리베이트 최대 −1.5bp이나 **기관 심사제**(institutional_services
  메일 지원, 볼륨 실적+스프레드/업타임 의무). OKX 최대 −0.3bp 동류. 업계 일반: 음수
  수수료는 **월 $100M+ 거래 기관** 대상.
- 결론: Binance VIP와 동일 — 이 계정 규모에서 도달 불가. 기각.

## 종합 — venue별 재개 여부

| venue | 도달가능 maker/taker | 스프레드(실측) | MM 재개조건(1) | 단기축 RT비용 |
|---|---|---|---|---|
| Binance VIP0 | 2.0 / 5.0bp | 0.053bp | ✗ (2.0 > 0.7) | ~10bp |
| Hyperliquid base~스테이킹 | 0.9~1.5 / 2.7~4.5bp | 0.418bp | ✗ (0.9 > 재산정 breakeven ~0.86*) | ~6~9bp |
| **Lighter standard** | **0 / 0bp** | **0.50~0.92bp** | **✓ (0 < 0.7)** | **~0.7bp+임팩트** |
| CEX MM 프로그램 | 기관 전용 | — | 도달불가 | — |

\* Hyperliquid breakeven은 자체 스프레드 0.418bp 반영 재산정: (0.418+1.3+0.04)/2 ≈ 0.88bp/leg.
최대 스테이킹 maker 0.9bp가 아슬아슬하게 걸쳐 있으나 AS=0 가정에서의 본전이라 실질 불성립.

## 리스크 (정직성)

1. **제로수수료 지속가능성**: Lighter의 standard 0/0은 포인트·에어드랍 시대의 보조금
   성격(수익원은 Premium 티어+청산수수료+USDC 리저브 수익). LIT 토큰 출시 후 요율표가
   바뀔 수 있다 — 다만 공개된 Premium 요율(0.4/2.8bp)이 폴백으로 명시돼 있어, 최악
   변경 시에도 Binance 대비 우위는 유지될 공산.
2. **얇은 호가 = AS 미지수**: touch 깊이가 Binance의 수십분의 일. 스프레드가 넓다는 건
   MM 수익원이 커진다는 뜻이지만 역선택도 같이 커질 수 있다. **자본 이동 전 resting AS
   실측이 필수 관문**(MM 게이트 재개조건 (2) 그대로).
3. **신생 구조 리스크**: 가동 수개월의 zk-rollup(시퀀서 중앙화, 크래시 구간 청산엔진
   미검증), 브리지/출금 지연. 실주문 자본은 이 리스크에 노출된다.
4. **신호 이식성**: 이 저장소의 모든 실측(스프레드 0.053bp, gross 상한 1.3bp, AS 참조점)은
   Binance 마이크로구조에서 나왔다. Lighter의 체결·호가 동학은 별도 수집·재측정 대상이며
   Binance 데이터로 대용할 수 없다.
5. **관할/계정**: 리뷰 소스상 No-KYC로 분류되나, 해외 파생상품 접근의 규제·세무 판단은
   전적으로 사용자 결정 영역이다. 이 문서는 그 판단을 하지 않는다.

## 무엇이 재개되지 않나

- **방향 알파 부재**는 비용 문제가 아니다 — venue를 바꿔도 40+ 라벨/8개 아키텍처 축의
  기각은 그대로다.
- **RDE 해당 정책**: OOS gross 자체가 음수(−4.15~−4.38%)였으므로 수수료 절감으로 부활하지
  않는다(아키텍처는 별개로 유효).
- **펀딩 캐리**: ETH 펀딩 붕괴는 시장 전반 현상 — venue 무관.

## 권고 및 다음 단계

**권고: Lighter를 1순위 후보로, 단 자본 이동 전 "measure-first" 순서를 지킬 것.**

1. **[엔지니어링, 승인 시 착수] Lighter ETH 섀도우 수집기**: 기존
   `maker_fill_shadow_worker` 패턴을 Lighter WS/API로 이식 + **체결 후 N-bar 드리프트
   기록 추가**(MM 게이트가 명시한 resting AS 실측 경로). 실주문·자본 없이 공개 데이터만.
   2~4주 수집 후 AS ≤ 0.35~1.0bp/leg 여부로 MM 게이트 재개조건 (2) 판정.
2. **동시에 스프레드/깊이 장기 샘플링**: 오늘 4회 샘플은 일요일 저변동 시간대 스냅샷 —
   파일럿 창 편향(H1 변동성 28퍼센타일이었던 전례)을 반복하지 않으려면 변동성 구간 포함
   연속 수집 필요.
3. **계정/자본/관할 결정은 사용자**: 1~2의 실측이 양성으로 나온 뒤에만 의미 있는 결정.
4. Binance 스택(라이브 봇, peg-maker 섀도우, 9월 체크포인트)은 이 검토와 무관하게 유지.

## ⚠️ 결정 반영 (같은 날 후속) — "Binance를 사용해야 해" + USDC-M 프로모션 발견

사용자가 **Binance 고정**을 확정 지시 — 위 Lighter 권고(1~3번)는 실행하지 않는다(기록으로만
보존). 이에 따라 Binance 내부 수수료 레버를 재조사한 결과 **판도를 바꾸는 발견**:

### Binance USDC-M 무기한 프로모션 — 전 등급 maker 0%

- 공식 공지(2025-12-10 시작, "until further notice", 2026-03까지 연장 이력 확인):
  **모든 USDC-마진 무기한에 대해 Regular User 포함 전 VIP 등급 maker 0.0000%**,
  taker Regular 4.0bp(BNB 할인 3.6bp — USDT-M의 5.0bp보다도 낮음).
- 종료 공지 검색: 2026-08-23 기준 **종료 공지 없음** — 유효할 공산이 크나, 계정 단위
  확정은 실패(레포 `.env`의 키가 commissionRate 조회에서 401 — 스테일 키 또는 IP 제한
  추정, 라이브 봇의 실제 키 소스는 보안상 추적 중단). **사용자가 로그인 후 수수료
  페이지에서 USDC-M maker 0% 표기를 확인하는 것이 확정 절차**(10초).
- **ETHUSDC 유동성 실측**(공개 API, 3샘플 + 24h 티커): 스프레드 **0.042bp(1틱,
  ETHUSDT와 동일)**, 일거래량 **$2.29B**(ETHUSDT $8.5B의 27%), touch 깊이 bid 12~16 /
  ask 0.8~2.7 ETH(USDT 페어의 ~1/8이나 이 계정 사이즈에는 충분).

### 함의 — Binance를 떠나지 않고 재개조건 (1) 충족 (프로모션 확정 시)

| 축 | 기존(ETHUSDT VIP0) | ETHUSDC 프로모션 시 |
|---|---|---|
| MM 재개조건(1) maker ≤0.7bp | ✗ (2.0bp) | **✓ (0bp)** — 조건(2) AS 실측만 남음 |
| peg-maker 집행비용/leg | 3.1~4.0bp (수수료 2 + 드리프트) | **~1.1~2.0bp (드리프트만)** — 모든 메이커 집행 전략의 비용 거의 반감 |
| taker/leg | 5.03bp 실측 | 4.0bp (BNB 3.6bp) |
| 단기축(왕복 ≤1.3bp 필요) | ✗ | ✗ (taker 기준 여전히 미달) |

방향알파 부재·RDE 정책·펀딩캐리는 여전히 안 열림(비용 문제 아님). ETHUSDC의 얇은 깊이
때문에 ETHUSDT 실측치(체결률, 드리프트, AS)를 그대로 이식하면 안 되고 재측정 필요.

### 수정된 권고 (Binance 고정 하)

1. **[사용자, 10초] 프로모션 계정 적용 확인**: 로그인 후 수수료 페이지에서 USDC-M
   maker 0% 확인 (+ 선택: BNB 수수료 할인 활성화 — taker 실측 5.03bp는 미적용 상태,
   무위험 10% 절감).
2. **[엔지니어링, 승인 시] ETHUSDC 섀도우 측정**: 기존 maker_fill_shadow 패턴을
   ETHUSDC로 이식 + 체결 후 N-bar 드리프트 기록(resting AS 실측 = MM 게이트 재개조건
   (2)) — 공개 WS 데이터만, 키·실주문 불요. 2~4주 후 AS ≤ 0.35~0.7bp/leg 여부 판정.
3. 프로모션 종료 리스크 상시 인지("until further notice") — 종료 시 이 절 전체 무효,
   Binance 표준 요율로 회귀.

## 출처

- Hyperliquid 수수료: hyperliquidguide.com/guides/fees, hiperwire.io, datawallet.com (base
  0.015%/0.045%, 리베이트 3티어 = 플랫폼 메이커 볼륨 점유율 0.5/1.5/3.0%+)
- Lighter 수수료/티어: docs.lighter.xyz/trading/trading-fees, perpdexguide.com,
  apidocs.lighter.xyz (Standard 0/0, Premium 0.004%/0.028%, 레이트리밋 60 req/min)
- Bybit/OKX MM: bybit.com 헬프센터 MM Incentive Program(기관 심사제, 리베이트 −0.015%까지),
  업계 관례 월 $100M+
- 스프레드 실측: Hyperliquid `POST api.hyperliquid.xyz/info {"type":"l2Book","coin":"ETH"}`
  3회, Lighter `GET mainnet.zklighter.elliot.ai/api/v1/orderBookOrders?market_id=0` 4회 +
  `orderBookDetails`(ETH=market_id 0, 수수료 필드 0.0000 확인), 2026-08-23 07:00 UTC 전후
