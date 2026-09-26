---
name: 이더리움 트레이딩 대시보드
description: 매매 중 띄워 두는 개인 ETH 선물 작업대 — 수급·청산·계좌·주문을 한 화면에
colors:
  night-desk: "#161a22"
  desk-panel: "rgba(35, 41, 50, 0.82)"
  desk-panel-raised: "rgba(46, 52, 62, 0.92)"
  chart-well: "#171b23"
  glass-film: "rgba(255, 255, 255, 0.05)"
  glass-film-strong: "rgba(255, 255, 255, 0.08)"
  lamp-text: "#eef0f6"
  dim-text: "#a6acc1"
  ink: "#f5f6fb"
  bid-green: "#5dbf83"
  ask-red: "#ef9387"
  alarm-amber: "#e99c57"
  steel-accent: "#dfe4ee"
  graphite-neutral: "#cbd1e3"
  turnover-blue: "#5aa9f5"
  book-depth-blue: "#7dd3fc"
  on-fill: "#12161d"
  line: "rgba(203, 209, 227, 0.18)"
  soft-line: "rgba(203, 209, 227, 0.095)"
  hover-line: "rgba(203, 209, 227, 0.38)"
  light-paper: "#dfe4ee"
  light-glass-panel: "rgba(255, 255, 255, 0.55)"
  light-text: "#16181d"
  light-dim-text: "#555c6c"
  light-bid-green: "#136436"
  light-ask-red: "#9c3227"
  light-alarm-amber: "#7a4809"
  light-steel-accent: "#2b3444"
typography:
  display:
    fontFamily: "Pretendard Variable, Pretendard, -apple-system, Noto Sans KR, sans-serif"
    fontSize: "clamp(28px, 4vw, 38px)"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "-0.02em"
    fontFeature: "tnum"
  clock:
    fontFamily: "Space Grotesk, Pretendard Variable, Noto Sans KR, sans-serif"
    fontSize: "12px"
    fontWeight: 500
    lineHeight: 1
    letterSpacing: "0.04em"
    fontFeature: "tnum"
  headline:
    fontFamily: "Space Grotesk, Pretendard Variable, Noto Sans KR, sans-serif"
    fontSize: "28px"
    fontWeight: 500
    lineHeight: 1.1
  title:
    fontFamily: "Space Grotesk, Pretendard Variable, Noto Sans KR, sans-serif"
    fontSize: "17px"
    fontWeight: 500
    letterSpacing: "-0.005em"
  figure:
    fontFamily: "Pretendard Variable, Pretendard, -apple-system, Noto Sans KR, sans-serif"
    fontSize: "40px"
    fontWeight: 600
    lineHeight: 1.02
    letterSpacing: "-0.025em"
  tile-value:
    fontFamily: "Pretendard Variable, Pretendard, -apple-system, Noto Sans KR, sans-serif"
    fontSize: "21px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "-0.015em"
  body:
    fontFamily: "Pretendard Variable, Pretendard, -apple-system, Noto Sans KR, sans-serif"
    fontSize: "16px"
    fontWeight: 400
  note:
    fontFamily: "Pretendard Variable, Pretendard, -apple-system, Noto Sans KR, sans-serif"
    fontSize: "11px"
    fontWeight: 600
    lineHeight: 1.55
  label:
    fontFamily: "Pretendard Variable, Pretendard, -apple-system, Noto Sans KR, sans-serif"
    fontSize: "11px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.12em"
  data:
    fontFamily: "JetBrains Mono, Pretendard Variable, ui-monospace, monospace"
    fontSize: "11.5px"
    fontWeight: 600
    lineHeight: 1
    fontFeature: "tnum"
rounded:
  card: "18px"
  card-light: "22px"
  tile: "12px"
  tile-light: "14px"
  control: "10px"
  note: "8px"
  chip: "7px"
  rail: "3px"
  pill: "999px"
spacing:
  hair: "4px"
  tight: "6px"
  base: "8px"
  cozy: "10px"
  group: "12px"
  card-gap: "16px"
components:
  panel:
    backgroundColor: "{colors.desk-panel}"
    rounded: "{rounded.card}"
    padding: "16px"
  tile:
    backgroundColor: "{colors.glass-film}"
    typography: "{typography.tile-value}"
    rounded: "{rounded.tile}"
    padding: "11px 12px 12px"
  chip:
    backgroundColor: "{colors.glass-film}"
    textColor: "{colors.dim-text}"
    typography: "{typography.data}"
    rounded: "{rounded.chip}"
    padding: "4px 9px"
    height: "24px"
  chip-on:
    textColor: "{colors.alarm-amber}"
    rounded: "{rounded.chip}"
  button-long:
    textColor: "{colors.bid-green}"
    rounded: "{rounded.pill}"
    padding: "10px 16px"
  button-short:
    textColor: "{colors.ask-red}"
    rounded: "{rounded.pill}"
    padding: "10px 16px"
  page-tab:
    textColor: "{colors.dim-text}"
    rounded: "{rounded.pill}"
    padding: "7px 11px"
  page-tab-active:
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
  badge-good:
    backgroundColor: "{colors.bid-green}"
    textColor: "{colors.on-fill}"
    rounded: "{rounded.pill}"
    padding: "7px 12px"
  note:
    backgroundColor: "{colors.glass-film}"
    textColor: "{colors.dim-text}"
    typography: "{typography.note}"
    rounded: "{rounded.note}"
    padding: "8px 10px"
  note-bad:
    textColor: "{colors.ask-red}"
    rounded: "{rounded.note}"
---

# Design System: 이더리움 트레이딩 대시보드

## Overview

**Creative North Star: "야간 트레이딩 데스크"**

불을 거의 끈 책상 위에 조용한 계기들이 놓여 있다. 표면은 어둡고 납작하며, 눈이 피곤하지 않은 무채색이 화면의 대부분을 덮는다. 색은 **장식이 아니라 신호**다 — 초록은 매수·상승·지지, 빨강은 매도·하락·저항, 주황은 경보. 그 밖의 모든 것은 한 가지 무채색 재료에서 밝기만 달리해 나온다. 그래서 색이 보이는 순간 그건 읽어야 할 무엇이다.

밀도가 장식을 이긴다. 한 화면에 수급 차트·풋프린트·청산·사분면·계좌·주문이 함께 산다. 숫자는 또렷하고(데이터는 등폭 숫자, 큰 단독 숫자는 비례 숫자), 설명은 작고 흐리다. 부품은 **조용하고 정밀하다** — 얇은 테두리와 옅은 필름 배경, 눌러야 하는 것은 분명하지만 도드라지지 않는다. 주문 버튼조차 채운 덩어리가 아니라 방향색 테두리와 글자다.

라이트 테마는 같은 책상에 불을 켠 것이 아니라 **재질이 바뀐다**: 애플 «Liquid Glass» — 흐림 유리 패널 뒤로 초록·빨강·주황의 부드러운 색 덩어리가 비친다. 다크에서는 유리 효과를 끈다(흐림 없음). 두 테마는 같은 토큰 이름을 공유하고 값만 갈아 끼운다.

**Key Characteristics:**
- 세 방향색(초록·빨강·주황) + 한 무채색 재료. 넷째 색(파랑)은 거래대금 선 하나 전용.
- 투명도는 `color-mix`, 명암은 `--lift`/`--shadow`/`--specular` — 흰색·검정·rgba 숫자를 직접 쓰지 않는다.
- 부품은 얇은 테두리 + 필름 배경. 채운 면은 상태 배지뿐.
- 데이터 숫자는 JetBrains Mono 등폭, 제목은 Space Grotesk, 본문·큰 숫자는 Pretendard.
- 다크 = 납작한 야간 표면, 라이트 = Liquid Glass.

## Colors

어두운 흑연 바탕 위에 방향을 뜻하는 세 가지 색만 켜지는 팔레트.

### Primary
- **매수 초록 (Bid Green)** (`bid-green`): 상승·매수·지지·롱·이익. 롱 진입/청산 버튼 테두리와 글자, 지지선, 이익 막대. 라이트는 `light-bid-green`.
- **매도 빨강 (Ask Red)** (`ask-red`): 하락·매도·저항·숏·손실·오류. 숏 버튼, 저항선, 손실 막대, «진입 불가» 같은 실패 문구. 라이트는 `light-ask-red`.

### Secondary
- **경보 주황 (Alarm Amber)** (`alarm-amber`): 주의·경보·선택된 칩·OI. 방향이 아니라 «지금 눈여겨봐라». 라이트는 `light-alarm-amber`.

### Tertiary
- **거래대금 파랑 (Turnover Blue)** (`turnover-blue`): **사분면 행의 거래대금 선 하나에만** 쓴다. 방향이 없는 크기라 방향색을 줄 수 없어서 들어온 예외다.
- **호가 깊이 하늘 (Book Depth Blue)** (`book-depth-blue`, `--book-depth`): 호가 프로파일 왼쪽(쌓인 호가 깊이) 막대 전용. 같은 이유(방향 없는 «양»)의 예외다.
- **청산 밀도 쿨 램프** (app.js `DENSITY_STOPS_DARK/LIGHT`): 밀도는 방향이 없어 초록·빨강·주황을 쓰면 캔들 방향으로 오독된다 — 단색 쿨 램프로 밝기만 올린다(배경 대비가 밀도에 따라 단조 증가하도록 테마별로 따로 잰 값). 비평(2026-09-26)이 «파랑 3곳»을 짚었지만 이 셋은 **허가된 예외**다.

### Neutral
- **야간 책상 (Night Desk)** (`night-desk`): 페이지 바닥. 명도 계단의 맨 아래.
- **책상 패널 (Desk Panel)** (`desk-panel`) · **올린 패널** (`desk-panel-raised`): 카드와 떠 있는 표면. 바닥에서 한 층씩(명도 ~6) 올라간다.
- **차트 우물 (Chart Well)** (`chart-well`): 차트 배경. 바닥보다 어두워지면 안 된다(차트가 가라앉는다).
- **유리 필름 (Glass Film)** (`glass-film`, `glass-film-strong`): 칩·타일·안내 상자의 옅은 배경.
- **등불 글자 (Lamp Text)** (`lamp-text`) · **흐린 글자 (Dim Text)** (`dim-text`, 최악 4.60:1): 본문과 보조 문구. 잉크(`ink`)는 가장 밝은 강조 글자.
- **강철 강조 (Steel Accent)** (`steel-accent`): 활성 탭·현재가·슬라이더 손잡이. 색이 아니라 **밝기**로 튄다.
- **흑연 (Graphite Neutral)** (`graphite-neutral`): 유일한 무채색 재료 — 선(`line`/`soft-line`/`hover-line`)과 틴트가 여기서 나온다.
- **채운 면 위 글자** (`on-fill`): 방향색을 채운 배지 위의 글자.

### Named Rules
**The Three Signals Rule.** 색은 초록·빨강·주황 셋뿐이다. 새 색이 필요하면 «셋 중 무엇인가»를 먼저 답한다 — 답이 없으면 그건 무채색이다.

**The Mix-Don't-Hardcode Rule.** 투명한 방향색은 `color-mix(in srgb, var(--bad) 22%, transparent)` 로만 만든다. `rgba(207,106,92,.22)` 처럼 숫자를 박으면 토큰을 고쳐도 따라오지 않는다(정리 전 하드코딩 사본 134회가 있었다).

**The Signed Lift Rule.** 띄우기·그늘은 `rgb(var(--lift) / a)`, `rgb(var(--shadow) / a)` 로. 라이트에서는 «띄우기»의 부호가 뒤집힌다(흰 유리 위의 흰 오버레이는 안 보인다).

## Typography

**Display Font:** Space Grotesk (with Pretendard Variable) — 제목·시계. 상단 현재가는 Pretendard.
**Body Font:** Pretendard Variable (with Pretendard, -apple-system, Noto Sans KR)
**Label/Mono Font:** JetBrains Mono (with Pretendard Variable, ui-monospace)

**Character:** 기하학적인 Space Grotesk 가 제목과 시계를 맡아 계기판의 표지판처럼 서고, 한글 가독성이 좋은 Pretendard 가 본문과 큰 숫자를, JetBrains Mono 가 줄을 맞춰야 하는 데이터 숫자를 맡는다.

### Hierarchy
- **Display** (600, clamp(28px, 4vw, 38px), 1, Pretendard): 상단 **현재가** + 5분봉 시가 대비. 화면에서 가장 큰 숫자는 지금 가격이다(2026-09-26 — 예전엔 벽시계였다). 1초마다 바뀌므로 등폭 숫자.
- **Clock** (500, 12px, Space Grotesk, 흐린 글자): 현재가 아래의 작은 시계.
- **Headline** (500, 28px, 1.1): 운영 화면 제목.
- **Title** (500, 17px): 카드 제목(«내 계좌», «진단»).
- **Figure** (600, 40px, 1.02, -0.025em): 순자산 같은 큰 단독 숫자. 디스플레이 서체도 등폭 숫자도 쓰지 않는다 — 비례 숫자.
- **Tile value** (700, 21px, 1): 계좌 타일 값(청산까지·증거금·노출).
- 최소 글자 크기는 **11px**다(10.5px 은 2026-09-26 전부 11px 로 올렸다).
- **Body** (400, 16px): 페이지 기본.
- **Note** (600, 11px, 1.55): 안내·경고 상자(entry-note), 차트 주석.
- **Label** (700, 11px, 0.12em, 대문자): 눈썹 이름표(«순자산»). 타일 이름은 600 11px.
- **Data** (600, 11.5px, 등폭): 칩·툴팁·가격·수량. 표 숫자는 항상 등폭.

### Named Rules
**The Tabular Data Rule.** 줄을 맞춰 비교하는 숫자(가격·수량·칩·툴팁)는 등폭(JetBrains Mono 또는 tabular-nums). 홀로 선 큰 숫자는 비례 숫자 — 등폭은 큰 크기에서 헐거워 보인다.

**The Small Voice Rule.** 설명은 11px 흐린 글자다. 크기가 아니라 **색(방향색)과 위치**가 중요도를 말한다.

## Layout

카드 격자 위에 차트가 큰 비중을 차지하는 한 페이지. 카드 사이는 `card-gap`(16px). 카드 안은 8·10·12px 의 촘촘한 간격이 주를 이룬다(가장 많이 쓰는 gap 8px). 계좌 카드는 넓은 화면에서 «청산 | 진입» 두 레인이 나란하고, **720px 이하**에서 한 열로 쌓인다(분기점 대부분이 720px; 보조 960·520·420px). 모바일 좌우 여백은 16px, 페이지 가로 스크롤 없음.

떠다니는 주문 버튼(#ofab)은 화면 위 고정층(z 900 — 차트 툴팁 1000 아래)이며, 펼치면 계좌 카드의 조작부를 **통째로 옮겨** 온다(복제하지 않는다).

## Elevation & Depth

다크는 **납작한 명도 계단**이다: 바닥 → 패널 → 올린 패널이 명도 ~6씩 올라가고, 카드에는 넓게 퍼지는 부드러운 그림자 하나와 윗면 1px 광택선(`--specular`)만 있다. 흐림 효과는 없다. 라이트는 **Liquid Glass**: 패널 알파를 낮추고 `blur(34px) saturate(200%)` 로 뒤의 색 덩어리를 굴절시키며, 그림자는 검정이 아니라 푸른 회색(`--shadow: 74 85 104`)이다. 호버로 카드를 띄우지 않는다.

### Shadow Vocabulary
- **카드 주변광** (`box-shadow: 0 20px 54px rgb(var(--shadow) / 0.28), inset 0 1px 0 var(--specular)`): 모든 패널.
- **떠 있는 조작부** (`box-shadow: 0 1px 2px rgb(var(--shadow) / 0.25), 0 12px 32px rgb(var(--shadow) / 0.36)`): 떠다니는 주문 버튼과 그 패널.
- **툴팁** (`box-shadow: 0 6px 18px rgb(var(--shadow) / 0.35)`): 차트·막대 툴팁.

### Named Rules
**The Still Card Rule.** 카드는 호버로 움직이지 않는다. 마우스가 지나가는 자리마다 카드가 뜨면 읽는 도중 화면이 흔들린다.

**The Glass Needs Something Behind Rule.** 라이트의 유리는 뒤에 색이 있어야 유리로 보인다 — 흰 바탕 위 blur 는 아무 일도 안 한다. 배경의 부드러운 방향색 메시가 그 «뒤»다.

## Shapes

부드럽고 연속적인 모서리. 카드는 크게(18px, 라이트 22px), 타일·패널 속 상자는 중간(12px, 라이트 14px), 버튼·안내 상자는 작게(8~10px), 칩은 7px, 탭·배지·주문 버튼·알약은 완전한 알약(999px). 레일(얇은 게이지 선)은 3px. 테두리는 거의 항상 1px 흑연 선(`line`/`soft-line`)이고, 빈 상태와 «가정» 값은 **점선** 테두리로 «아직 사실이 아님»을 말한다.

## Components

### Buttons
- **Shape:** 완전한 알약 (999px).
- **Long / Short:** 채우지 않는다 — 방향색 글자 + 방향색 45% 섞은 테두리(`color-mix`), 옅은 필름 배경. 호버에서 테두리가 방향색 100% 로 짙어진다. 12px 800 Pretendard, 10px 16px 안쪽 여백.
- **길게 누르기(hold-fire):** 실주문 버튼은 0.4초 누르는 동안 안쪽 채움 막대가 차오른다 — 클릭 한 번으로 돈이 나가지 않게.
- **Neutral(notify-btn):** 흑연 테두리 + `rgb(var(--lift) / .04)` 배경.
- **Focus:** 2px 강철 강조 윤곽선, 2px 띄움.

### Chips
- **Style:** 옅은 필름 배경, 투명 테두리, 흐린 등폭 글자(11.5px 600), 7px 모서리, 최소 높이 24px(WCAG 2.5.8).
- **State:** 선택되면 경보 주황 13% 섞은 배경 + 주황 글자 + 주황 45% 테두리. 입력이 잠기면 칩 묶음째 45% 투명·누를 수 없음.
- 칩은 숨긴 range 입력의 **얼굴**이다 — 떠다니는 패널에서는 같은 입력이 슬라이더로 드러난다(값이 어긋날 수 없다).

### Cards / Containers
- **Corner Style:** 18px (라이트 22px).
- **Background:** 패널 + 위에서 아래로 옅어지는 필름 그라디언트.
- **Shadow Strategy:** 카드 주변광(Elevation).
- **Border:** 1px `glass-edge`.
- **Internal Padding:** 16px.

### Tiles (계좌 타일)
- 필름 배경 + soft-line 테두리 + 12px 모서리. 이름(11px 600 흐린) → 값(21px 700, 방향색) → 3px 레일. 미리보기 값은 점선 테두리와 주황 «→ 뒤» 표기로 «가정»임을 말한다.

### Notes (안내·경고 상자)
- 11px 600, 8px 모서리, 필름 배경. 기본은 흐린 글자, 실패는 빨강 글자 + 빨강 35% 테두리, 진행 중 실주문은 주황. **실패는 절대 조용하지 않다.**

### Navigation
- 상단 페이지 탭: 알약, 흐린 11px 글자, 흑연 테두리. 활성 탭은 강철 강조 16% 배경 + 70% 테두리 + 잉크 글자.

### Tooltips
- 올린 패널 배경, 흑연 테두리, 8px 모서리, 11px 등폭 1.55 줄간격. 첫 줄이 결론(«9월 25일 $74.08 벌었음», 방향색 굵게), 나머지는 세부. 커서 즉시 뜬다(네이티브 title 1초 지연 금지). 키보드 포커스로도 같은 내용이 뜬다.
- ⚠️예외: 차트 호버 툴팁(`.chart-tooltip`)은 아직 6px 모서리·하드코딩 색(`rgba(45,51,61,.96)`, `#fff`)으로 이 규칙 밖이다 — 고칠 때 위 계좌 툴팁에 맞춘다.

### Floating Order Button (시그니처)
- 알약 막대: SVG 점 여섯 손잡이 + 포지션 글자(등폭 13px, 롱 초록·숏 빨강). 손잡이로만 끈다(길게 누르기 버튼 위에서 끌다 주문이 나가지 않게). 거친 포인터에서 44px.
- 패널: 야간 책상 배경, 12px(라이트 14px) 모서리, 버튼과 게이지만 — 설명 글자는 걷고 오류는 남긴다.

## Do's and Don'ts

### Do:
- **Do** 새 색이 필요하면 초록·빨강·주황 중 무엇인지 먼저 정하고, 투명도는 `color-mix(in srgb, var(--X) N%, transparent)` 로 만든다.
- **Do** 비교하는 숫자는 등폭(JetBrains Mono / tabular-nums), 큰 단독 숫자는 비례 숫자.
- **Do** 모서리·간격은 토큰(`--radius`, `--radius-sm`, `--card-gap`)을 쓴다 — 라이트 테마에서 값이 바뀐다.
- **Do** 실패·차단·지연은 빨강 안내 상자로 크게 말한다. 추정값·가정값은 점선 테두리로 사실과 구분한다.
- **Do** 새 버튼에는 `font-family: inherit` 을 **축약형 밖에서** 준다(버튼은 페이지 글꼴을 물려받지 않아 한글이 □ 가 된다).
- **Do** 대비는 가장 어두운 실제 바탕에서 잰다: 보조 글자 4.5:1 이상.

### Don't:
- **Don't** 흰색·검정·`rgba(…)` 숫자를 직접 쓴다 — `--lift`/`--shadow`/`--specular`/토큰으로.
- **Don't** 거래대금 파랑(`--turnover`)을 다른 곳에 쓴다.
- **Don't** 카드를 호버로 띄우거나 움직인다.
- **Don't** 주문 버튼을 방향색으로 채운 덩어리로 만든다 — 테두리와 글자가 방향을 말한다.
- **Don't** 다크 테마에 blur 를 켠다(현행 다크는 납작한 표면이다).
- **Don't** 방향색 글자·점에 발광(`text-shadow`/`box-shadow: 0 0 Npx`)을 넣는다 — 2026-09-26 83곳에서 걷었다. 색만으로 방향을 말한다.
- **Don't** 평상 상태(꺼진 세션·상한에 걸린 진입)를 빨강으로 칠한다 — 빨강은 실패·위험에만. 상한 막힘은 흐린 문구 + 흐린 버튼.
- **Don't** 점자·이모지 글자로 아이콘을 대신한다 — 인라인 SVG, `currentColor`.
- **Don't** `.top` 같은 일반 클래스명을 새로 만든다 — 전역 규칙과 충돌한다(접두사를 붙인다).
