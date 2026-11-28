# rtui 카드 목록 — 데스크톱 1920 (2026-11-29)

원본 = `dashboard/live/app.js`·`liq_tab.js`·`index.html`·`styles.css`(origin/main 66882817). 1920×1080 가로 = `html.fit1`: 카드가 위에서부터 Footprint · Macro Trend · News · Option · Account(각 창 높이), 풋프린트 차트 오른쪽 칸(`--fp-split .25`, 약 460px) = `#mcBody.mc-cmp`.
옮김: ✅ = rtui 에 있음 · 🟡 = 일부 · ⬜ = 없음 · ⛔ = 주문 경로·입력 · 보류. 크기 = 1920 화면에서 대략(px).

| 칸 | 원본 render | 데이터 주소 | 옮김 | 크기 |
|---|---|---|---|---|
| 맨 위 코인 탭 | renderSnapshotAssetTabs · renderAssetPop | (상태) | ⛔ 입력(코인 전환) | 450×40 |
| 맨 위 코인 이름 | renderSnapshotAssetTabs(#fpCoinName) | (상태) | ✅ `?card=head` (11-29 · 글만 · 탭 누르기 없음) | 120×30 |
| 보유 종목 손익 전광판 | renderPnlTicker | /api/binance-account · /rt 가격 | ✅ `?card=acct` | 600×28 |
| 발표 배지 · 급변 칩 · 세션 | renderMacroEventAlert · macroSpikeChips · buildSessionHtml | /api/session-alerts · /api/macro-spikes · 세션 = /api/state(원본 SSE) | ✅ `?card=head` (11-29) | 300×24 |
| 테마 · 알림 단추 · 알림 센터 | renderNotifyCenter | /api/notify-center · /api/push/* | ⛔ 입력 | 단추 |
| 카드 머리 보기·봉 단추 | renderChartWindowTabs | (상태) | 🟡 rtui 는 5m·30m·1h 만 | 400×34 |
| 머리 포지션 요약 | renderFpPos | /api/binance-account · 허브 가격 | ✅ `?card=head` (11-29 · 표시만) | 260×50 |
| 머리 Macro 요약 | mtHeadHtml(#fpRates) | /api/macro-trend · /api/macro-calendar(다음 발표) | ✅ `?card=mth` | 420×38 |
| 차트 위 미니맵 | renderChartPan | /api/market-history | ➖ 1920 화면엔 없음(넓은 화면 = .inrow → 7591 숨김 · 휴대폰도 숨김) -- 옮길 것 없음 | 480×34 |
| 선 주문 단추 · 대기 선 · 손절 선 | renderLineTool · renderArmedLines · renderSlCross | /api/manual-entry/* · /api/manual-bracket/* | ⛔ 주문 경로 · 보류 | 단추·선 |
| 지표 단추·팝업 | renderMarketCtx(#fpLineSwitch) | (상태) | ⛔ 입력 | 단추 |
| 풋프린트 차트(칸·캔들·델타·POC·축·알약) | renderCandleSvg | /api/footprint · /api/market-history · /api/footprint-week · @trade | ✅ 기본 | 1420×880 |
| 차트 레인 · 프로파일 · 스택 · 미완 경매 · 호버 · 청산 동그라미 | renderCandleSvg | 같은 것 · /api/liquidation-5m-* | ✅ | (위 안) |
| 차트 겹침 선(VWAP·EMA·스윙·OB·FVG 등) · 지표 표식 | renderCandleSvg · renderDirMarks · renderObFvg | 5분봉(/api/market-history · 1주일) · /api/swing-lines 등 -- 표식은 dirCalc 가 캔들로 계산(/api/dir-signals 안 씀) | ⬜ · 차트 카드 몫: «지표 방향» 판 줄(fpBoard.rows·cond = SMA50 일봉 추세 · 스윙 구조 돌파·«넘으면/깨면» · OB·FVG · 뉴욕 개장 돌파·CCI·RSI2·RSI 다이버전스)도 여기서 채운다 → dirb::Row·Cond 로 넘김 | (위 안) |
| 차트 «레벨 5분» 표식 | renderSrLean | /api/sr-lean | ⬜ | (위 안) |
| 왼쪽 위 «지표 방향» 판 | renderDirBoard · dirReadNow · dirReadH · dirTri | /api/situation(read · ETH 만 · 1초) + 판 줄은 차트 표식이 채움 | 🟡 `?card=dirb` (11-29 넷째 · 그리기 전부 · 줄 채우기 = 차트 카드 몫이라 실자료는 «지금 읽기»만) | 268×(줄 수) |
| 주문 상태 줄 | renderManualWait(#ordStat) | /api/manual-entry/status | ⛔ 주문 경로 · 보류 | 줄 |
| 오른쪽 칸 ①② 게이지 | renderMarketCtx(.mc-strip-wall) | /api/market-context | ✅ `?card=mc` | 460×128 |
| 오른쪽 칸 1초 수급 | renderSupply1s | /rt supply | ✅ `?card=s1` | 460×330 |
| ⑤ 종합 칩(ETH) | fcRowHtml(fcSum) | /api/dir4h · /api/trend · /api/open-rev | ✅ `?card=dirs` (11-29 둘째) | 460×22 |
| ⑤ 지금 추세(+7일 추세) | fcTrendCell · f1Html | /api/trend | ✅ `?card=dirs` (11-29 둘째) | 460×46 |
| ⑤ 15분 예측 | fcRowHtml sCell · prHit | /api/dir-short · /api/pred-record | ✅ `?card=dirs` | 460×42 |
| ⑤ 레벨 5분(ETH) | srLeanCell | /api/sr-lean | ✅ `?card=srl` · `?card=dirs` 안에도 | 460×42 |
| ⑤ 1시간 예측 | fcRowHtml sCell · prHit | /api/dir-short · /api/pred-record | ✅ `?card=dirs` | 460×42 |
| ⑤ 4h 예측 | fcRowHtml | /api/dir4h · /api/pred-record (/api/vol-forecast 는 툴팁에만 쓰여 안 부름) | ✅ `?card=dirs` (11-29 둘째) | 460×42 |
| ⑤ 횡보·추세 | fcRowHtml rbCell | /api/regime-band · /api/pred-record | ✅ `?card=dirs` (11-29 둘째) | 460×30 |
| ⑤ 위험 레이더(15분 폭 · 앞 1시간) | rrCells | /api/risk-radar · 허브 가격 | ✅ `?card=dirs` (11-29 둘째) | 460×88 |
| ⑤ 4h 상대 강도 표 | predXsecHtml | /api/pred-record | ✅ `?card=dirs` (11-29 둘째) | 460×122 |
| 보기: 틱 차트 · 체결 버블 · 일봉 풋프린트 | tick_charts · bubble_map.js · fp_daily.js | /api/footprint-week 등 | ⬜ | 차트 자리 |
| «청산» 탭 청산맵 | liq_tab.js CH.ladder | /api/liquidation-map(/tiers) · /api/liq-tab | ✅ `?card=liq` | 900×560 |
| «청산» 탭 남은 매물 일봉별 | liq_tab.js CH.remain | /api/liq-tab | ✅ `?card=rem` | 900×300 |
| «청산» 탭 오늘 누적 · 최대 고통 · 코인별 최대 고통 | CH.today · CH.pain · CH.quad · painKpi · coinTable | /api/liq-tab(all) · /api/liquidation-5m-history | ✅ `?card=liqx`(주기 = liqx1W 등 · 머리 단추·툴팁 빼고) | 3칸 × 340 |
| Macro Trend 카드 | renderMacroTrend · refreshMtToday | /api/macro-trend · /api/macro-today · /api/treasury-futures?h=24 · /api/xyz100 · /api/macro-calendar | ✅ `?card=mt` (본문 #mtBody · 카드 머리 제목 줄은 밖) | 1838×917 |
| News 카드 · ⑤ 다음 24시간 | renderNews · renderMarketCtx(#mcWhen) | /api/news · /api/market-context · /api/gex · /api/macro-calendar | ✅ `?card=news` (본문 #newsBody · 카드 머리 제목·건수 줄은 밖) | 1838×917 |
| Option 카드(코인 ETH·BTC·SOL·XRP) | renderOptions | /api/gex · /api/opt-smile · /api/market-history(실현 감마) · 허브 가격 | 🟡 `?card=opt` (11-29 셋째 · 주식 판 · 사다리 «내 선» 빼고) | 전폭×창 |
| Option 카드 주식(CBOE) 판 · 사다리 «내 선»(청산가·지정가) | renderOptions(o.src) · optEqKpiHtml · optTouchFront | /api/gex · /api/binance-account | ⬜ | (위 안) |
| 알림 센터 «전진 검증 중» 아이스버그 | renderIceChip(#ncIce -- Option 카드 아님) | /api/iceberg | ⬜ | 알림 센터 안 |
| Account 카드(잔고·곡선·순환·통계·달력) | renderSnapshotAccountCore · renderCycleCard | /api/binance-account · /api/cycle-path | ✅ `?card=acc` (머리 아래 본문 · 카드 머리 순자산 줄·요약 알약·강제 조회 단추는 밖) | 1872×934 |
| Account 메모 · 오답노트 | (acctNote·acctWrong) | /api/acct-notes · /api/acct-wrong | ⛔ 입력 | 탭 안 |

남은 칸: ⬜ 5 · ➖ 1(미니맵 -- 1920 에 안 보임) · 🟡 3(봉 단추 · Option -- 주식 판·내 선 빼고 · 지표 방향 판 -- 줄 채우기 빼고) · ⛔ 6(주문 경로·입력 · 보류) · ✅ 24줄. ⑤ 예측 줄 완성(11-29 둘째) = `?card=dirs` 가 fcRowHtml 전체(rtui/src/fc.rs).
11-29 둘째 정답지: compare/fc_js.mjs(원본 fcRowHtml 을 가짜 전역에서 · 지어낸 입력 8장) → compare/fc_layout.py(원본 styles.css · 헤드리스 크로미움 1920 · 폭 404 에서 글·상자·선 자리와 색을 잼) → `fc::t::js_parity`(645개 전부 하나씩 짝 · 높이 같음). 안 옮김: 칸 툴팁 · «?» 설명 팝업(단추 모양만).
11-29 둘째 고침: 오른쪽이 글만(«쌓는 중»)인 줄 높이 30 → 27(`srl::height` -- 원본 실측 27).
첫째(15분·1시간 예측)에 고침: 같은 칸 그리기(srl.rs scene)의 가로 자리 -- 넓은 화면 `#mcBody.mc-cmp .fc-c { gap: 2px }`·`.fc-t > .d4g` 가로 여백 0 을 놓쳐 «레벨 5분» 게이지·막대가 4~10px 어긋나 있었다(헤드리스 크로미움 실측으로 확인 · 시험 `srl::t::layout`).
11-29 셋째: «청산» 탭 아랫줄 세 칸 = `?card=liqx`(rtui/src/liqx.rs · 정답지 compare/liqx_js.mjs · `liqx::t::js_parity`). 안 옮김: 머리 꼬리표·주기 단추·머리 오른쪽 글 · 툴팁 · 좁은 화면 배치.
News(11-29): ⬜ 11 → 10. `?card=news` = news.rs(renderNews 머리 칩 «관련만»·«코인+매크로» · 기울기 1·6·12시간 · 코인별 12시간 · 고영향 상위 6(두 줄 자름) · 출처 줄 · 12시간 축 svg · 기사 목록(해킹·장애 고정 5 + 60건 · 말줄임 · 제목 누르면 새 탭 · 휠 스크롤) + ⑤ 다음 24시간(펀딩·옵션 만기·미국장·오늘·내일 발표 · 같은 분 합침 · mcGfx.timeline)). 자료 = /api/news 30초 · /api/market-context 1초 · /api/gex 1분 · /api/macro-calendar 6시간. 정답지 compare/news_js.mjs(원본 함수를 가짜 전역에서 · 지어낸 기사·일정 8장) → compare/news_layout.py(원본 styles.css · 헤드리스 1920 · html.fit1 · 실제 카드 구조 #newsBody 1838×917) → `news::t::js_parity` 글·칠·테·자르기 상자 1945개 ±0.07(글 x ±0.6 · 자르기 ±1) · svg 요소 수 976개 1e-6. 양성 대조 compare/news_mut.sh 원본 JS 8 · CSS 7 · Rust 5 = 20/20 잡음. 안 옮김: 카드 머리(제목·건수 줄) · 툴팁 · 밑줄·초점 테 · «?» 팝업 · 왼쪽 요약 스크롤 · 좁은 화면 두 줄 모양 · ⑤ 600 미만 목록.
Macro Trend(11-29): ⬜ 20 → 18 · ✅ 12줄. `?card=mt` = mt.rs(renderMacroTrend 표·범례·상세·오늘 24시간 차트·겹친 그림·관계 표·꼬리 줄 · 줄 누르면 그 지표) · `?card=mth` = 머리 «Macro» 요약(오른쪽 붙임). 정답지 compare/mt_js.mjs(원본 구역 통째를 가짜 전역에서 · 지어낸 입력 9장) → compare/mt_layout.py(원본 styles.css · 헤드리스 1920 · html.fit1) → `mt::t::js_parity` 글·칠·테·고리·svg 상자 ±0.07(글 x ±0.6) · svg 요소 수 1e-6. 안 옮김: 툴팁·«?» 팝업·올림/초점 테·상세 칸 스크롤(넘치면 잘림)·1500px 이하 한 단·휴대폰 판.
11-29 셋째 Option 카드(`?card=opt` · rtui/src/opt.rs): 정답지 = compare/opt_js.mjs(원본 renderOptions 와 부르는 함수 28개를 잘라 묶음 · 지어낸 입력 7장 ETH 다 있음/아직 없음/안 모으는 코인/BTC 지연·블록 0/SOL DVOL 없음·행사가 빈칸/XRP 작은 가격/ETH 22분 묵음) → compare/opt_layout.py(진짜 #optCard 틀 + 원본 styles.css · 헤드리스 크로미움 1920 html.fit1 · 카드 폭 1838 · 사다리 514 · 배율 .9 · renderOptions 두 번) → `opt::t::js_parity`: HTML 글·상자·선 하나씩 짝(fc 와 같은 허용) + svg 요소(사다리·만기·흐름) 전부 1e-6 · 높이 ±.02 · 7장 1,263개 · 양성 대조 원본 변이 JS 7/8 · CSS 5/5 · Rust 3/4(놓친 둘 = 지어낸 입력에 그 문턱을 넘는 값이 없음).
찾은 원본 결함(옮김 그대로 · 고치지 않음): 실현 감마 칸 격자 자리·13.5px 줄은 `[data-tip="rg"]`(ETH)만 -- XRP·BTC·SOL(rg_xrp 등)은 첫 빈 칸(cov 줄 · 지연 설명이 있으면 «gamma» 자리)으로 가고 12.5px 블록으로 그려진다.
안 옮김: 주식(CBOE) 판 · 사다리 «내 선»(청산가·지정가 + 닿음 칩) · 툴팁 · «?» 팝업·범위 칩·블록 펼치기 누르기(모양만) · fitLayout 의 사다리 높이·배율 되먹임(입력 h·고정 .9) · 좁은 화면 배치.
11-29 넷째 «지표 방향» 판(`?card=dirb` · rtui/src/dirb.rs): 원본 renderDirBoard 는 /api/dir-signals 를 안 쓴다(그건 ⑤ 지평 사다리 몫) -- 판 줄은 차트 그리기가 캔들로 채우고 아래 «지금 읽기»는 /api/situation 의 read(ETH 만 · 180초 넘으면 없음). 옮김 = 판 그리기 전부(머리 · 접기 단추 모양 · 위/아래 막대·개수 · 줄 홀짝 · 넘으면/깨면(마감 대기) · 지금 읽기 ▲▼·네모(80 이상 주황) · 덧말 말줄임(fit) · 설명 줄·켜진 것) + /api/situation 1초 묻기. 실자료 카드는 줄 자리가 비어(rows·cond = 차트 카드 몫) ETH «지금 읽기»만 보인다.
정답지: compare/dirb_js.mjs(원본 함수 10개를 잘라 fixture/dirb_src.js · 지어낸 입력 9장 = 다 있음/접힘/읽기만·말줄임/읽기 낡음 200초/ETH 아님/아무것도 없음/아래만·작은 네모·낡음 경계 150초/조건만/접힘+읽기) → compare/dirb_layout.py(진짜 svg + 원본 styles.css · 헤드리스 크로미움 · 계산된 fill/stroke · 원본 measureTextW 가 잰 폭 기록) → `dirb::t::js_parity` svg 요소 차례·수·글·색 ±3/255·알파 ±.003·자리 1e-6 · 높이 같음(원본 svg 요소 208개 · 투명 받이·빈 글 빼고 맞댐). 양성 대조 compare/dirb_mut.sh = 원본 변이 JS 12/12 · CSS 3/3 잡음.
안 옮김: 툴팁(fpTipOn) · 접기 누르기(늘 펼침 · 모양만) · 휴대폰 «지금 읽기» 목록(renderDirReadMob) · 판 자리 예약(fpBoardBox -- 차트 꼬리표 비키기). 상태는 공용 줄을 안 고치려 dirb.rs 스레드 지역(ponytail 주석).
11-29 다섯째 풋프린트 머리 작은 칸(`?card=head` · rtui/src/head.rs): ⬜ 10 → 6. 한 카드 = 1920×1080 페이지 자리 그대로(상자 0,0 = 페이지 왼쪽 위 · 1920×136 · 카드 레일 보임 · html.fit1) 네 덩이 -- 세션(buildSessionHtml · 맨 위 왼쪽) · 코인 이름(#fpCoinName · 맨 위 가운데) · 발표 배지 + 급변 칩(.session-alert-center · 코인 탭 왼쪽 · 넘친 칩은 둘째 줄로 숨음 · 경보 배지 맥박) · 포지션 요약(#fpPos · 카드 머리 정중앙). 자료 = /api/state 5초(세션 -- 원본은 SSE 밀기) · /api/session-alerts 30초(실패 = 숨김) · /api/macro-spikes 30초 · /api/binance-account 10초 + 허브 «모든 종목» 가격(acct_only 몫 그대로).
정답지: compare/head_js.mjs(원본 함수 15개를 잘라 가짜 전역에서 · 지어낸 계좌·가격·급변 8장 -- 실제 응답 안 받음) → compare/head_layout.py(진짜 index.html 스크립트만 뺀 것 + 원본 styles.css · 헤드리스 1920 · 원본 renderSnapshotAssetTabs·placeTopLeft 를 그 페이지에서 돌려 제자리에) → `head::t::js_parity` 덩이마다 글·칠 121개 짝(글 x ±0.6 · ±0.07) + 덩이 페이지 자리. 경보 칸 폭 = .snap-head flex 줄임(코인 탭과 «안쪽» 기준 폭 비례로 나눔 -- 탭 안쪽 여백 6 · 실측 393.0 으로 확인). 양성 대조 compare/head_mut.sh = 원본 JS 8 · CSS 6 · Rust 5 = 19/19 잡음.
안 옮김: 툴팁 · 코인 탭 누르기 · 세션 받기 전 «-» · 좁은 화면 배치 · 휴대폰 «다른 포지션» 칩. ponytail: 경보 칸 폭은 고정 칩 4개(eth·sol·xrp·sndk) 기준 상수 -- 지금 코인이 고정 밖이면(btc 등) 탭이 넓어져 실제 칸이 더 좁다(시험은 잰 재료로 맞춤).
11-29 여섯째 Account 카드(`?card=acc` · rtui/src/acc.rs): 곡선 칸(기간 칩 오늘·1주·1개월·전체 · 오늘 0~24시 svg · 누적 실현 계단 + 풀별 선 + 낙폭 + 최악 3건 svg) · 오른쪽 기둥(순자산/오늘 손익 · 풀 비중 막대 · 풀 줄 증거금·가용·지갑·미실현·손절 없음 · 승률·최대 낙폭) · 아래 탭 순환(여러 순환 탭 · 머리 · 타일 6 · 체결·평단·첫 진입 svg · 체결 칩 한 줄) · 성과 통계(풀별 막대 카드) · 달력·복기(달 넘김 · 주 합 · 고른 날 거래 목록 · 아래 알림 줄). 누르기 = 기간 칩 · 탭 · 순환 탭 · ‹ › · 날(보기만).
정답지: compare/acc_layout.py 하나 = 원본 계좌 구역 함수 20개를 잘라 진짜 #acctCard 틀(index.html)·원본 styles.css 로 헤드리스 크로미움 1920(html.fit1)에서 그대로 돌림(순환 칸은 원본처럼 그린 뒤 잰 나머지로 한 번 더) · 지어낸 계좌 15장(두 풀·통합·USDC 하나·불러오지 못함·빈 원장·34체결·잘린 순환·income·지난달·고른 날 …) → `acc::t::js_parity`(글·칠·테·윤곽·안쪽 고리·::before 점·::after 선·자르기 상자 1717개 + svg 수 6995개 1e-6). 양성 대조 compare/acc_mut.sh(JS 9 · CSS 6 · Rust 5).
안 옮김: 카드 머리 · 메모 칸·오답노트(⛔ 입력 -- 오답노트 탭은 모양만 · 메모 자리는 비움) · 툴팁 · 호버·초점 테 · 기간 칩·탭 기억(localStorage) · 좁은 화면. 상태는 공용 줄을 안 고치려 acc.rs 스레드 지역(news.rs 와 같은 꼴). 계좌 값은 정답지에 전부 지어낸 값(실제 응답 안 받음).
