# rtui 카드 목록 — 데스크톱 1920 (2026-11-29)

원본 = `dashboard/live/app.js`·`liq_tab.js`·`index.html`·`styles.css`(origin/main 66882817). 1920×1080 가로 = `html.fit1`: 카드가 위에서부터 Footprint · Macro Trend · News · Option · Account(각 창 높이), 풋프린트 차트 오른쪽 칸(`--fp-split .25`, 약 460px) = `#mcBody.mc-cmp`.
옮김: ✅ = rtui 에 있음 · 🟡 = 일부 · ⬜ = 없음 · ⛔ = 주문 경로·입력 · 보류. 크기 = 1920 화면에서 대략(px).

| 칸 | 원본 render | 데이터 주소 | 옮김 | 크기 |
|---|---|---|---|---|
| 맨 위 코인 탭 | renderSnapshotAssetTabs · renderAssetPop | (상태) | ⛔ 입력(코인 전환) | 450×40 |
| 맨 위 코인 이름 | renderSnapshotAssetTabs(#fpCoinName) | (상태) | ⬜ | 120×30 |
| 보유 종목 손익 전광판 | renderPnlTicker | /api/binance-account · /rt 가격 | ✅ `?card=acct` | 600×28 |
| 발표 배지 · 급변 칩 · 세션 | renderMacroEventAlert · macroSpikeChips | /api/macro-spikes · 세션 알림 | ⬜ | 300×24 |
| 테마 · 알림 단추 · 알림 센터 | renderNotifyCenter | /api/notify-center · /api/push/* | ⛔ 입력 | 단추 |
| 카드 머리 보기·봉 단추 | renderChartWindowTabs | (상태) | 🟡 rtui 는 5m·30m·1h 만 | 400×34 |
| 머리 포지션 요약 | renderFpPos | /api/binance-account | ⬜ | 260×50 |
| 머리 Macro 요약 | mtHeadHtml(#fpRates) | /api/macro-trend · /api/macro-calendar(다음 발표) | ✅ `?card=mth` | 420×38 |
| 차트 위 미니맵 | renderChartPan | /api/market-history | ⬜ | 480×34 |
| 선 주문 단추 · 대기 선 · 손절 선 | renderLineTool · renderArmedLines · renderSlCross | /api/manual-entry/* · /api/manual-bracket/* | ⛔ 주문 경로 · 보류 | 단추·선 |
| 지표 단추·팝업 | renderMarketCtx(#fpLineSwitch) | (상태) | ⛔ 입력 | 단추 |
| 풋프린트 차트(칸·캔들·델타·POC·축·알약) | renderCandleSvg | /api/footprint · /api/market-history · /api/footprint-week · @trade | ✅ 기본 | 1420×880 |
| 차트 레인 · 프로파일 · 스택 · 미완 경매 · 호버 · 청산 동그라미 | renderCandleSvg | 같은 것 · /api/liquidation-5m-* | ✅ | (위 안) |
| 차트 겹침 선(VWAP·EMA·스윙·OB·FVG 등) · 지표 표식 | renderCandleSvg · renderDirMarks · renderObFvg | /api/dir-signals 등 | ⬜ | (위 안) |
| 차트 «레벨 5분» 표식 | renderSrLean | /api/sr-lean | ⬜ | (위 안) |
| 왼쪽 위 «지표 방향» 판 | renderDirBoard | /api/dir-signals | ⬜ | 240×200 |
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
| News 카드 · ⑤ 다음 24시간 | renderNews · renderMarketCtx(#mcWhen) | /api/news · /api/market-context · /api/gex | ⬜ | 전폭×창 |
| Option 카드(코인 ETH·BTC·SOL·XRP) | renderOptions | /api/gex · /api/opt-smile · /api/market-history(실현 감마) · 허브 가격 | 🟡 `?card=opt` (11-29 셋째 · 주식 판 · 사다리 «내 선» 빼고) | 전폭×창 |
| Option 카드 주식(CBOE) 판 · 사다리 «내 선»(청산가·지정가) | renderOptions(o.src) · optEqKpiHtml · optTouchFront | /api/gex · /api/binance-account | ⬜ | (위 안) |
| 알림 센터 «전진 검증 중» 아이스버그 | renderIceChip(#ncIce -- Option 카드 아님) | /api/iceberg | ⬜ | 알림 센터 안 |
| Account 카드(잔고·곡선·순환·통계·달력) | renderSnapshotAccountCore · renderCycleCard | /api/binance-account · /api/cycle-path | ⬜ | 전폭×창 |
| Account 메모 · 오답노트 | (acctNote·acctWrong) | /api/acct-notes · /api/acct-wrong | ⛔ 입력 | 탭 안 |

남은 칸: ⬜ 12 · 🟡 2(봉 단추 · Option -- 주식 판·내 선 빼고) · ⛔ 6(주문 경로·입력 · 보류) · ✅ 19줄. ⑤ 예측 줄 완성(11-29 둘째) = `?card=dirs` 가 fcRowHtml 전체(rtui/src/fc.rs).
11-29 둘째 정답지: compare/fc_js.mjs(원본 fcRowHtml 을 가짜 전역에서 · 지어낸 입력 8장) → compare/fc_layout.py(원본 styles.css · 헤드리스 크로미움 1920 · 폭 404 에서 글·상자·선 자리와 색을 잼) → `fc::t::js_parity`(645개 전부 하나씩 짝 · 높이 같음). 안 옮김: 칸 툴팁 · «?» 설명 팝업(단추 모양만).
11-29 둘째 고침: 오른쪽이 글만(«쌓는 중»)인 줄 높이 30 → 27(`srl::height` -- 원본 실측 27).
첫째(15분·1시간 예측)에 고침: 같은 칸 그리기(srl.rs scene)의 가로 자리 -- 넓은 화면 `#mcBody.mc-cmp .fc-c { gap: 2px }`·`.fc-t > .d4g` 가로 여백 0 을 놓쳐 «레벨 5분» 게이지·막대가 4~10px 어긋나 있었다(헤드리스 크로미움 실측으로 확인 · 시험 `srl::t::layout`).
11-29 셋째: «청산» 탭 아랫줄 세 칸 = `?card=liqx`(rtui/src/liqx.rs · 정답지 compare/liqx_js.mjs · `liqx::t::js_parity`). 안 옮김: 머리 꼬리표·주기 단추·머리 오른쪽 글 · 툴팁 · 좁은 화면 배치.
Macro Trend(11-29): ⬜ 20 → 18 · ✅ 12줄. `?card=mt` = mt.rs(renderMacroTrend 표·범례·상세·오늘 24시간 차트·겹친 그림·관계 표·꼬리 줄 · 줄 누르면 그 지표) · `?card=mth` = 머리 «Macro» 요약(오른쪽 붙임). 정답지 compare/mt_js.mjs(원본 구역 통째를 가짜 전역에서 · 지어낸 입력 9장) → compare/mt_layout.py(원본 styles.css · 헤드리스 1920 · html.fit1) → `mt::t::js_parity` 글·칠·테·고리·svg 상자 ±0.07(글 x ±0.6) · svg 요소 수 1e-6. 안 옮김: 툴팁·«?» 팝업·올림/초점 테·상세 칸 스크롤(넘치면 잘림)·1500px 이하 한 단·휴대폰 판.
11-29 셋째 Option 카드(`?card=opt` · rtui/src/opt.rs): 정답지 = compare/opt_js.mjs(원본 renderOptions 와 부르는 함수 28개를 잘라 묶음 · 지어낸 입력 7장 ETH 다 있음/아직 없음/안 모으는 코인/BTC 지연·블록 0/SOL DVOL 없음·행사가 빈칸/XRP 작은 가격/ETH 22분 묵음) → compare/opt_layout.py(진짜 #optCard 틀 + 원본 styles.css · 헤드리스 크로미움 1920 html.fit1 · 카드 폭 1838 · 사다리 514 · 배율 .9 · renderOptions 두 번) → `opt::t::js_parity`: HTML 글·상자·선 하나씩 짝(fc 와 같은 허용) + svg 요소(사다리·만기·흐름) 전부 1e-6 · 높이 ±.02 · 7장 1,263개 · 양성 대조 원본 변이 JS 7/8 · CSS 5/5 · Rust 3/4(놓친 둘 = 지어낸 입력에 그 문턱을 넘는 값이 없음).
찾은 원본 결함(옮김 그대로 · 고치지 않음): 실현 감마 칸 격자 자리·13.5px 줄은 `[data-tip="rg"]`(ETH)만 -- XRP·BTC·SOL(rg_xrp 등)은 첫 빈 칸(cov 줄 · 지연 설명이 있으면 «gamma» 자리)으로 가고 12.5px 블록으로 그려진다.
안 옮김: 주식(CBOE) 판 · 사다리 «내 선»(청산가·지정가 + 닿음 칩) · 툴팁 · «?» 팝업·범위 칩·블록 펼치기 누르기(모양만) · fitLayout 의 사다리 높이·배율 되먹임(입력 h·고정 .9) · 좁은 화면 배치.
