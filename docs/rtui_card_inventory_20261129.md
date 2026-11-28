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
| 머리 Macro 요약 | mtHeadHtml(#fpRates) | /api/macro-trend · /api/macro-today | ⬜ | 420×40 |
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
| ⑤ 종합 칩(ETH) | fcRowHtml(fcSum) | /api/dir4h · /api/trend · /api/open-rev | ⬜ | 460×22 |
| ⑤ 지금 추세 | fcTrendCell | /api/trend | ⬜ | 460×42 |
| ⑤ 15분 예측 | fcRowHtml sCell · prHit | /api/dir-short · /api/pred-record | ✅ `?card=dirs` (이번) | 460×42 |
| ⑤ 레벨 5분(ETH) | srLeanCell | /api/sr-lean | ✅ `?card=srl` | 460×42 |
| ⑤ 1시간 예측 | fcRowHtml sCell · prHit | /api/dir-short · /api/pred-record | ✅ `?card=dirs` (이번) | 460×42 |
| ⑤ 4h 예측 | fcRowHtml | /api/dir4h · /api/vol-forecast · /api/pred-record | ⬜ | 460×42 |
| ⑤ 횡보·추세 | fcRowHtml rbCell | /api/regime-band · /api/pred-record | ⬜ | 460×30 |
| ⑤ 위험 레이더 | rrCells | /api/risk-radar | ⬜ | 460×60 |
| ⑤ 4h 상대 강도 표 | predXsecHtml | /api/pred-record | ⬜ | 460×80 |
| 보기: 틱 차트 · 체결 버블 · 일봉 풋프린트 | tick_charts · bubble_map.js · fp_daily.js | /api/footprint-week 등 | ⬜ | 차트 자리 |
| «청산» 탭 청산맵 | liq_tab.js CH.ladder | /api/liquidation-map(/tiers) · /api/liq-tab | ✅ `?card=liq` | 900×560 |
| «청산» 탭 남은 매물 일봉별 | liq_tab.js CH.remain | /api/liq-tab | ✅ `?card=rem` | 900×300 |
| «청산» 탭 오늘 누적 · 최대 고통 · 코인별 최대 고통 | CH.today · CH.pain · CH.quad | /api/liq-tab(all) · /api/liquidation-5m-history | ⬜ | 3칸 × 340 |
| Macro Trend 카드 | renderMacroTrend · refreshMtToday | /api/macro-trend · /api/macro-today · /api/treasury-futures | ⬜ | 전폭×창 |
| News 카드 · ⑤ 다음 24시간 | renderNews · renderMarketCtx(#mcWhen) | /api/news · /api/market-context · /api/gex | ⬜ | 전폭×창 |
| Option 카드 | renderOptions · renderIceChip | /api/gex · /api/opt-smile · /api/iceberg · /api/xyz100 | ⬜ | 전폭×창 |
| Account 카드(잔고·곡선·순환·통계·달력) | renderSnapshotAccountCore · renderCycleCard | /api/binance-account · /api/cycle-path | ⬜ | 전폭×창 |
| Account 메모 · 오답노트 | (acctNote·acctWrong) | /api/acct-notes · /api/acct-wrong | ⛔ 입력 | 탭 안 |

남은 칸: ⬜ 20 · 🟡 1(봉 단추) · ⛔ 6(주문 경로·입력 · 보류) · ✅ 10줄. 이번 = ⑤ 15분·1시간 예측(같은 함수 sCell 두 줄).
이번에 고침: 같은 칸 그리기(srl.rs scene)의 가로 자리 -- 넓은 화면 `#mcBody.mc-cmp .fc-c { gap: 2px }`·`.fc-t > .d4g` 가로 여백 0 을 놓쳐 «레벨 5분» 게이지·막대가 4~10px 어긋나 있었다(헤드리스 크로미움 실측으로 확인 · 시험 `srl::t::layout`).
