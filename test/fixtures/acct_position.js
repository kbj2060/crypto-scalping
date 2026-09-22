// 계좌 카드 «보유 중» 상태 (2026-09-22). 라이브에 포지션이 없을 때 그 화면을 재현한다.
// 🔴이 상태가 재현 불가였던 게 문제였다 -- 시안은 보유 중만 그렸는데 검증은 포지션 없음에서만
//   돌아서, 사용자가 «시안이랑 다른데»를 두 번 말할 때까지 몰랐다.
// 값은 2026-09-22 실제 라이브에서 뜬 것이다(ETHUSDC 롱 ×20).
() => {
  for (let i = 1; i < 99999; i++) window.clearInterval(i);   // 폴링이 덮지 않게
  latestBinanceAccount = {
    balance: { wallet: 2013.62, unrealized: 6.53, available: 1551.29,
               initial_margin: 403.00, margin: 2020.15, asset: "USDC" },
    exec_symbol: "ETHUSDC",
    positions: [{ symbol: "ETHUSDC", side: "LONG", qty: 3.423, leverage: 20,
                  entry_price: 2737.63, mark_price: 2739.42, liquidation_price: 2197.56,
                  notional: 9376.9, unrealized_pnl: 6.53 }],
  };
  // 🔴진입가와 마크가를 **일부러 벌려** 둔 두 번째 세트가 필요하면 mark_price 를 만진다.
  //   실제 값은 진입≈현재라 손잡이와 유령 눈금이 겹쳐 보인다 -- 벌어진 모습을 못 본 채
  //   «검증했다»고 하면 안 된다.
  entryProjPreview = {
    before: { liq_pct: 21.2, margin_used_pct: 20.0, exposure_x: 4.64 },
    after:  { liq_pct: 19.8, margin_used_pct: 25.0, exposure_x: 5.01 },
    __plan: { quantity: 0.856, price: 2739.42, target_leverage: 20 },
  };
  lastExitPositions = new Map([["ETHUSDC", latestBinanceAccount.positions[0]]]);
  renderSnapshotAccount();
  document.getElementById("snapExitRow").hidden = false;
  document.getElementById("snapExitLong").hidden = false;
  renderExitNow();
  const d = document.getElementById("snapEntryBox"); if (d) d.open = true;
}
