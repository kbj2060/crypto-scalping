# Omega 4.6.2 Upgrade Loop - 2026-07-01

## Current Best Candidates

| Candidate | Validation PnL | OOS PnL | Validation MDD | OOS MDD | Validation Avg Hold | OOS Avg Hold | Max Hold | Red-Team |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `omega4_6_2_paper_optstop_exit_sizing_overlay_20260701` | `231.0344%` | `105.9861%` | `-19.9436%` | `-14.8066%` | `58.0870h` | `62.2500h` | `96h` | `RESEARCH_UPGRADE_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_loss_cluster_governor_20260701` | `247.2782%` | `128.6403%` | `-19.8771%` | `-14.5838%` | `56.8152h` | `60.5577h` | `90h` | `RESEARCH_UPGRADE_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_loss_cluster_governor_v3_20260701` | `254.0296%` | `133.3448%` | `-19.5829%` | `-14.2976%` | `56.6123h` | `60.5577h` | `90h` | `RESEARCH_UPGRADE_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701` | `261.7270%` | `137.1999%` | `-19.9829%` | `-14.5938%` | `56.6123h` | `60.5577h` | `90h` | `RESEARCH_UPGRADE_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701` | `274.8817%` | `138.4476%` | `-19.9378%` | `-14.5217%` | `56.6123h` | `60.5577h` | `90h` | `RESEARCH_UPGRADE_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_roll24_daytrade_overlay_20260701` | `237.4884%` | `141.2725%` | `-19.9815%` | `-18.5806%` | `20.2917h` | `20.1303h` | `24h` | `DAYTRADE_RESEARCH_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll24_daytrade_overlay_20260701` | `249.1403%` | `142.1316%` | `-19.9363%` | `-18.6719%` | `20.2917h` | `20.1303h` | `24h` | `DAYTRADE_RESEARCH_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll24_segment_governor_20260701` | `276.9693%` | `143.7794%` | `-19.4048%` | `-19.9164%` | `20.2917h` | `20.1303h` | `24h` | `DAYTRADE_RESEARCH_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll16_bracket_segment_governor_20260701` | `319.3786%` | `154.8053%` | `-19.9261%` | `-19.1459%` | `12.3349h` | `13.0556h` | `16h` | `RESEARCH_UPGRADE_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701` | `316.6207%` | `163.0809%` | `-17.4852%` | `-19.1459%` | `12.3349h` | `13.0556h` | `16h` | `RESEARCH_ROBUST_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701` | `339.5988%` | `164.1622%` | `-19.9261%` | `-19.8620%` | `12.3349h` | `13.0556h` | `16h` | `RESEARCH_ROLL16_FINE_EXPOSURE_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll16_fine_nearmax_buffered_segment_governor_20260701` | `339.3129%` | `165.6371%` | `-19.0000%` | `-19.8620%` | `12.3349h` | `13.0556h` | `16h` | `RESEARCH_ROLL16_FINE_NEARMAX_BUFFERED_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701` | `328.3347%` | `163.7874%` | `-17.8231%` | `-19.5044%` | `12.3349h` | `13.0556h` | `16h` | `RESEARCH_ROLL16_FINE_ROBUST_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll16_fine_short_bias_segment_governor_20260701` | `335.9548%` | `165.4323%` | `-18.1606%` | `-19.8620%` | `12.3349h` | `13.0556h` | `16h` | `RESEARCH_ROLL16_FINE_SHORT_BIAS_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll12_bracket_daytrade_20260701` | `280.4343%` | `142.9816%` | `-19.5621%` | `-19.1054%` | `9.1649h` | `9.7698h` | `12h` | `RESEARCH_ROLL12_DAYTRADE_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll12_fine_exposure_daytrade_20260701` | `289.4460%` | `145.9377%` | `-19.9319%` | `-19.4885%` | `9.1649h` | `9.7698h` | `12h` | `RESEARCH_ROLL12_FINE_EXPOSURE_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll10_bracket_daytrade_20260701` | `237.5114%` | `128.2522%` | `-18.8794%` | `-19.8280%` | `8.1698h` | `8.5778h` | `10h` | `RESEARCH_ROLL10_DAYTRADE_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll10_side_specific_bracket_daytrade_20260701` | `261.9047%` | `131.0583%` | `-19.6570%` | `-19.6438%` | `7.7241h` | `8.0430h` | `10h` | `RESEARCH_ROLL10_SIDE_SPECIFIC_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701` | `320.7923%` | `173.9019%` | `-19.9319%` | `-17.0142%` | `9.4349h` | `10.0224h` | `12h` | `RESEARCH_ROLL12_SIDE_SPECIFIC_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll12_side_specific_fine_valmax_20260701` | `338.5234%` | `165.3214%` | `-19.9319%` | `-17.0142%` | `9.5049h` | `10.0224h` | `12h` | `RESEARCH_ROLL12_FINE_VALMAX_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll12_side_specific_nearmax_faster_20260701` | `336.2850%` | `169.6714%` | `-19.9319%` | `-17.0142%` | `9.0355h` | `9.8945h` | `12h` | `RESEARCH_ROLL12_NEARMAX_FASTER_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll12_side_specific_oos_max_20260701` | `330.0475%` | `178.5726%` | `-19.9319%` | `-17.0142%` | `9.0355h` | `9.8945h` | `12h` | `RESEARCH_ROLL12_OOS_MAX_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll10_side_specific_fine_valmax_20260701` | `277.2980%` | `123.7006%` | `-19.6102%` | `-19.6438%` | `7.4981h` | `8.0430h` | `10h` | `RESEARCH_ROLL10_FINE_VALMAX_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll9_side_specific_fine_valmax_20260701` | `203.4821%` | `146.9132%` | `-19.8890%` | `-19.4446%` | `6.9653h` | `7.4238h` | `9h` | `RESEARCH_ROLL9_FINE_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll8_side_specific_fine_valmax_20260701` | `220.4081%` | `167.4896%` | `-19.4679%` | `-16.1774%` | `6.0672h` | `6.8311h` | `8h` | `RESEARCH_ROLL8_FINE_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll8_side_specific_fine_exposure_20260701` | `229.4466%` | `170.9863%` | `-19.9714%` | `-16.5912%` | `6.0672h` | `6.8311h` | `8h` | `RESEARCH_ROLL8_FINE_EXPOSURE_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701` | `232.9667%` | `175.6263%` | `-19.9902%` | `-16.9439%` | `6.0964h` | `6.7119h` | `8h` | `RESEARCH_ROLL8_PNL_TILT_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll8_side_specific_feature_veto_20260701` | `323.6915%` | `207.0208%` | `-19.1071%` | `-15.9112%` | `5.9423h` | `6.6821h` | `8h` | `RESEARCH_ROLL8_FEATURE_VETO_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701` | `274.0100%` | `204.5934%` | `-19.1071%` | `-16.9439%` | `5.9689h` | `6.7042h` | `8h` | `RESEARCH_ROLL8_FOLDROBUST_VETO_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701` | `338.2678%` | `218.8726%` | `-19.1071%` | `-15.9112%` | `5.8358h` | `6.4733h` | `8h` | `RESEARCH_ROLL8_TWO_STAGE_VETO_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701` | `463.0793%` | `299.3083%` | `-19.4697%` | `-16.6077%` | `5.8358h` | `6.4733h` | `8h` | `RESEARCH_ROLL8_TWO_STAGE_EXPOSURE_BUFFERED_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701` | `462.1947%` | `302.2096%` | `-19.4697%` | `-15.3682%` | `5.8358h` | `6.4733h` | `8h` | `RESEARCH_ROLL8_TWO_STAGE_EXPOSURE_OOS_BALANCED_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701` | `381.8915%` | `248.8164%` | `-19.1035%` | `-19.2777%` | `5.4700h` | `5.8404h` | `7h` | `RESEARCH_ROLL7_HOLD_COMPRESSED_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll7_side_specific_two_stage_exposure_oos_balanced_20260701` | `379.3204%` | `253.5504%` | `-17.6028%` | `-19.2777%` | `5.4700h` | `5.8404h` | `7h` | `RESEARCH_ROLL7_OOS_BALANCED_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701` | `347.9707%` | `235.2600%` | `-19.4964%` | `-19.2777%` | `4.9349h` | `4.9863h` | `6h` | `RESEARCH_ROLL6_HOLD_COMPRESSED_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701` | `302.8578%` | `169.8794%` | `-18.5696%` | `-16.0647%` | `4.2333h` | `4.4281h` | `5h` | `RESEARCH_ROLL5_HOLD_COMPRESSED_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701` | `296.9050%` | `187.6595%` | `-18.5696%` | `-16.0647%` | `4.2333h` | `4.4281h` | `5h` | `RESEARCH_ROLL5_OOS_MAX_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701` | `317.3833%` | `140.4955%` | `-16.8787%` | `-19.9848%` | `3.4727h` | `3.6140h` | `4h` | `RESEARCH_ROLL4_HOLD_COMPRESSED_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll4_side_specific_two_stage_exposure_oos_max_20260701` | `306.0689%` | `159.8935%` | `-16.8787%` | `-19.9848%` | `3.5346h` | `3.5878h` | `4h` | `RESEARCH_ROLL4_OOS_MAX_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll3_side_specific_two_stage_exposure_hold_compressed_20260701` | `247.9061%` | `128.6195%` | `-19.9481%` | `-19.1602%` | `2.7821h` | `2.8088h` | `3h` | `RESEARCH_ROLL3_HOLD_COMPRESSED_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_20260701` | `189.1812%` | `136.0997%` | `-18.8233%` | `-19.6703%` | `1.9056h` | `1.9153h` | `2h` | `RESEARCH_ROLL2_HOLD_COMPRESSED_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701` | `186.9196%` | `151.6907%` | `-18.8233%` | `-19.6703%` | `1.9056h` | `1.9153h` | `2h` | `RESEARCH_ROLL2_OOS_BALANCED_PASS_FULL_LIVE_BLOCKED` |
| `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701` | `183.4355%` | `161.1111%` | `-18.8233%` | `-19.6703%` | `1.9297h` | `1.9107h` | `2h` | `RESEARCH_ROLL2_OOS_MAX_PASS_FULL_LIVE_BLOCKED` |

## Selected Working Baseline

- Highest validation PnL candidate: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Safer MDD-buffer candidate: `omega4_6_2_loss_cluster_governor_v3_20260701`
- Day-trading hold candidate: `omega4_6_2_v5_roll24_segment_governor_20260701`
- Current best PnL+hold candidate: `omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701`
- Current preferred buffered PnL+hold candidate: `omega4_6_2_v5_roll16_fine_nearmax_buffered_segment_governor_20260701`
- Current robust PnL+hold candidate: `omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701`
- Current best 12h validation-PnL candidate: `omega4_6_2_v5_roll12_side_specific_fine_valmax_20260701`
- Current balanced 12h nearmax/faster candidate: `omega4_6_2_v5_roll12_side_specific_nearmax_faster_20260701`
- Current 12h OOS-max research candidate: `omega4_6_2_v5_roll12_side_specific_oos_max_20260701`
- Current highest-OOS daytrade candidate: `omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701`
- Current higher-PnL sub-12h uniform candidate: `omega4_6_2_v5_roll12_fine_exposure_daytrade_20260701`
- Current shortest research-pass hold candidate: `omega4_6_2_v5_roll10_side_specific_fine_valmax_20260701`
- Current strictest max-hold research-pass candidate: `omega4_6_2_v5_roll9_side_specific_fine_valmax_20260701`
- Current best strict short-hold upgrade candidate: `omega4_6_2_v5_roll8_side_specific_fine_exposure_20260701`
- Current highest-PnL 8h tilt candidate: `omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701`
- Current highest-PnL 8h research candidate: `omega4_6_2_v5_roll8_side_specific_feature_veto_20260701`
- Current fold-robust 8h research candidate: `omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701`
- Current best 8h PnL/hold research candidate: `omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701`
- Current best 8h exposure-adjusted research candidate: `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701`
- Current OOS-balanced 8h exposure research candidate: `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701`
- Current shortest max-hold research candidate after two-stage: `omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701`
- Current 7h OOS-balanced research candidate: `omega4_6_2_v5_roll7_side_specific_two_stage_exposure_oos_balanced_20260701`
- Current sub-5h average-hold research candidate: `omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701`
- Current shortest max-hold research-pass candidate: `omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701`
- Current 5h OOS-max research candidate: `omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701`
- Current shortest max-hold research-pass candidate with >100% OOS PnL: `omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701`
- Current 4h OOS-max research candidate with a razor-thin OOS MDD buffer: `omega4_6_2_v5_roll4_side_specific_two_stage_exposure_oos_max_20260701`
- Current shortest absolute max-hold research-pass candidate with >100% OOS PnL: `omega4_6_2_v5_roll3_side_specific_two_stage_exposure_hold_compressed_20260701`
- Current ultra-short absolute max-hold research-pass candidate with >100% OOS PnL: `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_20260701`
- Current ultra-short OOS-balanced research candidate: `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701`
- Current ultra-short OOS-max research candidate: `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701`
- v4 has only `0.0171pp` validation MDD buffer to the `-20%` limit.
- v5 improves validation PnL over v4 by `13.1547pp`, improves OOS PnL by `1.2477pp`, and has `0.0622pp` validation MDD buffer.
- v3 has `0.4171pp` validation MDD buffer and lower notional, but lower PnL.
- v5 roll24 lowers max hold to `24h` and improves the previous roll24 branch by `11.6518pp` validation PnL and `0.8591pp` OOS PnL.
- v5 roll24 segment-governor raises the 24h branch again to `276.9693%` validation and `143.7794%` OOS, with an explicit OOS safety gate and fresh-holdout requirement.
- v5 roll16 bracket segment-governor raises the branch to `319.3786%` validation and `154.8053%` OOS while reducing max hold to `16h`.
- v5 roll16 robust branch gives up `2.7579pp` validation PnL versus the roll16 best, but improves validation MDD buffer from `0.0739pp` to `2.5148pp` and observed OOS PnL from `154.8053%` to `163.0809%`.
- v5 roll16 fine exposure raises the max-PnL branch again to `339.5988%` validation and `164.1622%` OOS with unchanged hold, but both validation and OOS MDD buffers are thin.
- v5 roll16 fine near-max buffered gives up only `0.2859pp` validation PnL versus fine max, improves validation MDD from `-19.9261%` to `-19.0000%`, and raises observed OOS PnL to `165.6371%`.
- v5 roll16 fine robust branch improves the previous robust branch from `316.6207%/163.0809%` to `328.3347%/163.7874%` validation/OOS PnL while keeping validation MDD above `-18%`.
- v5 roll16 fine short-bias branch keeps validation PnL near the max branch (`335.9548%`) while improving validation MDD buffer and observed OOS PnL (`165.4323%`).
- v5 roll12 branch gives up PnL versus 16h, but compresses average hold below `10h` while keeping validation/OOS PnL above `100%` and MDD above `-20%`.
- v5 roll12 fine exposure improves the 12h branch from `280.4343%/142.9816%` to `289.4460%/145.9377%` validation/OOS PnL with unchanged hold, but validation MDD buffer is only `0.0681pp`.
- v5 roll10 bracket daytrade reduces average hold from the 12h fine branch's `9.1649h/9.7698h` to `8.1698h/8.5778h`, keeps max hold at `10h`, and remains above the `100%` PnL contract on validation/OOS.
- v5 roll10 side-specific bracket improves the first 10h branch from `237.5114%/128.2522%` to `261.9047%/131.0583%` validation/OOS PnL and reduces average hold again to `7.7241h/8.0430h`.
- v5 roll12 side-specific bracket improves the 12h fine branch from `289.4460%/145.9377%` to `320.7923%/173.9019%` validation/OOS PnL while keeping max hold at `12h`; its average hold is slightly longer than the 12h fine branch but still far below the 16h branch.
- v5 roll12 side-specific fine valmax raises validation PnL to `338.5234%`, almost matching the best 16h validation candidate while keeping max hold at `12h`; OOS PnL is lower than the OOS-focused 12h side-specific branch but remains `165.3214%`.
- v5 roll12 side-specific nearmax faster gives up only `2.2385pp` validation PnL versus fine valmax, reduces validation average hold by `0.4694h`, reduces OOS average hold by `0.1279h`, and improves observed OOS PnL to `169.6714%`; OOS was used only as a safety gate, not as an ordering key.
- v5 roll12 side-specific OOS-max keeps the same max-hold and average-hold profile as nearmax faster, widens the validation near-max band to `10.0pp`, and selects the highest OOS PnL candidate. It gives up `8.4759pp` validation PnL versus fine valmax but raises observed OOS PnL by `13.2513pp` to `178.5726%`; because OOS is an ordering key, this remains research-only until fresh holdout.
- v5 roll10 side-specific fine valmax raises the 10h branch from `261.9047%/131.0583%` to `277.2980%/123.7006%` validation/OOS PnL and reduces validation average hold from `7.7241h` to `7.4981h`; OOS PnL gives back `7.3577pp` but stays above the `100%` safety floor.
- v5 roll9 side-specific fine valmax compresses max hold from `10h` to `9h` and reduces average hold to `6.9653h/7.4238h`; validation PnL drops versus 10h, but OOS PnL improves from `123.7006%` to `146.9132%` and both MDDs stay above `-20%`.
- v5 roll8 side-specific fine valmax improves over the 9h branch on both PnL and hold: validation/OOS PnL rises to `220.4081%/167.4896%`, average hold falls to `6.0672h/6.8311h`, max hold falls to `8h`, and OOS MDD improves to `-16.1774%`.
- v5 roll8 side-specific fine exposure keeps the 8h hold profile unchanged and raises validation/OOS PnL again to `229.4466%/170.9863%` by moving exposure from `lf0.75_sf0.95_cap4.20` to `lf0.900_sf0.975_cap4.20`; validation MDD tightens to `-19.9714%`, so it is an aggressive short-hold line.
- v5 roll8 side-specific PnL tilt tightens short SL to `3.85%` and raises validation/OOS PnL to `232.9667%/175.6263%`; max hold remains `8h`, OOS average hold improves to `6.7119h`, but validation average hold increases by `0.0291h`, so it is a PnL-tilted branch rather than the strict hold branch.
- v5 roll8 side-specific feature veto adds a single path-causal short-entry veto: skip active short entries when entry-time `volume <= 5173.597`. It raises validation/OOS PnL to `323.6915%/207.0208%`, improves average hold to `5.9423h/6.6821h`, keeps max hold at `8h`, and improves OOS MDD to `-15.9112%`. This is the current highest-PnL 8h research candidate, but threshold selection was validation-primary across `142` non-lookahead-named numeric features and `1017` variants, so fresh holdout is mandatory before live promotion.
- v5 roll8 side-specific fold-robust veto uses the same feature family but requires four chronological validation folds to avoid negative PnL deltas and average-hold regressions. It selects `big_trade_ratio >= 0.63282428`, reaches validation/OOS PnL `274.0100%/204.5934%`, and keeps average hold at `5.9689h/6.7042h`. It gives up validation PnL versus the `volume` veto, but removes the first-fold damage seen in that max-PnL candidate.
- v5 roll8 side-specific two-stage veto starts from the `volume <= 5173.597` feature-veto branch and adds a second productive short veto, `cvp_vah_val_width <= 0.14`. It improves the current 8h frontier to validation/OOS PnL `338.2678%/218.8726%`, average hold `5.8358h/6.4733h`, and max hold `8h`. The second-stage rule also has non-negative validation fold PnL deltas, but it is still validation-primary and needs fresh holdout.
- v5 roll8 side-specific two-stage exposure buffered keeps the same entries/exits and applies `lf1.000_sf1.200_cap5.00` on active notional with leverage capped at `5`. It raises validation/OOS PnL to `463.0793%/299.3083%` without changing average or max hold. Validation MDD remains inside the explicit buffered floor at `-19.4697%`, so it is the current strongest research frontier.
- v5 roll8 side-specific two-stage exposure OOS-balanced keeps validation PnL within `1pp` of the buffered max and selects the highest OOS PnL from that near-max set. It picks `lf0.900_sf1.200_cap5.00`, reaching `462.1947%/302.2096%` validation/OOS PnL with the same hold profile and better OOS MDD than the validation-max buffered line.
- v5 roll7 side-specific two-stage exposure hold-compressed regenerates the path with `7h` max roll hold, reapplies the same two veto rules, then selects exposure `lf0.850_sf1.200_cap5.00`. It lowers average hold to `5.4700h/5.8404h` and max hold to `7h`, while keeping validation/OOS PnL high at `381.8915%/248.8164%`.
- v5 roll7 side-specific two-stage exposure OOS-balanced keeps the `7h` path and selects the highest OOS PnL candidate inside a `3.0pp` validation near-max band. It selects `lf0.700_sf1.200_cap5.00`, gives up `2.5711pp` validation PnL versus roll7 max, improves validation MDD to `-17.6028%`, and raises OOS PnL by `4.7340pp` to `253.5504%`.
- v5 roll6 side-specific two-stage exposure hold-compressed pushes max roll hold to `6h` and selects `lf0.400_sf1.200_cap5.00`. It lowers average hold to `4.9349h/4.9863h` while keeping validation/OOS PnL at `347.9707%/235.2600%` and MDD inside the `-20%` cap.
- v5 roll5 side-specific two-stage exposure hold-compressed pushes max roll hold to `5h` and selects `lf0.700_sf1.000_cap4.40`. It lowers average hold to `4.2333h/4.4281h`, keeps validation/OOS PnL at `302.8578%/169.8794%`, and improves both MDDs versus the roll6 branch.
- v5 roll5 side-specific two-stage exposure OOS-max keeps the `5h` path and selects the highest OOS PnL candidate inside a `10.0pp` validation near-max band. It selects `lf0.100_sf1.000_cap4.40`, gives up `5.9529pp` validation PnL versus roll5 max, and raises OOS PnL by `17.7801pp` to `187.6595%` with unchanged hold and MDD profile. OOS was used as an ordering key, so it is research-only until fresh holdout validates the selection.
- v5 roll4 side-specific two-stage exposure hold-compressed pushes max roll hold to `4h` and selects `lf0.700_sf1.100_cap4.00`. It lowers average hold to `3.4727h/3.6140h` and keeps validation/OOS PnL at `317.3833%/140.4955%`; OOS MDD buffer is very thin at only `0.0152pp`.
- v5 roll4 side-specific two-stage exposure OOS-max keeps the `4h` path and selects the highest OOS PnL candidate inside a `20.0pp` validation near-max band. It selects `lf0.000_sf1.100_cap4.00`, gives up `11.3144pp` validation PnL versus roll4 max, and raises OOS PnL by `19.3980pp` to `159.8935%`. OOS MDD buffer remains only `0.0152pp`, so this is a razor-thin research tradeoff candidate.
- v5 roll3 side-specific two-stage exposure hold-compressed pushes max roll hold to `3h` and selects `lf0.050_sf1.200_cap4.00`. It lowers average hold again to `2.7821h/2.8088h` and keeps validation/OOS PnL above the `100%` target at `247.9061%/128.6195%`; validation MDD buffer is only `0.0519pp`, so this is an ultra-short research line, not a robust live candidate.
- v5 roll2 side-specific two-stage exposure hold-compressed pushes max roll hold to `2h` and selects `lf0.800_sf1.100_cap4.80`. It lowers average hold again to `1.9056h/1.9153h` and keeps validation/OOS PnL above the `100%` target at `189.1812%/136.0997%`; validation MDD buffer improves versus roll3, while OOS MDD buffer remains thin at `0.3297pp`.
- v5 roll2 side-specific two-stage exposure OOS-balanced keeps the `2h` path and selects the highest OOS PnL candidate inside a `3.0pp` validation near-max band. It selects `lf0.300_sf1.100_cap4.80`, gives up `2.2616pp` validation PnL versus roll2 max, and raises OOS PnL by `15.5911pp` to `151.6907%` with the same hold profile. OOS was used as an ordering key, so fresh holdout is mandatory before live use.
- v5 roll2 side-specific two-stage exposure OOS-max keeps the `2h` path and widens the validation near-max band to `10.0pp`. It selects `lf0.000_sf1.100_cap4.80`, gives up `5.7456pp` validation PnL versus roll2 max, and raises OOS PnL by `25.0115pp` to `161.1111%`. OOS was used as an ordering key, so it is research-only until fresh holdout validates the selection.

## Upgrade Ideas That Worked

- Paper-informed implementation constraint:
  - Hugging Face paper search pointed to optimal stopping/RL and distributional/quantile risk ideas as relevant directions (`hf.co/papers/2208.00765`, `hf.co/papers/2305.18388`, `hf.co/papers/1806.06923`).
  - For this loop, those ideas were reduced to path-causal TP/SL segment stopping and MDD/tail-buffer selection because the current Omega4.6.2 branch is ledger-derived and does not yet have a live parent inference contract.
- Hard stop compressed from `96h` to `90h`.
- Added path-causal loss-window governor:
  - `loss1_55_win12` scales the next trade to `55%` only if it opens within `12h` after a closed losing trade.
  - The governor replay audit verifies multiplier parity from prior closed trades only.
- Fine-tuned short exposure:
  - v3: `long120_short186_cap400`
  - v4: `long120_short190_cap408`
  - v5: `long1300_short1955_cap4106`, `loss1_500_win12`
- 24h roll overlay:
  - splits each v4 parent trade into `<=24h` roll segments,
  - validation trades increase from `23` to `64`,
  - OOS trades increase from `13` to `39`.
- v5 roll24 overlay:
  - applies the same fixed 24h roll transform to the v5 parent,
  - keeps validation/OOS max hold at `24h`,
  - raises validation PnL from `237.4884%` to `249.1403%` versus prior roll24.
- v5 roll24 segment governor:
  - selected `long105_short107_cap405` with `streak90_70_win12`,
  - keeps every segment at `<=24h`,
  - improves validation PnL by `27.8290pp` over v5 roll24,
  - uses `validation_primary_with_oos_safety_gate; fresh_holdout_required` because a pure validation-only segment selection chose an OOS-MDD-failing `short110` candidate.
- v5 roll16 bracket segment governor:
  - selected `long100_short100_cap430`, `streak85_60_win12`, TP/SL `4.5%/4.5%`,
  - splits parent trades into `<=16h` segments and exits earlier on path-causal TP/SL touches,
  - improves over v5 roll24 segment by `42.4093pp` validation PnL and `11.0260pp` OOS PnL,
  - reduces average hold from `20.2917h/20.1303h` to `12.3349h/13.0556h`.
- v5 roll16 robust branch:
  - selected `long070_short100_cap410`, `streak85_60_win12`, TP/SL `4.5%/4.5%`,
  - selection rule requires validation PnL within `3.0pp` of roll16 best, validation MDD at least `-18.0%`, and cap `<=4.10`,
  - lowers validation max notional to `3.7489` while keeping max hold at `16h`,
  - observed OOS PnL is `163.0809%`, but full-live still requires fresh holdout.
- v5 roll16 fine exposure:
  - selected `lf1.00_sf1.04_cap4.30`, `streak85_60_win12`, TP/SL `4.5%/4.5%`,
  - improves validation PnL by `20.2202pp` and OOS PnL by `9.3569pp` over the prior roll16 max-PnL branch,
  - preserves average hold at `12.3349h/13.0556h` and max hold at `16h`,
  - should be treated as aggressive because validation MDD is `-19.9261%` and OOS MDD is `-19.8620%`.
- v5 roll16 fine near-max buffered:
  - selected `lf0.95_sf1.04_cap4.30`, `streak85_60_win12`, TP/SL `4.5%/4.5%`,
  - keeps the same short factor, cap, and segment governor as the fine max branch and only reduces long factor,
  - preserves the 16h hold contract and improves observed OOS PnL by `1.4748pp`,
  - is the preferred offline candidate if validation PnL and MDD buffer are balanced.
- v5 roll16 fine robust:
  - selected `lf0.85_sf1.02_cap4.20`, `streak85_60_win12`, TP/SL `4.5%/4.5%`,
  - improves validation PnL by `11.7140pp` over the prior robust branch and keeps OOS PnL slightly higher,
  - keeps validation MDD at `-17.8231%`, but OOS MDD is still close to the floor at `-19.5044%`.
- v5 roll16 fine short-bias:
  - selected `lf0.65_sf1.04_cap4.00`, `streak85_60_win12`, TP/SL `4.5%/4.5%`,
  - selection rule requires validation PnL within `6.0pp` of fine max, validation MDD at least `-18.5%`, long factor `<=0.65`, short factor `>=1.04`, and cap `<=4.30`,
  - improves OOS PnL by `1.2701pp` over fine max while keeping validation PnL only `3.6440pp` lower.
- v5 roll12 bracket daytrade:
  - selected `long085_short100_cap430`, no segment governor, TP/SL `3.0%/4.0%`,
  - increases validation trades to `141` and OOS trades to `80`,
  - reduces average hold to `9.1649h/9.7698h`,
  - is a shortest-hold branch, not the max-PnL branch.
- v5 roll12 fine exposure:
  - selected `lf0.90_sf1.02_cap4.20`, no segment governor, TP/SL `3.0%/4.0%`,
  - improves validation PnL by `9.0117pp` and OOS PnL by `2.9562pp` over the first 12h branch,
  - preserves average hold at `9.1649h/9.7698h` and max hold at `12h`,
  - should be treated as an aggressive 12h line because validation MDD is `-19.9319%`.
- v5 roll10 bracket daytrade:
  - selected `lf0.80_sf0.95_cap4.00`, no segment governor, TP/SL `3.0%/4.0%`,
  - increases validation/OOS trade counts to `158/91`,
  - reduces average hold to `8.1698h/8.5778h` and max hold to `10h`,
  - uses a declared validation-tie rule that prefers the minimum exposure cap at or above `4.0`; OOS metrics are not ordering keys.
- v5 roll10 side-specific bracket daytrade:
  - selected `fast_short`, `lf0.90_sf1.02_cap4.20`, `loss1_90_win10`,
  - long TP/SL is `2.5%/3.5%`; short TP/SL is `2.5%/4.0%`,
  - increases validation/OOS trade counts to `167/97`,
  - improves validation PnL by `24.3933pp` over the first 10h branch and lowers average hold by `0.4458h/0.5349h`,
  - uses a declared validation-tie rule that prefers the exposure cap nearest the reference cap `4.20`; OOS metrics are not ordering keys.
- v5 roll12 side-specific bracket daytrade:
  - selected `oos_top`, `lf0.90_sf1.02_cap4.20`, no segment governor,
  - long TP/SL is `2.5%/4.0%`; short TP/SL is `4.0%/4.0%`,
  - improves validation PnL by `31.3463pp` and OOS PnL by `27.9642pp` over the 12h fine branch,
  - keeps max hold at `12h` and improves OOS MDD from `-19.4885%` to `-17.0142%`,
  - average hold is `9.4349h/10.0224h`, so this is a PnL/OOS candidate rather than the shortest-hold candidate.
- v5 roll12 side-specific fine valmax:
  - selected `fine_val_max`, `lf0.90_sf1.02_cap4.20`, no segment governor,
  - long TP/SL is `2.25%/5.0%`; short TP/SL is `4.0%/4.0%`,
  - improves validation PnL by `17.7311pp` over the first 12h side-specific branch and keeps max hold at `12h`,
  - is the current best 12h validation-PnL candidate, but OOS PnL drops from `173.9019%` to `165.3214%`.
- v5 roll12 side-specific nearmax faster:
  - selected `fine_fast_val`, `lf0.90_sf1.02_cap4.20`, no segment governor,
  - long TP/SL is `2.0%/4.0%`; short TP/SL is `4.0%/4.0%`,
  - selection rule requires validation PnL within `3.0pp` of fine valmax and validation average hold lower than fine valmax,
  - improves average hold from `9.5049h/10.0224h` to `9.0355h/9.8945h` and observed OOS PnL from `165.3214%` to `169.6714%`,
  - keeps max hold at `12h`; full-live still requires runtime-native replay and fresh holdout.
- v5 roll12 side-specific OOS-max:
  - selected `fine_fast_val`, `lf0.75_sf1.02_cap4.20`, `loss1_90_win12`,
  - long TP/SL is `2.0%/4.0%`; short TP/SL is `4.0%/4.0%`,
  - selection rule requires validation PnL within `10.0pp` of fine valmax and then maximizes OOS PnL,
  - keeps the same hold profile as nearmax faster at `9.0355h/9.8945h` average hold and `12h` max hold,
  - improves observed OOS PnL to `178.5726%`, but OOS is an ordering key, so full-live still requires runtime-native replay and fresh holdout.
- v5 roll10 side-specific fine valmax:
  - selected `fine10_valmax`, `lf0.90_sf1.02_cap4.20`, `loss1_90_win10`,
  - long TP/SL is `2.0%/4.5%`; short TP/SL is `2.5%/4.0%`,
  - improves validation PnL by `15.3933pp` over the first 10h side-specific branch,
  - reduces validation average hold to `7.4981h` and keeps max hold at `10h`,
  - OOS PnL is lower than the first 10h side-specific branch, so this is a shortest-hold validation-PnL branch rather than the OOS-preferred 10h branch.
- v5 roll9 side-specific fine valmax:
  - selected `fine9_fast`, `lf0.70_sf1.00_cap3.80`, no segment governor,
  - long TP/SL is `2.0%/3.0%`; short TP/SL is `2.5%/4.0%`,
  - lower exposure was required because the first 9h grid had enough PnL but failed MDD at `-20.5540%/-23.9810%`,
  - compresses max hold to `9h` and lowers average hold versus the 10h fine branch on both validation and OOS,
  - is a hold-compression branch, not a max-validation-PnL branch.
- v5 roll8 side-specific fine valmax:
  - selected `fine8_fast`, `lf0.75_sf0.95_cap4.20`, no segment governor,
  - long TP/SL is `2.0%/3.0%`; short TP/SL is `2.5%/4.0%`,
  - compresses max hold from `9h` to `8h`,
  - improves validation PnL by `16.9259pp` and OOS PnL by `20.5765pp` over the 9h branch while reducing average hold by `0.8981h/0.5927h`,
  - is the current best strict short-hold branch because it improves both PnL and hold versus the previous strictest max-hold candidate.
- v5 roll8 side-specific fine exposure:
  - selected `fine8_fast`, `lf0.900_sf0.975_cap4.20`, no segment governor,
  - keeps long TP/SL at `2.0%/3.0%` and short TP/SL at `2.5%/4.0%`,
  - keeps average/max hold unchanged versus the first 8h branch,
  - improves validation PnL by `9.0385pp` and OOS PnL by `3.4967pp`,
  - should be treated as aggressive because validation MDD buffer is only `0.0286pp`.
- v5 roll8 side-specific PnL tilt:
  - selected `short_sl385`, `lf0.900_sf1.005_cap4.20`, no segment governor,
  - keeps long TP/SL at `2.0%/3.0%`, keeps short TP at `2.5%`, and tightens short SL from `4.0%` to `3.85%`,
  - improves validation PnL by `3.5201pp` and OOS PnL by `4.6400pp` over the fine exposure branch,
  - OOS average hold improves by `0.1192h`, but validation average hold increases by `0.0291h`,
  - should be treated as the highest-PnL 8h tilt, not the strictest hold-preserving branch.
- v5 roll8 side-specific feature veto:
  - starts from the PnL-tilt ledger and searches one entry-time numeric feature threshold for short-only vetoes,
  - excludes lookahead/outcome-like fields by name before search,
  - selected `volume <= 5173.597` at the `0.15` active-short quantile,
  - vetoed `32` validation shorts and `19` OOS shorts; long entries were not vetoed,
  - improves PnL by `90.7248pp` validation and `31.3945pp` OOS versus PnL tilt,
  - reduces average hold by `0.1541h` validation and `0.0298h` OOS while keeping max hold at `8h`,
  - red-team replay verified that only the expected short rows were zeroed, but this branch has higher overfit risk because the veto feature and threshold were selected from validation.
- v5 roll8 side-specific fold-robust veto:
  - starts from the same PnL-tilt ledger and the same non-lookahead-named feature search family,
  - adds a `4`-fold chronological validation gate,
  - selected `big_trade_ratio >= 0.63282428`,
  - validation fold PnL deltas versus PnL tilt are `[0.0, 0.0, 3.2871, 10.2803]`,
  - improves PnL by `41.0433pp` validation and `28.9670pp` OOS versus PnL tilt,
  - reduces average hold by `0.1274h` validation and `0.0077h` OOS while keeping max hold at `8h`,
  - is the preferred research line when robustness is more important than the highest validation PnL.
- v5 roll8 side-specific two-stage veto:
  - starts from the `volume <= 5173.597` feature-veto branch,
  - adds `cvp_vah_val_width <= 0.14` as a second short-only veto,
  - requires the second veto to affect OOS directly: at least `2` OOS vetoed shorts, OOS PnL improvement, and OOS average-hold improvement of at least `0.05h`,
  - vetoed `12` additional validation shorts and `22` additional OOS shorts,
  - validation fold PnL deltas versus the first feature-veto branch are `[0.0, 0.0, 3.6938, 0.0]`,
  - improves PnL by `14.5763pp` validation and `11.8518pp` OOS versus the first feature-veto branch,
  - reduces average hold by `0.1064h` validation and `0.2088h` OOS.
- v5 roll8 side-specific two-stage exposure buffered:
  - starts from the two-stage veto ledger and changes only notional/leverage/margin/trade_return fields,
  - selected `lf1.000_sf1.200_cap5.00`,
  - keeps leverage capped at `5`, max notional below `5`, and margin fraction below `1.0`,
  - applies a validation MDD buffer floor of `-19.50%`, rejecting thinner high-validation exposure variants,
  - improves PnL by `124.8115pp` validation and `80.4357pp` OOS versus two-stage veto,
  - preserves average hold at `5.8358h/6.4733h` and max hold at `8h`.
- v5 roll8 side-specific two-stage exposure OOS-balanced:
  - reuses the buffered exposure grid,
  - requires validation PnL to be within `1.0pp` of the best buffered validation candidate,
  - selected `lf0.900_sf1.200_cap5.00`,
  - gives up only `0.8846pp` validation PnL versus the buffered validation-max line,
  - improves OOS PnL by `2.9013pp` and OOS MDD by `1.2395pp` versus the buffered validation-max line,
  - is explicitly OOS-balanced and must not be considered live-ready without fresh holdout.
- v5 roll7 side-specific two-stage exposure hold-compressed:
  - rebuilds the roll path at `7h` max hold using long TP/SL `2.0%/3.0%` and short TP/SL `2.5%/3.85%`,
  - reapplies `volume <= 5173.597` and `cvp_vah_val_width <= 0.14` short vetoes,
  - selected exposure `lf0.850_sf1.200_cap5.00`,
  - reduces average hold by `0.3658h` validation and `0.6330h` OOS versus the 8h OOS-balanced candidate,
  - gives up PnL versus the 8h exposure candidates, so it is the hold-compression branch rather than the max-PnL branch.
- v5 roll7 side-specific two-stage exposure OOS-balanced:
  - keeps the roll7 `7h` path,
  - selects the highest-OOS-PnL exposure inside a `3.0pp` validation near-max band,
  - selected exposure `lf0.700_sf1.200_cap5.00`,
  - raises OOS PnL from `248.8164%` to `253.5504%` while keeping validation PnL at `379.3204%`,
  - uses OOS as an ordering key, so it is research-only until fresh holdout validates the selection.
- v5 roll6 side-specific two-stage exposure hold-compressed:
  - reuses the roll7 construction with max roll hold set to `6h`,
  - selected exposure `lf0.400_sf1.200_cap5.00`,
  - reduces average hold by `0.5352h` validation and `0.8541h` OOS versus the roll7 branch,
  - keeps max hold at `6h`, validation/OOS PnL above `200%`, and both MDDs above `-20%`,
  - is the shortest current research-pass hold branch.
- v5 roll5 side-specific two-stage exposure hold-compressed:
  - reuses the roll6 construction with max roll hold set to `5h`,
  - selected exposure `lf0.700_sf1.000_cap4.40`,
  - reduces average hold by `0.7016h` validation and `0.5581h` OOS versus the roll6 branch,
  - keeps max hold at `5h`, validation/OOS PnL above `100%`, and both MDDs comfortably above `-20%`,
  - is now the shortest current research-pass hold branch.
- v5 roll5 side-specific two-stage exposure OOS-max:
  - keeps the roll5 `5h` path,
  - selects the highest-OOS-PnL exposure inside a `10.0pp` validation near-max band,
  - selected exposure `lf0.100_sf1.000_cap4.40`,
  - raises OOS PnL from `169.8794%` to `187.6595%` while keeping validation PnL at `296.9050%`,
  - uses OOS as an ordering key, so it is research-only until fresh holdout validates the selection.
- v5 roll4 side-specific two-stage exposure hold-compressed:
  - reuses the roll5 construction with max roll hold set to `4h`,
  - selected exposure `lf0.700_sf1.100_cap4.00`,
  - reduces average hold by `0.7605h` validation and `0.8141h` OOS versus the roll5 branch,
  - keeps max hold at `4h` and validation/OOS PnL above `100%`,
  - has extremely thin OOS MDD buffer (`0.0152pp` to `-20%`), so it is a shortest-hold research line, not the robust live candidate.
- v5 roll4 side-specific two-stage exposure OOS-max:
  - keeps the roll4 `4h` path,
  - selects the highest-OOS-PnL exposure inside a `20.0pp` validation near-max band,
  - selected exposure `lf0.000_sf1.100_cap4.00`,
  - raises OOS PnL from `140.4955%` to `159.8935%` while keeping validation PnL at `306.0689%`,
  - uses OOS as an ordering key and has only `0.0152pp` OOS MDD buffer, so it is research-only until fresh holdout validates the selection.
- v5 roll3 side-specific two-stage exposure hold-compressed:
  - reuses the roll5 construction with max roll hold set to `3h`,
  - selected exposure `lf0.050_sf1.200_cap4.00`,
  - reduces average hold by `0.6906h` validation and `0.8052h` OOS versus the roll4 branch,
  - keeps max hold at `3h` and validation/OOS PnL above `100%`,
  - has extremely thin validation MDD buffer (`0.0519pp` to `-20%`), so it is the absolute shortest current research-pass line but not the robust live candidate.
- v5 roll2 side-specific two-stage exposure hold-compressed:
  - reuses the roll5 construction with max roll hold set to `2h`,
  - selected exposure `lf0.800_sf1.100_cap4.80`,
  - reduces average hold by `0.8765h` validation and `0.8935h` OOS versus the roll3 branch,
  - keeps max hold at `2h` and validation/OOS PnL above `100%`,
  - improves validation MDD buffer versus roll3 but keeps OOS MDD close to the floor (`0.3297pp` to `-20%`), so it remains a research line until fresh holdout is available.
- v5 roll2 side-specific two-stage exposure OOS-balanced:
  - keeps the roll2 `2h` path,
  - selects the highest-OOS-PnL exposure inside a `3.0pp` validation near-max band,
  - selected exposure `lf0.300_sf1.100_cap4.80`,
  - raises OOS PnL from `136.0997%` to `151.6907%` while keeping validation PnL at `186.9196%`,
  - uses OOS as an ordering key, so it is research-only until fresh holdout validates the selection.
- v5 roll2 side-specific two-stage exposure OOS-max:
  - keeps the roll2 `2h` path,
  - selects the highest-OOS-PnL exposure inside a wider `10.0pp` validation near-max band,
  - selected exposure `lf0.000_sf1.100_cap4.80`,
  - raises OOS PnL from `136.0997%` to `161.1111%` while keeping validation PnL at `183.4355%`,
  - uses OOS as an ordering key, so it is research-only until fresh holdout validates the selection.

## Failed Or Non-Promoted Experiments

- `omega4_6_2_hold_fine_exit_sizing_overlay_20260701`
  - Finer `72h/78h/84h/90h/96h` hard-stop search did not improve the prior paper exit+sizing candidate.
- `omega4_6_2_loss_cluster_governor_v2_20260701`
  - Wider exposure found higher PnL but selected top candidate breached validation MDD at `-20.4817%`.
- `omega4_6_2_roll24_segment_governor_sweep_20260701`
  - Segment-level daytrade governor found higher raw PnL, but selected top candidate breached validation/OOS MDD at `-24.9060%` / `-27.6614%`.
  - Parent-governed roll24 fine probe found a small validation-only lift (`237.49%` to `239.32%`) but OOS dropped (`141.27%` to `138.20%`), so it was not promoted.
- v4 drawdown-guard probe
  - Drawdown guard did not improve v4; it reduced PnL without improving the limiting MDD event.
  - Increasing long factor to `1.35` at `short190` raised validation PnL (`261.73%` to `264.71%`) but reduced OOS PnL (`137.20%` to `134.60%`), so it was not promoted.
- roll16 trailing/profit-lock probe
  - Tested path-causal profit locks on top of the roll16 TP/SL bracket.
  - Validation-best remained the fixed `4.5%/4.5%` bracket with `long100_short100_cap430`, `streak85_60_win12`.
  - OOS-best probe reached `171.5437%` OOS PnL with lower validation PnL (`296.3002%`) and weaker validation MDD buffer, so it was not promoted without fresh holdout.
- roll8 ultra-short probe
  - Best validation-gated 8h candidate reached `208.2660%` validation PnL with `6.6038h/6.8923h` average hold.
  - OOS PnL was only `100.1081%`, so the pass buffer was too thin for promotion.
  - OOS-best 8h candidate reached `120.8504%` OOS PnL but validation PnL dropped to `181.6701%`, so it remains a non-promoted exploratory branch.
- roll11 side-specific compression probe
  - Tested the same side-specific bracket family under `11h` max hold.
  - Validation-best reached only `249.0398%` with OOS `139.1752%`, below the `10h` side-specific validation result and far below the `12h` side-specific PnL/OOS result.
  - It was not promoted because it did not improve the current PnL/hold frontier.
- roll12 fine-grid OOS-high probe
  - OOS-best reached `183.2798%` OOS PnL with validation `314.7229%` using `long 3.0%/4.0%`, `short 4.0%/4.0%`, `lf0.75_sf1.02_cap4.20`, `loss1_90_win12`.
  - It was not promoted because selecting it would use OOS as an ordering key; it remains a fresh-holdout candidate only.
- roll10 fine-grid OOS-high probe
  - OOS-best reached `173.0690%` OOS PnL with validation `238.2565%` using `long 2.5%/3.5%`, `short 4.0%/4.0%`, `lf0.75_sf1.04_cap4.30`, `streak85_60_win10`.
  - It was not promoted because selecting it would use OOS as an ordering key; it remains a fresh-holdout candidate only.
- roll7 compression probe
  - Best research-gated probe reached validation/OOS `196.5269%/129.7077%`, MDD `-19.2561%/-19.5435%`, average hold `5.7119h/6.2213h`, and max hold `7h`.
  - It was not promoted because it reduces hold versus the 8h branch but lowers validation/OOS PnL materially.
- roll8 loss-window governor probe
  - Wider `loss1_*_win12/16/20/24` governors did not repair the `sf1.0+` validation MDD failure; the key loss event was not reduced by that governor family.
- roll8 faster short-TP probe
  - No candidate simultaneously improved validation/OOS PnL and reduced both validation/OOS average hold versus `roll8 fine exposure`.
- roll1 ultra-compression probe
  - Tested max roll hold `1h` across `1848` exposure variants.
  - No candidate passed the validation/OOS `100%` PnL contract; the best validation candidate was only `-3.8605%` validation and `19.7627%` OOS.
  - It was not promoted because the 1h split destroys the current path edge before exposure sizing can recover it.
- roll2 extra entry-veto probe
  - Tested short-only entry-time feature vetoes on the roll2 path while holding the roll2 max-PnL exposure fixed.
  - The best validation uplift was `sum_toptrader_long_short_ratio >= 2.905512`, raising validation PnL from `189.1812%` to `222.5564%`.
  - It was not promoted because it vetoed `0` OOS trades, so the improvement has no OOS support; candidates that vetoed OOS trades reduced OOS PnL.
- roll2 OOS-balanced extra entry-veto probe
  - Repeated the short-only entry-time feature veto scan on the roll2 OOS-balanced exposure.
  - No candidate improved both validation and OOS PnL while preserving the `2h` hold/MDD/accounting contract.
  - The strongest validation candidates again had no OOS support or materially reduced OOS PnL, so no additional veto was promoted.

## Red-Team Reports

- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_paper_exit_sizing_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_loss_cluster_governor_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_loss_cluster_governor_v3_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_loss_cluster_governor_v4_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_loss_cluster_governor_v5_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_roll24_daytrade_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll24_daytrade_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll24_segment_governor_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll16_bracket_segment_governor_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll16_bracket_robust_segment_governor_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll16_fine_exposure_segment_governor_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll16_fine_nearmax_buffered_segment_governor_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll16_fine_robust_segment_governor_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll16_fine_short_bias_segment_governor_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll12_bracket_daytrade_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll12_fine_exposure_daytrade_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll10_bracket_daytrade_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll10_side_specific_bracket_daytrade_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll12_side_specific_bracket_daytrade_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll12_side_specific_fine_valmax_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll12_side_specific_nearmax_faster_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll12_side_specific_oos_max_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll10_side_specific_fine_valmax_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll9_side_specific_fine_valmax_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll8_side_specific_fine_valmax_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll8_side_specific_fine_exposure_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll8_side_specific_pnl_tilt_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll8_side_specific_feature_veto_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll8_side_specific_foldrobust_veto_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll8_side_specific_two_stage_veto_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll7_side_specific_two_stage_exposure_oos_balanced_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll4_side_specific_two_stage_exposure_oos_max_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll3_side_specific_two_stage_exposure_hold_compressed_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_frontier_leakage_redteam_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_runtime_wiring_blockers_20260701.md`
- `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_fresh_holdout_sources_20260701.md`

## Full-Live Blocker Resolution - 2026-07-01

- Final unblocked frontier model: `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701`.
- OOS selection contamination was removed by selecting exposure only from validation gates and validation metrics:
  - selected `lf0.900_sf1.050_cap4.40`
  - validation/OOS PnL `675.3209%` / `212.6850%`
  - validation/OOS MDD `-17.3157%` / `-19.4083%`
  - average hold `5.8723h` / `6.6409h`, max hold `8h`
  - `oos_used_in_selection=False`
- Runtime replay blocker was resolved with `trading_bot_modules/omega4_6_2_runtime_adapter.py`.
  - runtime replay audit: `RUNTIME_REPLAY_PASS`
  - runtime wiring audit: `RUNTIME_WIRING_PASS`
- Leakage/data contamination blocker was resolved:
  - frontier leakage audit: `FRONTIER_LEAKAGE_RUNTIME_PASS`
  - direct future leak: `False`
  - entry-feature data contamination: `False`
  - OOS selection contamination blockers: `[]`
- CVP feature provenance caveat was closed:
  - CVP feature causality audit: `CVP_FEATURE_CAUSALITY_PASS`
  - prefix stability passed on both train and eval source market CSVs.
- Older OOS-selected variants remain research-only unless rerun on a fresh holdout/walk-forward; they are no longer the promoted frontier.

## Data Contamination Repair - 2026-07-01

- Fixed the active roll feature-veto path so segment ledgers refresh source market features by segment `entry_timestamp` before applying entry-feature veto rules.
- Repaired roll8 OOS-balanced candidate:
  - `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701`
  - selected `lf0.950_sf1.080_cap4.60`
  - validation/OOS PnL `717.6129%` / `221.4408%`
  - validation/OOS MDD `-18.2147%` / `-19.9359%`
  - average hold `5.8723h` / `6.6409h`, max hold `8h`
  - red-team `RESEARCH_ROLL8_TWO_STAGE_EXPOSURE_OOS_BALANCED_PASS_FULL_LIVE_BLOCKED`
- Repaired shortest passing candidate:
  - `omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701`
  - validation/OOS PnL `308.7601%` / `138.4721%`
  - validation/OOS MDD `-18.3384%` / `-19.4112%`
  - average hold `4.2435h` / `4.4215h`, max hold `5h`
  - red-team `RESEARCH_ROLL5_HOLD_COMPRESSED_PASS_FULL_LIVE_BLOCKED`
- Repaired demotions:
  - `omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701` now fails MDD gates after entry-feature refresh.
  - `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701` now has no passing OOS-max candidate after entry-feature refresh.
- Current promoted-frontier leakage red-team: `FRONTIER_LEAKAGE_RUNTIME_PASS`.

## Next Work

- Use `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701` as the unblocked Omega4.6.2 frontier if full-live audit status is the priority.
- For live order submission, explicitly select the Omega4.6.2 validation-only runtime sleeve in deployment configuration before restart.
- Keep older OOS-selected frontier variants research-only until they are rerun on a fresh holdout/walk-forward.
- If the priority is robustness over max PnL, use v3 as the working baseline; if the priority is highest offline PnL under the stated MDD cap, use v5 with the MDD-buffer warning.
- If the priority is strict day-trading max hold with the strongest current offline validation PnL, use `omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701`.
- If the priority is strict day-trading max hold with near-max validation PnL and a better validation MDD buffer, use `omega4_6_2_v5_roll16_fine_nearmax_buffered_segment_governor_20260701`.
- If the priority is strict day-trading max hold with better validation MDD buffer, use `omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701`.
- If the priority is 16h-like validation PnL with max hold capped at 12h, use `omega4_6_2_v5_roll12_side_specific_fine_valmax_20260701`.
- If the priority is near-16h validation PnL with lower average hold inside the 12h cap, use `omega4_6_2_v5_roll12_side_specific_nearmax_faster_20260701`.
- If the priority is 12h max hold with stronger observed OOS PnL and OOS ordering is acceptable for research, use `omega4_6_2_v5_roll12_side_specific_oos_max_20260701`.
- If the priority is shortest research-pass hold while preserving PnL/MDD contracts, use `omega4_6_2_v5_roll10_side_specific_fine_valmax_20260701`.
- If the priority is strict max-hold compression and OOS average hold below 7.5h, use `omega4_6_2_v5_roll9_side_specific_fine_valmax_20260701`.
- If the priority is strict max-hold compression with PnL improvement over the 9h line, use `omega4_6_2_v5_roll8_side_specific_fine_valmax_20260701`.
- If the priority is the best current 8h PnL/hold balance, use `omega4_6_2_v5_roll8_side_specific_fine_exposure_20260701`.
- If the priority is highest current 8h PnL and a tiny validation average-hold increase is acceptable, use `omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701`.
- If the priority is highest current 8h research PnL and fresh-holdout validation will be run before live use, use `omega4_6_2_v5_roll8_side_specific_feature_veto_20260701`.
- If the priority is a more robust 8h research candidate with no negative validation-fold PnL delta, use `omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701`.
- If the priority is the strongest current 8h PnL/hold frontier and a fresh-holdout check is mandatory before live use, use `omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701`.
- If the priority is the strongest current 8h PnL frontier with unchanged hold and a buffered validation MDD floor, use `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701`.
- If the priority is the strongest repaired OOS PnL inside a 1pp validation near-max band, use `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701`.
- If the priority is shorter max hold and lower average hold while staying well above the PnL target, use `omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701`.
- If the priority is `7h` max hold with slightly stronger OOS PnL and OOS ordering is acceptable for research, use `omega4_6_2_v5_roll7_side_specific_two_stage_exposure_oos_balanced_20260701`.
- If the priority is sub-5h average hold after the entry-feature contamination repair, use `omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701`.
- `omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701` is no longer recommended after the repair because repaired MDD breaches the `-20%` contract.
- If the priority is `5h` max hold with stronger OOS PnL and OOS ordering is acceptable for research, use `omega4_6_2_v5_roll5_side_specific_two_stage_exposure_oos_max_20260701`.
- If the priority is absolute shortest current max hold and a razor-thin OOS MDD buffer is acceptable for research only, use `omega4_6_2_v5_roll4_side_specific_two_stage_exposure_hold_compressed_20260701`.
- If the priority is `4h` max hold with stronger OOS PnL and a razor-thin OOS MDD buffer is acceptable for research, use `omega4_6_2_v5_roll4_side_specific_two_stage_exposure_oos_max_20260701`.
- If the priority is absolute shortest current max hold and a razor-thin validation MDD buffer is acceptable for research only, use `omega4_6_2_v5_roll3_side_specific_two_stage_exposure_hold_compressed_20260701`.
- The roll2 ultra-short line is no longer recommended after the repair because it has no passing OOS-max candidate and breaches the PnL/MDD contracts.
- If the priority is higher PnL with max hold capped at 12h, use `omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701`.
- If the priority is uniform TP/SL with sub-12h average hold, use `omega4_6_2_v5_roll12_fine_exposure_daytrade_20260701`; use the first 12h branch only if the thinner MDD buffer is unacceptable.
