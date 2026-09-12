"""라벨이 배리어 선택의 산물인가, 아니면 더 일반적인 방향 성향인가."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/home/llewyn/crypto-scalping")
for p in (ROOT, ROOT/"scripts"): sys.path.insert(0, str(p))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega
import retest_omega4_6_1_extended_oos_20260706 as retest
import research_omega461_entry_condition_lift_20260910 as lift
import research_omega461_lagfree_direction_label_20260910 as lf

_fee, slip = omega._load_fee_slip()
frame = retest.load_frame_current("2026-01-01", "2026-08-30")
close, opn = frame["close"].to_numpy(float), frame["open"].to_numpy(float)
W = 288
POL = [(0.075,0.040,"배포 7.5/4.0"), (0.080,0.030,"스윕최고 8.0/3.0"),
       (0.030,0.015,"좁은 3.0/1.5"), (0.050,0.050,"대칭 5.0/5.0"),
       (0.150,0.080,"넓은 15/8")]
advs, labs = {}, {}
for tp, sl, nm in POL:
    o = lift._outcomes(close, opn, tp, sl, slip)
    a = lf._fwd_wr(o[:,0], W) - lf._fwd_wr(o[:,1], W)
    advs[nm] = a
    labs[nm] = np.where(np.isnan(a), 0, np.where(a > 0.10, 1, np.where(a < -0.10, -1, 0)))
    print(f"  {nm:18s} 롱 {int((labs[nm]==1).sum()):6,d}  숏 {int((labs[nm]==-1).sum()):6,d}  "
          f"현금 {int((labs[nm]==0).sum()):6,d}", flush=True)

names = [n for _,_,n in POL]
print("\n=== 연속 adv 상관 (스피어만) ===")
print(f"{'':18s}" + "".join(f"{n[:10]:>12s}" for n in names))
for a in names:
    row = f"{a:18s}"
    for b in names:
        m = np.isfinite(advs[a]) & np.isfinite(advs[b])
        row += f"{pd.Series(advs[a][m]).corr(pd.Series(advs[b][m]), method='spearman'):12.3f}"
    print(row)

print("\n=== 이산 라벨 부호 일치율 (둘 다 비현금인 봉만) ===")
print(f"{'':18s}" + "".join(f"{n[:10]:>12s}" for n in names))
for a in names:
    row = f"{a:18s}"
    for b in names:
        m = (labs[a]!=0) & (labs[b]!=0)
        row += f"{float((labs[a][m]==labs[b][m]).mean())*100:11.1f}%"
    print(row)
base = "배포 7.5/4.0"
others = [n for n in names if n != base]
ms = [float((labs[base][(labs[base]!=0)&(labs[n]!=0)]==labs[n][(labs[base]!=0)&(labs[n]!=0)]).mean()) for n in others]
print(f"\n배포 라벨 vs 나머지 4종 일치율 중앙 {np.median(ms)*100:.1f}%  (무작위 = 50%)")
