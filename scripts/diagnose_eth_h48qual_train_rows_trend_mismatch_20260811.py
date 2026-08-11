# 재현성 참고: 이 스크립트는 h48qual 자체 리서치 패널(fa_features.parquet, 145컬럼)에 의존하는데,
# 이 파일은 세션 scratchpad에만 있고 레포에 커밋되지 않았다 -- 기존 FINAL12 dedup 작업과 동일한
# 재현성 갭. docs/experiments/eth_h48qual_final12_feature_selection_20260811.md 참고.
import pandas as pd, numpy as np

ROOT = "/home/kbj20/crypto-scalping/"
TRAIN_CSV = ROOT + "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
H384_LABEL = ROOT + "tmp/eth_h384_conservative_padded_to_zigzag_timestamps_20260811/zigzag_action_labels_2025.csv"
ZZ_LABEL = ROOT + "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531/zigzag_action_labels_2025.csv"
REGIME3 = ROOT + "data/ensemble/supervised/eth_regime3_current_hmm_jmredesign_20260810_2025_maskedname.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")

def _read(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

train_all = _read(TRAIN_CSV)
train_raw = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
print(f"train_raw 전체 n={len(train_raw)}  범위 {train_raw['timestamp'].min()} ~ {train_raw['timestamp'].max()}")

A = train_raw.iloc[:30000]
B = train_raw.iloc[30000:45000]
print(f"\nWINDOW A (rows=30000 cap) n={len(A)}  {A['timestamp'].min()} ~ {A['timestamp'].max()}")
print(f"WINDOW B (30000~45000 추가분) n={len(B)}  {B['timestamp'].min()} ~ {B['timestamp'].max()}")

h384 = _read(H384_LABEL).rename(columns={"zigzag_action": "h384_action"})
zz = _read(ZZ_LABEL)
reg3 = _read(REGIME3)

def window_stats(win, name):
    ts = win[["timestamp"]]
    m_h384 = ts.merge(h384[["timestamp", "h384_action"]], on="timestamp", how="left")
    m_zz = ts.merge(zz[["timestamp", "zigzag_action"]], on="timestamp", how="left")
    reg_cols = [c for c in reg3.columns if c != "timestamp"]
    m_reg = ts.merge(reg3, on="timestamp", how="left")
    print(f"\n=== {name} (n={len(win)}) ===")
    print("h384(quality) 분포:", m_h384["h384_action"].value_counts(normalize=True).sort_index().round(3).to_dict(),
          " missing:", int(m_h384["h384_action"].isna().sum()))
    print("zigzag(direction) 분포:", m_zz["zigzag_action"].value_counts(normalize=True).sort_index().round(3).to_dict(),
          " missing:", int(m_zz["zigzag_action"].isna().sum()))
    print("regime3 컬럼:", reg_cols)
    for c in reg_cols:
        if pd.api.types.is_numeric_dtype(m_reg[c]):
            print(f"  {c}: mean={m_reg[c].mean():.4f} std={m_reg[c].std():.4f}")
    # basic price action
    close = win["close"].astype(float).to_numpy()
    ret = np.diff(close) / close[:-1]
    print(f"5분봉 수익률 std(변동성 proxy): {np.std(ret):.5f}")
    print(f"기간내 close 변화율: {(close[-1]/close[0]-1)*100:.2f}%")
    high = win["high"].astype(float).to_numpy(); low = win["low"].astype(float).to_numpy()
    tr = np.maximum(high-low, np.maximum(np.abs(high-np.roll(close,1)), np.abs(low-np.roll(close,1))))
    print(f"평균 true range/close: {(tr/np.where(close!=0,close,np.nan)).mean()*100:.4f}%")

window_stats(A, "WINDOW A (0~29999, rows=30000이 쓰는 구간)")
window_stats(B, "WINDOW B (30000~44999, rows=45000이 추가로 넣는 구간)")

print("\n\n=== VAL/OOS 구간 자체의 트렌드는 어느 쪽에 더 가까운가 ===")
EVAL_CSV = ROOT + "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
eval_all = _read(EVAL_CSV)

val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
print(f"VAL(2025-10-01~) n={len(val_raw)}  {val_raw['timestamp'].min()} ~ {val_raw['timestamp'].max()}  "
      f"close 변화율: {(val_raw['close'].iloc[-1]/val_raw['close'].iloc[0]-1)*100:.2f}%")
print(f"OOS(2026~) n={len(eval_all)}  {eval_all['timestamp'].min()} ~ {eval_all['timestamp'].max()}  "
      f"close 변화율: {(eval_all['close'].iloc[-1]/eval_all['close'].iloc[0]-1)*100:.2f}%")

for name, w in [("WINDOW A(30k)", A), ("WINDOW B(추가분)", B), ("A+B(45k 전체)", train_raw.iloc[:45000]),
                ("VAL", val_raw), ("OOS", eval_all)]:
    c = w["close"].astype(float)
    print(f"  {name}: {c.iloc[0]:.1f} -> {c.iloc[-1]:.1f}  ({(c.iloc[-1]/c.iloc[0]-1)*100:+.2f}%)  "
          f"min={c.min():.1f} max={c.max():.1f}")
