import pandas as pd

df_raw = pd.read_csv('/home/llewyn/crypto-scalping/data/training_features_5m.csv', usecols=['timestamp', 'session_us', 'hour_cos'])
df_rl = pd.read_csv('data/rl_training_data_full.csv', usecols=['session_us', 'hour_cos'])
diff_len = len(df_raw) - len(df_rl)

# RL 데이터셋은 원본 데이터에서 앞부분 일부가 잘려나간 형태입니다. 끝부분부터 정렬합니다.
df_raw_tail = df_raw.iloc[diff_len:].reset_index(drop=True)

print("=== 데이터 정렬 검증 ===")
match_rate = (df_raw_tail['hour_cos'].round(4) == df_rl['hour_cos'].round(4)).mean()
print(f"원본 vs RL데이터(끝부분) 정렬 일치율 (hour_cos 기준): {match_rate:.2%}")

print("\n=== 정확한 시간 매핑 샘플 검증 (UTC) ===")
# 정렬이 맞다면, df_rl의 session_us == 1.0인 곳의 df_raw_tail의 timestamp를 확인합니다.
active = df_raw_tail[df_rl['session_us'] == 1.0]
inactive = df_raw_tail[df_rl['session_us'] == 0.0]

print("[session_us == 1.0 (미국장 활성) 첫 3개]")
for idx, row in active.head(3).iterrows():
    print(f"  {row['timestamp']} | session_us: {row['session_us']} | hour_cos: {row['hour_cos']:.3f}")

print("\n[session_us == 0.0 (미국장 비활성) 첫 3개]")
for idx, row in inactive.head(3).iterrows():
    print(f"  {row['timestamp']} | session_us: {row['session_us']} | hour_cos: {row['hour_cos']:.3f}")

print("\n[임의의 활성 시간 샘플 3개]")
for idx, row in active.sample(3, random_state=42).iterrows():
    print(f"  {row['timestamp']} | session_us: {row['session_us']} | hour_cos: {row['hour_cos']:.3f}")
