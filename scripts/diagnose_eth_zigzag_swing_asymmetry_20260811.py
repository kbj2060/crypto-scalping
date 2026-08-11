"""eth_h48qual_zig075_direction_confidence_echo_check_20260811.md가 남긴 미검증 가설 검증:
direction_head가 숏 콜에 3~5pp 더 높은 confidence를 갖도록 학습된 게, 학습구간 자체에서 숏
zigzag 스윙이 롱 스윙보다 더 "교과서적"(가파르고 덜 눌림)이었기 때문 아닌가?

zigzag_action_labels_20260531 CSV(build_wave3_action_labels_20260531.py 산출)의 path_* 컬럼은
스윙 시작bar(zigzag_segment_id의 첫 행)에서 이미 "그 지점에 진입해서 스윙 끝까지 들고 갔을 때"의
mae/mfe/calmar/edge를 계산해둔 값이라, groupby(segment_id).first()만으로 스윙 단위 통계가 바로
나온다 -- 재계산 불필요, 라벨 빌더가 이미 검증한 값 재사용.

비교 지표: wave_bars(지속시간), |wave_return|(크기), steepness=|wave_return|/wave_bars(각도),
path_mae/|wave_return|(상대 눌림폭 -- 낮을수록 깔끔), path_calmar(수익/고통 비율 -- 높을수록
깔끔), path_edge(위험조정 엣지). Mann-Whitney U로 LONG vs SHORT 비교(confidence-echo 문서와
동일 검정)."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path("/home/kbj20/crypto-scalping")
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"

TRAIN_START, TRAIN_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-09-30")
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28")


def _read(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


frames = [_read(LABEL_DIR / "zigzag_action_labels_2024.csv"), _read(LABEL_DIR / "zigzag_action_labels_2025.csv")]
oos_path = LABEL_DIR / "zigzag_action_labels_2026.csv"
if oos_path.exists():
    frames.append(_read(oos_path))
all_df = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
print(f"[로드] {all_df['timestamp'].min()} ~ {all_df['timestamp'].max()}, n={len(all_df)}")


def swing_table(frame: pd.DataFrame) -> pd.DataFrame:
    sub = frame[frame["zigzag_segment_id"] >= 0].copy()
    # 주의 3가지, 전부 build_wave3_action_labels_20260531.py의 버퍼 적용부(223-245행) 직접
    # 확인: (1) transition_buffer(±2bar)가 label(zigzag_action_name)을 CASH로 덮어쓰지만
    # zigzag_segment_id는 유지 -- 모든 세그먼트 시작이 "전환점"이라 action에 단순 .first()를
    # 쓰면 거의 매번 버퍼의 CASH가 잡힌다. (2) 버퍼는 zigzag_path_mae/mfe/calmar/edge/return도
    # **0.0으로 같이 덮어쓴다** -- 이 컬럼들도 세그먼트 전체에서 상수가 아니라 버퍼 구간만 0이 됨.
    # zigzag_transition_buffer(0/1) 플래그로 버퍼 아닌 행만 남긴 뒤 그 중 첫 행을 써야 진짜
    # 방향·path 통계가 나온다(wave_return/wave_bars는 버퍼와 무관하게 세그먼트 전체 상수라
    # 버퍼 포함 여부 무관). (3) **`zigzag_segment_id`는 연도별 원본 CSV마다 0(또는 -1)부터
    # 다시 시작한다** -- 2024/2025 두 해를 concat한 뒤 segment_id 하나로만 groupby하면 서로
    # 무관한 두 해의 스윙이 같은 id로 합쳐진다(2024 id 0..1842 전부가 2025 id 범위와 겹침,
    # 직접 확인: 학습window에서 이 버그로 3443개 스윙이 1843개로 반토막 나고 2025 1~9월분
    # ~1600개가 통째로 유실됨 -- 2026-08-11 발견/수정). `year` 보조키로 연도를 구분해야 한다.
    sub["year"] = sub["timestamp"].dt.year
    gkey = ["year", "zigzag_segment_id"]
    clean = sub[sub["zigzag_transition_buffer"] == 0]
    action_map = clean.groupby(gkey)["zigzag_action_name"].first()
    path_cols = clean.groupby(gkey).agg(
        path_mae=("zigzag_path_mae", "first"),
        path_mfe=("zigzag_path_mfe", "first"),
        path_calmar=("zigzag_path_calmar", "first"),
        path_edge=("zigzag_path_edge", "first"),
    )
    seg = sub.groupby(gkey).agg(
        start_ts=("timestamp", "first"),
        wave_bars=("zigzag_wave_bars", "first"),
        wave_return=("zigzag_wave_return", "first"),
        atr_pct=("zigzag_atr_pct", "first"),
    ).reset_index()
    seg["action"] = seg.set_index(gkey).index.map(action_map)
    seg = seg.merge(path_cols, on=gkey, how="left")
    seg = seg[seg["action"].isin(["LONG", "SHORT"])].copy()
    seg["abs_return"] = seg["wave_return"].abs()
    seg["steepness"] = seg["abs_return"] / seg["wave_bars"].clip(lower=1)
    seg["relative_mae"] = seg["path_mae"] / seg["abs_return"].clip(lower=1e-6)
    seg["overshoot_ratio"] = seg["path_mfe"] / seg["abs_return"].clip(lower=1e-6)
    return seg


def compare(seg: pd.DataFrame, label: str):
    long_s = seg[seg["action"] == "LONG"]
    short_s = seg[seg["action"] == "SHORT"]
    print(f"\n--- {label}: LONG 스윙 {len(long_s)}개 vs SHORT 스윙 {len(short_s)}개 ---")
    metrics = [
        ("wave_bars", "지속시간(bar)", False),
        ("abs_return", "스윙크기(|%|)", False),
        ("steepness", "각도(|%|/bar)", False),
        ("relative_mae", "상대눌림폭(낮을수록 깔끔)", True),
        ("path_calmar", "calmar(높을수록 깔끔)", False),
        ("path_edge", "위험조정엣지", False),
        ("overshoot_ratio", "오버슈트비(1.0=끝점이 극값)", True),
    ]
    for col, name, lower_is_cleaner in metrics:
        l = long_s[col].replace([np.inf, -np.inf], np.nan).dropna()
        s = short_s[col].replace([np.inf, -np.inf], np.nan).dropna()
        u, p = mannwhitneyu(s, l, alternative="two-sided")
        direction = "SHORT가 더 깔끔" if (s.median() < l.median()) == lower_is_cleaner else "LONG이 더 깔끔"
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {name:>28}: LONG 중앙값={l.median():+.5f}  SHORT 중앙값={s.median():+.5f}  "
              f"MWU p={p:.2e}{sig:<3} -> {direction}")


for label, start, end in [("학습구간(2024+2025 1~9월)", TRAIN_START, TRAIN_END),
                           ("VAL(2025-10~12)", VAL_START, VAL_END),
                           ("OOS(2026-01~02)", OOS_START, OOS_END)]:
    window = all_df[(all_df["timestamp"] >= start) & (all_df["timestamp"] <= end)]
    if window.empty:
        print(f"\n[스킵] {label}: 데이터 없음")
        continue
    seg = swing_table(window)
    compare(seg, label)
