import pandas as pd
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def update_rl_features_only_sessions(features_csv: str, rl_csv: str):
    logger.info("🚀 데이터 로드 시작...")
    # 원본 파일에서 필요한 세션 데이터만 가져오기
    df_features = pd.read_csv(features_csv, usecols=['session_asia', 'session_europe', 'session_us', 'is_hour_open'])
    
    # RL 데이터셋 로드
    df_rl = pd.read_csv(rl_csv)
    
    logger.info(f"📊 원본 데이터 크기: {len(df_features)} 행 | RL 데이터 크기: {len(df_rl)} 행")
    
    # RL 데이터셋이 원본의 "최신 데이터 N개" 또는 "일부 스킵 후의 데이터"일 수 있습니다.
    # 하지만 기본적으로 train_rl_agent.csv는 Feature 데이터를 그대로 iterrows 또는 루프로 추출합니다.
    # 만약 개수가 다르면 RL 데이터에 timestamp가 없으므로 완벽한 매핑이 불가능할 수 있습니다.
    
    # 그러나 확인 결과 RL 데이터는 모델 환경에 맞도록 잘라진 형태일 가능성이 큽니다.
    # 두 데이터의 개수가 정확히 일치한다면 그대로 덮어씁니다.
    if len(df_features) == len(df_rl):
        logger.info("✅ 행 개수가 일치하므로 다이렉트 1:1 매핑을 진행합니다.")
        for col in ['session_asia', 'session_europe', 'session_us', 'is_hour_open']:
            if col in df_rl.columns:
                df_rl[col] = df_features[col]
                logger.info(f"  - {col} 컬럼 업데이트 완료")
            else:
                logger.warning(f"  - RL 데이터에 {col} 컬럼이 없어 추가하지 않았습니다.")
                
    else:
        # 개수가 다를 경우, 주로 앞의 N개(히스토리 부족)를 자르고 뒤부터 맞췄을 가능성이 매우 높습니다 (ex. lookback window 288개 제외)
        diff = len(df_features) - len(df_rl)
        logger.warning(f"⚠️ 행 개수가 다릅니다. (원본이 {diff}개 더 많음)")
        logger.info(f"윈도우 크기(ex. 288 등) 만큼 앞에서 스킵되었다고 가정하고 꼬리(끝)부터 길이를 맞춰 매핑합니다.")
        
        # 끝에서부터 RL 데이터 개수만큼 가져오기
        df_cut = df_features.iloc[-len(df_rl):].reset_index(drop=True)
        
        for col in ['session_asia', 'session_europe', 'session_us', 'is_hour_open']:
            if col in df_rl.columns:
                df_rl[col] = df_cut[col]
                logger.info(f"  - {col} 컬럼 꼬리 매핑 (마지막 {len(df_rl)}개) 완료")
            else:
                pass
                
    logger.info(f"💾 {rl_csv} 파일 덮어쓰기 저장 중...")
    df_rl.to_csv(rl_csv, index=False)
    logger.info("🎉 덮어쓰기가 완료되었습니다!")

if __name__ == "__main__":
    FEATURES_CSV = "/home/llewyn/crypto-scalping/data/training_features_5m.csv"
    RL_CSV = "/home/llewyn/crypto-scalping/data/ensemble/rl_training_data_full.csv"
    update_rl_features_only_sessions(FEATURES_CSV, RL_CSV)
