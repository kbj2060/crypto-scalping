#!/bin/bash

# 프로젝트 디렉토리로 이동
cd /content/drive/MyDrive/crypto-scalping/

# 필요한 라이브러리 설치
echo "라이브러리 설치 시작..."
pip install pandas-ta python-binance

# TA-Lib은 일반 pip install로 안될 수 있으므로 전처리 설치 포함 (필요 시)
pip install TA-Lib

# TD3 학습 실행
echo "TD3 모델 학습을 시작합니다."
python TD3/train_td3.py