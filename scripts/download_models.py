"""
download_models.py
- 파운데이션 모델들을 로컬 디스크에 영구 보관하기 위한 다운로더
- 실행: python download_models.py
"""
import os
from huggingface_hub import snapshot_download

# 다운로드할 파운데이션 모델 목록 (HF Repo ID)
MODELS_TO_DOWNLOAD = {
    "chronos-2": "amazon/chronos-2",
    "kronos-small": "NeoQuasar/Kronos-small",
    "kronos-tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
    "timesfm-2.0": "google/timesfm-2.0-500m-pytorch",
    "moirai-2.0-small": "Salesforce/moirai-2.0-R-small"
}

LOCAL_DIR = "./models"

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print("=" * 60)
    print("📥 파운데이션 모델 로컬 다운로드 시작 (최초 1회만 실행)")
    print("=" * 60)

    for local_name, hf_repo in MODELS_TO_DOWNLOAD.items():
        target_path = os.path.join(LOCAL_DIR, local_name)
        print(f"\n▶ 다운로드 중: {hf_repo} -> {target_path}")
        
        # snapshot_download는 이미 다운로드된 파일이 있으면 자동으로 건너뜁니다.
        snapshot_download(
            repo_id=hf_repo,
            local_dir=target_path,
            local_dir_use_symlinks=False # 심볼릭 링크 대신 실제 파일 복사(안정성)
        )
        print(f"✅ {local_name} 완료!")

    print("\n🎉 모든 파운데이션 모델이 로컬에 저장되었습니다!")
    print("이제 ensemble_forecast.py 실행 시 HTTP 통신 없이 즉시 로드됩니다.")

if __name__ == "__main__":
    main()