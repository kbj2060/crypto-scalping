import torch
from transformers import AutoConfig

def check_final():
    print("🚀 앙상블 모델 최종 점검...\n")

    # 1. Chronos-Bolt
    try:
        AutoConfig.from_pretrained("amazon/chronos-bolt-tiny", trust_remote_code=True)
        print("✅ Chronos-Bolt: Ready")
    except Exception as e: print(f"❌ Chronos-Bolt: {e}")

    # 2. Kronos (LlamaTokenizer로 명시적 로드)
    try:
        from transformers import LlamaTokenizer
        LlamaTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k", use_fast=False)
        print("✅ Kronos: Ready")
    except Exception as e: print(f"❌ Kronos: {e}")

    # 3. Lag-Llama (패키지명은 lag_llama로 임포트될 수 있음)
    try:
        try:
            import lag_llama
        except ImportError:
            import lagllama
        print("✅ Lag-Llama: Ready")
    except Exception as e: print(f"❌ Lag-Llama: {e}")

if __name__ == "__main__":
    check_final()