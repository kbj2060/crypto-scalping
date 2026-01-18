"""
API 키 검증 스크립트
"""
import os
from dotenv import load_dotenv
from binance.client import Client

load_dotenv()

api_key = os.getenv('BINANCE_API_KEY', '')
secret_key = os.getenv('BINANCE_SECRET_KEY', '')
testnet = os.getenv('BINANCE_TESTNET', 'False').lower() == 'true'

print("=" * 60)
print("바이낸스 API 키 검증")
print("=" * 60)
print(f"API Key: {api_key[:10]}..." if api_key else "API Key: (비어있음)")
print(f"Secret Key: {'설정됨' if secret_key else '(비어있음)'}")
print(f"Testnet: {testnet}")
print("=" * 60)

if not api_key or not secret_key:
    print("❌ API 키가 설정되지 않았습니다!")
    print(".env 파일에 BINANCE_API_KEY와 BINANCE_SECRET_KEY를 입력하세요.")
    exit(1)

if api_key == 'your_api_key_here' or secret_key == 'your_secret_key_here':
    print("❌ API 키를 실제 값으로 변경해주세요!")
    exit(1)

try:
    if testnet:
        client = Client(api_key, secret_key, testnet=True)
        print("✅ 테스트넷 연결 성공")
    else:
        client = Client(api_key, secret_key)
        print("✅ 실제 거래소 연결 성공")
    
    # 선물 계정 정보 확인
    try:
        account = client.futures_account()
        print(f"✅ 선물 계정 접근 성공")
        print(f"   총 자산: {float(account['totalWalletBalance']):.2f} USDT")
        print(f"   사용 가능 자산: {float(account['availableBalance']):.2f} USDT")
    except Exception as e:
        print(f"❌ 선물 계정 접근 실패: {e}")
        print("\n⚠️  선물 거래 권한이 없을 수 있습니다.")
        print("   바이낸스에서 API 키 설정을 확인하세요:")
        if testnet:
            print("   https://testnet.binancefuture.com/en/futures/BTCUSDT")
        else:
            print("   https://www.binance.com/en/my/settings/api-management")
        print("   → 'Enable Futures' 옵션이 체크되어 있어야 합니다.")
        raise
    
    # 심볼 정보 확인
    try:
        exchange_info = client.futures_exchange_info()
        print(f"✅ 선물 거래소 정보 조회 성공")
        print(f"   거래 가능 심볼 수: {len(exchange_info['symbols'])}")
    except Exception as e:
        print(f"⚠️  선물 거래소 정보 조회 실패: {e}")
    
    # 간단한 데이터 조회 테스트
    try:
        ticker = client.futures_symbol_ticker(symbol='ETHUSDT')
        print(f"✅ 가격 조회 성공: ETHUSDT = {float(ticker['price']):.2f} USDT")
    except Exception as e:
        print(f"⚠️  가격 조회 실패: {e}")
    
    # 레버리지 설정 테스트
    try:
        client.futures_change_leverage(symbol='ETHUSDT', leverage=10)
        print(f"✅ 레버리지 설정 성공: 10x")
    except Exception as e:
        print(f"⚠️  레버리지 설정 실패: {e}")
        print("   (이것은 정상일 수 있습니다 - 이미 설정되어 있거나 권한 문제)")
    
except Exception as e:
    error_msg = str(e)
    print("=" * 60)
    print("❌ API 연결 실패!")
    print("=" * 60)
    
    if "API-key" in error_msg or "-2015" in error_msg:
        print("가능한 원인:")
        print("1. API 키가 잘못되었습니다")
        print("2. API 키에 선물 거래 권한이 없습니다")
        print("   → 바이낸스에서 API 키 생성 시 'Enable Futures' 옵션을 체크하세요")
        print("3. IP 화이트리스트가 설정되어 있습니다")
        print("   → 바이낸스 API 설정에서 IP 제한을 해제하거나 현재 IP를 추가하세요")
        print("4. 테스트넷/실거래소 API 키가 일치하지 않습니다")
        print("   → 테스트넷: https://testnet.binancefuture.com/")
        print("   → 실거래소: https://www.binance.com/")
    else:
        print(f"에러: {error_msg}")
    
    print("=" * 60)
    exit(1)

print("=" * 60)
print("✅ 모든 검증 완료!")
print("=" * 60)
