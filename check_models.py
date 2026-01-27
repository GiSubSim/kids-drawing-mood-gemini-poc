import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("🔍 사용 가능한 모델 리스트 확인 중...")
try:
    # 1. 모델 목록 가져오기
    for m in genai.list_models():
        # 2. 'generateContent' (채팅/그림분석) 기능이 있는 모델만 출력
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            
except Exception as e:
    print(f"❌ 에러 발생: {e}")