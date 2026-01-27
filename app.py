
import streamlit as st
import os
import json
import time
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from prompts import SYSTEM_PROMPT

# 1. 환경 설정 및 API 키 로드
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    st.error("❌ GOOGLE_API_KEY가 .env 파일에 없습니다.")

# --- [비용 설정 (Gemini 2.5 Flash 표준 유료 등급 기준)] ---
PRICE_PER_1M_INPUT_TOKENS = 0.30    # 입력 토큰 100만개당 $0.30
PRICE_PER_1M_OUTPUT_TOKENS = 2.50   # 출력 토큰 100만개당 $2.50

# 2. Gemini API 호출 함수
def analyze_images_with_gemini(image_files, persona):
    
    # 모델: 최신 Gemini 2.5 Flash 사용
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT
    )
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
    ]

    content_inputs = [f"사용자가 선택한 페르소나: {persona}\n위 페르소나 말투로 JSON 포맷에 맞춰 답변해줘."]
    
    for img_file in image_files:
        image = Image.open(img_file)
        content_inputs.append(image)

    try:
        print("\n" + "="*50)
        print(">>> Gemini API 요청 시작...")
        start_time = time.time()
        
        # API 호출
        response = model.generate_content(
            content_inputs,
            safety_settings=safety_settings,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2
            }
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # --- [토큰 및 비용 정밀 계산] ---
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        
        # 비용 계산
        input_cost = (input_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
        output_cost = (output_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT_TOKENS
        total_cost = input_cost + output_cost

        # 1. 성능 및 비용 로그 출력
        print(f">>> 소요 시간: {elapsed_time:.2f}초")
        print(f">>> 토큰 사용: 입력 {input_tokens} / 출력 {output_tokens}")
        print(f">>> 예상 비용: ${total_cost:.6f}")
        
        # 2. API 응답 텍스트(JSON) 로그 출력 (추가된 부분)
        print("-" * 30)
        print(">>> API 응답 텍스트:")
        try:
            # 보기 좋게 들여쓰기해서 출력
            parsed_json = json.loads(response.text)
            print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
        except:
            # JSON 파싱 실패시 원본 텍스트 출력
            print(response.text)
        print("="*50 + "\n")

        return {
            "data": json.loads(response.text),
            "meta": {
                "time": elapsed_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": total_cost
            }
        }

    except Exception as e:
        st.error(f"API 호출 중 오류 발생: {e}")
        print(f">>> 오류 상세: {e}")
        return None

# --- [Streamlit UI 코드 (기존과 동일)] ---
st.set_page_config(page_title="아트봉봉 그림 분석 데모", layout="wide")

st.title("🎨 아트봉봉 AI 그림 분석 데모 (Gemini Ver.)")
st.markdown("아이의 그림(1~4장)을 올리고 페르소나를 선택하면 AI가 시각적 특징을 분석해줍니다.")

st.sidebar.header("📊 분석 현황판")
st.sidebar.info("이미지를 업로드하고 분석을 시작하면 여기에 통계가 표시됩니다.")

st.header("1. 그림 업로드 (최대 4장)")
uploaded_files = st.file_uploader("PNG, JPG 파일을 선택하세요", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 4:
        st.warning("최대 4장까지만 업로드 가능합니다. 앞의 4장만 사용합니다.")
        uploaded_files = uploaded_files[:4]
    
    cols = st.columns(len(uploaded_files))
    for idx, file in enumerate(uploaded_files):
        cols[idx].image(file, caption=f"그림 {idx+1}", use_container_width=True)

    st.header("2. 페르소나 선택")
    
    personas = {
        "마음박사 페페": "🐧 마음박사 페페 (따뜻한 공감형)",
        "카리스마 샤샤": "😎 카리스마 샤샤 (쿨한 멘토형)",
        "칭찬봇 피코": "🤖 칭찬봇 피코 (데이터 분석형)",
        "현실친구 라봉이": "🦁 현실친구 라봉이 (솔직한 친구형)"
    }
    
    selected_persona_key = st.radio("분석할 캐릭터를 골라주세요:", list(personas.keys()), horizontal=True)
    
    if st.button("🚀 그림 분석 시작하기", type="primary"):
        with st.spinner(f"'{selected_persona_key}'가 그림을 분석 중입니다... 잠시만 기다려주세요!"):
            
            result = analyze_images_with_gemini(uploaded_files, personas[selected_persona_key])
            
            if result:
                st.success("분석 완료!")
                st.divider()
                
                meta = result["meta"]
                st.sidebar.empty()
                st.sidebar.header("⏱️ 성능 및 비용")
                
                st.sidebar.metric("⏳ 소요 시간", f"{meta['time']:.2f} 초")
                st.sidebar.metric("💰 예상 비용 (USD)", f"${meta['cost']:.5f}")
                
                st.sidebar.markdown("---")
                st.sidebar.markdown(f"**🔢 토큰 사용량**")
                st.sidebar.text(f"입력(Total): {meta['input_tokens']:,}")
                st.sidebar.text(f"출력(Total): {meta['output_tokens']:,}")
                st.sidebar.caption("※ 입력 토큰에는 이미지 크기에 따른 타일링(추가 토큰)이 자동 반영되었습니다.")
                
                krw_cost = meta['cost'] * 1450
                st.sidebar.markdown(f"**🇰🇷 원화 환산:** 약 `{krw_cost:.2f}원`")

                result_json = result["data"]
                analysis = result_json.get("analysis_result", {})
                commentary = result_json.get("character_commentary", "")
                
                mind_expr = analysis.get("mind_expression", "알 수 없음")
                word_cloud = analysis.get("word_cloud", [])
                colors = analysis.get("top_5_colors", [])
                energy = analysis.get("energy_chart", {})

                st.subheader("💖 그림의 분위기")
                st.info(f"**[{mind_expr}]**")

                st.subheader("☁️ 무드 키워드 (Top 5)")
                wc_cols = st.columns(5)
                for i, word in enumerate(word_cloud):
                    wc_cols[i].markdown(f"#### #{word}")

                st.divider()

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("🎨 Top 5 색상 (면적 기준)")
                    for color in colors:
                        st.markdown(
                            f"""
                            <div style="
                                background-color: {color};
                                width: 100%;
                                height: 40px;
                                border-radius: 5px;
                                margin-bottom: 5px;
                                border: 1px solid #ddd;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                color: #555;
                                font-weight: bold;
                                font-size: 12px;
                            ">{color}</div>
                            """, 
                            unsafe_allow_html=True
                        )

                with col2:
                    st.subheader("⚡ 비주얼 스타일 차트")
                    if energy:
                        df = pd.DataFrame(dict(
                            r=list(energy.values()),
                            theta=list(energy.keys())
                        ))
                        fig = px.line_polar(df, r='r', theta='theta', line_close=True, range_r=[0, 100])
                        fig.update_traces(fill='toself')
                        st.plotly_chart(fig, use_container_width=True)

                st.divider()

                st.subheader(f"📢 {selected_persona_key}의 감상평")
                persona_emoji = {
                    "마음박사 페페": "🐧",
                    "카리스마 샤샤": "😎",
                    "칭찬봇 피코": "🤖",
                    "현실친구 라봉이": "🦁"
                }
                st.markdown(
                    f"""
                    <div style="
                        background-color: #f0f2f6;
                        padding: 20px;
                        border-radius: 10px;
                        border-left: 5px solid #4CAF50;
                        white-space: pre-line; 
                        line-height: 1.6;
                        font-size: 16px;
                    ">
                    <h3 style="margin-top:0;">{persona_emoji.get(selected_persona_key, "🤖")} {selected_persona_key}</h3>
                    {commentary}
                    </div>
                    """,
                    unsafe_allow_html=True
                )



# import streamlit as st
# import os
# import json
# import pandas as pd
# import plotly.express as px
# from dotenv import load_dotenv
# import google.generativeai as genai
# from PIL import Image
# from prompts import SYSTEM_PROMPT

# # 1. 환경 설정 및 API 키 로드
# load_dotenv()
# google_api_key = os.getenv("GOOGLE_API_KEY")

# # Gemini 설정
# if google_api_key:
#     genai.configure(api_key=google_api_key)
# else:
#     st.error("❌ GOOGLE_API_KEY가 .env 파일에 없습니다.")

# # 2. Gemini API 호출 함수
# def analyze_images_with_gemini(image_files, persona):
    
#     # 1) 모델 초기화 (Gemini 1.5 Pro 사용 권장)
#     # system_instruction에 프롬프트를 넣어 강력하게 지시합니다.
#     model = genai.GenerativeModel(
#         model_name="gemini-2.5-flash",
#         system_instruction=SYSTEM_PROMPT
#     )
    
#     # 2) 안전 설정 (Safety Settings)
#     # 아동 그림이나 창의적 표현이 차단되지 않도록 필터를 완화합니다.
#     safety_settings = [
#         {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
#         {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
#         {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
#         {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
#     ]

#     # 3) 입력 데이터 준비
#     # Gemini는 PIL.Image 객체 리스트를 바로 받습니다.
#     content_inputs = [f"사용자가 선택한 페르소나: {persona}\n위 페르소나 말투로 JSON 포맷에 맞춰 답변해줘."]
    
#     for img_file in image_files:
#         # Streamlit의 UploadedFile(BytesIO)을 PIL Image로 변환
#         image = Image.open(img_file)
#         content_inputs.append(image)

#     try:
#         print(">>> Gemini API 요청 시작...")
        
#         # 4) 콘텐츠 생성 요청
#         # generation_config에서 JSON 응답을 강제합니다.
#         response = model.generate_content(
#             content_inputs,
#             safety_settings=safety_settings,
#             generation_config={
#                 "response_mime_type": "application/json",
#                 "temperature": 0.1  # <-- 0.0 ~ 0.2 강력 추천 (안 적으면 1.0으로 작동함)
#             }
#         )
        
#         # --- [디버깅 로그] ---
#         print(f">>> API 응답 텍스트:\n{response.text}")
#         # -------------------

#         # 5) JSON 파싱
#         return json.loads(response.text)

#     except Exception as e:
#         st.error(f"API 호출 중 오류 발생: {e}")
#         print(f">>> 오류 상세: {e}")
#         return None


# # --- [Streamlit UI 시작] ---
# # (이 아래 UI 코드는 기존과 동일합니다. 함수 이름만 바뀌었습니다.)

# st.set_page_config(page_title="아트봉봉 그림 분석 데모", layout="wide")

# st.title("🎨 아트봉봉 AI 그림 분석 데모 (Gemini Ver.)")
# st.markdown("아이의 그림(1~4장)을 올리고 페르소나를 선택하면 AI가 시각적 특징을 분석해줍니다.")

# # Step 1: 이미지 업로드
# st.header("1. 그림 업로드 (최대 4장)")
# uploaded_files = st.file_uploader("PNG, JPG 파일을 선택하세요", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# if uploaded_files:
#     if len(uploaded_files) > 4:
#         st.warning("최대 4장까지만 업로드 가능합니다. 앞의 4장만 사용합니다.")
#         uploaded_files = uploaded_files[:4]
    
#     # 업로드된 이미지 미리보기
#     cols = st.columns(len(uploaded_files))
#     for idx, file in enumerate(uploaded_files):
#         cols[idx].image(file, caption=f"그림 {idx+1}", use_container_width=True)

#     # Step 2: 페르소나 선택
#     st.header("2. 페르소나 선택")
    
#     personas = {
#         "마음박사 페페": "🐧 마음박사 페페 (따뜻한 공감형)",
#         "카리스마 샤샤": "😎 카리스마 샤샤 (쿨한 멘토형)",
#         "칭찬봇 피코": "🤖 칭찬봇 피코 (데이터 분석형)",
#         "현실친구 라봉이": "🦁 현실친구 라봉이 (솔직한 친구형)"
#     }
    
#     selected_persona_key = st.radio("분석할 캐릭터를 골라주세요:", list(personas.keys()), horizontal=True)
    
#     # 분석하기 버튼
#     if st.button("🚀 그림 분석 시작하기", type="primary"):
#         with st.spinner(f"'{selected_persona_key}'가 그림을 분석 중입니다... 잠시만 기다려주세요!"):
            
#             # --- [변경된 함수 호출] ---
#             result_json = analyze_images_with_gemini(uploaded_files, personas[selected_persona_key])
#             # -----------------------
            
#             if result_json:
#                 st.success("분석 완료!")
#                 st.divider()

#                 # --- 결과 출력 화면 ---
                
#                 # 데이터 파싱
#                 analysis = result_json.get("analysis_result", {})
#                 commentary = result_json.get("character_commentary", "")
                
#                 mind_expr = analysis.get("mind_expression", "알 수 없음")
#                 word_cloud = analysis.get("word_cloud", [])
#                 colors = analysis.get("top_5_colors", [])
#                 energy = analysis.get("energy_chart", {})

#                 # (1) 마음 표현 텍스트
#                 st.subheader("💖 그림의 분위기")
#                 st.info(f"**[{mind_expr}]**")

#                 # (2) 워드 클라우드 (리스트 출력)
#                 st.subheader("☁️ 무드 키워드 (Top 5)")
#                 st.write("그림에서 가장 많이 느껴지는 분위기 단어들입니다.")
                
#                 # 가로로 뱃지처럼 나열
#                 wc_cols = st.columns(5)
#                 for i, word in enumerate(word_cloud):
#                     wc_cols[i].markdown(f"#### #{word}")

#                 st.divider()

#                 # (3) & (4) 차트와 색상 (2단 컬럼 구성)
#                 col1, col2 = st.columns([1, 1])

#                 with col1:
#                     st.subheader("🎨 Top 5 색상 (면적 기준)")
#                     # 색상 보여주기 (HTML/CSS 활용)
#                     for color in colors:
#                         st.markdown(
#                             f"""
#                             <div style="
#                                 background-color: {color};
#                                 width: 100%;
#                                 height: 40px;
#                                 border-radius: 5px;
#                                 margin-bottom: 5px;
#                                 border: 1px solid #ddd;
#                                 display: flex;
#                                 align-items: center;
#                                 justify-content: center;
#                                 color: #555;
#                                 font-weight: bold;
#                                 font-size: 12px;
#                             ">{color}</div>
#                             """, 
#                             unsafe_allow_html=True
#                         )

#                 with col2:
#                     st.subheader("⚡ 비주얼 스타일 차트")
#                     # Plotly Radar Chart (방사형 차트) 그리기
#                     if energy:
#                         df = pd.DataFrame(dict(
#                             r=list(energy.values()),
#                             theta=list(energy.keys())
#                         ))
#                         fig = px.line_polar(df, r='r', theta='theta', line_close=True, range_r=[0, 100])
#                         fig.update_traces(fill='toself')
#                         st.plotly_chart(fig, use_container_width=True)

#                 st.divider()

#                 # (5) 페르소나 분석 결과 (줄바꿈 처리)
#                 st.subheader(f"📢 {selected_persona_key}의 감상평")
                
#                 # 페르소나별 아이콘/이미지
#                 persona_emoji = {
#                     "마음박사 페페": "🐧",
#                     "카리스마 샤샤": "😎",
#                     "칭찬봇 피코": "🤖",
#                     "현실친구 라봉이": "🦁"
#                 }
                
#                 # 말풍선 스타일로 출력
#                 st.markdown(
#                     f"""
#                     <div style="
#                         background-color: #f0f2f6;
#                         padding: 20px;
#                         border-radius: 10px;
#                         border-left: 5px solid #4CAF50;
#                         white-space: pre-line; 
#                         line-height: 1.6;
#                         font-size: 24px;
#                     ">
#                     <h3 style="margin-top:0;">{persona_emoji.get(selected_persona_key, "🤖")} {selected_persona_key}</h3>
#                     {commentary}
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )