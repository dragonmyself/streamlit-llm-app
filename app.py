import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# --- 環境変数の読み込み ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- LLM回答関数 ---
def get_llm_response(user_input: str, expert_type: str) -> str:
    """入力テキストと専門家タイプを受け取り、LLMからの回答を返す"""
    
    if expert_type == "医師":
        system_prompt = "あなたは経験豊富な内科医です。専門用語を使わずに、患者に分かりやすく説明してください。"
    elif expert_type == "経営コンサルタント":
        system_prompt = "あなたは外資系の経営コンサルタントです。ロジカルに課題解決のアドバイスをしてください。"
    else:
        system_prompt = "あなたは有能なアシスタントAIです。分かりやすく回答してください。"

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.5,
        openai_api_key=OPENAI_API_KEY
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]
    response = llm(messages)
    return response.content

# --- Streamlit UI ---
st.title("🧑‍💻 LangChain × Streamlit デモアプリ")
st.write("このアプリでは、入力した質問に対してAIが専門家の視点で回答します。")
st.write("ラジオボタンから専門家の種類を選び、テキストを入力して送信してください。")

# ラジオボタン（専門家の種類）
expert_type = st.radio(
    "専門家の種類を選択してください",
    ("医師", "経営コンサルタント")
)

# 入力フォーム
user_input = st.text_input("質問を入力してください")

# 実行ボタン
if st.button("送信"):
    if user_input:
        response = get_llm_response(user_input, expert_type)
        st.subheader("AIからの回答")
        st.write(response)
    else:
        st.warning("テキストを入力してください。")
