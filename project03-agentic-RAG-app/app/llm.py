import streamlit as st
from langchain_openai import ChatOpenAI
from .config import GROQ_API_KEY

@st.cache_resource
def load_llm():
    return get_llm()

def get_llm():
    return ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
    )