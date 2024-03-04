from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watsonx_ai.foundation_models import Model
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.title('🔗 IBM Chat App')

