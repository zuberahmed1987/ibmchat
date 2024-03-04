from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

ibm_url = os.getenv("IBM_WATSONX_URL")
ibm_apikey = os.getenv("IBM_WATSONX_API_KEY")
model_id = os.getenv("IBM_WATSONX_MODEL_ID")
project_id = os.getenv("IBM_WATSONX_PROJECT_ID")
space_id = os.getenv("IBM_WATSONX_SPACE_ID")
debug = os.getenv("DEBUG", False)
log_level = os.getenv("LOG_LEVEL", "INFO")

#logger = logging.getLogger(__name__)
#logging.basicConfig(level=log_level)

credentials = {
    "url": ibm_url,
    "apikey": ibm_apikey
}

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
    GenParams.MAX_NEW_TOKENS: 4096,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1
}

model = WatsonxLLM(
    model_id=model_id,
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id,
    params=parameters
    )

def generate_response(input_text):
    prompt_1 = PromptTemplate(
        input_variables=["question"],
        template="Answer the following question: {question}"
    )

    llm_chain = LLMChain(llm=model, prompt=prompt_1, output_key='answer')
    st.info(llm_chain(input_text))

def main():
    st.set_page_config(
        page_title="IBM Watsonx Chatui by HCL",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )
    st.markdown("<h1 style='text-align: center; color: blue;'>Chat with IBM Watsonx AI </h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,1])
    with col2:
        st.info("Chat Below")
        input_text = st.text_area("Enter your query")
            if input_text is not None:
            if st.button("Chat with IBM Watsonx AI"):
                st.info("Your Query: "+input_text)
                result = generate_response(input_text)
                st.success(result)
    #with st.form('my_form'):
    #    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    #    submitted = st.form_submit_button('Submit')
    #    generate_response(text)

if __name__ == "__main__":
    main()


