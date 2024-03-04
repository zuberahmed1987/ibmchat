from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watsonx_ai.foundation_models import Model
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

ibm_url = os.getenv("IBM_WATSONX_URL")
ibm_apikey = os.getenv("IBM_WATSONX_API_KEY")
model_id = os.getenv("IBM_WATSONX_MODEL_ID")
project_id = os.getenv("IBM_WATSONX_PROJECT_ID")
space_id = os.getenv("IBM_WATSONX_SPACE_ID")
debug = os.getenv("DEBUG", False)
log_level = os.getenv("LOG_LEVEL", "INFO")

logger = logging.getLogger(__name__)
logging.basicConfig(level=log_level)

st.set_page_config(
    page_title="IBM Watsonx Chatui by HCL",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

def get_credentials():
	return {
		"url" : ibm_url,
		"apikey" : ibm_apikey
	}

model = Model(
    model_id = model_id,
    params = None,
    credentials = get_credentials(),
    project_id = project_id,
    space_id = space_id
)

prompt_template = "What color is the {flower}?"

parameters = {
  "decoding_method": "sample",
  "max_new_tokens": 4096,
  "min_new_tokens": 1,
  "temperature": 0.2,
  "repetition_penalty": 1
}

def generate_response(input_text):
    #llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    llm_chain = LLMChain(llm=model.to_langchain(), prompt=PromptTemplate.from_template(prompt_template))
    st.info(llm_chain(input_text))
  
with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    generate_response(text)




