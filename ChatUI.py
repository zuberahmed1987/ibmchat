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

def get_credentials():
    return {
        "url" : ibm_url,
        "apikey" : ibm_apikey
    }

@st.cache_resource
@st.spinner('Loading Model...')
def load_model():
    parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 4000,
    "min_new_tokens": 1,
    "temperature": 0.2,
    "repetition_penalty": 1
    }
    
    # Call IBM Watsonx
    model = Model(
        model_id = model_id,
        params = parameters,
        credentials = get_credentials(),
        project_id = project_id,
        space_id = space_id
    )
    return model

def response_generator(generated_response):
    for chunk in generated_response:
        yield chunk


def main():
    st.set_page_config(
        page_title="IBM Watsonx Chatui by HCL",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    with st.sidebar:
        model = load_model()
        if st.button('Clear Conversation'):
            st.session_state.conversation = None
            st.session_state.chat_history = None
            st.session_state.messages = []
            st.toast('Chat History Cleared!')
            
    st.header('IBM Watsonx AI Chatbot')
    st.write('Allows users to interact with the IBM watsonx AI LLM')
    user_query = st.chat_input(placeholder="Ask me anything!")
    system_messages = "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown."
    prompt_input = f"""system: {system_messages}
user: {user_query}
assistant: """
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    if user_query:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_query)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})
        
        #with st.chat_message("assistant"):
        #    response = model.generate_text(prompt=prompt_input)
        #    st.markdown(response)
        #    st.session_state.messages.append({"role": "assistant", "content": response})
                
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            genrated_stream = model.generate_text_stream(prompt=prompt_input)
            response = st.write_stream(response_generator(genrated_stream))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
	
if __name__ == "__main__":
    main()


