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
	
parameters = {
"decoding_method": "sample",
"max_new_tokens": 4000,
"min_new_tokens": 1,
"temperature": 0.2,
"repetition_penalty": 1,
"stop_sequences": ["<end_of_code>"]
}

# Call IBM Watsonx
model = Model(
    model_id = model_id,
    params = parameters,
    credentials = get_credentials(),
    project_id = project_id,
    space_id = space_id
)

def generate_response(input_text):
    for chunk in generated_response:
        yield chunk


def main():
    st.set_page_config(
        page_title="IBM Watsonx Chatui by HCL",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.header('IBM Watsonx AI Chatbot')
    st.write('Allows users to interact with the IBM watsonx AI LLM')
    user_query = st.chat_input(placeholder="Ask me anything!")
    system_messages = "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown."
    prompt_input = f"""system: {system_messages}
User: create dockerfile for python app
assisassistant: 
```Dockerfile
FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 80
CMD ["python", "app.py"]
```
<end_of_code>
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
        
        with st.chat_message("assistant"):
            #response = model.generate_text_stream(prompt=prompt_input, params=parameters, guardrails=True)
            #response = generate_response(response)
            response = model.generate_text(prompt=prompt_input, params=parameters, guardrails=True)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    #input_text = st.text_area("Enter your query")
    #if input_text is not None:
    #    st.info("User: "+input_text)
    #    result = generate_response(input_text)
    #    st.success('AI: ' +result)


    #with st.form('my_form'):
    #    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    #    submitted = st.form_submit_button('Submit')
    #    generate_response(text)

if __name__ == "__main__":
    main()


