import os
import requests
import streamlit as st 
from streamlit_chat import message
import tempfile
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Set your Hugging Face API token
os.environ['HUGGINGFACE_HUB_TOKEN'] = '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'
LLAMA_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B"
SENTENCE_TRANSFORMER_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_HUB_TOKEN']}"}

def query_huggingface_api(url, payload):
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

# Loading the model
def load_llm():
    llm = CTransformers(
        model="meta-llama/Meta-Llama-3-8B",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5,
        use_auth_token=os.environ['HUGGINGFACE_HUB_TOKEN']
    )
    return llm

st.title("K9 REPORTER")

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    # Prepare data for embeddings
    sentences = [record.page_content for record in data]

    embeddings_response = query_huggingface_api(SENTENCE_TRANSFORMER_API_URL, {
        "inputs": {
            "source_sentence": sentences[0],  # Use the first sentence as the source for simplicity
            "sentences": sentences
        }
    })

    st.write("Embeddings Response:", embeddings_response)  # Print the embeddings response to inspect its structure

    # Extract embeddings from the response
    embeddings = [item['embedding'] for item in embeddings_response]
    
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    llm = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]
        
    # container for the chat history
    response_container = st.container()
    # container for the user's text input
    container = st.container()

    use_local_model = False  # Set to True to use the local model, False to use the API

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            if use_local_model:
                output = conversational_chat(user_input)
            else:
                api_response = query_huggingface_api(LLAMA_API_URL, {"inputs": user_input})
                output = api_response.get('generated_text', "Sorry, I couldn't generate a response.")

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
