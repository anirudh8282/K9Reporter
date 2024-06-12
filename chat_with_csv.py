import os
from transformers import LlamaForCausalLM, LlamaTokenizer
import streamlit as st
from streamlit_chat import message
import tempfile
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Set your Hugging Face API token
os.environ['HUGGINGFACE_HUB_TOKEN'] = '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'

# Loading the model
def load_llm():
    # Load the tokenizer and model from Hugging Face
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=True)
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=True)
    
    # Ensure the model is moved to the appropriate device (e.g., CPU or GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return model, tokenizer, device

st.title("K9 REPORTER")

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    # use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    model, tokenizer, device = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=model, retriever=db.as_retriever(), tokenizer=tokenizer, device=device)

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

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
