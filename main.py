"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import faiss
import pickle
from utils.openai_client import OPENAI_API_KEY
from dotenv import load_dotenv

# loading environment variables
load_dotenv()

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = VectorDBQAWithSourcesChain.from_llm(
    llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), vectorstore=store)


# From here down is all the StreamLit UI.
st.set_page_config(page_title="NLP Book QA Bot", page_icon=":robot:")
st.header("NLP Book QA Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "What is deep learning?", key="input")
    return input_text


user_input = get_text()

if user_input:

    st.session_state.past.append(user_input)
    result = chain({"question": user_input})
    output = f"Answer: {result['answer']}\nSources: {result['sources']}"
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i],
                is_user=True, key=str(i) + "_user")
