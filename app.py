
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM 
import streamlit as st

CHROMA_PATH = "chroma"

@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen1.5-1.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

tokenizer, model = load_model()
db = load_vector_db()


SYSTEM_PROMPT = """
Instruction:
You are a chatbot for the National Highways stand at the Big Bang Careers Fair.
You help children aged 10-14 learn about opportunities, apprenticeships and the company.
Respond to the given question using only the context provided below and this given system context. If you don't know the answer, say you don't know. Do not make up an answer.
Respond with a couple of sentences using simple language, but do not stray from the context of the question. 


Question:
{question}

Context:
{context}

Answer:
"""


def query_data(question):
    results = db.similarity_search_with_score(question, k=2)

    if len(results) == 0 or results[0][1] < 0.5:
        print("No relevant documents found.")
        return

    context_text = "\n\n---\n\n".join([result[0].page_content for result in results])


    prompt = SYSTEM_PROMPT.format(context=context_text, question=question)
    
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
    )

    print(outputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(response)
    st.info(response)


st.title("Big Bang Careers Fair")

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Hi, what is a T-Level?",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.info("Loading...")
        query_data(text)
