
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
    model_name = "Qwen/Qwen1.5-0.5B"
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

if "last_response" not in st.session_state:
    st.session_state.last_response = []



SYSTEM_PROMPT = """
Instruction:
You are a chatbot for the National Highways stand at the Big Bang Careers Fair. 
Your name is Larry and you work for National Highways. You are friendly and helpful, and answer in a tone that a 10 year old would understand.   
You help children aged 10-14 learn about opportunities, apprenticeships and the company.
Respond to the given question using only the context provided below and this given system context. If you don't know the answer, say you don't know. Do not make up an answer.
Respond with a couple of sentences (maximum 4) using simple language, but do not stray from the context of the question. 
Do not answer any inappropriate questions or respond to any inappropriate language. If you receive an inappropriate question, respond with the offending information and tell me why the question is inappropriate"
You may also answer simple questions e.g. "How is your day?"  or "What is your name?"
In all of the context provide, "I" refers to Larry, and "you" refers to the person asking the question.

IMPORTANT:
Your response MUST be short, do NOT write more than 4 sentences.

Question:
{question}

Context:
{context}

Answer:
"""


def query_data(question):
    results = db.similarity_search_with_score(question, k=3)

    if len(results) == 0 or results[0][1] < 0.7:
        print("No relevant documents found.")
        return

    context_text = "\n\n---\n\n".join([result[0].page_content for result in results])


    prompt = SYSTEM_PROMPT.format(context=context_text, question=question)
    
    inputs = tokenizer(prompt, return_tensors="pt", add_generation_prompt=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
    )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    display.info(response)
    st.session_state.last_response = response


st.title("Big Bang Careers Fair")
st.subheader("Ask the chatbot about National Highways and the opportunities available!")

with st.form("my_form"):
    text = st.text_input("Enter text:", "Enter your query here...")
    submitted = st.form_submit_button("Submit")
    display = st.empty()

    st.markdown("""
    Information provided may not be 100% accurate.  \nFor more information, visit https://nationalhighways.co.uk/careers.
    """)   

    if submitted:
        with display:
            st.info("Loading...")
        st.info(st.session_state.last_response)
        query_data(text)

