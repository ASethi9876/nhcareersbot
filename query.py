
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM 
import streamlit as st

CHROMA_PATH = "chroma"
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
    embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}  
        )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    results = db.similarity_search_with_score(question, k=2)

    if len(results) == 0 or results[0][1] < 0.5:
        print("No relevant documents found.")
        return

    context_text = "\n\n---\n\n".join([result[0].page_content for result in results])

    model_name = "Qwen/Qwen1.5-1.8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = SYSTEM_PROMPT.format(context=context_text, question=question)
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,

    )

    print(outputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(response)
    return

def parse_input():
    pass 

st.title("Big Bang Careers Fair")


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Hi, what is a T-Level?",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        query_data(text)
