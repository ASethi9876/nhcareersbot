
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 


CHROMA_PATH = "chroma"
SYSTEM_PROMPT = """ Yo  what's going on? Tell me about the benefits of AI and be chatty about it. Give me at least 3 benefits and 1 drawback.
"""
SYSTEM_PROMP = """
Instruction:
FIRST SAY HI HOW ARE YOU
You are a chatbot for the National Highways stand at the Big Bang Careers Fair.
You help children aged 10-14 learn about opportunities, apprenticeships and the company.
Respond to the given question using only the context provided below and this given system context. If you don't know the answer, say you don't know. Do not make up an answer.
Always start your answers with Hi I'm Johnny! and end with is there anything else?"


Question:
{question}

Context:
{context}

Answer (in 2-4 warm and human-like sentences):
"""

question = "YO WHAT job to do if i love information?"

def query_data():
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

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    #prompt = SYSTEM_PROMPT.format(context=context_text, question=question)
    prompt = SYSTEM_PROMPT  
    print(prompt)
    print(len(prompt))

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,       
        num_beams=1,           
        no_repeat_ngram_size=3,
    )

    print(outputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(response)
    return

def parse_input():
    pass 


query_data()