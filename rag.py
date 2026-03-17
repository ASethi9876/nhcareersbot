from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_PATH = "files"
CHROMA_PATH = "chroma"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt",  loader_cls=lambda path: TextLoader(path, encoding="utf-8"))
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  
    )

    vector_store = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    vector_store.persist()
    return vector_store


docs = load_documents()
docs = split_documents(docs)
create_vector_store(docs) 

