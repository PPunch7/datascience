from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from .embeddings import get_embeddings

def build_vectorstore():
    loader = PyPDFLoader("crew_data/doc.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)
    embeddings = get_embeddings()

    return FAISS.from_documents(docs, embeddings)