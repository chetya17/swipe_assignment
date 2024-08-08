from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings



def load_vectorDB(pdf_dir):
    docs = []
    for file in os.listdir(pdf_dir):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, file)
            loader = PyPDFLoader(pdf_path)
            print(loader.load())
            docs.extend(loader.load())
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(docs)
    return chunked_documents



data = load_vectorDB(pdf_dir="./pdf files")


for d in data:
    print(d.page_content)
    break



embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(data, embeddings)


db.save_local("./pdf_db")

