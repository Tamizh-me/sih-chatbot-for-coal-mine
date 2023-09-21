import os
import textwrap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader


loader = DirectoryLoader('docs', glob='./*.pdf', loader_cls=PyPDFLoader)
documents = loader.load()
len(documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"})

embedding = instructor_embeddings
faiss= 'faiss_db'

plain_faiss = FAISS.from_documents(documents=texts, embedding=embedding)
plain_faiss.save_local(faiss)

reload_faiss = FAISS.load_local(faiss, embeddings=embedding)
retriever = reload_faiss.as_retriever(search_kwargs={"k": 5})



