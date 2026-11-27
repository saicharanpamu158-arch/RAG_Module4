from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from convert_to_chunks import chunks

embeddings=OllamaEmbeddings(
    model="nomic-embed-text"
    )
#load the database
db=Chroma.from_documents(documents= chunks ,embedding=embeddings)
#query the database
#print("datase created")
#print("total",db._collection.count())