from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
load = PyPDFLoader("policy.pdf")
save= load.load()
splitter= RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap =20,
    length_function=len,
    separators=["\n\n"]
)
chunks=splitter.split_documents(save)
embeddings=OllamaEmbeddings(model="nomic-embed-text")
vector = Chroma.from_documents(chunks, embeddings)
question =("how many leaves I am supposed to take")
results = vector.similarity_search(question, k=2)
