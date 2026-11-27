from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
save=PyPDFLoader("policy.pdf")
load= save.load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap= 10,
    length_function=len,
    separators=["\n\n\n", "\n"]
)
#split documents
chunks=splitter.split_documents(load)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text")
db=Chroma.from_documents(documents= chunks ,embedding=embeddings)
question = "how many sick leaves i can take in a year"
results = db.similarity_search(question, k=2)