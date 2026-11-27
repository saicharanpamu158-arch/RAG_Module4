from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from convert_to_chunks import chunks
from database import db
question = "how many sick leaves i can take in a year"
results = db.similarity_search(question, k=1)
for result in results:
    print(result.page_content)

#context="\n".join(result.page_content for result in results)
#print(context)



