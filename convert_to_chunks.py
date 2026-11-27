#convert the document to chunks
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
save=PyPDFLoader("C:\AI\RAG\policy.pdf")
load= save.load()
#print(f"{len(load)}")

#create text splitter
splitter=RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap= 10,
    length_function=len,
    separators=["\n\n\n", "\n"]
)
#split documents
chunks=splitter.split_documents(load)
#print("Total chunks", len(chunks))

'''for i, chunk in enumerate(chunks):
    print(f"total chunks {i+1}:")
    print(chunk.page_content)
'''