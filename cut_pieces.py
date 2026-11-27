#convert the document to chunks
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
save=TextLoader("C:\AI\RAG\sample.txt")
load= save.load()
print(f"{len(load)}")

#create text splitter
splitter=RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap= 10,
    length_function=len,
    separators=["\n\n\n", "\n"]

)
#split documents
chunks=splitter.split_documents(load)
print("Total chunks", len(chunks))
print(chunks[0].page_content[:150])
'''
for i, chunk in enumerate(chunks):
    print(f"total chunks {i+1}:")
    print(chunk.page_content)

'''

