from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
save=PyPDFLoader("policy.pdf")
load= save.load()
print(f"{len(load)}")
for i, doc in enumerate(load):
    print(f"{i+1}:")
    print(doc.page_content)
    print(f"{doc.metadata}")
    