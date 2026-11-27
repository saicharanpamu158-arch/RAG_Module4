from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
import streamlit as st
st.title("PDF BOT")
question=st.text_input("Ask your question")
save=PyPDFLoader("policy.pdf")
load= save.load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap= 80,
    length_function=len,
    separators=["\n\n","\n"," "]

)
#split documents
chunks=splitter.split_documents(load)
embeddings=OllamaEmbeddings(
    model="nomic-embed-text"
    )
#load the database
db=Chroma.from_documents(documents= chunks ,embedding=embeddings)
if question:
    
    results = db.similarity_search(question, k=1)
    context="\n".join(result.page_content for result in results)
#print(context)
    llama=ChatOllama(model="llama3.2:1b",
    temperature=0)
    prompt=ChatPromptTemplate.from_template(""" Answer the question using the provided context.
    if you dont find information, say the information is not displayed.
    Context:{context}
    Question:{question}
    """)

    chain= prompt | llama
    response=chain.invoke({"context":context,"question":question})
    #print(response.content)
    st.write(response.content)


