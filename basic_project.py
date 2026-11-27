from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from ask import context,question
import streamlit as st
st.title("HELLO")
question=st.chat_input("answer")

llama=ChatOllama(model="llama3.2:1b",
    temperature=0)
prompt=ChatPromptTemplate.from_template(""" Answer the question using the provided context
Context:{context}
Question:{question}
""")

if question:
    chain= prompt | llama
    response=chain.invoke({"context":context,"question":question})
    st.write(response.content)