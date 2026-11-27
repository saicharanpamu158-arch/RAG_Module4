
# STEP 1: IMPORT LIBRARIES

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# STEP 2: STREAMLIT UI

st.title("📚 RAG PDF Chatbot")

file = st.file_uploader("Upload your PDF", type="pdf")


# STEP 3: PROCESS PDF

if file:

    if st.button("Process PDF"):

        with open("temp.pdf", "wb") as f:
            f.write(file.getbuffer())

        # Load PDF

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # Split into chunks

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)

        # Create embeddings

        embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        # Store in ChromaDB

        db = Chroma.from_documents(
            chunks,
            embeddings
        )

        st.session_state.db = db

        st.success("PDF processed successfully!")


# STEP 4: ASK QUESTIONS

if "db" in st.session_state:

    question = st.chat_input("Ask a question about your PDF")

    if question:

        st.chat_message("user").write(question)

        # Similarity search

        results = st.session_state.db.similarity_search(
            question,
            k=3
        )

        # Combine retrieved chunks

        context = "\n".join(
            result.page_content for result in results
        )

        # Create prompt

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question using the following context.

            If the answer is not available in the context,
            say that it is not available in the document.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )

        # Connect LLM

        llm = ChatOllama(model="llama3.2")

        # Generate answer

        chain = prompt | llm

        response = chain.invoke({
            "context": context,
            "question": question
        })

        st.chat_message("assistant").write(
            response.content
        )
