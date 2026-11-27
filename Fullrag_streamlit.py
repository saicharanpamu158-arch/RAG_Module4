
import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# STEP 1: PAGE CONFIGURATION

#st.set_page_config(page_title="PDF RAG Chatbot")

st.title("📄 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions about it")


# STEP 2: UPLOAD PDF

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# STEP 3: PROCESS PDF

if st.button("Process PDF"):

    if uploaded_file is not None:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            pdf_path = temp_file.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
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

        # Create Chroma database
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        # Store database in session state
        st.session_state.db = db

        st.success("PDF processed successfully!")


# STEP 4: ASK QUESTIONS

if "db" in st.session_state:

    question = st.chat_input("Ask a question about your PDF")

    if question:

        # Similarity search
        results = st.session_state.db.similarity_search(
            question, k=3
        )

        # Create context
        context = "\n".join(
            result.page_content for result in results
        )

        # Create prompt
        prompt = ChatPromptTemplate.from_template("""
        You are PDF question- answering assistant.
        Instructions:
        1.Answer questions only using the providedcontext.
        2.If the context does not contain relevant information in the document.
        say:"I couldn't find this information in the document."
        3.Do not guess unrelated context to answer questions.
        4.If the user sends a greeting like hi and hello, respond
        politely and ask how you can help with the document.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """)

        # Connect Ollama LLM
        llm = ChatOllama(
            model="llama3.2:1b",
            temperature=0
        )

        # Connect prompt and LLM
        chain = prompt | llm

        # Generate answer
        response = chain.invoke({
            "context": context,
            "question": question
        })

        st.write("### Answer")
        st.write(response.content)
