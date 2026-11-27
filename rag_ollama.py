
from datasets import Dataset
from ragas import evaluate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)




# 1. Configure Ollama LLM
ollama_llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

# 2. Configure Ollama Embeddings
ollama_embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# 3. Wrap models for RAGAS
llm = LangchainLLMWrapper(ollama_llm)

embeddings = LangchainEmbeddingsWrapper(
    ollama_embeddings
)

# 4. Sample evaluation questions
eval_questions = [
    "What is RAG?",
    "What is an embedding?",
    "What is a vector database?"
]

# 5. Generated answers from RAG application
eval_answers = [
    "RAG stands for Retrieval Augmented Generation.",
    "Embeddings convert text into numerical vectors.",
    "A vector database stores embeddings for similarity search."
]

# 6. Retrieved contexts
eval_contexts = [
    ["RAG retrieves relevant documents and provides context to an LLM to generate answers."],
    ["Embeddings represent text as numerical vectors."],
    ["Vector databases store vector embeddings and support similarity search."]
]

# 7. Expected correct answers
eval_ground_truths = [
    "RAG retrieves relevant information and uses it to generate answers through an LLM.",
    "Embeddings convert text into numerical vector representations.",
    "A vector database stores embeddings and enables similarity search."
]

# 8. Create evaluation dataset
eval_dataset = Dataset.from_dict({
    "question": eval_questions,
    "answer": eval_answers,
    "contexts": eval_contexts,
    "ground_truth": eval_ground_truths
})

# 9. Run RAGAS evaluation
result = evaluate(
    eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=llm,
    embeddings=embeddings
)

# 10. Display results
print("\nRAGAS Evaluation Results:")
print(result.to_pandas())
