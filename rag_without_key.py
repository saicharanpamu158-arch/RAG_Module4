from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
# Sample evaluation data
eval_questions = [
    "What is RAG?",
    "What is an embedding?"
]

eval_answers = [
    "RAG stands for Retrieval Augmented Generation.",
    "An embedding converts text into numerical vectors."
]

eval_contexts = [
    ["RAG retrieves relevant information and uses an LLM to generate answers."],
    ["Embeddings represent text as numerical vectors."]
]

eval_ground_truths = [
    "RAG retrieves relevant information and generates answers using an LLM.",
    "Embeddings convert text into numerical vector representations."
]

eval_dataset = Dataset.from_dict({
    "question": eval_questions,
    "answer": eval_answers,
    "contexts": eval_contexts,
    "ground_truth": eval_ground_truths,
})

result = evaluate(eval_dataset, metrics=[
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
])

print(result.to_pandas())