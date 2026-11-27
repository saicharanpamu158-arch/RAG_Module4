from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from ask import question,context
llm=ChatOllama(
    model="llama3.2:1b",
    temperature=0
)
prompt=ChatPromptTemplate.from_template("""Answer the question using only the context provided.
If the answer is not available in the context say it is not available in document
if the answer is present do not say it is not available.
context:{context}
question:{question}
""")
#question
chain=prompt | llm
response=chain.invoke({
    "context":context,
    "question":question
})
print(response.content)