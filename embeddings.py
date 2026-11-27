from langchain_ollama import OllamaEmbeddings

load = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url= "http://localhost:11434"
)
vector= load.embed_query("Hello world Hello")
vector2 =load.embed_query("Hello world Hello")
print(f"{len(vector)}")
print(vector[:3])
print(vector[:3])