from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import dotenv
import os

# 0. Read environmental variables
flag_env = dotenv.load_dotenv(".env")
assert flag_env, "Failed to load .env file"

# 1. Set up Ollama LLM and embedding objects
llm = OllamaLLM(model="gemma3:1b", base_url=f"http://localhost:{os.getenv('OLLAMA_PORT_HOST')}")
embedding = OllamaEmbeddings(base_url=f"http://localhost:{os.getenv('OLLAMA_PORT_HOST')}", model="nomic-embed-text")

# 2. Load and split documents (some text examples from LangChains website)
docs = [
    Document(page_content="LangChain is a composable framework to build with LLMs. LangGraph is the orchestration framework for controllable agentic workflows."),
    Document(page_content="The largest community building the future of LLM apps"),
    Document(page_content="LangChain’s flexible abstractions and AI-first toolkit make it the #1 choice for developers when building with GenAI. Join 1M+ builders standardizing their LLM app development in LangChain's Python and JavaScript frameworks.")
]

# 3. Connect to Qdrant and upload documents
qdrant = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embedding,
    url=f"http://localhost",
    port=os.getenv('QDRANT_PORT_HOST'),
    prefer_grpc=False,
    collection_name="rag-collection",
)

# 4. Create retriever and RAG chain
retriever = qdrant.as_retriever()

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 5. Run a query asking about the documents
query = "What is LangChain? Give a short summary."
result = rag_chain.invoke(query)

print("\nAnswer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print("-", doc.page_content)
