from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import app.utils.documents_loader as dl
import dotenv
import os

# 0. Read environmental variables and set paths to documents
flag_env = dotenv.load_dotenv(".env")
assert flag_env, "Failed to load .env file"
pdf_directory = "app/data"
vdb_collection_name="pdf-rag-collection"

# 1) Set up Ollama LLM and embedding objects
llm = OllamaLLM(model="gemma3:1b", base_url=f"http://localhost:{os.getenv('OLLAMA_PORT_HOST')}")
embedding = OllamaEmbeddings(base_url=f"http://localhost:{os.getenv('OLLAMA_PORT_HOST')}", model="nomic-embed-text")

# 2) Load documents
docs = dl.load_pdf_documents(pdf_directory)

if docs:
    # 3) Split the documents into chunks
    split_docs = dl.split_documents(documents=docs, chunk_size=1000, chunk_overlap=200)
    print(f"Loaded {len(docs)} documents, split into {len(split_docs)} chunks")

    # 4) Connect to Qdrant and upload documents
    qdrant = QdrantVectorStore.from_documents(
        documents=split_docs,
        embedding=embedding,
        url="http://localhost",
        port=os.getenv('QDRANT_PORT_HOST'),
        prefer_grpc=False,
        collection_name=vdb_collection_name,
    )

    # 5) Create retriever and RAG chain
    retriever = qdrant.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # 6) Run a query asking about the documents
    query = "Summarize the nature of these documents."
    result = rag_chain.invoke(query)

    print("\nAnswer:", result["result"])
    print("\nSources (showing first 3):")
    for i, doc in enumerate(result["source_documents"][:3]):
        print(f"- Source {i + 1}: {doc.page_content[:150]}...")
else:
    print("No documents were loaded. Please check your PDF directory.")
