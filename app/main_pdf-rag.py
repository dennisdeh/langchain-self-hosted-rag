from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dotenv
import os
import glob

# 0. Read environmental variables
flag_env = dotenv.load_dotenv(".env")
assert flag_env, "Failed to load .env file"

# 1. Set up Ollama LLM and embedding objects
llm = OllamaLLM(model="gemma3:1b", base_url=f"http://localhost:{os.getenv('OLLAMA_PORT_HOST')}")
embedding = OllamaEmbeddings(base_url=f"http://localhost:{os.getenv('OLLAMA_PORT_HOST')}", model="nomic-embed-text")


# 2. Load PDF documents
def load_pdf_documents(pdf_directory: str):
    """
    Load all PDF documents from the specified directory
    """
    pdf_files = glob.glob(f"{pdf_directory}/*.pdf")

    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return []

    all_docs = []
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}")
        try:
            loader = UnstructuredLoader(file_path=pdf_file, strategy="hi_res")
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

    return all_docs


# 3. Split documents
def split_documents(documents):
    """
    Split documents into smaller chunks for better processing
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)


# 4. Main processing
pdf_directory = "app/data"  # Change this to your PDF directory path
docs = load_pdf_documents(pdf_directory)

if docs:
    # Split the documents into chunks
    split_docs = split_documents(docs)
    print(f"Loaded {len(docs)} documents, split into {len(split_docs)} chunks")

    # 5. Connect to Qdrant and upload documents
    qdrant = QdrantVectorStore.from_documents(
        documents=split_docs,
        embedding=embedding,
        url=f"http://localhost",
        port=os.getenv('QDRANT_PORT_HOST'),
        prefer_grpc=False,
        collection_name="pdf-rag-collection",
    )

    # 6. Create retriever and RAG chain
    retriever = qdrant.as_retriever()

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # 7. Run a query asking about the documents
    query = "Summarize the key points from these documents."
    result = rag_chain.invoke(query)

    print("\nAnswer:", result["result"])
    print("\nSources (showing first 3):")
    for i, doc in enumerate(result["source_documents"][:10]):
        print(f"- Source {i + 1}: {doc.page_content[:150]}...")
else:
    print("No documents were loaded. Please check your PDF directory.")
