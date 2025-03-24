from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob


def load_pdf_documents(path: str):
    """
    Load all PDF documents from the specified path
    """
    pdf_files = glob.glob(f"{path}/*.pdf")

    if not pdf_files:
        print(f"No PDF files found in {path}")
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


def split_documents(documents, chunk_size:int=1000, chunk_overlap:int=200):
    """
    Split documents into smaller chunks for better processing
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

