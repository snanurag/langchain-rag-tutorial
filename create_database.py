from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
import os
import nltk
import ssl
import shutil

# Setup SSL
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY", "your-default-api-key")

CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Only delete ChromaDB if necessary (prevents unnecessary re-processing)
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)  

def download_nltk_data():
    """Ensures required NLTK resources are available."""
    required_packages = ["punkt", "punkt_tab", "averaged_perceptron_tagger_eng"]
    for package in required_packages:
        try:
            nltk.data.find(f"tokenizers/{package}")
        except LookupError:
            nltk.download(package)

download_nltk_data()


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    """Loads documents efficiently from the directory."""
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = list(loader.lazy_load())  # Lazy loading for better memory usage
    return documents


def split_text(documents: list[Document]):
    """Splits documents into smaller chunks for better embedding accuracy."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # Increased for better efficiency
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Debug print first chunk
    if chunks:
        print(chunks[0].page_content)
        print(chunks[0].metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    """Stores document embeddings in ChromaDB."""
    
    # Ensure ChromaDB directory exists
    os.makedirs(CHROMA_PATH, exist_ok=True)

    # Use OpenAI's latest large-scale embedding model
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
