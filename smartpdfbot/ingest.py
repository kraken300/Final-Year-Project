from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
    ids = [str(i) for i in range(1, len(text_chunks) + 1)]

    vectordb = Chroma.from_texts(
        texts = text_chunks,
        embedding=embeddings,
        persist_directory = "vector_db",
        collection_name = "docs",
        ids=ids
    )
    vectordb.persist()

def delete():
    # Specify the directory to be deleted
    directory = 'vector_db'

    try:
        # Remove the directory
        shutil.rmtree(directory)
        print(f'Directory {directory} has been deleted successfully')
    except FileNotFoundError:
        print(f'Directory {directory} not found')
    except OSError as e:
        print(f'Error: {directory} : {e.strerror}')