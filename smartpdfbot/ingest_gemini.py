from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import shutil
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GEMINI_API_KEY"]=os.getenv("GEMINI_API_KEY")

def extract_text_from_pdf(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def create_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def store_into_vectordb(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key=os.environ["GEMINI_API_KEY"])

    ids = [str(i) for i in range(1, len(text_chunks) + 1)]

    vectordb = Chroma.from_texts(
        texts = text_chunks,
        embedding=embeddings,
        persist_directory = "../vector_db",
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