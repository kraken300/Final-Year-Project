from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GEMINI_API_KEY"]=os.getenv("GEMINI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Embedding Function
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key=os.environ["GEMINI_API_KEY"])

vector_db =  Chroma(
    persist_directory = "../vector_db",
    collection_name = "docs",
    embedding_function = embeddings
)

# create chat model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=os.environ["GEMINI_API_KEY"], temperature=0.5)

# from langchain_community.chat_models import ChatOllama
# llm = ChatOllama(model="gemma:2b")
# llm=Ollama(model="gemma:2b")

memory = ConversationBufferMemory(
    return_messages=True, memory_key="chat_history"
)

# create retriever chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    memory=memory,
    retriever = vector_db.as_retriever(
        search_kwargs={"fetch_k":4,"k":3},search_type="mmr"
        # search_kwargs={"k":2},search_type="similarity"
    ),
    chain_type = "refine",
    verbose=True
)

def rag_function(question: str) -> str:
    """
    This function takes in user question or prompt and returns a response

    :param: question: String value of the question or the prompt from the user.
    :returns: String value of the answer to the user question.
    """
    
    response = qa_chain.invoke({"question":question})

    return (response.get("answer"))

