from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_chroma import Chroma
from typing import Literal
import os
from dotenv import load_dotenv
from utils import chunker, load_process
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()
# load your api key
open_api_key = os.getenv("OPENAI_API_KEY")

if not open_api_key:
    raise ValueError("No API KEY FOUND! Contact the right authorities")

print("Successfully loaded your API KEY!")

#initialise your model

llm = ChatOpenAI(
    model = "gpt-5-nano",
    temperature= 0.7,
    api_key=open_api_key
)

print(f"Initialized Model: {llm.model_name}")

#load document
file_path = "documents\\pharmaceutics-16-00637.pdf"

# loader = PyPDFLoader(file_path)
# pages = []

# async for page in loader.alazy_load():
#     pages.append(page)

async def load_process(file_path):
    loader = PyPDFLoader(file_path)
    pages = []

    async for page in loader.alazy_load():
        pages.append(page)
 
    return pages

print(f"loaded {len(pages)} pages from the pdf")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap= 100
    )

split_doc = text_splitter.split_documents(pages)
# processed = load_process(file_path)

# #chunk document
# chunked = chunker(processed)

# vector-store
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",api_key=open_api_key
)
print("Embedding model on!")

chroma_path = "chroma_db"
vectorstore = Chroma(
    collection_name="curcumn_docs",
    persist_directory=chroma_path,
    embedding_function=embeddings
)

vectorstore.add_documents(documents=split_doc)
print(f"✅ Vector store created with {len(split_doc)} chunks")
print(f"   Persisted to: {chroma_path}")


