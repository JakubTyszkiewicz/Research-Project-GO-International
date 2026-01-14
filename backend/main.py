import os
import shutil
import glob
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="International Processes Chatbot RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Configuration ---
DATA_PATH = "/data"
DB_PATH = "/app/db"

# Mount /data directory to serve PDFs
# Accessible via http://localhost:8000/static/...
app.mount("/static", StaticFiles(directory=DATA_PATH), name="static")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL_NAME = "llama3"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Efficient sentence-transformer model

# --- Global Variables ---
vectorstore = None
retriever = None
llm = None
rag_chain = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# --- Initialization Logic ---

def initialize_rag():
    global vectorstore, retriever, llm, rag_chain
    
    print("Initializing RAG system...")
    
    # 1. Setup Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # 2. Check persistence / Ingest Data
    # Simple check: if DB folder exists and has files (sqlite3), we assume it's populated.
    # Chroma creates a sqlite3 file.
    db_exists = os.path.exists(DB_PATH) and any(os.scandir(DB_PATH))

    if db_exists:
        print(f"Loading existing vector store from {DB_PATH}...")
        vectorstore = Chroma(
            persist_directory=DB_PATH, 
            embedding_function=embeddings
        )
    else:
        print(f"Index not found in {DB_PATH}. Ingesting data from {DATA_PATH}...")
        if not os.path.exists(DATA_PATH):
             print(f"Warning: Data path {DATA_PATH} does not exist. Starting with empty index.")
             # Create directory so Chroma doesn't fail on persist if we were to add documents later
             os.makedirs(DB_PATH, exist_ok=True)
             vectorstore = Chroma(
                persist_directory=DB_PATH, 
                embedding_function=embeddings
            )
        else:
            # Recursive load using manual glob for better error handling
            pdf_files = glob.glob(os.path.join(DATA_PATH, "**/*.pdf"), recursive=True)
            print(f"Found {len(pdf_files)} PDF files in {DATA_PATH}")

            docs = []
            for pdf_path in pdf_files:
                try:
                    loader = PyPDFLoader(pdf_path)
                    file_docs = loader.load()
                    docs.extend(file_docs)
                    # print(f"Loaded: {os.path.basename(pdf_path)}")
                except Exception as e:
                    print(f"SKIPPING: Error loading {os.path.basename(pdf_path)}: {e}")
            
            if not docs:
                print("No PDF documents found.")
            else:
                print(f"Loaded {len(docs)} documents.")
                
                # Split text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(docs)
                
                # Fix metadata paths to be relative to DATA_PATH for cleaner citations
                for doc in splits:
                    source_abs = doc.metadata.get("source", "")
                    if source_abs.startswith(DATA_PATH):
                        # Make relative, e.g., /data/models/module1/doc.pdf -> models/module1/doc.pdf
                        # Careful with path separators if running on windows vs linux
                        rel_path = os.path.relpath(source_abs, DATA_PATH)
                        doc.metadata["source"] = rel_path.replace("\\", "/") # Ensure forward slashes

                print(f"Creating embeddings for {len(splits)} chunks...")
                
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory=DB_PATH
                )
                print("Vector store created and saved.")

    # 3. Setup Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4. Setup LLM
    llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)

    # 5. Setup Chain
    # System Prompt: Dutch output, Domain focus, Auto-formatting.
    template = """Je bent een AI-assistent voor studenten van Howest. Je helpt hen met vragen over internationale stages, studieprogramma's in het buitenland en visumprocedures.

Gebruik de volgende context om de vraag te beantwoorden.
Antwoord ALTIJD in het Nederlands.

INSTRUCTIES VOOR OPMAAK:
1. Gebruik Markdown om de tekst leesbaar te maken.
2. Gebruik **vetgedrukte tekst** voor belangrijke datums, deadlines of kernbegrippen.
3. Gebruik bullet points (-) of genummerde lijsten voor opsommingen.
4. Voeg witregels toe tussen paragrafen.
5. Zet GEEN bronvermeldingen (zoals [Bron: ...]) in de tekst zelf; deze worden door het systeem toegevoegd.

Als je het antwoord niet weet op basis van de context, zeg dan dat je het niet weet. Ga geen informatie verzinnen.

Context:
{context}

Vraag:
{question}

Antwoord:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(f"Inhoud: {doc.page_content}\nBron: {doc.metadata.get('source', 'Onbekend')}" for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG System Initialized.")

@app.on_event("startup")
async def startup_event():
    initialize_rag()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet.")
    
    try:
        # Run inference
        answer = rag_chain.invoke(request.question)
        
        # We can try to extract sources from the retrieval step if we want structured output,
        # but the prompt instructs the LLM to include them in the text. 
        # For this simple implementation, we assume the LLM handles citations in text.
        # However, to be thorough, let's fetch the docs to return relevant sources in the API response too.
        docs = retriever.invoke(request.question)
        unique_sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))
        
        return QueryResponse(answer=answer, sources=unique_sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
