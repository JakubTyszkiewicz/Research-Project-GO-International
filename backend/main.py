import os
import shutil
import glob
import uuid
import asyncio
from datetime import datetime
from typing import List, Optional
from operator import itemgetter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# RAG & LangChain Imports
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Database
from database import SessionLocal, ChatMessage, ChatSession, init_db

app = FastAPI(title="International Processes Chatbot RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
try:
    DATA_PATH = os.environ["DATA_PATH"]
    DB_PATH = os.environ["DB_PATH"]
    OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
    APP_USER = os.environ["APP_USER"]
    MODEL_NAME = os.environ["MODEL_NAME"]
    EMBEDDING_MODEL_NAME = os.environ["EMBEDDING_MODEL_NAME"]
except KeyError as e:
    raise RuntimeError(f"Missing required environment variable: {e}")

# Mount Static
app.mount("/static", StaticFiles(directory=DATA_PATH), name="static")

# Init DB
init_db()

# --- Global Variables ---
vectorstore = None
retriever = None
llm = None
rag_chain = None

# --- Models ---
class CreateSessionRequest(BaseModel):
    user_id: str = APP_USER

class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime

class QueryRequest(BaseModel):
    question: str
    user_id: str = APP_USER
    session_id: str

class HistoryResponse(BaseModel):
    id: int
    role: str
    content: str
    sources: List[str] = []

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# --- RAG Initialization ---
def initialize_rag():
    global vectorstore, retriever, llm, rag_chain
    print("Initializing RAG system...")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Check persistence
    db_exists = os.path.exists(DB_PATH) and any(os.scandir(DB_PATH))

    if db_exists:
        print(f"Loading existing vector store from {DB_PATH}...")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        print(f"Index not found. Ingesting data from {DATA_PATH}...")
        os.makedirs(DB_PATH, exist_ok=True)
        pdf_files = glob.glob(os.path.join(DATA_PATH, "**/*.pdf"), recursive=True)
        
        docs = []
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(pdf_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {pdf_path}: {e}")
        
        if docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            # Fix metadata paths
            for doc in splits:
                source_abs = doc.metadata.get("source", "")
                if source_abs.startswith(DATA_PATH):
                    doc.metadata["source"] = os.path.relpath(source_abs, DATA_PATH).replace("\\", "/")
            
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_PATH)
        else:
            print("No documents found. Initializing empty.")
            vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)

    template = """Je bent een AI-assistent voor studenten van Howest. Je helpt hen met vragen over internationale stages, studieprogramma's in het buitenland en visumprocedures.

Je hebt toegang tot de voorgaande conversatie met de student:
{history}

Gebruik de volgende context om de vraag te beantwoorden.
Antwoord ALTIJD in het Nederlands.

INSTRUCTIES VOOR OPMAAK:
1. Gebruik Markdown om de tekst leesbaar te maken.
2. Gebruik **vetgedrukte tekst** voor belangrijke datums, deadlines of kernbegrippen.
3. Gebruik bullet points (-) of genummerde lijsten voor opsommingen.

Als je het antwoord niet weet, zeg dan dat je het niet weet.

Context:
{context}

Vraag:
{question}

Antwoord:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(f"Inhoud: {doc.page_content}\nBron: {doc.metadata.get('source', 'Onbekend')}" for doc in docs)

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs, 
            "question": itemgetter("question"), 
            "history": itemgetter("history")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Warmup to force model load
    print("Warming up Ollama model (forcing tensor load)...")
    try:
        # Simple non-chain usage just to trigger load
        llm.invoke("Hello")
        print("Ollama model loaded and ready.")
    except Exception as e:
        print(f"Warning: Model warmup failed: {e}")

    print("RAG System Initialized.")

@app.on_event("startup")
async def startup_event():
    # Run initialization synchronously (blocking)
    # This prevents the backend from accepting requests until the model is loaded
    await asyncio.to_thread(initialize_rag)

# --- Endpoints ---

@app.post("/sessions", response_model=ChatSessionResponse)
async def create_session(request: CreateSessionRequest):
    db = SessionLocal()
    try:
        session_id = str(uuid.uuid4())
        new_session = ChatSession(
            id=session_id,
            user_id=request.user_id,
            title="Nieuwe Chat"
        )
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        return ChatSessionResponse(
            id=new_session.id,
            title=new_session.title,
            created_at=new_session.created_at
        )
    finally:
        db.close()

@app.get("/sessions", response_model=List[ChatSessionResponse])
async def get_sessions(user_id: str = APP_USER):
    db = SessionLocal()
    try:
        sessions = db.query(ChatSession)\
            .filter(ChatSession.user_id == user_id)\
            .order_by(ChatSession.created_at.desc())\
            .all()
        return [
            ChatSessionResponse(id=s.id, title=s.title, created_at=s.created_at)
            for s in sessions
        ]
    finally:
        db.close()

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        db.delete(session) # Configured cascade delete in database.py
        db.commit()
        return {"status": "deleted"}
    finally:
        db.close()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet.")
    
    db = SessionLocal()
    try:
        # Check if session exists
        session = db.query(ChatSession).filter(ChatSession.id == request.session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # 1. Fetch History (scoped to session)
        history_msgs = db.query(ChatMessage)\
            .filter(ChatMessage.session_id == request.session_id)\
            .order_by(ChatMessage.timestamp.desc())\
            .limit(10)\
            .all()
        
        # Reverse to get chronological order for the prompt
        history_msgs.reverse()
        history_text = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in history_msgs])

        # --- Chit-Chat Detection ---
        # Detect simple greetings/thanks to avoid confused RAG responses
        low_q = request.question.lower().strip()
        is_chitchat = any(x in low_q for x in ["dank u", "dankjewel", "bedankt", "hallo", "goeiemorgen", "goedemiddag", "merci"]) and len(low_q) < 20
        
        answer = ""
        unique_sources = []

        if is_chitchat:
            # Simple direct response, no retrieval
            simple_prompt = f"""
Je bent een behulpzame AI assistent.
De gebruiker zegt: "{request.question}"
Reageer kort en beleefd in het Nederlands (bijv. "Graag gedaan!", "Hallo! Waarmee kan ik helpen?").
"""
            answer = llm.invoke(simple_prompt)
            unique_sources = []
        else:
            # 2. Run Inference (RAG)
            answer = rag_chain.invoke({"question": request.question, "history": history_text})
            
            # 3. Retrieve Sources (for UI)
            docs = retriever.invoke(request.question)
            unique_sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))
        
        # 4. Save User Message
        user_msg = ChatMessage(
            session_id=request.session_id, # Linked to session
            role="user",
            content=request.question
        )
        db.add(user_msg)
        
        # 5. Save Assistant Message
        ai_msg = ChatMessage(
            session_id=request.session_id, # Linked to session
            role="assistant",
            content=answer,
            sources=",".join(unique_sources)
        )
        db.add(ai_msg)

        # 6. Auto-Update Title if this is the first interaction
        # We check if there were no previous messages (history empty before this exchange)
        if len(history_msgs) == 0:
            # Simple heuristic: Use the first 30 chars of the question as title
            new_title = request.question[:30] + ("..." if len(request.question) > 30 else "")
            session.title = new_title
            
        db.commit()
        
        return QueryResponse(answer=answer, sources=unique_sources)
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/history", response_model=List[HistoryResponse])
async def get_history(session_id: str):
    db = SessionLocal()
    try:
        msgs = db.query(ChatMessage)\
            .filter(ChatMessage.session_id == session_id)\
            .order_by(ChatMessage.timestamp.asc())\
            .all()
        
        response = []
        for m in msgs:
            srcs = m.sources.split(",") if m.sources else []
            srcs = [s for s in srcs if s] 
            response.append(HistoryResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                sources=srcs
            ))
        return response
    finally:
        db.close()

@app.get("/health")
async def health():
    return {"status": "ok"}
