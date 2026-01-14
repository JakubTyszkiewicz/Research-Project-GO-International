import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# Fetch URL from environment
try:
    # Use a new DB file to avoid migration conflicts with the old schema
    SQL_DB_URL = os.environ["SQL_DB_URL"]
except KeyError:
    # Fail fast if env var is missing
    raise RuntimeError("Missing required environment variable: SQL_DB_URL")

# Create Engine
engine = create_engine(SQL_DB_URL, connect_args={"check_same_thread": False})

# Create Session Factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base Class for Models
Base = declarative_base()

# --- Models ---
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True, index=True) # UUID
    user_id = Column(String, index=True)
    title = Column(String, default="Nieuwe Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), index=True)
    role = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    sources = Column(Text, nullable=True) # Stored as comma-separated string
    
    session = relationship("ChatSession", back_populates="messages")

# Helper to create tables
def init_db():
    Base.metadata.create_all(bind=engine)
