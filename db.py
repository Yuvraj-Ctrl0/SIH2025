from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, LargeBinary, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import numpy as np
from .crypto_utils import encrypt_json, decrypt_json

# Database URL (SQLite)
DB_URL = "sqlite:///attendance.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

def now():
    return datetime.utcnow()

# -------------------------
# MODELS
# -------------------------
class Student(Base):
    __tablename__ = "students"  # Fixed: double underscores
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    roll = Column(String, nullable=False, unique=True)
    embedding_enc = Column(LargeBinary, nullable=False)
    salt = Column(LargeBinary, nullable=False)

class Attendance(Base):
    __tablename__ = "attendance"  # Fixed: double underscores
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, nullable=False)
    roll = Column(String, nullable=False)
    name = Column(String, nullable=False)
    timestamp = Column(DateTime, default=now)
    confidence = Column(Float, default=0.0)
    session_id = Column(String, nullable=False)
    liveness_pass = Column(Boolean, default=True)

class Event(Base):
    __tablename__ = "events"  # Fixed: double underscores
    id = Column(Integer, primary_key=True, index=True)
    room = Column(String, nullable=False)
    kind = Column(String, nullable=False)
    payload = Column(Text, default="{}")
    timestamp = Column(DateTime, default=now)

# -------------------------
# INIT DB
# -------------------------
def init_db():
    Base.metadata.create_all(bind=engine)

# -------------------------
# DATABASE OPERATIONS
# -------------------------
def add_student_with_embedding(name: str, roll: str, embedding, passphrase: str):
    """Add a student with encrypted embedding to database"""
    if not passphrase:
        raise ValueError("Passphrase required for encryption")
    
    # Convert single embedding to list format for consistency
    embedding_list = [embedding.tolist() if hasattr(embedding, 'tolist') else embedding]
    token, salt = store_encrypted_embedding(passphrase, embedding_list)
    
    db = SessionLocal()
    try:
        student = Student(name=name, roll=roll, embedding_enc=token, salt=salt)
        db.add(student)
        db.commit()
        db.refresh(student)
        return student
    finally:
        db.close()

def load_all_students_embeddings(passphrase: str):
    """Load all student embeddings from database"""
    if not passphrase:
        print("Warning: No passphrase provided, cannot decrypt embeddings")
        return []
    
    db = SessionLocal()
    try:
        students = db.query(Student).all()
        result = []
        for student in students:
            try:
                embeddings_list = load_decrypted_embeddings(passphrase, student.embedding_enc, student.salt)
                # Use the first embedding (assuming multiple were stored)
                if embeddings_list:
                    result.append({
                        "id": student.id,
                        "name": student.name,
                        "roll": student.roll,
                        "embedding": embeddings_list[0]  # First embedding
                    })
            except Exception as e:
                print(f"Failed to decrypt embedding for {student.name}: {e}")
        return result
    finally:
        db.close()

def add_attendance(student_id: int, roll: str, name: str, confidence: float, session_id: str = "default"):
    """Add attendance record"""
    db = SessionLocal()
    try:
        attendance = Attendance(
            student_id=student_id,
            roll=roll,
            name=name,
            confidence=confidence,
            session_id=session_id
        )
        db.add(attendance)
        db.commit()
        return attendance
    finally:
        db.close()

# -------------------------
# ENCRYPTION HELPERS
# -------------------------
def store_encrypted_embedding(passphrase, emb_list):
    token, salt = encrypt_json(passphrase, emb_list)
    return token, salt

def load_decrypted_embeddings(passphrase, token, salt):
    lst = decrypt_json(passphrase, token, salt)
    return [np.array(x, dtype="float32") for x in lst]

# -------------------------
# SCRIPT ENTRYPOINT
# -------------------------
if __name__ == "__main__":  # Fixed: double underscores
    print("📦 Initializing database...")
    init_db()
    print("✅ Database initialized successfully (attendance.db created)")