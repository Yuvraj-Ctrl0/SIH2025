import numpy as np
from .db import SessionLocal, init_db
from .db import Student, store_encrypted_embedding
from .config import ADMIN_PASSPHRASE

def create_sample(count=3, passphrase=None):
    """Create sample students with random embeddings"""
    
    # Use provided passphrase or get from config
    if passphrase is None:
        passphrase = ADMIN_PASSPHRASE
    
    if not passphrase:
        raise ValueError("No passphrase provided. Set SC_ADMIN_PASSPHRASE environment variable.")
    
    print(f"🧪 Creating {count} sample students...")
    
    init_db()
    db = SessionLocal()
    
    try:
        created_count = 0
        for i in range(count):
            name = f"Demo Student {i+1}"
            roll = f"DS{i+1:03d}"
            
            # Check if student already exists
            existing = db.query(Student).filter(Student.roll == roll).first()
            if existing:
                print(f"⚠️  Student {roll} already exists, skipping...")
                continue
            
            # Generate random face embeddings (128-dimensional for face_recognition)
            embeddings = [np.random.rand(128).astype('float32').tolist() for _ in range(3)]
            
            # Encrypt and store
            token, salt = store_encrypted_embedding(passphrase, embeddings)
            
            student = Student(
                name=name,
                roll=roll,
                embedding_enc=token,
                salt=salt
            )
            
            db.add(student)
            created_count += 1
            print(f"✅ Created: {name} ({roll})")
        
        db.commit()
        print(f"🎉 Successfully created {created_count} sample students!")
        return created_count
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error creating sample students: {e}")
        raise e
    finally:
        db.close()

if __name__ == '__main__':
    import os
    try:
        passphrase = os.getenv('SC_ADMIN_PASSPHRASE')
        if not passphrase:
            print("❌ Error: SC_ADMIN_PASSPHRASE environment variable not set!")
            print("Set it with: export SC_ADMIN_PASSPHRASE='YourPassword' (Linux/Mac)")
            print("Or: $env:SC_ADMIN_PASSPHRASE = 'YourPassword' (PowerShell)")
        else:
            create_sample(3, passphrase)
    except Exception as e:
        print(f"❌ Failed to create samples: {e}")