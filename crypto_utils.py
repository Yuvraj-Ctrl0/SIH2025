import json
import base64
import os
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet

backend = default_backend()

def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a Fernet key from passphrase + salt."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
        backend=backend
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))

def encrypt_json(passphrase: str, obj) -> tuple[bytes, bytes]:
    """Encrypts Python object (list/dict) into bytes + returns (token, salt)."""
    salt = os.urandom(16)
    key = _derive_key(passphrase, salt)
    f = Fernet(key)
    data = json.dumps(obj).encode()
    token = f.encrypt(data)
    return token, salt

def decrypt_json(passphrase: str, token: bytes, salt: bytes):
    """Decrypts token back into Python object."""
    key = _derive_key(passphrase, salt)
    f = Fernet(key)
    data = f.decrypt(token)
    return json.loads(data.decode())
