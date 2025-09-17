import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration settings
CAMERA_SOURCE = os.getenv('CAMERA_SOURCE', '0')
ROOM_ID = os.getenv('ROOM_ID', 'room101')
ADMIN_PASSPHRASE = os.getenv('SC_ADMIN_PASSPHRASE', None)
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.60'))
FRAMES_REQUIRED = int(os.getenv('FRAMES_REQUIRED', '6'))

# Debug: Print if passphrase is loaded (don't do this in production!)
if ADMIN_PASSPHRASE:
    print(f"✅ Admin passphrase loaded (length: {len(ADMIN_PASSPHRASE)})")
else:
    print("❌ Warning: SC_ADMIN_PASSPHRASE not found in environment!")