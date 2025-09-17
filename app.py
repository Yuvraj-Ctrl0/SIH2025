from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .recognizer import RecognizerThread
from .mqtt_client import MqttBridge
from .db import init_db, SessionLocal, Attendance
from .sample_data import create_sample
from .config import ROOM_ID, ADMIN_PASSPHRASE
import uuid, os

app = FastAPI(title='Smart Classroom Complete')
app.mount('/static', StaticFiles(directory='server/static'), name='static')

SESSION_ID = f"{ROOM_ID}-{uuid.uuid4().hex[:6]}"
recognizer = None; mqtt = None

@app.on_event('startup')
def on_start():
    global recognizer, mqtt, SESSION_ID
    init_db()
    mqtt = MqttBridge()
    recognizer = RecognizerThread(session_id=SESSION_ID, mqtt_publish=mqtt.publish_cmd)
    recognizer.start()

@app.get('/')
def index():
    return FileResponse('server/static/index.html')

@app.get('/api/status')
def status():
    s = recognizer.snapshot() if recognizer else {'state':'STOPPED'}
    db = SessionLocal()
    try:
        rows = db.query(Attendance).filter(Attendance.session_id==SESSION_ID).all()
        people = [{'roll':r.roll,'name':r.name,'ts':r.timestamp.isoformat(),'conf':r.confidence} for r in rows]
    finally:
        db.close()
    s['people'] = people; s['session_id'] = SESSION_ID
    return JSONResponse(s)

@app.post('/api/command')
async def command(req: Request):
    payload = await req.json()
    mqtt.publish_cmd(payload)
    return {'ok': True, 'sent': payload}

@app.post('/api/enroll_sample')
async def enroll_sample(payload: dict):
    n = int(payload.get('count',3))
    create_sample(n, os.getenv('SC_ADMIN_PASSPHRASE'))
    return {'ok':True,'created':n}
