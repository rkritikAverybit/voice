import os
import json
import base64
import asyncio
import logging
from datetime import datetime
from typing import Dict, List

import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import AsyncOpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("mindful_plus_tts")

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
    REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
    SYSTEM_PROMPT = (
        "You are Mindful+, a calm, supportive voice companion. "
        "Speak warmly in simple English. Keep replies short (2–3 sentences). "
        "Offer gentle grounding or breathing when useful. "
        "Acknowledge feelings without judgment. Do not give medical advice."
    )

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing in environment")
        return True

# Memory (light)
MEMORY: List[Dict[str, str]] = []

class ChatMessage(BaseModel):
    content: str
    context: str = ""

class ChatResponse(BaseModel):
    success: bool
    response: str
    timestamp: str

Config.validate()
transport = httpx.AsyncHTTPTransport(retries=2)
http_client = httpx.AsyncClient(transport=transport, timeout=60.0, follow_redirects=True)
client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY, http_client=http_client)

app = FastAPI(title="Mindful+ Human Voice", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def home():
    return FileResponse("static/index.html")

@app.get("/api/health")
async def health():
    return {"status": "ok", "key": bool(Config.OPENAI_API_KEY)}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(msg: ChatMessage):
    try:
        messages = [{"role":"system","content":Config.SYSTEM_PROMPT}] + MEMORY[-10:]
        messages.append({"role":"user","content":msg.content})
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=400
        )
        text = resp.choices[0].message.content
        MEMORY.extend([{"role":"user","content":msg.content},{"role":"assistant","content":text}])
        if len(MEMORY)>10: del MEMORY[:-10]
        return ChatResponse(success=True, response=text, timestamp=datetime.now().isoformat())
    except Exception as e:
        log.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------- REALTIME + TTS -------------
SESSIONS: Dict[str, Dict] = {}

class RealtimeClient:
    def __init__(self, sid: str, on_text_done, on_error):
        self.sid = sid
        self.ws = None
        self.on_text_done = on_text_done
        self.on_error = on_error
        self.ready = asyncio.Event()

    async def connect(self):
        headers = {
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }
        self.ws = await websockets.connect(
            Config.REALTIME_URL, extra_headers=headers, ping_interval=20, ping_timeout=10
        )
        asyncio.create_task(self._recv())
        await self._send({
            "type": "session.update",
            "session": {
                "modalities": ["audio","text"],
                "instructions": Config.SYSTEM_PROMPT,
                "voice": "none",                # disable realtime voice
                "input_audio_format": "pcm16",
                "output_audio_format": "none",  # we'll handle TTS manually
                "input_audio_transcription": {"model":"whisper-1"},
                "turn_detection": {
                    "type":"server_vad","silence_duration_ms":650,
                    "prefix_padding_ms":250,"create_response":True
                },
                "temperature":0.7,
                "max_response_output_tokens":2048
            }
        })
        self.ready.set()

    async def _send(self,obj:dict):
        if self.ws: await self.ws.send(json.dumps(obj))

    async def _recv(self):
        try:
            async for raw in self.ws:
                e=json.loads(raw)
                t=e.get("type")
                if t=="response.audio_transcript.done":
                    tr=e.get("transcript") or ""
                    if tr: await self.on_text_done(tr)
                elif t=="error":
                    err=e.get("error",{})
                    await self.on_error(f"{err.get('code','')}: {err.get('message','Unknown error')}")
        except Exception as ex:
            await self.on_error(str(ex))

    async def send_audio(self,b64:str):
        await self.ready.wait()
        await self._send({"type":"input_audio_buffer.append","audio":b64})

    async def commit(self):
        await self._send({"type":"input_audio_buffer.commit"})
        await self._send({"type":"response.create","response":{"modalities":["audio","text"]}})

    async def close(self):
        try:
            if self.ws: await self.ws.close()
        except Exception: pass

@app.websocket("/ws/voice/{sid}")
async def ws_voice(ws: WebSocket, sid: str):
    await ws.accept()
    SESSIONS[sid] = {"ws": ws, "client": None}
    await safe_send(ws, {"type": "connected", "session_id": sid})

    async def on_text_final(text: str):
        # send the text
        await safe_send(ws, {"type":"transcript","text":text})
        # generate a human TTS voice
        try:
            speech = await client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="verse",      # or alloy, cora, shimmer
                input=text
            )
            audio_b64 = base64.b64encode(await speech.read()).decode("utf-8")
            await safe_send(ws, {"type":"audio_response_tts","data":audio_b64})
        except Exception as e:
            await send_err(sid,f"TTS failed: {e}")

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            t = msg.get("type")
            if t=="start_session":
                c=RealtimeClient(sid,on_text_done=on_text_final,on_error=lambda e: asyncio.create_task(send_err(sid,e)))
                SESSIONS[sid]["client"]=c
                await c.connect()
                await safe_send(ws,{"type":"session_started"})
            elif t=="audio_data":
                if c:=SESSIONS[sid].get("client"):
                    await c.send_audio(msg.get("data",""))
            elif t=="stop_session":
                if c:=SESSIONS[sid].get("client"):
                    await c.commit()
                    await c.close()
                break
            elif t=="ping":
                await safe_send(ws,{"type":"pong","ts":datetime.utcnow().isoformat()})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await send_err(sid,str(e))
    finally:
        await cleanup(sid)

async def safe_send(ws: WebSocket, payload: dict):
    try: await ws.send_json(payload)
    except Exception: pass

async def send_err(sid: str, err: str):
    s=SESSIONS.get(sid)
    if not s: return
    await safe_send(s["ws"],{"type":"error","message":err})

async def cleanup(sid: str):
    s=SESSIONS.pop(sid,None)
    if not s: return
    try: await safe_send(s["ws"],{"type":"session_closed"})
    except Exception: pass
    if c:=s.get("client"):
        try: await c.close()
        except Exception: pass

if __name__=="__main__":
    import uvicorn
    uvicorn.run("backend:app",host="0.0.0.0",port=8000,reload=True)
