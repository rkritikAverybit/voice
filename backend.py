
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
log = logging.getLogger("mindful_plus")

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
    REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
    VOICE = os.getenv("VOICE", "alloy")
    INPUT_AUDIO_FORMAT = "pcm16"
    OUTPUT_AUDIO_FORMAT = "pcm16"
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

class ChatMessage(BaseModel):
    content: str
    context: str = ""

class ChatResponse(BaseModel):
    success: bool
    response: str
    timestamp: str

# light memory (last 10 messages total)
MEMORY: List[Dict[str, str]] = []

class OpenAIService:
    def __init__(self):
        transport = httpx.AsyncHTTPTransport(retries=2)
        http_client = httpx.AsyncClient(transport=transport, timeout=30.0, follow_redirects=True)
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY, http_client=http_client)

    async def text_reply(self, msg: str, context: str = "") -> str:
        messages = [{"role": "system", "content": Config.SYSTEM_PROMPT}]
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        history = MEMORY[-10:]
        messages.extend(history)
        messages.append({"role": "user", "content": msg})
        try:
            resp = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=400
            )
            text = resp.choices[0].message.content
            MEMORY.append({"role": "user", "content": msg})
            MEMORY.append({"role": "assistant", "content": text})
            if len(MEMORY) > 10:
                del MEMORY[:-10]
            return text
        except Exception as e:
            log.error(f"Text API error: {e}")
            return "I'm having trouble responding right now. Please try again."

class RealtimeClient:
    def __init__(self, sid: str, on_audio, on_text_delta, on_text_done, on_error):
        self.sid = sid
        self.ws = None
        self.audio_buf = bytearray()
        self.on_audio = on_audio
        self.on_text_delta = on_text_delta
        self.on_text_done = on_text_done
        self.on_error = on_error
        self.ready = asyncio.Event()

    async def connect(self):
        headers = {
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }
        self.ws = await websockets.connect(
            Config.REALTIME_URL,
            extra_headers=headers,
            ping_interval=20,
            ping_timeout=10
        )
        asyncio.create_task(self._recv())

        await self._send({
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": Config.SYSTEM_PROMPT,
                "voice": Config.VOICE,
                "input_audio_format": Config.INPUT_AUDIO_FORMAT,
                "output_audio_format": Config.OUTPUT_AUDIO_FORMAT,
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "silence_duration_ms": 650,
                    "prefix_padding_ms": 250,
                    "create_response": True
                },
                "temperature": 0.7,
                "max_response_output_tokens": 2048
            }
        })
        self.ready.set()

    async def _send(self, msg: dict):
        if self.ws:
            await self.ws.send(json.dumps(msg))

    async def _recv(self):
        try:
            async for raw in self.ws:
                e = json.loads(raw)
                t = e.get("type")
                if t == "response.audio.delta":
                    delta_b64 = e.get("delta", "")
                    if delta_b64:
                        self.audio_buf.extend(base64.b64decode(delta_b64))
                elif t == "response.audio.done":
                    if self.audio_buf:
                        await self.on_audio(bytes(self.audio_buf))
                        self.audio_buf.clear()
                elif t == "response.audio_transcript.delta":
                    delta = e.get("delta") or ""
                    if delta:
                        await self.on_text_delta(delta)
                elif t == "response.audio_transcript.done":
                    transcript = e.get("transcript") or ""
                    if transcript:
                        await self.on_text_done(transcript)
                elif t == "error":
                    err = e.get("error", {})
                    await self.on_error(f"{err.get('code','')}: {err.get('message','Unknown error')}")
        except Exception as ex:
            await self.on_error(str(ex))

    async def send_audio(self, audio_b64: str):
        await self.ready.wait()
        await self._send({"type": "input_audio_buffer.append", "audio": audio_b64})

    async def commit(self):
        await self._send({"type": "input_audio_buffer.commit"})
        await self._send({"type": "response.create", "response": {"modalities": ["audio", "text"]}})

    async def close(self):
        try:
            if self.ws:
                await self.ws.close()
        except Exception:
            pass

# FastAPI app
Config.validate()
app = FastAPI(title="Mindful+ API", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# serve static
app.mount("/static", StaticFiles(directory="static"), name="static")

# connections
SESSIONS: Dict[str, Dict] = {}
svc = OpenAIService()

@app.get("/")
async def home():
    return FileResponse("static/index.html")

@app.get("/api/health")
async def health():
    return {"status": "healthy", "sessions": len(SESSIONS), "key": bool(Config.OPENAI_API_KEY)}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(msg: ChatMessage):
    try:
        text = await svc.text_reply(msg.content, msg.context)
        return ChatResponse(success=True, response=text, timestamp=datetime.now().isoformat())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/voice/{sid}")
async def ws_voice(ws: WebSocket, sid: str):
    await ws.accept()
    SESSIONS[sid] = {"ws": ws, "client": None}
    await safe_send(ws, {"type": "connected", "session_id": sid})
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            t = msg.get("type")

            if t == "start_session":
                client = RealtimeClient(
                    sid,
                    on_audio=lambda a: asyncio.create_task(send_audio(sid, a)),
                    on_text_delta=lambda d: asyncio.create_task(send_text_delta(sid, d)),
                    on_text_done=lambda txt: asyncio.create_task(send_text_done(sid, txt)),
                    on_error=lambda e: asyncio.create_task(send_err(sid, e))
                )
                SESSIONS[sid]["client"] = client
                await client.connect()
                await safe_send(ws, {"type": "session_started"})

            elif t == "audio_data":
                if c := SESSIONS[sid].get("client"):
                    await c.send_audio(msg.get("data", ""))

            elif t == "stop_session":
                if c := SESSIONS[sid].get("client"):
                    await c.commit()
                    await c.close()
                break

            elif t == "ping":
                await safe_send(ws, {"type": "pong", "ts": datetime.now().isoformat()})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await safe_send(ws, {"type": "error", "message": str(e)})
    finally:
        await cleanup(sid)

async def safe_send(ws: WebSocket, payload: dict):
    try:
        await ws.send_json(payload)
    except Exception:
        pass

async def send_audio(sid: str, audio: bytes):
    s = SESSIONS.get(sid)
    if not s: return
    await safe_send(s["ws"], {"type": "audio_response", "data": audio.hex()})

async def send_text_delta(sid: str, delta: str):
    s = SESSIONS.get(sid)
    if not s: return
    await safe_send(s["ws"], {"type": "transcript_delta", "delta": delta})

async def send_text_done(sid: str, text: str):
    s = SESSIONS.get(sid)
    if not s: return
    await safe_send(s["ws"], {"type": "transcript", "text": text})

async def send_err(sid: str, err: str):
    s = SESSIONS.get(sid)
    if not s: return
    await safe_send(s["ws"], {"type": "error", "message": err})

async def cleanup(sid: str):
    s = SESSIONS.pop(sid, None)
    if not s: return
    try:
        await safe_send(s["ws"], {"type": "session_closed"})
    except Exception:
        pass
    if c := s.get("client"):
        try:
            await c.close()
        except Exception:
            pass

# local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
