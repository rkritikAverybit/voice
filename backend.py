import asyncio, base64, json, logging, os
from datetime import datetime
from typing import Dict, Callable
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import AsyncOpenAI
import websockets, httpx
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
    REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
    SYSTEM_PROMPT = (
        "You are Mindful — a calm, supportive voice companion. Speak warmly, "
        "use simple English, and keep replies short (2–3 sentences). Offer brief mindfulness "
        "reminders or grounding when appropriate. Never sound clinical or overly cheerful."
    )
    INPUT_AUDIO_FORMAT = "pcm16"
    OUTPUT_AUDIO_FORMAT = "pcm16"
    VOICE = "alloy"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mindful")

class ChatMessage(BaseModel):
    content: str
    context: str = ""

class ChatResponse(BaseModel):
    success: bool
    response: str
    timestamp: str

class OpenAIRealtimeClient:
    def __init__(self, sid: str, on_audio, on_text, on_error):
        self.sid, self.ws = sid, None
        self.on_audio, self.on_text, self.on_error = on_audio, on_text, on_error
        self.audio_buf = bytearray()
        self.session_ready = asyncio.Event()

    async def connect(self):
        headers = {
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }
        self.ws = await websockets.connect(Config.REALTIME_URL, extra_headers=headers)
        asyncio.create_task(self._recv())
        await self._send({
            "type": "session.update",
            "session": {
                "modalities": ["audio"],
                "instructions": Config.SYSTEM_PROMPT,
                "voice": Config.VOICE,
                "input_audio_format": Config.INPUT_AUDIO_FORMAT,
                "output_audio_format": Config.OUTPUT_AUDIO_FORMAT,
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {"type": "server_vad", "silence_duration_ms": 600, "create_response": True}
            }
        })
        self.session_ready.set()

    async def _send(self, msg): 
        if self.ws: await self.ws.send(json.dumps(msg))

    async def _recv(self):
        try:
            async for m in self.ws:
                e = json.loads(m); t = e.get("type")
                if t == "response.audio.delta":
                    self.audio_buf.extend(base64.b64decode(e.get("delta", "")))
                elif t == "response.audio.done":
                    await self.on_audio(bytes(self.audio_buf)); self.audio_buf.clear()
                elif t == "response.audio_transcript.done":
                    await self.on_text(e.get("transcript", ""))
                elif t == "error":
                    await self.on_error(str(e))
        except Exception as ex:
            logger.error(f"Receive loop error: {ex}")
            await self.on_error(str(ex))

    async def send_audio(self, b64):
        await self.session_ready.wait()
        await self._send({"type": "input_audio_buffer.append", "audio": b64})

    async def commit(self):
        await self._send({"type": "input_audio_buffer.commit"})
        await self._send({"type": "response.create", "response": {"modalities": ["audio"]}})

    async def close(self):
        try:
            await self.ws.close()
        except Exception: pass

class OpenAIService:
    def __init__(self):
        client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(retries=2), timeout=30)
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY, http_client=client)

    async def chat(self, message, context=""):
        try:
            msgs = [{"role": "system", "content": Config.SYSTEM_PROMPT},
                    {"role": "user", "content": message}]
            if context: msgs.insert(1, {"role": "system", "content": f"Context: {context}"})
            res = await self.client.chat.completions.create(model="gpt-4o-mini", messages=msgs, max_tokens=400)
            return res.choices[0].message.content
        except Exception as e:
            return "I'm having trouble connecting right now."

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")
sessions: Dict[str, dict] = {}
service = OpenAIService()

@app.get("/")
async def root(): return FileResponse("static/index.html")

@app.post("/api/chat")
async def chat(msg: ChatMessage):
    text = await service.chat(msg.content, msg.context)
    return ChatResponse(success=True, response=text, timestamp=datetime.now().isoformat())

@app.websocket("/ws/voice/{sid}")
async def ws_voice(ws: WebSocket, sid: str):
    await ws.accept(); sessions[sid] = {"ws": ws, "client": None}
    await ws.send_json({"type": "connected", "sid": sid})
    try:
        while True:
            data = json.loads(await ws.receive_text())
            if data["type"] == "start_session":
                client = OpenAIRealtimeClient(
                    sid,
                    on_audio=lambda a: asyncio.create_task(send_audio(sid, a)),
                    on_text=lambda t: asyncio.create_task(send_text(sid, t)),
                    on_error=lambda e: asyncio.create_task(send_err(sid, e))
                )
                sessions[sid]["client"] = client
                await client.connect()
                await ws.send_json({"type": "session_started"})
            elif data["type"] == "audio_data":
                if c := sessions[sid].get("client"):
                    await c.send_audio(data["data"])
            elif data["type"] == "stop_session":
                if c := sessions[sid].get("client"):
                    await c.commit(); await c.close()
                break
    except WebSocketDisconnect:
        pass
    finally:
        await cleanup(sid)

async def send_audio(sid, audio: bytes):
    s = sessions.get(sid); 
    if not s: return
    try: await s["ws"].send_json({"type": "audio_response", "data": audio.hex()})
    except Exception: pass

async def send_text(sid, text: str):
    s = sessions.get(sid); 
    if not s: return
    try: await s["ws"].send_json({"type": "transcript", "text": text})
    except Exception: pass

async def send_err(sid, err: str):
    s = sessions.get(sid)
    if s: 
        try: await s["ws"].send_json({"type": "error", "message": err})
        except Exception: pass

async def cleanup(sid):
    if sid in sessions:
        if c := sessions[sid].get("client"):
            await c.close()
        try: await sessions[sid]["ws"].send_json({"type": "session_closed"})
        except Exception: pass
        sessions.pop(sid, None)

@app.get("/api/health")
async def health(): return {"status": "healthy", "sessions": len(sessions)}
