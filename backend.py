import asyncio, base64, json, logging, os
from datetime import datetime
from typing import Dict, List, Callable, Awaitable

import httpx, websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import AsyncOpenAI

# ──────────────────────────────────────────────────────────────────────────────
# ENV & LOG
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("mindful_plus")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime")
    REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
    VOICE = os.getenv("VOICE", "verse")
    INPUT_AUDIO_FORMAT = "pcm16"
    OUTPUT_AUDIO_FORMAT = "pcm16"
    SAMPLE_RATE = 24000

    # Cookbook-style, labeled system prompt. Keep these sections short and explicit.
    SYSTEM_PROMPT = """
# Role & Objective
You are Mindful+, a calm, supportive voice assistant for everyday wellbeing check-ins.
Your objective is to listen, reflect back clearly, and offer a small grounding nudge when helpful.

# Personality & Tone
- Friendly, warm, concise, never saccharine.
- Speak naturally like a person.
- Use simple words, avoid jargon.

# Language
- The conversation must be only in English.
- Do not switch languages even if the user does.

# Length
- 2–3 sentences per response.

# Pacing
- Deliver your audio quickly but do not sound rushed.
- If the user sounds upset, slow slightly and keep sentences shorter.

# Variety
- Do not repeat the same opener or sentence twice.
- Vary confirmations and fillers.

# Unclear Audio
- Only respond to clear audio or text.
- If audio is unintelligible, briefly say: "I didn’t catch that. Can you repeat?"

# Safety & Escalation
- Do not give medical advice.
- If user asks for urgent help or mentions self-harm, say you are limited and suggest contacting a trusted person or local emergency services.
""".strip()

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing")
        return True

# ──────────────────────────────────────────────────────────────────────────────
# TEXT CHAT MODELS
# ──────────────────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    content: str
    context: str = ""

class ChatResponse(BaseModel):
    success: bool
    response: str
    timestamp: str

MEMORY: List[Dict[str, str]] = []

class OpenAIService:
    def __init__(self):
        transport = httpx.AsyncHTTPTransport(retries=2)
        self.client = AsyncOpenAI(
            api_key=Config.OPENAI_API_KEY,
            http_client=httpx.AsyncClient(transport=transport, timeout=25.0),
        )

    async def text_reply(self, msg: str, context: str = "") -> str:
        history = MEMORY[-10:]
        messages = [{"role": "system", "content": Config.SYSTEM_PROMPT}] + history + [
            {"role": "user", "content": msg}
        ]
        if context:
            messages.insert(1, {"role": "system", "content": f"Context: {context}"})

        try:
            resp = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=400,
                temperature=0.7,
            )
            text = resp.choices[0].message.content
            MEMORY.extend([{"role": "user", "content": msg}, {"role": "assistant", "content": text}])
            #del MEMORY[:-10] if len(MEMORY) > 10 else None
            if len(MEMORY) > 10:
                del MEMORY[:-10]

            return text
        except Exception as e:
            log.error(f"Text API error: {e}")
            return "Sorry, I’m having trouble responding right now."

# ──────────────────────────────────────────────────────────────────────────────
# REALTIME CLIENT
# ──────────────────────────────────────────────────────────────────────────────
OnAudio = Callable[[bytes], Awaitable[None]]
OnText = Callable[[str], Awaitable[None]]

class RealtimeClient:
    def __init__(self, sid: str, on_audio: OnAudio, on_text_delta: OnText, on_text_done: OnText, on_error: OnText):
        self.sid = sid
        self.ws = None
        self.on_audio = on_audio
        self.on_text_delta = on_text_delta
        self.on_text_done = on_text_done
        self.on_error = on_error
        self.audio_buf = bytearray()
        self.ready = asyncio.Event()
        self.response_in_progress = False
        self.last_commit_time = 0.0
        self.commit_cooldown = 0.5

    async def connect(self):
        headers = {
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }
        try:
            self.ws = await websockets.connect(
                Config.REALTIME_URL,
                extra_headers=headers,
                ping_interval=10,
                ping_timeout=5,
                max_size=8 * 1024 * 1024,
            )
            asyncio.create_task(self._recv())

            # Initial session setup: modalities, whisper, VAD tuned per cookbook.
            await self._send({
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "instructions": Config.SYSTEM_PROMPT,
                    "voice": Config.VOICE,
                    "input_audio_format": Config.INPUT_AUDIO_FORMAT,
                    "output_audio_format": Config.OUTPUT_AUDIO_FORMAT,
                    "input_audio_transcription": {"model": "whisper-1", "language": "en"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.3,
                        "silence_duration_ms": 800,
                        "prefix_padding_ms": 300,
                        "create_response": True
                    },
                    "temperature": 0.8,
                    "max_response_output_tokens": 1024
                }
            })
            self.ready.set()
            log.info(f"✅ Realtime session ready for {self.sid}")
            return True
        except Exception as e:
            log.error(f"Realtime connection failed: {e}")
            return False

    async def _send(self, msg: dict):
        if self.ws and self.ws.open:
            await self.ws.send(json.dumps(msg))

    async def _recv(self):
        try:
            async for raw in self.ws:
                e = json.loads(raw)
                t = e.get("type")
                if t == "response.audio.delta":
                    delta_b64 = e.get("delta") or ""
                    if delta_b64:
                        self.audio_buf.extend(base64.b64decode(delta_b64))
                elif t == "response.audio.done":
                    if self.audio_buf:
                        await self.on_audio(bytes(self.audio_buf))
                        self.audio_buf.clear()
                    self.response_in_progress = False
                elif t == "response.audio_transcript.delta":
                    d = e.get("delta") or ""
                    if d:
                        await self.on_text_delta(d)
                elif t == "response.audio_transcript.done":
                    tr = e.get("transcript") or ""
                    if tr:
                        await self.on_text_done(tr)
                elif t == "response.done":
                    self.response_in_progress = False
                elif t == "error":
                    msg = (e.get("error") or {}).get("message", "Unknown error")
                    await self.on_error(msg)
                    self.response_in_progress = False
        except Exception as ex:
            log.error(f"[{self.sid}] recv error: {ex}")
            await self.on_error(str(ex))

    async def send_audio(self, audio_b64: str):
        await self.ready.wait()
        await self._send({"type": "input_audio_buffer.append", "audio": audio_b64})

    async def commit(self):
        # Guard: avoid overlapping commits and too-frequent commits
        now = asyncio.get_event_loop().time()
        if self.response_in_progress or (now - self.last_commit_time) < self.commit_cooldown:
            return
        self.response_in_progress = True
        self.last_commit_time = now
        try:
            await self._send({"type": "input_audio_buffer.commit"})
            await self._send({"type": "response.create", "response": {"modalities": ["audio", "text"]}})
        except Exception as e:
            self.response_in_progress = False
            log.error(f"Commit error: {e}")

    async def close(self):
        try:
            if self.ws:
                await self.ws.close()
        except Exception:
            pass

# ──────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ──────────────────────────────────────────────────────────────────────────────
Config.validate()

app = FastAPI(title="Mindful+ Voice (Cookbook-style)", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve /static and root page
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/api/health")
async def health():
    return {"status": "ok", "sessions": len(SESSIONS), "key": bool(Config.OPENAI_API_KEY)}

class ChatAPI:
    svc = OpenAIService()

    @staticmethod
    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(m: ChatMessage):
        try:
            text = await ChatAPI.svc.text_reply(m.content, m.context)
            return ChatResponse(success=True, response=text, timestamp=datetime.now().isoformat())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

SESSIONS: Dict[str, Dict] = {}

@app.websocket("/ws/voice/{sid}")
async def ws_voice(ws: WebSocket, sid: str):
    await ws.accept()
    SESSIONS[sid] = {"ws": ws, "client": None, "connected": datetime.now()}
    await safe_send(ws, {"type": "connected", "session_id": sid})
    log.info("connection open")

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            t = msg.get("type")

            if t == "start_session":
                if SESSIONS[sid].get("client"):
                    continue
                client = RealtimeClient(
                    sid,
                    on_audio=lambda a: asyncio.create_task(send_audio(sid, a)),
                    on_text_delta=lambda d: asyncio.create_task(send_text_delta(sid, d)),
                    on_text_done=lambda txt: asyncio.create_task(send_text_done(sid, txt)),
                    on_error=lambda e: asyncio.create_task(send_err(sid, e)),
                )
                SESSIONS[sid]["client"] = client
                if await client.connect():
                    await safe_send(ws, {"type": "session_started"})
                else:
                    await safe_send(ws, {"type": "error", "message": "Failed to connect to OpenAI"})

            elif t == "audio_data":
                data = msg.get("data", "")
                # For debugging you can log size; avoid spamming logs in prod
                # log.info(f"🎙️ chunk {len(data)}B")
                if c := SESSIONS[sid].get("client"):
                    await c.send_audio(data)

            elif t == "stop_session":
                if c := SESSIONS[sid].get("client"):
                    await c.commit()   # crucial: finalize turn so model responds
                    await c.close()
                break

            elif t == "ping":
                await safe_send(ws, {"type": "pong", "ts": datetime.now().isoformat()})

    except WebSocketDisconnect:
        log.info("client disconnected")
    except Exception as e:
        log.error(f"websocket error: {e}")
        await safe_send(ws, {"type": "error", "message": str(e)})
    finally:
        await cleanup(sid)
        log.info("connection closed")

# ──────────────────────────────────────────────────────────────────────────────
# WS helpers
# ──────────────────────────────────────────────────────────────────────────────
async def safe_send(ws: WebSocket, payload: dict):
    try:
        await ws.send_json(payload)
    except Exception:
        pass

async def send_audio(sid: str, audio: bytes):
    if s := SESSIONS.get(sid):
        await safe_send(s["ws"], {"type": "audio_response", "data": audio.hex()})

async def send_text_delta(sid: str, delta: str):
    if s := SESSIONS.get(sid):
        await safe_send(s["ws"], {"type": "transcript_delta", "delta": delta})

async def send_text_done(sid: str, text: str):
    if s := SESSIONS.get(sid):
        await safe_send(s["ws"], {"type": "transcript", "text": text})

async def send_err(sid: str, err: str):
    if s := SESSIONS.get(sid):
        await safe_send(s["ws"], {"type": "error", "message": err})

async def cleanup(sid: str):
    s = SESSIONS.pop(sid, None)
    if not s:
        return
    try:
        await safe_send(s["ws"], {"type": "session_closed"})
    except Exception:
        pass
    if c := s.get("client"):
        try:
            await c.close()
        except Exception:
            pass

# ──────────────────────────────────────────────────────────────────────────────
# DEV ENTRY
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
