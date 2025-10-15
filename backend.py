"""
Mindful+ v2 — FastAPI backend
- Serves static index
- /api/chat (non-stream) and /api/chat_stream (SSE-like chunked streaming over fetch)
- /ws/voice/{sid} bridging to OpenAI Realtime with ["audio","text"]
- Accepts 'persona' and 'emotion_hint' to steer session instructions (no DB; local memory only per process)
"""

import os, json, base64, asyncio, logging
from datetime import datetime
from typing import Dict, List

import httpx, websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import AsyncOpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("mindful_plus_v2")

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
    REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
    VOICE = os.getenv("VOICE", "alloy")
    SYSTEM_PROMPT_BASE = (
        "You are Mindful+, a calm, supportive voice companion. "
        "Speak warmly in simple English. Keep replies short (2–3 sentences). "
        "Offer gentle grounding or breathing when useful. Acknowledge feelings without judgment. "
        "Do not give medical advice."
    )

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing")
        return True

# lightweight in-process memory
MEMORY: List[Dict[str,str]] = []   # last 10 turns total

class ChatMessage(BaseModel):
    content: str
    context: str = ""

class ChatResponse(BaseModel):
    success: bool
    response: str
    timestamp: str

# OpenAI client
transport = httpx.AsyncHTTPTransport(retries=2)
http_client = httpx.AsyncClient(transport=transport, timeout=60.0, follow_redirects=True)
ocli = AsyncOpenAI(api_key=Config.OPENAI_API_KEY, http_client=http_client)

app = FastAPI(title="Mindful+ v2", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/api/health")
async def health():
    return {"status":"ok","key":bool(Config.OPENAI_API_KEY)}

def build_persona(context: str) -> str:
    # parse context like: [persona:calm] additional text...
    persona = "calm"
    if "[persona:" in context:
        try:
            tag = context.split("[persona:",1)[1]
            persona = tag.split("]",1)[0].strip()
        except:
            pass
    tone = {
        "calm": "Warm, slow, supportive. Keep 2–3 short sentences. Gentle grounding when useful.",
        "friendly": "Upbeat, friendly, concise. Encourage lightly. Keep it simple and kind.",
        "mentor": "Clear, practical, respectful. Offer one actionable tip, succinctly."
    }.get(persona, "Warm and concise.")
    return tone

def system_prompt_with(context: str) -> str:
    return Config.SYSTEM_PROMPT_BASE + " " + build_persona(context)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(msg: ChatMessage):
    # non-stream fallback
    try:
        messages = [{"role":"system","content": system_prompt_with(msg.context)}]
        history = MEMORY[-10:]
        messages += history
        messages.append({"role":"user","content": msg.content})
        resp = await ocli.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=400,
            temperature=0.7,
        )
        text = resp.choices[0].message.content
        MEMORY.append({"role":"user","content":msg.content})
        MEMORY.append({"role":"assistant","content":text})
        if len(MEMORY)>10: del MEMORY[:-10]
        return ChatResponse(success=True, response=text, timestamp=datetime.now().isoformat())
    except Exception as e:
        log.error(f"/api/chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat_stream")
async def chat_stream(req: Request):
    body = await req.json()
    content = body.get("content","")
    context = body.get("context","")
    messages = [{"role":"system","content": system_prompt_with(context)}] + MEMORY[-10:] + [{"role":"user","content":content}]

    async def gen():
        try:
            stream = await ocli.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=400,
                stream=True
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
            # update memory at end (we need the full text; rebuild quickly)
            # Simpler path: run one more non-stream call for final text, or accumulate client-side.
            # We'll trust client accumulation; just mark done:
            MEMORY.append({"role":"user","content":content})
            if len(MEMORY)>10: del MEMORY[:-10]
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

# -------- Realtime bridge --------
SESSIONS: Dict[str, Dict] = {}

class RTClient:
    def __init__(self, sid: str):
        self.sid = sid
        self.ws = None
        self.audio_buf = bytearray()
        self.persona = "calm"

    async def connect(self):
        headers = {"Authorization": f"Bearer {Config.OPENAI_API_KEY}", "OpenAI-Beta": "realtime=v1"}
        self.ws = await websockets.connect(Config.REALTIME_URL, extra_headers=headers, ping_interval=20, ping_timeout=10)
        await self.send({
            "type":"session.update",
            "session":{
                "modalities":["audio","text"],
                "instructions": Config.SYSTEM_PROMPT_BASE + " " + build_persona(f"[persona:{self.persona}]"),
                "voice": Config.VOICE,
                "input_audio_format":"pcm16",
                "output_audio_format":"pcm16",
                "input_audio_transcription":{"model":"whisper-1"},
                "turn_detection":{"type":"server_vad","silence_duration_ms":650,"prefix_padding_ms":250,"create_response":True},
                "temperature":0.7,
                "max_response_output_tokens":2048
            }
        })
        asyncio.create_task(self.recv_loop())

    async def send(self, obj: dict):
        if self.ws:
            await self.ws.send(json.dumps(obj))

    async def recv_loop(self):
        try:
            async for raw in self.ws:
                e = json.loads(raw)
                t = e.get("type")
                if t == "response.audio.delta":
                    delta_b64 = e.get("delta","")
                    if delta_b64: self.audio_buf.extend(base64.b64decode(delta_b64))
                elif t == "response.audio.done":
                    if self.audio_buf:
                        await send_audio(self.sid, bytes(self.audio_buf))
                        self.audio_buf.clear()
                elif t == "response.audio_transcript.delta":
                    d = e.get("delta") or ""
                    if d: await send_text_delta(self.sid, d)
                elif t == "response.audio_transcript.done":
                    tr = e.get("transcript") or ""
                    if tr: await send_text_done(self.sid, tr)
                elif t == "error":
                    err = e.get("error", {})
                    await send_err(self.sid, f"{err.get('code','')}: {err.get('message','Unknown error')}")
        except Exception as ex:
            await send_err(self.sid, str(ex))

@app.websocket("/ws/voice/{sid}")
async def ws_voice(ws: WebSocket, sid: str):
    await ws.accept()
    SESSIONS[sid] = {"ws": ws, "rt": None}
    await safe_send(ws, {"type":"connected","session_id":sid})
    try:
      while True:
        data = await ws.receive_text()
        msg = json.loads(data)
        t = msg.get("type")
        if t == "start_session":
            rt = RTClient(sid)
            SESSIONS[sid]["rt"] = rt
            await rt.connect()
            await safe_send(ws, {"type":"session_started"})
        elif t == "persona":
            p = (msg.get("persona") or "calm").lower()
            if SESSIONS[sid].get("rt"):
                SESSIONS[sid]["rt"].persona = p
                await SESSIONS[sid]["rt"].send({"type":"session.update","session":{"instructions": Config.SYSTEM_PROMPT_BASE + " " + build_persona(f"[persona:{p}]")}})
        elif t == "emotion_hint":
            mood = (msg.get("data",{}).get("mood") or "calm")
            if SESSIONS[sid].get("rt"):
                extra = f" User currently sounds {mood}."
                await SESSIONS[sid]["rt"].send({"type":"session.update","session":{"instructions": Config.SYSTEM_PROMPT_BASE + " " + build_persona(f"[persona:{SESSIONS[sid]['rt'].persona}]") + extra}})
        elif t == "audio_data":
            if SESSIONS[sid].get("rt"):
                await SESSIONS[sid]["rt"].send({"type":"input_audio_buffer.append","audio": msg.get("data","")})
        elif t == "stop_session":
            if SESSIONS[sid].get("rt"):
                await SESSIONS[sid]["rt"].send({"type":"input_audio_buffer.commit"})
                await SESSIONS[sid]["rt"].send({"type":"response.create","response":{"modalities":["audio","text"]}})
            break
        elif t == "ping":
            await safe_send(ws, {"type":"pong","ts": datetime.utcnow().isoformat()})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await safe_send(ws, {"type":"error","message":str(e)})
    finally:
        await cleanup(sid)

async def safe_send(ws: WebSocket, payload: dict):
    try: await ws.send_json(payload)
    except Exception: pass

async def send_audio(sid: str, audio: bytes):
    s = SESSIONS.get(sid)
    if not s: return
    await safe_send(s["ws"], {"type":"audio_response","data": audio.hex()})

async def send_text_delta(sid: str, delta: str):
    s = SESSIONS.get(sid)
    if not s: return
    await safe_send(s["ws"], {"type":"transcript_delta","delta": delta})

async def send_text_done(sid: str, text: str):
    s = SESSIONS.get(sid)
    if not s: return
    await safe_send(s["ws"], {"type":"transcript","text": text})

async def send_err(sid: str, err: str):
    s = SESSIONS.get(sid)
    if not s: return
    await safe_send(s["ws"], {"type":"error","message": err})

async def cleanup(sid: str):
    s = SESSIONS.pop(sid, None)
    if not s: return
    try: await safe_send(s["ws"], {"type":"session_closed"})
    except Exception: pass
    rt = s.get("rt")
    if rt and rt.ws:
        try: await rt.ws.close()
        except Exception: pass

# Local dev run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
