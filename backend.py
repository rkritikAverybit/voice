"""
Mindful Voice API - Complete Backend with Frontend
Deploy on Render.com
"""

import asyncio
import base64
import json
import logging
import os
from datetime import datetime
from typing import Callable, Dict

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import AsyncOpenAI  # ✅ updated import for new OpenAI SDK

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================

class Config:
    """Application configuration"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
    REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
    API_TIMEOUT = 30
    VOICE = "alloy"
    INPUT_AUDIO_FORMAT = "pcm16"
    OUTPUT_AUDIO_FORMAT = "mp3"
    
    SYSTEM_PROMPT = """
# Role & Objective
You are "Mindful" — a gentle, supportive mindfulness voice companion designed to help users find calm and presence.

# Core Personality & Tone
- Warm, calm, and genuinely encouraging without being overly cheerful
- Natural speech with appropriate soft pauses for reflection
- Responses should be 2-3 sentences maximum unless guiding an exercise
- Never clinical or detached; always personal and present

# Language Guidelines
- Respond only in English unless explicitly requested otherwise
- Use simple, accessible language avoiding jargon
- Speak as if having a gentle conversation with a friend

# Mindfulness Techniques
- When appropriate, offer brief mindfulness exercises
- For breathing: "Let's breathe together... Inhale slowly... And exhale gently..."
- For grounding: Guide attention to present sensations

# Emotional Support
- Acknowledge emotions without minimizing
- Offer support through presence: "I'm here with you"
- Suggest practical steps when user seems ready

# Safety Protocol
- If user expresses severe distress: Acknowledge, suggest professional support
- For anxiety/panic: Immediately offer grounding or breathing exercise
- Never provide medical advice or diagnosis
""".strip()
    
    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("❌ OPENAI_API_KEY is required")
        return True

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== DATA MODELS ====================

class ChatMessage(BaseModel):
    content: str
    context: str = ""

class ChatResponse(BaseModel):
    success: bool
    response: str
    timestamp: str

# ==================== OPENAI REALTIME CLIENT ====================

class OpenAIRealtimeClient:
    """OpenAI Realtime API WebSocket client"""
    
    def __init__(
        self,
        session_id: str,
        on_audio_response: Callable,
        on_transcript: Callable,
        on_error: Callable
    ):
        self.session_id = session_id
        self.ws = None
        self.on_audio_response = on_audio_response
        self.on_transcript = on_transcript
        self.on_error = on_error
        self.session_ready = asyncio.Event()
        self.running = False
        self.audio_buffer = bytearray()
        
    async def connect(self):
        """Connect to OpenAI Realtime API"""
        try:
            headers = {
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            logger.info(f"🔄 Connecting to OpenAI for {self.session_id}")
            
            self.ws = await websockets.connect(
                Config.REALTIME_URL,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.running = True
            asyncio.create_task(self._receive_loop())
            
            # Configure session immediately
            await self._configure_session()
            
            logger.info(f"✅ Connected to OpenAI")
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            await self.on_error(str(e))
            raise
    
    async def _send_message(self, message: dict):
        """Send message to OpenAI"""
        if self.ws:
            await self.ws.send(json.dumps(message))
    
    async def _receive_loop(self):
        """Receive messages from OpenAI"""
        try:
            async for message in self.ws:
                event = json.loads(message)
                await self._handle_event(event)
        except Exception as e:
            logger.error(f"❌ Receive error: {e}")
            await self.on_error(str(e))
    
    async def _handle_event(self, event: dict):
        """Handle incoming events"""
        event_type = event.get("type")
        
        if event_type == "session.updated":
            self.session_ready.set()
            logger.info(f"✅ Session ready for {self.session_id}")
            
        elif event_type == "response.audio.delta":
            delta_b64 = event.get("delta", "")
            if delta_b64:
                audio_bytes = base64.b64decode(delta_b64)
                self.audio_buffer.extend(audio_bytes)
                
        elif event_type == "response.audio.done":
            if self.audio_buffer:
                await self.on_audio_response(bytes(self.audio_buffer))
                self.audio_buffer.clear()
                
        elif event_type == "conversation.item.created":
            item = event.get("item", {})
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "audio":
                        transcript = content.get("transcript", "")
                        if transcript:
                            await self.on_transcript(transcript)
        
        elif event_type == "response.done":
            response = event.get("response", {})
            for output in response.get("output", []):
                if output.get("type") == "message":
                    for content in output.get("content", []):
                        if content.get("type") == "audio":
                            transcript = content.get("transcript", "")
                            if transcript:
                                await self.on_transcript(transcript)
        
        elif event_type == "error":
            error_msg = event.get("error", {}).get("message", "Unknown error")
            logger.error(f"❌ OpenAI error: {error_msg}")
            await self.on_error(error_msg)
    
    async def _configure_session(self):
        """Configure session settings"""
        await self._send_message({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": Config.SYSTEM_PROMPT,
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 700
                }
            }
        })
        logger.info(f"⚙️ Session configured for {self.session_id}")
    
    async def send_audio(self, audio_data: str):
        """Send audio data to OpenAI"""
        if not self.session_ready.is_set():
            await self.session_ready.wait()
        
        await self._send_message({
            "type": "input_audio_buffer.append",
            "audio": audio_data
        })
    
    async def close(self):
        """Close connection"""
        self.running = False
        if self.ws:
            await self.ws.close()
            logger.info(f"🔌 OpenAI client closed for {self.session_id}")

# ==================== OPENAI SERVICE ====================


from openai import AsyncOpenAI
import httpx
import logging

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        """
        Force OpenAI to use a clean httpx.AsyncClient without 'proxies' argument
        (avoids crash on Python 3.13 + httpx>=0.28)
        """
        # ✅ Manually create a simple HTTPX client
        transport = httpx.AsyncHTTPTransport(retries=2)
        http_client = httpx.AsyncClient(transport=transport, timeout=30.0, follow_redirects=True)

        # ✅ Pass it safely into AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=Config.OPENAI_API_KEY,
            http_client=http_client
        )

    async def create_realtime_client(self, session_id: str, on_audio_response, on_transcript, on_error):
        client = OpenAIRealtimeClient(session_id, on_audio_response, on_transcript, on_error)
        await client.connect()
        return client

    async def send_text_completion(self, message: str, context: str = "") -> str:
        """
        Safe wrapper for text completions
        """
        try:
            messages = [{"role": "system", "content": Config.SYSTEM_PROMPT}]
            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})
            messages.append({"role": "user", "content": message})

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Text completion error: {e}")
            return "I'm having trouble connecting right now. Please try again later."

# ==================== CONNECTION MANAGER ====================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    async def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

# ==================== FASTAPI APPLICATION ====================

Config.validate()

app = FastAPI(title="Mindful Voice API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connection_manager = ConnectionManager()
openai_service = OpenAIService()
active_sessions: Dict[str, dict] = {}

# ==================== STATIC FILES ====================

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

# ==================== API ROUTES ====================

@app.post("/api/chat", response_model=ChatResponse)
async def text_chat(message: ChatMessage):
    try:
        logger.info(f"📩 Chat: {message.content[:50]}...")
        response = await openai_service.send_text_completion(message.content, message.context)
        return ChatResponse(success=True, response=response, timestamp=datetime.now().isoformat())
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str):
    await connection_manager.connect(websocket, session_id)
    logger.info(f"✅ Client connected: {session_id}")
    
    active_sessions[session_id] = {"websocket": websocket, "openai_client": None, "connected_at": datetime.now()}
    
    try:
        await websocket.send_json({"type": "connected", "session_id": session_id})
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            await handle_voice_message(session_id, message, websocket)
    except WebSocketDisconnect:
        logger.info(f"🔌 Disconnected: {session_id}")
        await cleanup_session(session_id)
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        await cleanup_session(session_id)

# ==================== HANDLERS ====================

async def handle_voice_message(session_id: str, message: dict, websocket: WebSocket):
    msg_type = message.get("type")
    
    if msg_type == "start_session":
        await start_openai_session(session_id, websocket)
    elif msg_type == "audio_data":
        session = active_sessions.get(session_id)
        if session and session["openai_client"]:
            await session["openai_client"].send_audio(message.get("data"))
    elif msg_type == "stop_session":
        await cleanup_session(session_id)
    elif msg_type == "ping":
        await websocket.send_json({"type": "pong"})

async def start_openai_session(session_id: str, websocket: WebSocket):
    try:
        session = active_sessions.get(session_id)
        if not session:
            return
        
        openai_client = await openai_service.create_realtime_client(
            session_id=session_id,
            on_audio_response=lambda audio: handle_ai_audio(session_id, audio),
            on_transcript=lambda text: handle_ai_transcript(session_id, text),
            on_error=lambda err: handle_ai_error(session_id, err)
        )
        
        session["openai_client"] = openai_client
        await websocket.send_json({"type": "session_started", "message": "Voice session ready"})
    except Exception as e:
        logger.error(f"❌ Failed to start: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})

async def handle_ai_audio(session_id: str, audio_data: bytes):
    session = active_sessions.get(session_id)
    if session:
        try:
            await session["websocket"].send_json({"type": "audio_response", "data": audio_data.hex()})
        except Exception as e:
            logger.error(f"❌ Error sending audio: {e}")

async def handle_ai_transcript(session_id: str, transcript: str):
    session = active_sessions.get(session_id)
    if session:
        try:
            await session["websocket"].send_json({"type": "transcript", "text": transcript})
        except Exception as e:
            logger.error(f"❌ Error sending transcript: {e}")

async def handle_ai_error(session_id: str, error: str):
    session = active_sessions.get(session_id)
    if session:
        try:
            await session["websocket"].send_json({"type": "error", "message": error})
        except Exception as e:
            logger.error(f"❌ Error sending error: {e}")



async def cleanup_session(session_id: str):
    """Cleanup session resources"""
    session = active_sessions.get(session_id)
    if not session:
        logger.info(f"⚠️ Session {session_id} already cleaned up")
        return  # ✅ Early return if already deleted
    
    # Close OpenAI client
    if session.get("openai_client"):
        try:
            await session["openai_client"].close()
            logger.info(f"🔌 OpenAI client closed for {session_id}")
        except Exception as e:
            logger.error(f"❌ Error closing OpenAI client: {e}")
    
    # Disconnect WebSocket
    try:
        await connection_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"❌ Error disconnecting WebSocket: {e}")
    
    # Remove from active sessions
    try:
        del active_sessions[session_id]
        logger.info(f"🧹 Cleaned up session {session_id}")
    except KeyError:
        logger.info(f"⚠️ Session {session_id} already removed")


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "openai_key_set": bool(Config.OPENAI_API_KEY)
    }

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Mindful Voice API Starting...")
    logger.info(f"📍 OpenAI Key: {'✅ Set' if Config.OPENAI_API_KEY else '❌ Missing'}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down...")
    for session_id in list(active_sessions.keys()):
        await cleanup_session(session_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
