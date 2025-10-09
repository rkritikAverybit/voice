"""
Mindful Voice API - Complete Backend
Single file implementation with all features
"""

import asyncio
import base64
import json
import logging
import os
from datetime import datetime
from typing import Callable, Dict, Optional

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================

class Config:
    """Application configuration"""
    
    # OpenAI Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
    REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
    
    # API Settings
    API_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    
    # Voice Settings
    VOICE = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    INPUT_AUDIO_FORMAT = "pcm16"
    OUTPUT_AUDIO_FORMAT = "mp3"
    
    # System Prompt
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

# Conversational Variety
- Never use the same opening phrase twice in a row
- Vary acknowledgments: "I understand" → "I hear you" → "That makes sense"
- Mix response structures to avoid predictability

# Mindfulness Techniques
- When appropriate, offer brief mindfulness exercises
- For breathing: "Let's breathe together... Inhale slowly... (pause 2s)... And exhale gently... (pause 2s)"
- For grounding: Guide attention to present sensations
- For body scan: Start from feet or head, move systematically

# Emotional Support
- Acknowledge emotions without minimizing: "That sounds challenging" not "Don't worry"
- Offer support through presence: "I'm here with you"
- Suggest practical steps when user seems ready

# Safety Protocol
- If user expresses severe distress: Acknowledge, suggest professional support
- For anxiety/panic: Immediately offer grounding or breathing exercise
- Never provide medical advice or diagnosis
""".strip()
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("❌ OPENAI_API_KEY is required in .env file")
        return True


# ==================== LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== DATA MODELS ====================

class ChatMessage(BaseModel):
    """Chat message model"""
    content: str
    context: str = ""


class ChatResponse(BaseModel):
    """Chat response model"""
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
            
            logger.info(f"🔄 Connecting to OpenAI for session {self.session_id}")
            
            self.ws = await websockets.connect(
                Config.REALTIME_URL,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.running = True
            
            # Start receiver task
            asyncio.create_task(self._receive_loop())
            
            # Initialize session
            await self._send_message({"type": "session.create"})
            
            logger.info(f"✅ Connected to OpenAI for {self.session_id}")
            
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
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"🔌 OpenAI connection closed for {self.session_id}")
            
        except Exception as e:
            logger.error(f"❌ Receive loop error: {e}")
            await self.on_error(str(e))
    
    async def _handle_event(self, event: dict):
        """Handle incoming events"""
        event_type = event.get("type")
        
        if event_type == "session.created":
            await self._configure_session()
            
        elif event_type == "session.updated":
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
                "instructions": Config.SYSTEM_PROMPT,
                "voice": Config.VOICE,
                "input_audio_format": Config.INPUT_AUDIO_FORMAT,
                "output_audio_format": Config.OUTPUT_AUDIO_FORMAT,
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 700
                },
                "input_audio_transcription": {
                    "model": "whisper-1"
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

class OpenAIService:
    """Service for OpenAI API interactions"""
    
    async def create_realtime_client(
        self,
        session_id: str,
        on_audio_response: Callable,
        on_transcript: Callable,
        on_error: Callable
    ) -> OpenAIRealtimeClient:
        """Create and connect a realtime client"""
        client = OpenAIRealtimeClient(
            session_id,
            on_audio_response,
            on_transcript,
            on_error
        )
        await client.connect()
        return client
    
    async def send_text_completion(self, message: str, context: str = "") -> str:
        """Send text completion request using OpenAI Chat API"""
        try:
            import openai
            
            client = openai.AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
            
            messages = [
                {"role": "system", "content": Config.SYSTEM_PROMPT}
            ]
            
            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})
            
            messages.append({"role": "user", "content": message})
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ Text completion error: {e}")
            return "I'm having trouble connecting right now. Please try again in a moment."


# ==================== WEBSOCKET MANAGER ====================

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept and store WebSocket connection"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"✅ Connection accepted: {session_id}")
    
    async def disconnect(self, session_id: str):
        """Remove WebSocket connection"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"🔌 Connection removed: {session_id}")


# ==================== FASTAPI APPLICATION ====================

# Validate configuration
Config.validate()

# Initialize FastAPI
app = FastAPI(
    title="Mindful Voice API",
    description="Real-time voice and chat mindfulness companion",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
connection_manager = ConnectionManager()
openai_service = OpenAIService()

# Store active sessions
active_sessions: Dict[str, dict] = {}


# ==================== API ROUTES ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🪷 Mindful Voice API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/api/chat",
            "voice": "/ws/voice/{session_id}",
            "health": "/api/health",
            "metrics": "/api/metrics"
        }
    }


@app.post("/api/chat", response_model=ChatResponse)
async def text_chat(message: ChatMessage):
    """Handle text-based chat requests"""
    try:
        logger.info(f"📩 Chat request: {message.content[:50]}...")
        
        response = await openai_service.send_text_completion(
            message.content,
            message.context
        )
        
        return ChatResponse(
            success=True,
            response=response,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str):
    """Handle WebSocket connections for voice chat"""
    await connection_manager.connect(websocket, session_id)
    logger.info(f"✅ Client connected: {session_id}")
    
    # Initialize session
    active_sessions[session_id] = {
        "websocket": websocket,
        "openai_client": None,
        "connected_at": datetime.now(),
        "messages_sent": 0,
        "messages_received": 0
    }
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Main message loop
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            active_sessions[session_id]["messages_received"] += 1
            await handle_voice_message(session_id, message, websocket)
            
    except WebSocketDisconnect:
        logger.info(f"🔌 Client disconnected: {session_id}")
        await cleanup_session(session_id)
        
    except Exception as e:
        logger.error(f"❌ WebSocket error for {session_id}: {e}")
        await cleanup_session(session_id)


async def handle_voice_message(session_id: str, message: dict, websocket: WebSocket):
    """Handle incoming voice messages"""
    msg_type = message.get("type")
    logger.info(f"📩 {msg_type} from {session_id}")
    
    if msg_type == "start_session":
        await start_openai_session(session_id, websocket)
        
    elif msg_type == "audio_data":
        session = active_sessions.get(session_id)
        if session and session["openai_client"]:
            await session["openai_client"].send_audio(message.get("data"))
        
    elif msg_type == "stop_session":
        await cleanup_session(session_id)
        
    elif msg_type == "ping":
        await websocket.send_json({
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })


async def start_openai_session(session_id: str, websocket: WebSocket):
    """Initialize OpenAI Realtime API connection"""
    try:
        session = active_sessions.get(session_id)
        if not session:
            return
        
        logger.info(f"🔄 Starting OpenAI session for {session_id}")
        
        # Create OpenAI client with callbacks
        openai_client = await openai_service.create_realtime_client(
            session_id=session_id,
            on_audio_response=lambda audio: handle_ai_audio(session_id, audio),
            on_transcript=lambda text: handle_ai_transcript(session_id, text),
            on_error=lambda err: handle_ai_error(session_id, err)
        )
        
        session["openai_client"] = openai_client
        
        await websocket.send_json({
            "type": "session_started",
            "message": "Voice session initialized successfully",
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"✅ OpenAI session started for {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Failed to start OpenAI session: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Failed to initialize: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })


async def handle_ai_audio(session_id: str, audio_data: bytes):
    """Handle audio response from OpenAI"""
    session = active_sessions.get(session_id)
    if session:
        try:
            await session["websocket"].send_json({
                "type": "audio_response",
                "data": audio_data.hex(),
                "timestamp": datetime.now().isoformat()
            })
            session["messages_sent"] += 1
            logger.info(f"🔊 Sent audio to {session_id}")
        except Exception as e:
            logger.error(f"❌ Error sending audio: {e}")


async def handle_ai_transcript(session_id: str, transcript: str):
    """Handle transcript from OpenAI"""
    session = active_sessions.get(session_id)
    if session:
        try:
            await session["websocket"].send_json({
                "type": "transcript",
                "text": transcript,
                "timestamp": datetime.now().isoformat()
            })
            logger.info(f"📝 Sent transcript to {session_id}: {transcript[:50]}...")
        except Exception as e:
            logger.error(f"❌ Error sending transcript: {e}")


async def handle_ai_error(session_id: str, error: str):
    """Handle errors from OpenAI"""
    session = active_sessions.get(session_id)
    if session:
        try:
            await session["websocket"].send_json({
                "type": "error",
                "message": error,
                "timestamp": datetime.now().isoformat()
            })
            logger.error(f"❌ OpenAI error for {session_id}: {error}")
        except Exception as e:
            logger.error(f"❌ Error sending error message: {e}")


async def cleanup_session(session_id: str):
    """Cleanup session resources"""
    session = active_sessions.get(session_id)
    if session:
        # Close OpenAI client
        if session["openai_client"]:
            try:
                await session["openai_client"].close()
            except Exception as e:
                logger.error(f"❌ Error closing OpenAI client: {e}")
        
        # Disconnect WebSocket
        await connection_manager.disconnect(session_id)
        
        # Remove from active sessions
        del active_sessions[session_id]
        logger.info(f"🧹 Cleaned up session {session_id}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "openai_key_set": bool(Config.OPENAI_API_KEY),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/metrics")
async def get_metrics():
    """Get server metrics"""
    total_messages_sent = sum(s.get("messages_sent", 0) for s in active_sessions.values())
    total_messages_received = sum(s.get("messages_received", 0) for s in active_sessions.values())
    
    return {
        "active_connections": len(active_sessions),
        "total_messages_sent": total_messages_sent,
        "total_messages_received": total_messages_received,
        "sessions": list(active_sessions.keys()),
        "timestamp": datetime.now().isoformat()
    }


# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("=" * 60)
    logger.info("🚀 Mindful Voice API Starting...")
    logger.info("=" * 60)
    logger.info(f"📍 OpenAI API Key: {'✅ Set' if Config.OPENAI_API_KEY else '❌ Missing'}")
    logger.info(f"🎤 Voice Model: {Config.REALTIME_MODEL}")
    logger.info(f"🔊 Voice: {Config.VOICE}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("🛑 Shutting down...")
    
    # Cleanup all active sessions
    for session_id in list(active_sessions.keys()):
        await cleanup_session(session_id)
    
    logger.info("👋 Shutdown complete")


# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    
    print("🪷 Mindful Voice API")
    print("=" * 60)
    
    if not Config.OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY not found in .env file")
        print("Please create a .env file with your OpenAI API key")
        exit(1)
    
    print("✅ Configuration valid")
    print("🚀 Starting server...")
    print("=" * 60)
    
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
