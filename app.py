
# app1_streamlit_cloud_final_autoplay_fixed_loop.py
import os, io, base64, tempfile, time, urllib.request, re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import soundfile as sf

from pydub import AudioSegment
from pydub.utils import which

import nest_asyncio
nest_asyncio.apply()

try:
    import edge_tts, asyncio
except Exception:
    edge_tts = None
    asyncio = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

def ensure_ffmpeg():
    ffmpeg_dir = "/tmp/ffmpeg_bin"
    ffmpeg_bin = os.path.join(ffmpeg_dir, "ffmpeg")
    ffprobe_bin = os.path.join(ffmpeg_dir, "ffprobe")
    if which("ffmpeg") and which("ffprobe"):
        return True
    os.makedirs(ffmpeg_dir, exist_ok=True)
    base_url = "https://github.com/eugeneware/ffmpeg-static/releases/latest/download"
    try:
        if not os.path.exists(ffmpeg_bin):
            urllib.request.urlretrieve(f"{base_url}/ffmpeg-linux-x64", ffmpeg_bin)
            os.chmod(ffmpeg_bin, 0o755)
        if not os.path.exists(ffprobe_bin):
            urllib.request.urlretrieve(f"{base_url}/ffprobe-linux-x64", ffprobe_bin)
            os.chmod(ffprobe_bin, 0o755)
        os.environ["PATH"] += os.pathsep + ffmpeg_dir
        AudioSegment.converter = ffmpeg_bin
        AudioSegment.ffprobe = ffprobe_bin
        return True
    except Exception as e:
        st.warning(f"ffmpeg setup issue: {e}")
        return False

ensure_ffmpeg()

AUDIO_DIR = Path("audio_responses"); AUDIO_DIR.mkdir(exist_ok=True)
AUDIO_TTL_HOURS = 6
MAX_CONTEXT_MESSAGES = 15
MODEL_LIST = ["openai/gpt-4o-mini","meta-llama/llama-3.2-3b-instruct","meta-llama/llama-3.2-1b-instruct"]

VOICE_OPTIONS = {
    "Jenny (Female, Calm)": "en-US-JennyNeural",
    "Guy (Male, Warm)": "en-US-GuyNeural",
    "Aria (Female, Friendly)": "en-US-AriaNeural",
    "Davis (Male, Professional)": "en-US-DavisNeural",
    "Neerja (Female, Indian)": "en-IN-NeerjaNeural",
    "Prabhat (Male, Indian)": "en-IN-PrabhatNeural",
}

INTENT_PATTERNS = {
    "greeting": ["hello","hi","hey","good morning","good evening"],
    "meditation": ["meditate","meditation","breathing","breath","relax","calm"],
    "mood_check": ["how am i","feeling","mood","emotions","anxious","stressed"],
    "gratitude": ["grateful","thankful","appreciate","gratitude"],
    "sleep": ["sleep","rest","tired","insomnia","can't sleep"],
    "exercise": ["exercise","workout","movement","yoga","walk"],
    "farewell": ["goodbye","bye","see you","talk later"],
}

st.set_page_config(page_title="Mindful Wellness (Cloud)", page_icon="🪷", layout="wide")

def ensure_state():
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("user_profile", {"voice_preference":"en-US-JennyNeural","response_length":"medium","topics_discussed":[],"total_conversations":0})
    ss.setdefault("conversation_state","idle")
    ss.setdefault("audio_response_path", None)
    ss.setdefault("webrtc_frames", [])
ensure_state()

def cleanup_old_audio():
    cutoff = datetime.now() - timedelta(hours=AUDIO_TTL_HOURS)
    for f in AUDIO_DIR.glob("response_*.mp3"):
        try:
            if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                f.unlink(missing_ok=True)
        except Exception: pass

def detect_intent(txt:str)->str:
    t = txt.lower()
    for intent, pats in INTENT_PATTERNS.items():
        if any(p in t for p in pats): return intent
    return "general"

def update_conversation_stats(intent:str):
    prof = st.session_state.user_profile
    if intent not in prof["topics_discussed"]:
        prof["topics_discussed"].append(intent)
    prof["total_conversations"] += 1

def autoplay_audio_html(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        return f'''
            <audio autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        '''
    except Exception as e:
        st.warning(f"Audio playback issue: {e}")
        return ""

@st.cache_resource
def init_openai_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error("Missing OPENROUTER_API_KEY in Secrets."); st.stop()
    if OpenAI is None:
        st.error("openai package missing."); st.stop()
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
openai_client = init_openai_client()

def get_system_prompt(intent="general", user_profile=None):
    base = ("You are a warm, gentle mindfulness companion named 'Mindful'. "
            "Speak kindly. Keep responses brief and calming. No medical advice.")
    intent_map = {
        "greeting":" Respond warmly and ask how they're feeling today.",
        "meditation":" Offer a short calming breathing tip (1-2 mins).",
        "mood_check":" Acknowledge feelings and suggest one gentle next step.",
        "gratitude":" Encourage a small gratitude reflection.",
        "sleep":" Share simple wind-down advice (no medical).",
        "farewell":" Close kindly and invite them back."
    }
    extra = intent_map.get(intent,"")
    if user_profile:
        rl = user_profile.get("response_length","medium")
        lens = {"short":" Keep it 1–2 sentences.","medium":" Keep it 2–4 sentences.","long":" Use 4–6 sentences."}
        extra += lens.get(rl,"")
    return base+extra

def build_context(intent="general"):
    msgs = st.session_state.messages.copy()
    if len(msgs)>25:
        recent = msgs[-15:]; old = msgs[:-15]
        topics = [detect_intent(m["content"]) for m in old if m["role"]=="user"]
        summary = "Earlier we discussed: " + ", ".join(list(dict.fromkeys(topics))[:5])
        msgs = [{"role":"system","content":get_system_prompt(intent, st.session_state.user_profile)},
                {"role":"system","content":summary}] + recent
    else:
        msgs.insert(0, {"role":"system","content":get_system_prompt(intent, st.session_state.user_profile)})
    if len(msgs)>MAX_CONTEXT_MESSAGES:
        msgs = [msgs[0]] + msgs[-MAX_CONTEXT_MESSAGES:]
    return msgs

def get_ai_reply(user_message:str)->str:
    st.session_state.conversation_state="thinking"
    intent = detect_intent(user_message); update_conversation_stats(intent)
    st.session_state.messages.append({"role":"user","content":user_message})
    ctx = build_context(intent)
    try:
        resp = openai_client.chat.completions.create(
            model=MODEL_LIST[0], messages=ctx, max_tokens=250, temperature=0.7,
            extra_headers={"HTTP-Referer":"https://mindful-bot.streamlit.app","X-Title":"Mindful Wellness Chatbot"}
        )
        reply = resp.choices[0].message.content
    except Exception:
        reply = "I'm having trouble connecting — but take a deep breath and try again soon."
    st.session_state.messages.append({"role":"assistant","content":reply})
    return reply

def ndarray_to_int16(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[0]
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr, -1.0, 1.0); arr = (arr * 32767.0).astype(np.int16)
    elif arr.dtype != np.int16:
        arr = arr.astype(np.int16)
    return arr

def save_pcm_to_wav(pcm_frames: bytes, file_path: str, sample_rate: int = 48000):
    audio_array = np.frombuffer(pcm_frames, dtype=np.int16)
    if audio_array.size == 0: return False
    if np.max(np.abs(audio_array)) < 300:
        st.info("🕊️ The recording seems very quiet. Try speaking closer to the mic.")
    sf.write(file_path, audio_array, sample_rate, subtype='PCM_16')
    return True

import audioop
def transcribe_audio_bytes(raw_wav_bytes:bytes)->Optional[str]:
    if sr is None:
        st.error("SpeechRecognition not installed."); return None
    try:
        try:
            rms = audioop.rms(raw_wav_bytes, 2)
            if rms < 120:
                st.warning("🕊️ Too much silence detected. Please try recording again."); return None
        except Exception: pass
        r = sr.Recognizer(); r.energy_threshold = 200; r.dynamic_energy_threshold = True
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(raw_wav_bytes); tmp_path = tmp.name
        text = None
        for attempt in range(3):
            try:
                with sr.AudioFile(tmp_path) as src:
                    audio = r.record(src)
                    text = r.recognize_google(audio, language="en-US")
                break
            except sr.UnknownValueError:
                st.warning("🤔 Could not understand the audio. Try again clearly."); break
            except sr.RequestError:
                st.info(f"🌐 Retrying transcription... ({attempt+1}/3)"); time.sleep(1.0); continue
        os.unlink(tmp_path)
        return text.strip() if text else None
    except Exception as e:
        st.warning(f"⚠️ Transcription issue: {e}"); return None

def audio_upload_to_text(uploaded)->Optional[str]:
    if sr is None:
        st.error("SpeechRecognition not installed."); return None
    try:
        r = sr.Recognizer(); r.energy_threshold = 200; r.dynamic_energy_threshold = True
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.getbuffer()); tmp_path = tmp.name
        text = None
        for attempt in range(3):
            try:
                with sr.AudioFile(tmp_path) as src:
                    audio = r.record(src)
                    text = r.recognize_google(audio, language="en-US")
                break
            except sr.UnknownValueError:
                st.warning("🤔 Could not understand the audio. Try again clearly."); break
            except sr.RequestError:
                st.info(f"🌐 Retrying transcription... ({attempt+1}/3)"); time.sleep(1.0); continue
        os.unlink(tmp_path)
        return text.strip() if text else None
    except Exception as e:
        st.error(f"Couldn't process audio: {e}"); return None

def tts_sentence_to_mp3(text:str, voice:str, out_path:Path)->bool:
    try:
        if edge_tts and asyncio:
            async def _run():
                tts = edge_tts.Communicate(text, voice=voice, rate="-10%", pitch="-2Hz")
                await tts.save(str(out_path))
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            nest_asyncio.apply(loop)
            loop.run_until_complete(_run())
            return True
        elif gTTS:
            gTTS(text=text, lang="en").save(str(out_path)); return True
        else:
            return False
    except Exception as e:
        st.warning(f"TTS error: {e}"); return False

def stream_tts_response(text: str, voice: Optional[str] = None):
    voice = voice or st.session_state.user_profile.get("voice_preference","en-US-JennyNeural")
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    sentence_paths = []
    for i, s in enumerate(parts):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = AUDIO_DIR / f"part_{ts}_{i}.mp3"
        if tts_sentence_to_mp3(s, voice, out):
            sentence_paths.append(str(out))
    combined = AudioSegment.silent(duration=400)
    for p in sentence_paths:
        try:
            seg = AudioSegment.from_file(p, format="mp3")
            combined += seg + AudioSegment.silent(duration=800)
        except Exception as e:
            st.warning(f"Merge error: {e}")
    final_path = AUDIO_DIR / f"response_{int(time.time())}.mp3"
    combined.export(final_path, format="mp3")
    st.session_state.audio_response_path = str(final_path)
    for p in sentence_paths:
        try: os.remove(p)
        except: pass
    return [str(final_path)]

def record_browser_audio_ui():
    st.markdown("#### Browser Voice Recorder")
    st.caption("Click Start to record, Stop when done, then press Transcribe. If mic doesn't start, allow microphone permission in your browser.")
    try:
        ctx = webrtc_streamer(
            key="webrtc-audio",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"audio": True, "video": False},
        )
    except AttributeError:
        st.warning("⚠️ Browser mic stream stopped unexpectedly. Click Restart Recorder."); ctx = None
    except Exception as e:
        st.error(f"🎤 Mic setup issue: {e}"); ctx = None
    if (ctx is None) or (ctx and not ctx.state.playing):
        st.button("🔁 Restart Recorder", on_click=lambda: st.rerun())
    if ctx and ctx.state.playing:
        frames = ctx.audio_receiver.get_frames(timeout=1)
        for f in frames:
            arr = f.to_ndarray()
            if isinstance(arr, np.ndarray):
                if arr.ndim == 2: arr = arr[0]
                if arr.dtype in (np.float32, np.float64):
                    arr = np.clip(arr, -1.0, 1.0); arr = (arr * 32767.0).astype(np.int16)
                elif arr.dtype != np.int16:
                    arr = arr.astype(np.int16)
                st.session_state.webrtc_frames.append(arr.tobytes())
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Clear Recording Buffer", use_container_width=True):
            st.session_state.webrtc_frames = []; st.success("Cleared recorded audio.")
    with c2:
        if st.button("Transcribe Recording", use_container_width=True):
            if not st.session_state.webrtc_frames:
                st.warning("No audio recorded yet.")
            else:
                raw_pcm = b"".join(st.session_state.webrtc_frames)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    ok = save_pcm_to_wav(raw_pcm, tmp.name, sample_rate=48000)
                    tmp.seek(0); wav_bytes = tmp.read() if ok else b""
                if not wav_bytes:
                    st.warning("No audio captured. Please try again.")
                else:
                    with st.spinner("Transcribing recording..."):
                        text = transcribe_audio_bytes(wav_bytes)
                    if text:
                        st.success(f"Transcribed: {text}")
                        with st.spinner("Thinking..."): reply = get_ai_reply(text)
                        with st.spinner("Responding..."):
                            stream_tts_response(reply)
                        st.rerun()

st.markdown("<div style='text-align:center; padding: 20px;'><h1>Mindful Wellness AI Assistant</h1><p>Streamlit Cloud–ready: browser mic + file upload</p></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Settings")
    st.markdown("### Voice Selection")
    selected_voice = st.selectbox("Choose AI voice:", options=list(VOICE_OPTIONS.keys()), index=0)
    st.session_state.user_profile["voice_preference"] = VOICE_OPTIONS[selected_voice]
    st.markdown("### Response Length")
    rl = st.radio("Style:", options=["short","medium","long"], index=1, horizontal=True)
    st.session_state.user_profile["response_length"] = rl
    st.markdown("---"); st.markdown("### Your Journey")
    st.metric("Conversations", st.session_state.user_profile["total_conversations"])
    if st.session_state.user_profile["topics_discussed"]:
        st.write("**Topics explored:**")
        for t in st.session_state.user_profile["topics_discussed"][:5]: st.write(f"• {t.capitalize()}")
    st.markdown("---")
    if st.button("Reset Conversation", use_container_width=True):
        st.session_state.messages = []; st.rerun()
    st.caption("Powered by OpenRouter API")

col1, col2 = st.columns([2,1])
with col1:
    st.markdown("### Conversation")
    chat = st.container(height=420)
    with chat:
        for i, m in enumerate(st.session_state.messages):
            if m["role"]=="user":
                with st.chat_message("user", avatar="🧘"):
                    st.markdown(f"**You:** {m['content']}")
            else:
                with st.chat_message("assistant", avatar="🌿"):
                    st.markdown(m["content"])
                    if (
                        i == len(st.session_state.messages) - 1
                        and st.session_state.get("audio_response_path")
                        and os.path.exists(st.session_state.audio_response_path)
                    ):
                        html = autoplay_audio_html(st.session_state.audio_response_path)
                        if html:
                            st.markdown(html, unsafe_allow_html=True)
                            st.session_state.audio_response_path = None  # 🔄 stop loop

    if prompt := st.chat_input("Type your message..."):
        with st.spinner("Reflecting..."): reply = get_ai_reply(prompt)
        with st.spinner("Preparing voice response..."):
            stream_tts_response(reply)
        st.rerun()

with col2:
    st.markdown("### Voice Controls (Cloud-friendly)")
    record_browser_audio_ui()
    st.markdown("---"); st.markdown("**Or upload a voice note**")
    uploaded_audio = st.file_uploader("Upload audio (wav/mp3/m4a/flac)", type=["wav","mp3","m4a","flac"], label_visibility="collapsed")
    if uploaded_audio:
        with st.spinner("🎧 Transcribing your voice note..."):
            text = audio_upload_to_text(uploaded_audio)

        if text:
            st.success(f"Transcribed: {text}")

            with st.spinner("💬 Thinking..."):
                reply = get_ai_reply(text)

            with st.spinner("🎤 Responding..."):
                stream_tts_response(reply)

            # 🧘 Reset audio path AFTER rerun-trigger to stop looping playback
            current_audio = st.session_state.get("audio_response_path")
            #st.rerun()
            st.session_state.audio_response_path = None


cleanup_old_audio()

st.markdown("<hr/><div style='text-align:center; color:#777; padding: 20px;'><p>Each breath is a new beginning</p></div>", unsafe_allow_html=True)