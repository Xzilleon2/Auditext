"""
Whisper transcription app: FastAPI backend + Streamlit UI in a single file.

Run with:
    streamlit run app.py

The FastAPI server starts in a background thread inside the Streamlit process,
so it works on Streamlit Community Cloud / Hugging Face Spaces (one process only).
You can also run the API on its own:
    uvicorn app:api --host 0.0.0.0 --port 8000
"""

import threading
import time

import requests
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import pipeline

MODEL_ID = "GrayMuff/whisper_finetune_16bit"
API_HOST = "127.0.0.1"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"

# --------------------------------------------------------------------------- #
# FastAPI backend
# --------------------------------------------------------------------------- #
api = FastAPI(title="Whisper Transcription API")

_asr = None
_asr_lock = threading.Lock()


def get_asr():
    """Load the model once, on first use."""
    global _asr
    with _asr_lock:
        if _asr is None:
            use_cuda = torch.cuda.is_available()
            _asr = pipeline(
                "automatic-speech-recognition",
                model=MODEL_ID,
                torch_dtype=torch.float16 if use_cuda else torch.float32,
                device=0 if use_cuda else -1,
                chunk_length_s=30,  # handles audio longer than 30 seconds
            )
    return _asr


@api.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "loaded": _asr is not None}


@api.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    try:
        asr = get_asr()
        result = asr(
            audio_bytes,
            generate_kwargs={"language": "english", "task": "transcribe"},
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    return {"filename": file.filename, "text": result["text"].strip()}


# --------------------------------------------------------------------------- #
# Start FastAPI in a background thread (once per Streamlit process)
# --------------------------------------------------------------------------- #
def _start_api_server():
    config = uvicorn.Config(api, host=API_HOST, port=API_PORT, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(50):  # wait up to ~5s for startup
        if server.started:
            break
        time.sleep(0.1)
    return server


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def main():
    import streamlit as st

    st.set_page_config(page_title="Audio to Text", page_icon="🎙️", layout="centered")

    @st.cache_resource
    def ensure_server():
        return _start_api_server()

    ensure_server()

    # Session state
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("transcript", "")
    st.session_state.setdefault("source_name", "")

    st.title("🎙️ Audio to Text")
    st.caption(f"Powered by `{MODEL_ID}`")

    # ---- Input ----
    audio = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg", "webm"],
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if audio is not None:
        st.audio(audio)

    col1, col2 = st.columns(2)
    confirm = col1.button(
        "Transcribe", type="primary", use_container_width=True, disabled=audio is None
    )
    clear = col2.button("Clear input", use_container_width=True)

    if clear:
        st.session_state.uploader_key += 1  # resets the uploader widget
        st.session_state.transcript = ""
        st.session_state.source_name = ""
        st.rerun()

    if confirm and audio is not None:
        with st.spinner("Transcribing... the first run also loads the model, so it may take a while."):
            try:
                resp = requests.post(
                    f"{API_URL}/transcribe",
                    files={"file": (audio.name, audio.getvalue(), audio.type)},
                    timeout=600,
                )
                resp.raise_for_status()
                st.session_state.transcript = resp.json()["text"]
                st.session_state.source_name = audio.name
            except requests.HTTPError:
                detail = resp.json().get("detail", resp.text)
                st.error(f"Server error: {detail}")
            except requests.RequestException as e:
                st.error(f"Could not reach the transcription API: {e}")

    # ---- Output ----
    st.subheader("Transcription")
    st.text_area(
        "Transcription output",
        value=st.session_state.transcript,
        height=250,
        placeholder="Your transcription will appear here...",
        label_visibility="collapsed",
    )

    base = (st.session_state.source_name.rsplit(".", 1)[0] or "transcription")
    st.download_button(
        "Download transcription (.txt)",
        data=st.session_state.transcript,
        file_name=f"{base}.txt",
        mime="text/plain",
        disabled=not st.session_state.transcript,
        use_container_width=True,
    )


try:
    from streamlit import runtime

    if runtime.exists():
        main()
except ImportError:
    pass
