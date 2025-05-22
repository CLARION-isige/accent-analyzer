import os
import io
import streamlit as st
import speech_recognition as sr
from mistralai import Mistral
import yt_dlp
import tempfile
import shutil
import subprocess
import platform
import sys
import logging
import requests
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from secrets
mistralai_api_key = st.secrets.get("MISTRALAI_API_KEY")
client = Mistral(api_key=mistralai_api_key) if mistralai_api_key else None

# Audio settings
AUDIO_FORMAT = "wav"
SAMPLE_RATE = 16000  # Optimal for speech recognition
CHANNELS = 1         # Mono is better for speech recognition

def is_ffmpeg_installed():
    """Check if ffmpeg is available in the system path."""
    try:
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            # Verify ffmpeg works by checking version
            result = subprocess.run(
                [ffmpeg_path, '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return result.returncode == 0
        return False
    except Exception:
        return False

def stream_audio_to_memory(video_url):
    """
    Stream audio directly to memory without saving files locally.
    Returns audio data in WAV format.
    """
    if not video_url or not video_url.strip():
        raise ValueError("Video URL cannot be empty")

    temp_dir = tempfile.mkdtemp()
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'no_warnings': True,
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': AUDIO_FORMAT,
                'preferredquality': '192',
            }],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            if not info:
                raise RuntimeError("Could not extract video information")

            ydl.download([video_url])

            # Find the downloaded file
            audio_file = next(
                (os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith('audio.')),
                None
            )

            if not audio_file:
                raise RuntimeError("Could not find downloaded audio file")

            # Read file into memory
            with open(audio_file, 'rb') as f:
                audio_bytes = f.read()

        # Move conversion outside of the try-finally to ensure the temp directory is not yet deleted
        ext = audio_file.split('.')[-1].lower()
        if ext != 'wav':
            audio_buffer = convert_audio_format(audio_bytes, input_format=ext, output_format='wav')
        else:
            audio_buffer = io.BytesIO(audio_bytes)

        audio_buffer.seek(0)
        return audio_buffer

    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(f"Failed to extract audio: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)



def convert_audio_format(audio_data, input_format='mp4', output_format='wav'):
    """Convert audio between formats using pydub"""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format=input_format)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS)

        temp_buffer = io.BytesIO()
        audio.export(temp_buffer, format=output_format)
        temp_buffer.seek(0)  # Rewind the buffer for reading
        return temp_buffer
    except Exception as e:
        raise RuntimeError(f"Audio conversion failed: {str(e)}")



def transcribe_audio(audio_data):
    """Transcribe audio from memory using SpeechRecognition"""
    recognizer = sr.Recognizer()
    
    try:
        # Create an in-memory file-like object
        with sr.AudioFile(audio_data) as source:
            audio = recognizer.record(source)
        
        # Use Google Web Speech API
        return recognizer.recognize_google(audio)
    
    except sr.UnknownValueError:
        raise RuntimeError("Speech recognition could not understand audio")
    except sr.RequestError as e:
        raise RuntimeError(f"Could not request results from speech recognition service; {e}")
    except Exception as e:
        raise RuntimeError(f"Error during transcription: {str(e)}")

def analyze_accent(transcription):
    """Analyze accent using MistralAI"""
    if not client:
        raise ValueError("MistralAI client is not configured")
        
    if not transcription or not isinstance(transcription, str):
        raise ValueError("Transcription must be a non-empty string")

    prompt = f"""
    Analyze the following English speech transcription and provide:
    1. Classification of the accent (e.g., British, American, Australian, etc.)
    2. A confidence score (0-100%) indicating how certain you are about the accent classification
    3. Notable phonetic or linguistic features that support your classification
    4. Possible regions where this accent might be common

    Provide the response in clear markdown format with headings for each section.

    Transcription: "{transcription}"
    """
    
    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": "You are a language expert who analyzes accents and provides detailed explanations."},
                {"role": "user", "content": prompt}
            ]
        )
        
        if not response or not response.choices:
            raise ValueError("Invalid or empty response from MistralAI API")
            
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Error generating analysis: {str(e)}")

def main():
    st.set_page_config(page_title="Accent Analysis Tool", page_icon="🎤")
    st.title("🎤 English Accent Detection Tool")
    st.write("Analyze accents from audio/video URLs without downloading files locally.")
    
    # System checks
    if not is_ffmpeg_installed():
        st.warning("FFmpeg is not properly installed. Some audio processing may fail.")
    
    if not client:
        st.error("MistralAI API key is not configured. Some functionality will be limited.")

    # Input section
    with st.form("audio_input_form"):
        video_url = st.text_input(
            "Enter video/audio URL (YouTube, Loom, MP4, etc.):",
            placeholder="https://www.youtube.com/watch?v=... or https://www.loom.com/share/..."
        )
        
        submitted = st.form_submit_button("Analyze Accent")
    
    if submitted and video_url:
        try:
            # Step 1: Stream audio
            with st.status("Step 1: Extracting audio...", expanded=True) as status:
                audio_buffer = stream_audio_to_memory(video_url)
                if audio_buffer.getbuffer().nbytes == 0:
                    raise RuntimeError("No audio data was extracted")
                st.success("Audio extracted successfully")
                status.update(label="Audio extraction complete", state="complete")

            # Step 2: Transcribe audio
            with st.status("Step 2: Transcribing audio...", expanded=True) as status:
                transcription = transcribe_audio(audio_buffer)
                st.text_area("Transcription", transcription, height=150)
                st.success("Transcription completed")
                status.update(label="Transcription complete", state="complete")

            # Step 3: Analyze accent
            with st.status("Step 3: Analyzing accent...", expanded=True) as status:
                if not client:
                    st.error("MistralAI API not configured - cannot analyze accent")
                else:
                    analysis = analyze_accent(transcription)
                    st.markdown(analysis)
                    st.success("Analysis completed")
                status.update(label="Analysis complete", state="complete")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)  # Show detailed error in debug mode

    # Instructions and examples
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
        **Supported Sources:**
        - YouTube videos (any public URL)
        - Loom recordings
        - Direct links to MP4, MP3, WAV files
        - Most platforms supported by yt-dlp
        """)

    # Debug info in sidebar
    if st.sidebar.checkbox("Show debug info"):
        st.sidebar.subheader("System Information")
        st.sidebar.text(f"Python: {sys.version}")
        st.sidebar.text(f"Platform: {platform.platform()}")
        st.sidebar.text(f"FFmpeg: {'Available' if is_ffmpeg_installed() else 'Missing'}")
        st.sidebar.text(f"Temp dir: {tempfile.gettempdir()}")

if __name__ == "__main__":
    main()