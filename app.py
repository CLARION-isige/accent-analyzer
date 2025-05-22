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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from secrets
mistralai_api_key = st.secrets["MISTRALAI_API_KEY"]
client = Mistral(api_key=mistralai_api_key)

# Check if running on Streamlit Cloud
def is_streamlit_cloud():
    return os.environ.get('STREAMLIT_SHARING', '') == 'true' or os.environ.get('STREAMLIT_CLOUD', '') == 'true'

# Check if ffmpeg is installed
def is_ffmpeg_installed():
    ffmpeg_path = shutil.which('ffmpeg')
    logger.info(f"FFmpeg found at: {ffmpeg_path}")
    return ffmpeg_path is not None

# Function to stream audio from URL
def stream_audio(video_url):
    if not video_url or not video_url.strip():
        raise ValueError("Video URL cannot be empty")

    # Create a temporary directory that's guaranteed to be writable
    temp_dir = tempfile.mkdtemp()
    temp_file_base = os.path.join(temp_dir, "audio")
    temp_audio_path = f"{temp_file_base}.wav"
    
    # Configure yt-dlp options with simpler output template
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'extract_audio': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': temp_file_base
    }
    
    # Try to find ffmpeg in common locations for cloud environments
    ffmpeg_paths = [
        '/usr/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
        '/app/.heroku/python/bin/ffmpeg',
        '/app/.apt/usr/bin/ffmpeg',
        shutil.which('ffmpeg')
    ]
    
    for path in ffmpeg_paths:
        if path and os.path.exists(path):
            ydl_opts['ffmpeg_location'] = path
            st.write(f"Using FFmpeg at: {path}")  # Debug info
            break

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            
        # Find the created WAV file in the temp directory
        for file in os.listdir(temp_dir):
            if file.endswith('.wav'):
                return os.path.join(temp_dir, file)
                
        raise FileNotFoundError("Failed to extract audio stream")
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(f"Failed to extract audio: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while streaming: {str(e)}")
    finally:
        # We'll clean up the temp directory after transcription is complete
        # Cleanup any partial downloads
        partial_path = f"{temp_audio_path}.part"
        if os.path.exists(partial_path):
            os.remove(partial_path)

# Transcribe audio
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "API unavailable"
    except Exception as e:
        return f"Error during transcription: {str(e)}"

# Function to generate confidence score and summary using GPT
def generate_confidence_and_summary(transcription):
    if not mistralai_api_key:
        raise ValueError("MistralAI API key is not configured")
        
    if not transcription or not isinstance(transcription, str):
        raise ValueError("Transcription must be a non-empty string")

    prompt = f"""
    Analyze the following transcription and provide:
    1. Classification of the accent (e.g., British, American, Australian, etc.)
    2. A confidence score (0-100%) indicating how likely the speaker has an English accent.
    3. A short summary or explanation of the accent classification.

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
        
        if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
            raise ValueError("Invalid or empty response from MistralAI API")
            
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Error generating analysis: {str(e)}")

# Streamlit app
def main():
    st.title("English Accent Detection Tool")
    st.write("Upload a video URL to analyze the speaker's accent.")
    
    # Debug information
    if st.sidebar.checkbox("Show debug info"):
        st.sidebar.write("System information:")
        st.sidebar.write(f"- Platform: {platform.platform()}")
        st.sidebar.write(f"- Python: {sys.version}")
        st.sidebar.write(f"- Temp directory: {tempfile.gettempdir()}")
        st.sidebar.write(f"- FFmpeg installed: {is_ffmpeg_installed()}")
        st.sidebar.write(f"- Working directory: {os.getcwd()}")
        st.sidebar.write(f"- Directory writable: {os.access(os.getcwd(), os.W_OK)}")
        st.sidebar.write(f"- Temp dir writable: {os.access(tempfile.gettempdir(), os.W_OK)}")

    # Check MistalAI API key
    if not mistralai_api_key:
        st.error("⚠️ MistralAI API key is not configured. Please set the mistralai_api_key environment variable.")
        return

    # Input: Video URL
    video_url = st.text_input("Enter the video URL (e.g., YouTube, Loom or direct MP4 link):")

    if video_url:
        audio_path = None
        try:
            with st.spinner("Streaming audio..."):
                audio_path = stream_audio(video_url)

            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(audio_path)
                if transcription in ["Could not understand audio", "API unavailable"]:
                    st.error(f"❌ Transcription failed: {transcription}")
                    return
                st.success("✅ Transcription completed")
                st.write("Transcription:", transcription)
                
            with st.spinner("Analyzing accent..."):
                gpt_response = generate_confidence_and_summary(transcription)
                st.success("✅ Accent analysis completed")
                st.markdown("### Accent Analysis:")
                st.markdown(gpt_response)

        except ValueError as e:
            st.error(f"❌ Invalid input: {str(e)}")
        except RuntimeError as e:
            st.error(f"❌ Processing error: {str(e)}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
        finally:
            # Cleanup temporary files
            try:
                if audio_path and os.path.exists(audio_path):
                    # Remove the file
                    os.remove(audio_path)
                    # Also remove the parent directory if it's a temp dir
                    parent_dir = os.path.dirname(audio_path)
                    if os.path.exists(parent_dir) and tempfile.gettempdir() in parent_dir:
                        shutil.rmtree(parent_dir, ignore_errors=True)
            except Exception:
                pass  # Ignore cleanup errors

if __name__ == "__main__":
    main()