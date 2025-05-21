import os
import io
import streamlit as st
import speech_recognition as sr
from dotenv import load_dotenv
from mistralai import Mistral 
import yt_dlp
import ffmpeg
import tempfile

mistralai_api_key = st.secrets["MISTRALAI_API_KEY"]
client = Mistral(api_key=mistralai_api_key)

# Function to stream audio from URL
def stream_audio(video_url):
    if not video_url or not video_url.strip():
        raise ValueError("Video URL cannot be empty")

    # Create a temporary file for the audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name

    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'extract_audio': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': temp_audio_path[:-4]  # Remove .wav as yt-dlp adds it
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract audio info without downloading the video
            info = ydl.extract_info(video_url, download=True)
            
        if not os.path.exists(temp_audio_path):
            raise FileNotFoundError("Failed to extract audio stream")
            
        return temp_audio_path
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(f"Failed to extract audio: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while streaming: {str(e)}")
    finally:
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

    # Check MistalAI API key first
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
                    os.remove(audio_path)
            except Exception:
                pass  # Ignore cleanup errors

if __name__ == "__main__":
    main()