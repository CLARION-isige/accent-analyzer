# English Accent Detection Tool

A Streamlit application that analyzes audio from videos to detect and evaluate English accents.

## Overview

This tool allows users to input a video URL (YouTube, Loom, or direct MP4 links), extracts the audio, transcribes it, detects the language, and analyzes the speaker's accent. It provides:

1. Text transcription of the audio
2. Language detection with confidence score
3. English accent analysis with confidence score and detailed explanation

## Features

- **Audio Extraction**: Streams audio from various video sources without downloading the entire video
- **Speech Recognition**: Transcribes spoken content to text
- **Accent Analysis**: Uses Mistral AI model to analyze the speaker's accent and provide detailed feedback

## Requirements

- Python 3.8+
- MistralAI API key (for accent analysis)
- FFmpeg (for audio processing)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Audio_Analyzer
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv audio_env
   source audio_env/bin/activate  # On Windows: audio_env\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your MistralAI API key:
   ```
   MISTRALAI_API_KEY=your_mistralai_api_key_here
   ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Enter a video URL in the input field and wait for the analysis to complete

## How It Works

1. **Audio Extraction**: The application uses yt-dlp to extract audio from the provided video URL
2. **Transcription**: The extracted audio is transcribed using Google's Speech Recognition API
3. **Language Detection**: The transcription is analyzed using the papluca/xlm-roberta-base-language-detection model
4. **Accent Analysis**: GPT-4 analyzes the transcription to determine the likelihood of an English accent and provides a detailed explanation

## Limitations

- Audio quality affects transcription accuracy
- Short audio samples may not provide enough data for accurate accent analysis
- The tool currently only evaluates English accents

## Dependencies

- yt-dlp: For extracting audio from videos
- SpeechRecognition: For transcribing audio to text
- streamlit: For the web interface
- transformers: For language detection
- Mistral: For accent analysis using mistral-large-latest model
- python-dotenv: For environment variable management
- ffmpeg-python: For audio processing
