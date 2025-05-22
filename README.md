# English Accent Detection Tool

A Streamlit application that analyzes the accent of speakers in videos.

## Prerequisites

- Python 3.8+
- FFmpeg (required for audio extraction)

## Deployment on Streamlit Cloud

When deploying to Streamlit Cloud, the `packages.txt` file will automatically install FFmpeg.

## Local Development

### FFmpeg Installation

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

#### macOS
```bash
brew install ffmpeg
```

#### Windows
Download from [FFmpeg official website](https://ffmpeg.org/download.html) and add to your PATH.

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.streamlit/secrets.toml` file with your MistralAI API key:
```toml
MISTRALAI_API_KEY = "your-api-key-here"
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Enter a video URL (YouTube, Loom, or direct MP4 link) and the app will:
1. Extract the audio
2. Transcribe the speech
3. Analyze the accent using MistralAI