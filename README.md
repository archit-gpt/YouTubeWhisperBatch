# YouTube Whisper Batch

`YouTubeWhisperBatch` is a Python tool that automates the transcription of YouTube videos from a channel. It utilizes the [Whisper](https://github.com/openai/whisper) model for transcription and [yt-dlp](https://github.com/yt-dlp/yt-dlp) for downloading audio. The tool can batch process multiple videos from a YouTube channel, saving transcripts in a specified directory with logging and progress tracking.

## Features

- **Batch Download and Transcription**: Downloads audio from multiple YouTube videos and transcribes them using the Whisper model.
- **CUDA Support**: Automatically detects and uses CUDA if available for faster processing.
- **Logging and Progress Tracking**: Logs processing information, calculates estimated completion time, and shows progress in real time.
- **Organized Transcripts**: Saves transcripts in a dedicated directory with detailed information, including title, URL, and duration of each video.

## Requirements

- Python 3.8+
- [Whisper](https://github.com/openai/whisper) (for transcription)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) (for YouTube audio extraction)
- PyTorch (with CUDA support for GPU acceleration, optional)

### Python Libraries
Install the necessary libraries with:
```bash
pip install yt-dlp whisper torch
