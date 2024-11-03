import yt_dlp
import whisper
import os
from pathlib import Path
import logging
import concurrent.futures
from datetime import datetime, timedelta
import tempfile
import time
import torch
import re

class YouTubeWhisperBatch:
    def __init__(self, whisper_model="base", max_workers=1):
        # Existing initialization code remains the same
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'youtube_transcripts')
        self.output_dir = Path(desktop_path)
        self.transcript_dir = self.output_dir / "transcripts"
        
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.whisper_model = whisper.load_model(whisper_model, device=self.device)
        
        self.max_workers = max_workers
        
        # Create directories
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'processing.log'),
                logging.StreamHandler()
            ]
        )
        
        # Initialize timing statistics
        self.total_duration = 0
        self.processed_duration = 0
        self.start_time = None
        
        print(f"📂 Transcripts will be saved to: {self.transcript_dir}")
        print(f"💻 Using device: {self.device}")

    def format_time(self, seconds):
        """Convert seconds to human-readable time format"""
        return str(timedelta(seconds=int(seconds)))

    def estimate_remaining_time(self):
        """Estimate remaining processing time based on current progress"""
        if self.processed_duration == 0:
            return "Calculating..."
        
        elapsed_time = time.time() - self.start_time
        rate = self.processed_duration / elapsed_time
        remaining_duration = self.total_duration - self.processed_duration
        remaining_seconds = remaining_duration / rate if rate > 0 else 0
        
        return self.format_time(remaining_seconds)

    def clean_filename(self, title):
        """Convert title to valid filename."""
        clean_title = re.sub(r'[<>:"/\\|?*]', '', title)
        clean_title = clean_title.replace(' ', '_')
        clean_title = re.sub(r'_+', '_', clean_title)
        clean_title = clean_title.strip('_')
        if len(clean_title) > 255:
            clean_title = clean_title[:255]
        return clean_title

    def process_video(self, video_info):
        """Process a single video: download audio, transcribe, and cleanup"""
        video_url = f"https://www.youtube.com/watch?v={video_info['id']}"
        video_duration = video_info.get('duration', 0)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                    }],
                    'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    video_id = info['id']
                    video_title = info['title']
                    
                    clean_title = self.clean_filename(video_title)
                    transcript_path = self.transcript_dir / f"{clean_title}.txt"
                    
                    if transcript_path.exists():
                        logging.info(f"⏭️  Skipping (already transcribed): {video_title}")
                        self.processed_duration += video_duration
                        self.update_progress()
                        return transcript_path

                    logging.info(f"🎯 Processing: {video_title}")
                    
                    audio_path = os.path.join(temp_dir, f"{video_id}.mp3")
                    
                    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                        raise Exception("Audio file not downloaded correctly")

                    time.sleep(1)
                    
                    result = self.whisper_model.transcribe(
                        audio_path,
                        fp16=False,
                        language='en'
                    )
                    
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        f.write(f"Title: {video_title}\n")
                        f.write(f"URL: {video_url}\n")
                        f.write(f"Duration: {self.format_time(video_duration)}\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(result['text'])
                    
                    self.processed_duration += video_duration
                    self.update_progress()
                    logging.info(f"✅ Transcribed: {video_title}")
                    return transcript_path
                
        except Exception as e:
            logging.error(f"❌ Error processing {video_url}: {str(e)}")
            return None

    def update_progress(self):
        """Update and display progress information"""
        if self.total_duration > 0:
            progress = (self.processed_duration / self.total_duration) * 100
            elapsed = self.format_time(time.time() - self.start_time)
            remaining = self.estimate_remaining_time()
            
            print(f"\rProgress: {progress:.1f}% | Elapsed: {elapsed} | Remaining: {remaining}", end="")

    def process_channel(self, channel_url):
        """Process all videos from a YouTube channel."""
        ydl_opts = {
            'extract_flat': True,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist'
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logging.info(f"🔍 Fetching channel information...")
                channel_info = ydl.extract_info(channel_url, download=False)
                if not channel_info:
                    logging.error(f"❌ Could not fetch channel info for {channel_url}")
                    return []
                
                entries = []
                if 'entries' in channel_info:
                    for playlist in channel_info['entries']:
                        if 'entries' in playlist:
                            entries.extend(playlist['entries'])
                
                valid_videos = [video for video in entries if video and video.get('id')]
                
                # Calculate total duration
                self.total_duration = sum(video.get('duration', 0) for video in valid_videos)
                estimated_time = self.format_time(self.total_duration * 1.5)  # Factor in processing overhead
                
                logging.info(f"📊 Found {len(valid_videos)} videos to process")
                logging.info(f"⏱️  Total video duration: {self.format_time(self.total_duration)}")
                logging.info(f"⏳ Estimated processing time: {estimated_time}")
                
                if not valid_videos:
                    logging.error("❌ No valid videos found in channel")
                    return []
                
                # Start timing
                self.start_time = time.time()
                
                # Process videos sequentially
                transcript_paths = []
                for video in valid_videos:
                    result = self.process_video(video)
                    if result:
                        transcript_paths.append(result)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Final statistics
                total_time = time.time() - self.start_time
                average_speed = self.total_duration / total_time if total_time > 0 else 0
                
                print("\n\n=== Processing Complete ===")
                print(f"Total videos processed: {len(transcript_paths)}")
                print(f"Total time taken: {self.format_time(total_time)}")
                print(f"Processing speed: {average_speed:.2f}x realtime")
                
                return transcript_paths
                
        except Exception as e:
            logging.error(f"❌ Error processing channel {channel_url}: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    processor = YouTubeWhisperBatch(
        whisper_model="base",
        max_workers=1
    )
    
    channel_url = "https://www.youtube.com/@danmartell"
    transcripts = processor.process_channel(channel_url)