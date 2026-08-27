import os
import yt_dlp
import uuid
import shutil
from pathlib import Path

TEMP_DIR = Path("/tmp/automix_dsp_backend")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

class DownloaderService:
    @staticmethod
    def is_url(path_or_url: str) -> bool:
        return path_or_url.startswith("http://") or path_or_url.startswith("https://")

    @staticmethod
    def get_audio_path(path_or_url: str) -> str:
        """Returns the local file path for the audio, downloading if necessary."""
        if not DownloaderService.is_url(path_or_url):
            if not os.path.exists(path_or_url):
                raise FileNotFoundError(f"Local file not found: {path_or_url}")
            return path_or_url

        # It's a URL, download it
        video_id = str(uuid.uuid4())
        output_template = str(TEMP_DIR / f"{video_id}.%(ext)s")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([path_or_url])
            
        # The postprocessor changes the extension to .wav
        wav_path = str(TEMP_DIR / f"{video_id}.wav")
        if not os.path.exists(wav_path):
             raise FileNotFoundError(f"Failed to download or convert to wav: {wav_path}")
        return wav_path

    @staticmethod
    def cleanup():
        """Removes the temp directory to clean up space."""
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
