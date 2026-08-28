import os
import yt_dlp
import hashlib
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIO_CACHE_DIR = Path("cache/audio")

class DownloaderService:
    @staticmethod
    def init_cache():
        AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Audio cache initialized at %s", AUDIO_CACHE_DIR)

    @staticmethod
    def is_url(path_or_url: str) -> bool:
        return path_or_url.startswith("http://") or path_or_url.startswith("https://")

    @staticmethod
    def get_audio_path(path_or_url: str) -> str:
        """Returns the local file path for the audio, from cache if available, downloading if necessary."""
        if not DownloaderService.is_url(path_or_url):
            if not os.path.exists(path_or_url):
                logger.error(f"Local file not found: {path_or_url}")
                raise FileNotFoundError(f"Local file not found: {path_or_url}")
            logger.info(f"Using local file: {path_or_url}")
            return path_or_url

        # Generate a unique hash for the URL
        url_hash = hashlib.md5(path_or_url.encode()).hexdigest()
        wav_path = str(AUDIO_CACHE_DIR / f"{url_hash}.wav")

        if os.path.exists(wav_path):
            logger.info(f"Audio cache HIT for URL: {path_or_url} -> {wav_path}")
            return wav_path
            
        logger.info(f"Audio cache MISS for URL: {path_or_url}. Downloading...")

        output_template = str(AUDIO_CACHE_DIR / f"{url_hash}.%(ext)s")
        
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
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([path_or_url])
        except Exception as e:
            logger.error(f"Download failed for {path_or_url}: {e}")
            raise

        if not os.path.exists(wav_path):
            logger.error(f"Conversion failed or file missing: {wav_path}")
            raise FileNotFoundError(f"Failed to download or convert to wav: {wav_path}")
            
        logger.info(f"Successfully downloaded and cached: {wav_path}")
        return wav_path

    @staticmethod
    def clear_cache():
        """Removes the audio cache directory."""
        if AUDIO_CACHE_DIR.exists():
            shutil.rmtree(AUDIO_CACHE_DIR)
            logger.info("Audio cache directory cleared.")
        DownloaderService.init_cache()
