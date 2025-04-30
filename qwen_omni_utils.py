import os
import tempfile
from typing import List, Dict, Tuple, Any

import numpy as np
import requests
import soundfile as sf
from PIL import Image
from moviepy import VideoFileClip

# ---------------------------------------------------------------------------
# Public helpers (load_image, load_audio, extract_audio_from_video)
# These follow exactly the I/O conventions expected by Qwen2.5‑OmniProcessor
# – NumPy arrays for image & audio, string paths for video.
# ---------------------------------------------------------------------------
TARGET_SR = 24_000  # sample rate used in the official Qwen notebook


def download_media(url: str) -> str:
    """Download any http/https asset to a temp file and return its local path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    suffix = os.path.splitext(url)[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fp:
        for chunk in response.iter_content(8192):
            fp.write(chunk)
    return fp.name


# ----------------------- IMAGE --------------------------------------------

def load_image(path_or_url: str) -> np.ndarray:
    if path_or_url.startswith(("http://", "https://")):
        path = download_media(path_or_url)
    else:
        path = path_or_url
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


# ----------------------- AUDIO --------------------------------------------

def load_audio(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Return mono 1‑D float32 NumPy array normalised to ‑1…1."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if np.abs(audio).max() > 0:
        audio = audio / np.abs(audio).max()
    if sr != target_sr:
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * target_sr / sr))
    return audio


def extract_audio_from_video(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Extract audio track from a video – slow but reliable ffmpeg via moviepy."""
    with VideoFileClip(path) as clip:
        if clip.audio is None:
            return np.array([], dtype=np.float32)
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        clip.audio.write_audiofile(tmp_wav, fps=target_sr, verbose=False, logger=None)
    audio = load_audio(tmp_wav, target_sr)
    os.unlink(tmp_wav)
    return audio


# ---------------------- CONVERSATION PARSE --------------------------------

def process_mm_info(conversation: List[Dict[str, Any]], use_audio_in_video: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Walk over a chat message list and collect media into python‑native types."""
    audios: List[np.ndarray] = []
    images: List[np.ndarray] = []
    videos: List[str] = []

    for msg in conversation:
        for item in msg.get("content", []):
            if item["type"] == "image":
                images.append(load_image(item["image"]))

            elif item["type"] == "audio":
                audios.append(load_audio(item["audio"]))

            elif item["type"] == "video":
                videos.append(item["video"])
                if use_audio_in_video:
                    source = item["video"]
                    if source.startswith(("http://", "https://")):
                        local = download_media(source)
                        audio = extract_audio_from_video(local)
                        os.unlink(local)
                    else:
                        audio = extract_audio_from_video(source)
                    if audio.size:
                        audios.append(audio)

    return audios, images, videos
