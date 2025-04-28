import os
import base64
import tempfile
from typing import List, Dict, Tuple, Optional, Any, Union
import requests
from PIL import Image
import torch
import numpy as np
import soundfile as sf
from moviepy import VideoFileClip

def download_media(url: str) -> str:
    """Download media from URL to a temporary file."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Create a temporary file with the appropriate extension
    suffix = os.path.splitext(url)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
    
    return temp_file.name

def load_image(image_path_or_url: str) -> np.ndarray:
    """Load image from path or URL."""
    if image_path_or_url.startswith(('http://', 'https://')):
        image_path = download_media(image_path_or_url)
    else:
        image_path = image_path_or_url
    
    image = Image.open(image_path).convert('RGB')
    return np.array(image)

def load_audio(audio_path: str, target_sr: int = 24000) -> np.ndarray:
    """Load audio from path with resampling if needed."""
    audio, sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    
    # Ensure audio is 1D
    audio = audio.reshape(-1)
    
    # Normalize audio
    audio = audio.astype(np.float32)
    if np.abs(audio).max() > 0:
        audio = audio / np.abs(audio).max()
    
    # Resample if needed
    if sr != target_sr:
        # For simplicity, we'll just use scipy resample in a real project
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * target_sr / sr))
    
    # Reshape to match expected dimensions (add channel dimension)
    audio = audio.reshape(1, -1)
    
    return audio

def extract_audio_from_video(video_path: str) -> np.ndarray:
    """Extract audio from video file."""
    video = VideoFileClip(video_path)
    if video.audio is not None:
        # Create temporary audio file
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # Write audio to temporary file
        video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        
        # Load audio
        audio = load_audio(temp_audio_path)
        
        # Clean up
        os.unlink(temp_audio_path)
        return audio
    return np.array([], dtype=np.float32)

def process_mm_info(conversation: List[Dict[str, Any]], use_audio_in_video: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Process multimedia information from conversation.
    
    Args:
        conversation: List of conversation messages
        use_audio_in_video: Whether to extract audio from videos
        
    Returns:
        Tuple of (audios, images, videos) as tensors
    """
    audios = []
    images = []
    videos = []
    
    for message in conversation:
        if "content" in message:
            for content in message["content"]:
                if content["type"] == "image":
                    image_path = content.get("image")
                    if image_path:
                        img = load_image(image_path)
                        images.append(torch.tensor(img).cpu())
                
                elif content["type"] == "audio":
                    audio_path = content.get("audio")
                    if audio_path:
                        audio = load_audio(audio_path)
                        audios.append(torch.tensor(audio).cpu())
                
                elif content["type"] == "video":
                    video_path = content.get("video")
                    if video_path:
                        # For video, we just store the path for now
                        # In a real implementation, you would process the video frames
                        videos.append(video_path)
                        
                        # Extract audio from video if requested
                        if use_audio_in_video:
                            if video_path.startswith(('http://', 'https://')):
                                local_path = download_media(video_path)
                                audio = extract_audio_from_video(local_path)
                                os.unlink(local_path)  # Clean up
                            else:
                                audio = extract_audio_from_video(video_path)
                            
                            if len(audio) > 0:
                                audios.append(torch.tensor(audio).cpu())
    
    # Ensure all tensors are on CPU before returning
    audios = [tensor.cpu() if tensor.device.type != 'cpu' else tensor for tensor in audios]
    images = [tensor.cpu() if tensor.device.type != 'cpu' else tensor for tensor in images]
    
    return audios, images, videos