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
        # But this is a placeholder - you would implement proper resampling here
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * target_sr / sr))
    
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

def batch_process_multimodal(
    processor,
    model,
    conversations: List[List[Dict[str, Any]]],
    use_audio_in_video: bool = False,
    max_new_tokens: int = 512,
    do_sample: bool = True,
    return_audio: bool = False
) -> List[str]:
    """
    Process multiple conversations with various media types in batch mode.
    
    Args:
        processor: Qwen2_5OmniProcessor instance
        model: Qwen2_5OmniForConditionalGeneration instance
        conversations: List of conversations, each containing messages
        use_audio_in_video: Whether to extract audio from videos
        max_new_tokens: Maximum number of tokens to generate
        do_sample: Whether to use sampling in generation
        return_audio: Whether to return audio output
        
    Returns:
        List of generated responses
    """
    # Prepare text inputs
    texts = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    
    # Process all multimedia information
    all_audios = []
    all_images = []
    all_videos = []
    
    for conversation in conversations:
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
        all_audios.extend(audios)
        all_images.extend(images)
        all_videos.extend(videos)
    
    # Create model inputs
    inputs = processor(
        text=texts,
        audio=all_audios,
        images=all_images,
        videos=all_videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video
    )
    
    # Move inputs to model device and dtype
    inputs = {k: v.to(model.device).to(model.dtype) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    
    # Generate responses
    with torch.no_grad():
        if return_audio:
            output_ids, audio_output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_audio_in_video=use_audio_in_video,
                return_dict_in_generate=False,
                return_audio=True
            )
            responses = processor.batch_decode(
                output_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            return responses, audio_output
        else:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_audio_in_video=use_audio_in_video,
                return_dict_in_generate=False,
                return_audio=False
            )
            responses = processor.batch_decode(
                output_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            return responses