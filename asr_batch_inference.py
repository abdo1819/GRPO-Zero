import torch
import os
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import batch_process_multimodal

def batch_asr_inference(
    model_path: str,
    audio_paths: list,
    batch_size: int = 4,
    device: str = "cuda",
    dtype: str = "bfloat16"
):
    """
    Perform batch ASR inference on a list of audio files.
    
    Args:
        model_path: Path to the Qwen2.5-Omni model
        audio_paths: List of paths to audio files
        batch_size: Maximum number of audios to process in a single batch
        device: Device to run inference on ("cuda" or "cpu")
        dtype: Data type for inference ("bfloat16", "float16", or "float32")
        
    Returns:
        List of transcriptions for each audio file
    """
    # Initialize model and processor
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype_map.get(dtype, torch.bfloat16),
        device_map=device
    )
    
    # Create ASR conversations for each audio file
    conversations = []
    for audio_path in audio_paths:
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
            
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a speech recognition system. Your task is to transcribe the given audio into text."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please transcribe the following audio into text. Return the transcription in <answer> </answer> tags."},
                    {"type": "audio", "audio": audio_path},
                ]
            }
        ]
        conversations.append(conversation)
    
    # Process in batches to avoid OOM
    all_transcriptions = []
    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(conversations) + batch_size - 1)//batch_size}")
        
        try:
            responses = batch_process_multimodal(
                processor=processor,
                model=model,
                conversations=batch,
                use_audio_in_video=False,
                max_new_tokens=256,
                do_sample=False,  # For ASR, we want deterministic results
                return_audio=False
            )
            
            # Extract the transcriptions from the responses
            for response in responses:
                # Extract text between <answer> and </answer> tags
                import re
                match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if match:
                    transcription = match.group(1).strip()
                else:
                    transcription = "Failed to extract transcription"
                all_transcriptions.append(transcription)
        
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            # Add empty transcriptions for this batch
            all_transcriptions.extend(["Error: " + str(e)] * len(batch))
    
    return all_transcriptions

# Example usage
if __name__ == "__main__":
    model_path = "PATH_TO_YOUR_MODEL"  # Replace with actual model path
    
    # List of audio files to transcribe
    audio_paths = [
        "/path/to/audio1.wav",
        "/path/to/audio2.wav",
        "/path/to/audio3.wav",
        # Add more paths as needed
    ]
    
    # Run batch ASR inference
    transcriptions = batch_asr_inference(
        model_path=model_path,
        audio_paths=audio_paths,
        batch_size=4
    )
    
    # Print transcriptions
    for i, transcription in enumerate(transcriptions):
        print(f"\n=== Transcription for {os.path.basename(audio_paths[i])} ===")
        print(transcription) 