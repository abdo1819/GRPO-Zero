import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import batch_process_multimodal

# Initialize model and processor
model_path = "PATH_TO_YOUR_MODEL"  # Replace with your model path
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Sample conversations for batch inference

# Conversation with video only
conversation1 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "/path/to/video.mp4"},
        ]
    }
]

# Conversation with audio only
conversation2 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "/path/to/audio.wav"},
        ]
    }
]

# Conversation with pure text
conversation3 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Who are you?"}
        ]
    }
]

# Conversation with mixed media
conversation4 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/path/to/image.jpg"},
            {"type": "video", "video": "/path/to/video.mp4"},
            {"type": "audio", "audio": "/path/to/audio.wav"},
            {"type": "text", "text": "What are the elements can you see and hear in these medias?"},
        ],
    }
]

# ASR conversation
conversation5 = [
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
            {"type": "audio", "audio": "/path/to/transcribe.wav"},
        ]
    }
]

# Combine messages for batch processing
conversations = [conversation1, conversation2, conversation3, conversation4, conversation5]

# Process all conversations in batch mode
responses = batch_process_multimodal(
    processor=processor,
    model=model,
    conversations=conversations,
    use_audio_in_video=True,
    max_new_tokens=512,
    do_sample=True,
    return_audio=False
)

# Print responses
for i, response in enumerate(responses):
    print(f"\n=== Response {i+1} ===")
    print(response)

# Example with audio response
print("\n=== With Audio Response ===")
text_responses, audio_responses = batch_process_multimodal(
    processor=processor,
    model=model,
    conversations=[conversation3],  # Just using the text conversation for example
    use_audio_in_video=True,
    max_new_tokens=512,
    do_sample=True,
    return_audio=True
)

print(text_responses[0])
print(f"Audio response shape: {audio_responses.shape}") 