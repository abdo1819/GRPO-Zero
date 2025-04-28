from transformers import AutoTokenizer, Qwen2_5OmniProcessor
from typing import Dict, List, Optional, Any
import torch

from qwen_omni_utils import process_mm_info

class Tokenizer:
    """Tokenizer using Hugging Face's AutoTokenizer or Qwen2_5OmniProcessor"""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id

    def encode_chat(self, messages: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    def encode_chat_with_response_prompt(
        self, messages: List[Dict[str, str]], prompt: str
    ) -> str:
        messages.append({"role": "assistant", "content": prompt})
        return self.encode_chat(messages)

    def tokenize(self, text: str):
        return self.tokenizer(text, return_tensors="pt")
        
    def process_multimodal(self, conversation: List[Dict[str, Any]], use_audio_in_video: bool = False):
        """Process multimodal inputs using Qwen processor"""
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
        inputs = self.processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=use_audio_in_video
        )
        return inputs, text

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
