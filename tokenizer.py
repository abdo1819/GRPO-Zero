from transformers import AutoTokenizer
from typing import Dict, List

class Tokenizer:
    """Tokenizer using Hugging Face's AutoTokenizer"""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
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

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
