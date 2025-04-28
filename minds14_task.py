import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from data_types import MiniBatch
from tokenizer import Tokenizer

SYSTEM_MESSAGE = (
    "You are a speech recognition system. Your task is to transcribe the given audio into text."
)

USER_TEMPLATE = (
    "Please transcribe the following audio into text. "
    "Return the transcription in <answer> </answer> tags."
)

RESPONSE_PROMPT = "Let me transcribe this audio.\n<think>"


def reward_function(episode: Episode, target_transcription: str) -> Dict[str, float]:
    """
    Compute reward for ASR task.
    The reward consists of two components:
    1. Format reward (0.1): Correctly using <answer> tags
    2. Transcription reward (1.0): Word Error Rate (WER) based reward
    """
    # Extract the transcription from the model's response
    answer_match = re.search(r"<answer>(.*?)</answer>", episode.text, re.DOTALL)
    if not answer_match:
        return {"format_reward": 0.0, "transcription_reward": 0.0, "answer_reward": 0.0}
    
    predicted_transcription = answer_match.group(1).strip()
    
    # Format reward
    format_reward = 0.1
    
    # Compute Word Error Rate (WER) based reward
    # For simplicity, we'll use exact match for now
    # In practice, you'd want to use a proper WER calculation
    transcription_reward = 1.0 if predicted_transcription.lower() == target_transcription.lower() else 0.0
    
    # Total reward
    answer_reward = format_reward + transcription_reward
    
    return {
        "format_reward": format_reward,
        "transcription_reward": transcription_reward,
        "answer_reward": answer_reward,
    }


class MInDS14Dataset(Dataset):
    """Prepare MInDS-14 dataset for ASR training"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        language: str = "en-US",
        split: str = "train",
        test_size: int = 100,
    ):
        # Load dataset from Hugging Face
        self.dataset = load_dataset("PolyAI/minds14", language)
        self.split = split
        
        # Convert to list for easier indexing
        self.data = list(self.dataset[split])
        
        # If test split, take the last test_size examples
        if split == "test":
            self.data = self.data[-test_size:]
        else:
            self.data = self.data[:-test_size]
            
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Get audio path and transcription
        audio_path = item["path"]
        transcription = item["transcription"]
        
        # Encode the prefix with the audio path
        item.update(self.encode_prefix(audio_path, transcription))
        return item

    def encode_prefix(self, audio_path: str, transcription: str):
        """Prefix is the *actual* input to the model."""
        user_message = USER_TEMPLATE.format(audio_path=audio_path)
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
            "transcription": transcription,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        audio_paths = [item["path"] for item in batch]
        transcriptions = [item["transcription"] for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        return MiniBatch(
            audio_paths=audio_paths,
            transcriptions=transcriptions,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        ) 