from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Episode:
    """Store all relevant information of an episode."""

    prefix: str
    text: str
    prefix_token_ids: List[int]
    prefix_tokens: List[str]
    generated_token_ids: List[int]
    is_finished: bool
    reward: float
    reward_info: Dict[str, float]


@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]
    audio_paths: List[str]
    transcriptions: List[str]
    # New fields for multimodal support
    images: Optional[List[str]] = None
    videos: Optional[List[str]] = None
    use_audio_in_video: bool = False
