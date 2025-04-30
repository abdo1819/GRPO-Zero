from typing import List, Dict, Any
import copy

from transformers import AutoTokenizer, Qwen2_5OmniProcessor

from qwen_omni_utils import process_mm_info


class Tokenizer:
    """Wrapper around Hugging‑Face tokenizer + Qwen Omni processor that adds a few
    conveniences (pad‑token fix, multimodal helper, safe message copy)."""

    def __init__(self, model_path: str):
        # trust_remote_code makes sure custom Qwen special tokens are registered
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

        # Some checkpoints ship without an explicit pad‑token.  We alias it to EOS
        # so downstream cross‑entropy loss never receives -100 / None.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id

    # ---------------------------------------------------------------------
    #  Chat‑style helpers
    # ---------------------------------------------------------------------

    def encode_chat(self, messages: List[Dict[str, str]]) -> str:
        """Return the **formatted** chat template (string) – *not* tokenised."""
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def encode_chat_with_response_prompt(self, messages: List[Dict[str, str]], prompt: str) -> str:
        """Same as above but appends the assistant‑side *response prompt*.
        We deep‑copy to avoid mutating caller's list in‑place."""
        msgs = copy.deepcopy(messages)
        msgs.append({"role": "assistant", "content": prompt})
        return self.encode_chat(msgs)

    # ---------------------------------------------------------------------
    #  Token ↔ ID helpers
    # ---------------------------------------------------------------------

    def tokenize(self, text: str):
        return self.tokenizer(text, return_tensors="pt")

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ---------------------------------------------------------------------
    #  Multimodal helper that mirrors the reference notebook exactly
    # ---------------------------------------------------------------------

    def process_multimodal(self, conversation: List[Dict[str, Any]], *, use_audio_in_video: bool = False):
        """Create the processor inputs **and** the prefixed text string in one call."""
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        return inputs, text

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
