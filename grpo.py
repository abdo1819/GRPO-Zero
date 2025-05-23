import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List, Dict, Any, Optional

import numpy as np
import torch
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from data_types import Episode, MiniBatch
from tokenizer import Tokenizer
from qwen_omni_utils import process_mm_info


@torch.no_grad()
def rollout(
    model: Qwen2_5OmniForConditionalGeneration,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Episode]:
    """Generate responses using the Qwen2.5-Omni model for audio inputs."""
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # Prepare batches - we'll process each example separately for clarity
    episodes = []
    
    for i, prefix in enumerate(batch.prefix):
        # Create conversation format for each example
        conversation = [
            {"role": "system", "content": [
                {"type": "text", "text": "You are a speech recognition system. Your task is to transcribe the given audio into text."}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "Please transcribe the following audio into text. Return the transcription in <answer> </answer> tags."},
                {"type": "audio", "audio": batch.audio_paths[i]}
            ]},
        ]
        
        # Process each example num_answer_per_question times
        for j in range(num_answer_per_question):
            print(f"\r* Generating trajectory: {i*num_answer_per_question+j+1}/{len(batch.prefix)*num_answer_per_question}", 
                  flush=True, end="")
            
            # Convert conversation to model inputs
            text = tokenizer.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # Process multimedia inputs
            audios, images, videos = process_mm_info(
                conversation, 
                use_audio_in_video=batch.use_audio_in_video
            )
            
            # Create model inputs
            inputs = tokenizer.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=batch.use_audio_in_video,
                device=device,
                
            )
            
            # Move inputs to the correct device
            # inputs = {k: v.to(device).to(dtype) if isinstance(v, torch.Tensor) else v 
            #          for k, v in inputs.items()}
            device = next(model.parameters()).device        # first parameter’s device → cuda:0
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate completion with audio support
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_gen_len,
                do_sample=True,
                use_audio_in_video=batch.use_audio_in_video,
                # return_dict_in_generate=True,
            )
        
            # Extract text and audio from outputs
            text_ids = outputs.sequences
            audio = outputs.audio_output
            
            # Decode the generated IDs
            generated_text = tokenizer.processor.batch_decode(
                text_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Subtract the prompt to get only the generated part
            prefix_text = tokenizer.processor.batch_decode(
                inputs["input_ids"], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Get only the generated part
            if generated_text.startswith(prefix_text):
                generated_text = generated_text[len(prefix_text):]
            
            # Get the generated token IDs
            prefix_token_ids = batch.prefix_token_ids[i]
            # For simplicity, we'll just use tokenizer to tokenize the generated part
            generated_token_ids = tokenizer.tokenizer.encode(
                generated_text, 
                add_special_tokens=False
            )
            
            # Check if the generation is complete (has the EOS token)
            is_finished = end_token_id in generated_token_ids
            if is_finished:
                # Truncate at EOS token
                end_pos = generated_token_ids.index(end_token_id) + 1
                generated_token_ids = generated_token_ids[:end_pos]
                generated_text = tokenizer.tokenizer.decode(
                    generated_token_ids, 
                    skip_special_tokens=True
                )
            
            # Calculate reward
            rewards = reward_function(
                response=prefix + generated_text,
                numbers=[],  # Not used for ASR
                target=batch.transcriptions[i],
                end_token=end_token,
            )
            
            # Create episode
            episode = Episode(
                prefix=prefix,
                text=prefix + generated_text,
                prefix_token_ids=prefix_token_ids,
                prefix_tokens=[tokenizer.tokenizer.decode([tid]) for tid in prefix_token_ids],
                generated_token_ids=generated_token_ids,
                is_finished=is_finished,
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            
            episodes.append(episode)
    
    # Clean up
    gc.collect()
    torch.cuda.empty_cache()
    
    # Clear the output line
    print("\r", end=" " * 100, flush=True)
    return episodes


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """Normalize rewards per group. A group is defined by the prefix."""
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)
    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


def update_policy(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """Update the policy using the GRPO algorithm."""
    episodes = normalize_rewards_per_group(episodes)
    # sort episodes by token length for efficient (micro-)batching
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy = 0.0

    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            
            # Create inputs dict for Qwen2.5-Omni
            model_inputs = {
                "input_ids": input_token_ids,
                "attention_mask": torch.ones_like(input_token_ids),
            }
            
            outputs = model(**model_inputs)
            logits = outputs.logits.float()

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy = entropy + (token_entropy * target_masks).sum() / num_target_tokens

        obj = log_probs * batch_advantages[:, None]
        # per-token objective
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()

    # update the policy
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }
