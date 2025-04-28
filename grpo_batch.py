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
from qwen_omni_utils import batch_process_multimodal


@torch.no_grad()
def rollout_batch(
    model: Qwen2_5OmniForConditionalGeneration,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 8,
) -> List[Episode]:
    """Generate responses using the Qwen2.5-Omni model for audio inputs in batches."""
    end_token = tokenizer.eos_token
    
    episodes = []
    total_examples = len(batch.prefix) * num_answer_per_question
    
    # Process in smaller batches to avoid memory issues
    for batch_start in range(0, len(batch.prefix), batch_size):
        batch_end = min(batch_start + batch_size, len(batch.prefix))
        print(f"\r* Generating trajectories: {batch_start * num_answer_per_question + 1}-"
              f"{min(batch_end * num_answer_per_question, total_examples)}/{total_examples}", 
              flush=True, end="")
        
        # Create conversations for this batch
        conversations = []
        examples_indices = []
        
        for i in range(batch_start, batch_end):
            for j in range(num_answer_per_question):
                # Create conversation format
                conversation = [
                    {"role": "system", "content": [
                        {"type": "text", "text": "You are a speech recognition system. Your task is to transcribe the given audio into text."}
                    ]},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Please transcribe the following audio into text. Return the transcription in <answer> </answer> tags."},
                        {"type": "audio", "audio": batch.audio_paths[i]}
                    ]},
                ]
                conversations.append(conversation)
                examples_indices.append(i)
        
        # Process batch with multimodal batch processor
        responses = batch_process_multimodal(
            processor=tokenizer.processor,
            model=model,
            conversations=conversations,
            use_audio_in_video=batch.use_audio_in_video,
            max_new_tokens=max_gen_len,
            do_sample=True,
            return_audio=False
        )
        
        # Process responses
        for idx, (response, example_idx) in enumerate(zip(responses, examples_indices)):
            print(f"\r* Processing response: {idx+1}/{len(responses)}", flush=True, end="")
            
            prefix = batch.prefix[example_idx]
            prefix_text = tokenizer.processor.batch_decode(
                [tokenizer.processor.encode(prefix)], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Extract generated text (remove prefix)
            full_text = response
            if full_text.startswith(prefix_text):
                generated_text = full_text[len(prefix_text):]
            else:
                generated_text = full_text
            
            # Get token IDs
            prefix_token_ids = batch.prefix_token_ids[example_idx]
            generated_token_ids = tokenizer.tokenizer.encode(
                generated_text, 
                add_special_tokens=False
            )
            
            # Check if the generation is complete (has EOS token)
            is_finished = tokenizer.eos_token_id in generated_token_ids
            if is_finished:
                end_pos = generated_token_ids.index(tokenizer.eos_token_id) + 1
                generated_token_ids = generated_token_ids[:end_pos]
                generated_text = tokenizer.tokenizer.decode(
                    generated_token_ids, 
                    skip_special_tokens=True
                )
            
            # Calculate reward
            rewards = reward_function(
                response=prefix + generated_text,
                numbers=[],  # Not used for ASR
                target=batch.transcriptions[example_idx],
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


# Use the same normalize_rewards_per_group, compute_entropy, and update_policy
# functions from the original grpo.py file
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