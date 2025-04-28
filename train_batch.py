import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from pathlib import Path

from transformers import Qwen2_5OmniForConditionalGeneration
from minds14_task import MInDS14Dataset, reward_function
from grpo_batch import rollout_batch, update_policy
from optimizer import MemoryEfficientAdamW
from tokenizer import Tokenizer

def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])
    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{current_time}")
    tokenizer = Tokenizer(str(pretrained_model_path))

    # Load model
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        pretrained_model_path,
        torch_dtype=dtype,
        device_map="auto"
    ).train()

    train_dataset = MInDS14Dataset(
        tokenizer=tokenizer,
        language=config["data"]["language"],
        split="train",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=MInDS14Dataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )

    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        enabled=config["training"]["memory_efficient_adamw"],
    )

    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Starting training with batched rollout...")
    for step, batch in enumerate(train_dataloader, start=1):
        print(f"Step {step}: Processing batch...")
        
        # Use batched rollout function
        episodes = rollout_batch(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
            batch_size=config.get("training", {}).get("batch_rollout_size", 4)
        )
        
        if config["training"]["skip_unfinished_episodes"]:
            episodes = [episode for episode in episodes if episode.is_finished]
        
        results = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=config["training"]["micro_batch_size"],
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=config["training"]["max_grad_norm"],
            device=device,
            dtype=dtype,
        )
        
        # Compute and log metrics
        reward = [episode.reward for episode in episodes]
        formatted_reward = [episode.reward_info["format_reward"] for episode in episodes]
        answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
        num_finished_episodes = sum(episode.is_finished for episode in episodes)
        
        mean_reward = np.mean(reward)
        success_rate = np.mean(answer_reward)
        format_reward = np.mean(formatted_reward)
        
        print(
            f"Step {step}, mean_reward: {mean_reward:.2f}, "
            f"train success_rate: {success_rate:.2f}, "
            f"grad_norm: {results['grad_norm']:.2f}, "
            f"num_finished_episodes: {num_finished_episodes}, "
            f"entropy: {results['entropy']:.2f}"
        )
        
        # Save checkpoint
        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            torch.save(model.state_dict(), output_file)
            print(f"Saved checkpoint to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config) 