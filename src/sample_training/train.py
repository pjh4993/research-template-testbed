"""Minimal SFT training script for Tier 2 script-runner E2E validation.

Trains a tiny model for a few steps to verify the full pipeline:
git fetch → uv install → python -m → wandb logging.
"""

import argparse

import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def make_dummy_dataset(size: int = 20) -> Dataset:
    """Create a tiny dataset for smoke testing."""
    samples = [
        {"text": f"Question: What is {i} + {i}? Answer: {i + i}"}
        for i in range(size)
    ]
    return Dataset.from_list(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample SFT training")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--wandb-project", default="research-template-testbed")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project or "research-template-testbed",
        entity=args.wandb_entity or None,
        name=args.run_id or None,
    )

    print(f"Loading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = make_dummy_dataset()
    print(f"Dataset size: {len(dataset)}")

    training_args = SFTConfig(
        output_dir="/tmp/sft-output",
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        logging_steps=1,
        save_strategy="no",
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"Training for {args.max_steps} steps...")
    trainer.train()
    print("Training complete.")

    wandb.finish()


if __name__ == "__main__":
    main()
