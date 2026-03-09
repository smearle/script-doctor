"""
Train a small causal language model (GPT-2) on (source game, targeted diff) pairs.

Each training example is:
    <|source|>
    {original PuzzleScript game text}
    <|startdiff|>
    {targeted unified diff hunks}
    <|enddiff|>

The model learns: given a game file, generate a plausible edit as a unified diff.
At inference, feed `<|source|>\\n{game}\\n<|startdiff|>\\n` and sample a diff completion.

Usage:
    python train_diff_lm.py [--model_name gpt2] [--epochs 3] [--batch_size 4] [--max_length 2048]
"""

import argparse
import glob
import os
import re

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

DIFFS_DIR = os.path.join("puzzlescript_data", "diffs")
CLEAN_DATA_DIR = os.path.join("puzzlescript_data", "clean_data")
OUTPUT_DIR = os.path.join("models", "diff_lm")

# Special tokens
BOS_SOURCE = "<|source|>"
BOS_DIFF = "<|startdiff|>"
EOS_DIFF = "<|enddiff|>"


def extract_hunks(diff_text: str) -> str:
    """Extract just the hunk headers and +/- lines from a unified diff,
    dropping the --- / +++ headers and reducing context to <=3 lines."""
    lines = diff_text.splitlines()
    hunks = []
    in_hunk = False
    context_budget = 0

    for line in lines:
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        if line.startswith("@@"):
            in_hunk = True
            hunks.append(line)
            context_budget = 3  # allow a few context lines after hunk header
            continue
        if not in_hunk:
            continue
        if line.startswith("+") or line.startswith("-"):
            hunks.append(line)
            context_budget = 3  # reset context budget after a change line
        elif line.startswith(" ") or line == "":
            if context_budget > 0:
                hunks.append(line)
                context_budget -= 1
        # else: skip

    return "\n".join(hunks)


def load_paired_data(max_source_lines: int = 300, max_diff_lines: int = 200) -> list[str]:
    """Load (source game text, targeted diff) pairs.

    For each .diff file, find the corresponding 'before' game file in clean_data/
    using the timestamp encoded in the diff filename.
    """
    diff_files = glob.glob(os.path.join(DIFFS_DIR, "**", "*.diff"), recursive=True)
    texts = []
    stats = {"empty_diff": 0, "no_source": 0, "too_long_source": 0,
             "too_long_diff": 0, "no_hunks": 0, "ok": 0}

    for diff_path in diff_files:
        # Read diff
        with open(diff_path, "r", errors="replace") as f:
            diff_content = f.read().strip()
        if not diff_content:
            stats["empty_diff"] += 1
            continue

        # Extract targeted hunks (drop full-file context)
        hunks = extract_hunks(diff_content)
        if not hunks.strip():
            stats["no_hunks"] += 1
            continue
        if hunks.count("\n") > max_diff_lines:
            stats["too_long_diff"] += 1
            continue

        # Find the 'before' source file
        # diff path: puzzlescript_data/diffs/{user}/{game}/{before}__to__{after}.diff
        # source:    puzzlescript_data/clean_data/{user}/{game}/{before}.txt
        rel_path = os.path.relpath(diff_path, DIFFS_DIR)
        parts = rel_path.rsplit(os.sep, 1)
        if len(parts) != 2:
            stats["no_source"] += 1
            continue
        user_game_dir, diff_filename = parts
        before_ts = diff_filename.split("__to__")[0]
        source_path = os.path.join(CLEAN_DATA_DIR, user_game_dir, f"{before_ts}.txt")

        if not os.path.isfile(source_path):
            stats["no_source"] += 1
            continue

        with open(source_path, "r", errors="replace") as f:
            source_text = f.read().strip()

        if source_text.count("\n") > max_source_lines:
            stats["too_long_source"] += 1
            continue

        # Build training example: source + targeted diff
        example = f"{BOS_SOURCE}\n{source_text}\n{BOS_DIFF}\n{hunks}\n{EOS_DIFF}"
        texts.append(example)
        stats["ok"] += 1

    print(f"Loaded {stats['ok']} paired examples")
    print(f"  Skipped: {stats['empty_diff']} empty diffs, {stats['no_source']} no source, "
          f"{stats['too_long_source']} source too long, {stats['too_long_diff']} diff too long, "
          f"{stats['no_hunks']} no hunks")
    return texts


def tokenize_fn(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a diff language model on (source, diff) pairs")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Base model (default: gpt2 = 124M params)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max token sequence length per example")
    parser.add_argument("--max_source_lines", type=int, default=300,
                        help="Skip games with more source lines than this")
    parser.add_argument("--max_diff_lines", type=int, default=200,
                        help="Skip diffs with more hunk lines than this")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # GPT-2 has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special delimiter tokens
    special_tokens = {"additional_special_tokens": [BOS_SOURCE, BOS_DIFF, EOS_DIFF]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens")

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # Load paired (source, diff) data
    texts = load_paired_data(
        max_source_lines=args.max_source_lines,
        max_diff_lines=args.max_diff_lines,
    )
    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.shuffle(seed=42)

    # 95/5 train/eval split
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    train_dataset = train_dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train",
    )
    eval_dataset = eval_dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing eval",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=not args.resume,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    checkpoint = None
    if args.resume:
        checkpoints = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
        if checkpoints:
            checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"Resuming from {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
