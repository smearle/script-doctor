"""
Train a small causal language model (GPT-2) on (full source, canonical diff) pairs.

Each training example is:
    <|before|>
    {full PuzzleScript source before the edit}
    <|diff|>
    {canonical diff hunks}
    <|enddiff|>

Each hunk in the canonical diff is formatted as:
    @@ -{orig_start},{n_removed} +{new_start},{n_added} @@
    -{removed line 1}
    ...
    +{added line 1}
    ...
"""

import argparse
import difflib
import glob
import json
import os

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

TRAINING_JSONL = os.path.join("puzzlescript-analysis", "training_hunks.jsonl")
OUTPUT_DIR = os.path.join("models", "diff_lm")

BOS_BEFORE = "<|before|>"
BOS_DIFF = "<|diff|>"
EOS_DIFF = "<|enddiff|>"


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _splitlines_keepends(text: str) -> list[str]:
    return _normalize_text(text).splitlines(keepends=True)


def _format_hunk_range(start: int, count: int) -> str:
    return f"{start},{count}"


def canonicalize_diff(original_text: str, updated_text: str) -> str:
    """Return canonical edit hunks with header, then '-' lines, then '+' lines."""
    original_lines = _splitlines_keepends(original_text)
    updated_lines = _splitlines_keepends(updated_text)
    matcher = difflib.SequenceMatcher(a=original_lines, b=updated_lines)

    hunks = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        removed = original_lines[i1:i2]
        added = updated_lines[j1:j2]
        header = (
            f"@@ -{_format_hunk_range(i1 + 1, len(removed))} "
            f"+{_format_hunk_range(j1 + 1, len(added))} @@"
        )
        hunk_lines = [header]
        hunk_lines.extend(f"-{line}" for line in removed)
        hunk_lines.extend(f"+{line}" for line in added)
        hunks.append("".join(hunk_lines).rstrip("\n"))

    return "\n".join(hunks)


def load_training_data(
    jsonl_path: str,
    max_source_lines: int = 200,
    max_diff_lines: int = 100,
) -> list[str]:
    """Load JSONL records with {source, updated} full-file pairs."""
    texts = []
    stats = {"ok": 0, "source_long": 0, "diff_long": 0, "empty": 0}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            if "source" not in rec or "updated" not in rec:
                raise ValueError(
                    f"Expected JSONL records with 'source' and 'updated' keys, got: {sorted(rec.keys())}"
                )

            source = _normalize_text(rec["source"])
            updated = _normalize_text(rec["updated"])
            canonical_diff = canonicalize_diff(source, updated)

            if not canonical_diff:
                stats["empty"] += 1
                continue
            if source.count("\n") > max_source_lines:
                stats["source_long"] += 1
                continue
            if canonical_diff.count("\n") > max_diff_lines:
                stats["diff_long"] += 1
                continue

            texts.append(
                f"{BOS_BEFORE}\n{source.rstrip()}\n"
                f"{BOS_DIFF}\n{canonical_diff}\n{EOS_DIFF}"
            )
            stats["ok"] += 1

    print(f"Loaded {stats['ok']} training examples from {jsonl_path}")
    print(
        f"  Skipped: {stats['empty']} empty, "
        f"{stats['source_long']} source too long, "
        f"{stats['diff_long']} diff too long"
    )
    return texts


def tokenize_fn(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a diff language model on (source, canonical diff) pairs")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Base model (default: gpt2 = 124M params)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Max token sequence length per example")
    parser.add_argument("--max_source_lines", type=int, default=200,
                        help="Skip samples with more source lines than this")
    parser.add_argument("--max_diff_lines", type=int, default=100,
                        help="Skip diffs with more hunk lines than this")
    parser.add_argument("--training_jsonl", type=str, default=TRAINING_JSONL,
                        help="Path to full-file JSONL training data")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens = {"additional_special_tokens": [BOS_BEFORE, BOS_DIFF, EOS_DIFF]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens")

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    print(f"Loading training data from {args.training_jsonl}")
    texts = load_training_data(
        args.training_jsonl,
        max_source_lines=args.max_source_lines,
        max_diff_lines=args.max_diff_lines,
    )
    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.shuffle(seed=42)

    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    n_samples = min(10, len(texts))
    for si in range(n_samples):
        with open(os.path.join(samples_dir, f"sample_{si:02d}.txt"), "w", encoding="utf-8") as f:
            f.write(dataset["text"][si])
    print(f"Saved {n_samples} training samples to {samples_dir}/")

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
        dataloader_num_workers=0,
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

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
