#!/usr/bin/env python3
"""
Auto-generated LLM Fine-Tuning Script (TRL)
Generated at: 2026-03-05T21:16:37.127885
Model: mistralai/Mistral-7B-v0.3
Method: full
"""

import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ─── Configuration ───────────────────────────────────────────────────
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
DATASET_PATH = "./training_data.jsonl"
OUTPUT_DIR = "./output"
MAX_SEQ_LENGTH = 4096

# ─── Load Dataset ────────────────────────────────────────────────────
print("📥 Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# Split into train/validation
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"   Train samples: {len(train_dataset)}")

# ─── Load Model & Tokenizer ─────────────────────────────────────────
print("🔧 Loading model: mistralai/Mistral-7B-v0.3")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# ─── Training Arguments ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=0.0002,
    warmup_ratio=0.03,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    report_to="none",
    optim="adamw_torch",
)

# ─── Formatting Function ────────────────────────────────────────────
def formatting_func(example):
    """Format training examples based on template."""

    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        text = f"""### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"""
    else:
        text = f"""### Instruction:\n{instruction}\n\n### Response:\n{output}"""
    return text


# ─── Train ───────────────────────────────────────────────────────────
print("🚀 Starting training...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    formatting_func=formatting_func,
    max_seq_length=MAX_SEQ_LENGTH,
)

trainer.train()

# ─── Save ────────────────────────────────────────────────────────────
print("💾 Saving model...")
trainer.save_model(OUTPUT_DIR + "/final_model")
tokenizer.save_pretrained(OUTPUT_DIR + "/final_model")

print("✅ Training complete!")
print(f"   Model saved to: {OUTPUT_DIR}/final_model")
