"""
Fine-Tune Config Module
=======================
Generates ready-to-run LoRA/SFT training configurations and scripts
for LLM fine-tuning via HuggingFace TRL or Unsloth.
"""

import os
import json
from typing import Dict, Any, Optional
from datetime import datetime


class FineTuneConfig:
    """
    Generates LoRA/SFT training configurations and runnable training scripts.

    Does NOT import or require transformers/peft/trl at config generation time.
    The generated script handles all imports when actually executed on GPU.

    Example:
    --------
    >>> config = FineTuneConfig(
    ...     model_name="meta-llama/Llama-3-8B",
    ...     method="lora"
    ... )
    >>> config.set_lora_params(r=16, alpha=32)
    >>> config.set_training_params(epochs=3, learning_rate=2e-4)
    >>> config.export("./training_output")
    """

    # Presets for popular models
    MODEL_PRESETS = {
        "llama-3-8b": {
            "model_name": "meta-llama/Meta-Llama-3-8B",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "max_seq_length": 2048,
        },
        "llama-3-70b": {
            "model_name": "meta-llama/Meta-Llama-3-70B",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "max_seq_length": 2048,
        },
        "mistral-7b": {
            "model_name": "mistralai/Mistral-7B-v0.3",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "max_seq_length": 4096,
        },
        "phi-3-mini": {
            "model_name": "microsoft/Phi-3-mini-4k-instruct",
            "target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
            "max_seq_length": 4096,
        },
        "gemma-7b": {
            "model_name": "google/gemma-7b",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "max_seq_length": 2048,
        },
    }

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        method: str = "lora",
        backend: str = "trl",
    ):
        """
        Initialize the fine-tune config generator.

        Parameters
        ----------
        model_name : str
            HuggingFace model name or path.
        method : str
            Fine-tuning method: 'lora', 'qlora', or 'full'.
        backend : str
            Training backend: 'trl' or 'unsloth'.
        """
        if method not in ("lora", "qlora", "full"):
            raise ValueError(f"Invalid method '{method}'. Choose: lora, qlora, full")
        if backend not in ("trl", "unsloth"):
            raise ValueError(f"Invalid backend '{backend}'. Choose: trl, unsloth")

        self.model_name = model_name
        self.method = method
        self.backend = backend

        # Apply preset if available
        preset = self._find_preset(model_name)

        # LoRA parameters
        self.lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": preset.get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            if preset
            else ["q_proj", "k_proj", "v_proj", "o_proj"],
        }

        # Training parameters
        self.training_config = {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_steps": 100,
            "save_total_limit": 3,
            "fp16": False,
            "bf16": True,
            "max_grad_norm": 0.3,
            "max_seq_length": preset.get("max_seq_length", 2048) if preset else 2048,
            "output_dir": "./output",
            "report_to": "none",
        }

        # QLoRA-specific
        if method == "qlora":
            self.quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            }
        else:
            self.quantization_config = {}

        # Dataset config
        self.dataset_config = {
            "dataset_path": "./data.jsonl",
            "text_field": "text",
            "template": "alpaca",
            "max_samples": None,
            "validation_split": 0.1,
        }

    def _find_preset(self, model_name: str) -> Optional[Dict]:
        """Find a matching model preset."""
        name_lower = model_name.lower()
        for key, preset in self.MODEL_PRESETS.items():
            if key in name_lower or preset["model_name"].lower() == name_lower:
                return preset
        return None

    def set_lora_params(self, **kwargs):
        """Update LoRA parameters."""
        for key, value in kwargs.items():
            if key == "r":
                self.lora_config["r"] = value
            elif key == "alpha":
                self.lora_config["lora_alpha"] = value
            elif key == "dropout":
                self.lora_config["lora_dropout"] = value
            elif key == "target_modules":
                self.lora_config["target_modules"] = value
            elif key in self.lora_config:
                self.lora_config[key] = value

    def auto_configure_lora(self, model_size_b: float) -> None:
        """
        Auto-configure LoRA hyperparameters based on model size (in Billions of params).
        Heuristics:
        - < 3B: r=8, alpha=16
        - 3B - 10B: r=16, alpha=32
        - > 10B: r=32, alpha=64
        """
        if model_size_b < 3.0:
            self.set_lora_params(r=8, alpha=16)
        elif model_size_b <= 10.0:
            self.set_lora_params(r=16, alpha=32)
        else:
            self.set_lora_params(r=32, alpha=64)

    def set_training_params(self, **kwargs):
        """Update training parameters."""
        for key, value in kwargs.items():
            if key == "epochs":
                self.training_config["num_train_epochs"] = value
            elif key == "learning_rate" or key == "lr":
                self.training_config["learning_rate"] = value
            elif key == "batch_size":
                self.training_config["per_device_train_batch_size"] = value
            elif key in self.training_config:
                self.training_config[key] = value

    def set_dataset_params(self, **kwargs):
        """Update dataset configuration."""
        for key, value in kwargs.items():
            if key in self.dataset_config:
                self.dataset_config[key] = value

    def get_full_config(self) -> Dict[str, Any]:
        """Return the complete configuration as a dictionary."""
        config = {
            "model": {
                "model_name": self.model_name,
                "method": self.method,
                "backend": self.backend,
            },
            "lora": self.lora_config if self.method in ("lora", "qlora") else {},
            "quantization": self.quantization_config if self.method == "qlora" else {},
            "training": self.training_config,
            "dataset": self.dataset_config,
            "generated_at": datetime.now().isoformat(),
        }
        return config

    def export(
        self,
        output_dir: str,
        generate_script: bool = True,
        dataset_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Export training configuration and optionally a training script.

        Parameters
        ----------
        output_dir : str
            Directory to save config and script files.
        generate_script : bool
            If True, also generates a runnable train.py script.
        dataset_path : str, optional
            Override dataset path in config.

        Returns
        -------
        dict
            Paths to generated files.
        """
        os.makedirs(output_dir, exist_ok=True)

        if dataset_path:
            self.dataset_config["dataset_path"] = dataset_path

        # Save config JSON
        config = self.get_full_config()
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        result = {"config": config_path}

        # Generate training script
        if generate_script:
            script_path = os.path.join(output_dir, "train.py")
            script = self._generate_training_script(config)
            with open(script_path, "w") as f:
                f.write(script)
            result["script"] = script_path

        # Generate requirements
        req_path = os.path.join(output_dir, "requirements_training.txt")
        reqs = self._generate_requirements()
        with open(req_path, "w") as f:
            f.write(reqs)
        result["requirements"] = req_path

        print(f"✅ Training config exported to: {output_dir}")
        for key, path in result.items():
            print(f"   • {key}: {os.path.basename(path)}")

        return result

    def _generate_training_script(self, config: Dict) -> str:
        """Generate a runnable Python training script."""
        if self.backend == "unsloth":
            return self._generate_unsloth_script(config)
        else:
            return self._generate_trl_script(config)

    def _generate_trl_script(self, config: Dict) -> str:
        """Generate a TRL-based training script."""
        lora = config.get("lora", {})
        training = config["training"]
        dataset = config["dataset"]

        script = f'''#!/usr/bin/env python3
"""
Auto-generated LLM Fine-Tuning Script (TRL)
Generated at: {config["generated_at"]}
Model: {self.model_name}
Method: {self.method}
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
MODEL_NAME = "{self.model_name}"
DATASET_PATH = "{dataset["dataset_path"]}"
OUTPUT_DIR = "{training["output_dir"]}"
MAX_SEQ_LENGTH = {training["max_seq_length"]}

# ─── Load Dataset ────────────────────────────────────────────────────
print("📥 Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
'''

        if dataset.get("max_samples"):
            script += f"""
# Limit samples
dataset = dataset.select(range(min({dataset["max_samples"]}, len(dataset))))
"""

        if dataset.get("validation_split", 0) > 0:
            script += f"""
# Split into train/validation
dataset = dataset.train_test_split(test_size={dataset["validation_split"]})
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
"""
        else:
            script += """
train_dataset = dataset
eval_dataset = None
"""

        script += f"""
print(f"   Train samples: {{len(train_dataset)}}")

# ─── Load Model & Tokenizer ─────────────────────────────────────────
print("🔧 Loading model: {self.model_name}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
"""

        if self.method == "qlora":
            script += """
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)
"""
        else:
            script += """
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
"""

        if self.method in ("lora", "qlora"):
            script += f'''
# ─── LoRA Configuration ─────────────────────────────────────────────
lora_config = LoraConfig(
    r={lora.get("r", 16)},
    lora_alpha={lora.get("lora_alpha", 32)},
    lora_dropout={lora.get("lora_dropout", 0.05)},
    bias="{lora.get("bias", "none")}",
    task_type="{lora.get("task_type", "CAUSAL_LM")}",
    target_modules={lora.get("target_modules", [])},
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
'''

        script += f'''
# ─── Training Arguments ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs={training["num_train_epochs"]},
    per_device_train_batch_size={training["per_device_train_batch_size"]},
    gradient_accumulation_steps={training["gradient_accumulation_steps"]},
    learning_rate={training["learning_rate"]},
    warmup_ratio={training["warmup_ratio"]},
    weight_decay={training["weight_decay"]},
    lr_scheduler_type="{training["lr_scheduler_type"]}",
    logging_steps={training["logging_steps"]},
    save_steps={training["save_steps"]},
    save_total_limit={training["save_total_limit"]},
    fp16={training["fp16"]},
    bf16={training["bf16"]},
    max_grad_norm={training["max_grad_norm"]},
    report_to="{training["report_to"]}",
    optim="adamw_torch",
)

# ─── Formatting Function ────────────────────────────────────────────
def formatting_func(example):
    """Format training examples based on template."""
'''

        if dataset.get("template") == "chatml":
            script += """
    messages = example.get("messages", [])
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text += f"<|im_start|>{role}\\n{content}<|im_end|>\\n"
    return text

"""
        elif dataset.get("template") == "sharegpt":
            script += """
    conversations = example.get("conversations", [])
    text = ""
    for turn in conversations:
        role = "Human" if turn["from"] == "human" else "Assistant"
        text += f"{role}: {turn['value']}\\n\\n"
    return text

"""
        else:  # alpaca
            script += '''
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        text = f"""### Instruction:\\n{instruction}\\n\\n### Input:\\n{input_text}\\n\\n### Response:\\n{output}"""
    else:
        text = f"""### Instruction:\\n{instruction}\\n\\n### Response:\\n{output}"""
    return text

'''

        script += """
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
"""
        return script

    def _generate_unsloth_script(self, config: Dict) -> str:
        """Generate an Unsloth-based training script."""
        lora = config.get("lora", {})
        training = config["training"]
        dataset = config["dataset"]

        script = f'''#!/usr/bin/env python3
"""
Auto-generated LLM Fine-Tuning Script (Unsloth)
Generated at: {config["generated_at"]}
Model: {self.model_name}
Method: {self.method}
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ─── Configuration ───────────────────────────────────────────────────
MODEL_NAME = "{self.model_name}"
DATASET_PATH = "{dataset["dataset_path"]}"
OUTPUT_DIR = "{training["output_dir"]}"
MAX_SEQ_LENGTH = {training["max_seq_length"]}

# ─── Load Model (Unsloth optimized) ─────────────────────────────────
print("🔧 Loading model with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect
    load_in_4bit={str(self.method == "qlora")},
)

# ─── Apply LoRA ──────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r={lora.get("r", 16)},
    lora_alpha={lora.get("lora_alpha", 32)},
    lora_dropout={lora.get("lora_dropout", 0.05)},
    target_modules={lora.get("target_modules", [])},
    bias="{lora.get("bias", "none")}",
)

# ─── Load Dataset ────────────────────────────────────────────────────
print("📥 Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
'''

        if dataset.get("validation_split", 0) > 0:
            script += f"""
dataset = dataset.train_test_split(test_size={dataset["validation_split"]})
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
"""
        else:
            script += """
train_dataset = dataset
eval_dataset = None
"""

        script += f'''
# ─── Formatting Function ────────────────────────────────────────────
def formatting_func(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    if input_text:
        return f"### Instruction:\\n{{instruction}}\\n\\n### Input:\\n{{input_text}}\\n\\n### Response:\\n{{output}}"
    return f"### Instruction:\\n{{instruction}}\\n\\n### Response:\\n{{output}}"

# ─── Train ───────────────────────────────────────────────────────────
print("🚀 Starting training with Unsloth...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs={training["num_train_epochs"]},
        per_device_train_batch_size={training["per_device_train_batch_size"]},
        gradient_accumulation_steps={training["gradient_accumulation_steps"]},
        learning_rate={training["learning_rate"]},
        warmup_ratio={training["warmup_ratio"]},
        weight_decay={training["weight_decay"]},
        lr_scheduler_type="{training["lr_scheduler_type"]}",
        logging_steps={training["logging_steps"]},
        save_steps={training["save_steps"]},
        fp16={training["fp16"]},
        bf16={training["bf16"]},
        optim="adamw_8bit",
        report_to="{training["report_to"]}",
    ),
    formatting_func=formatting_func,
    max_seq_length=MAX_SEQ_LENGTH,
)

trainer.train()

# ─── Save ────────────────────────────────────────────────────────────
print("💾 Saving model...")
model.save_pretrained(OUTPUT_DIR + "/final_model")
tokenizer.save_pretrained(OUTPUT_DIR + "/final_model")

# Optional: Save to GGUF for llama.cpp
# model.save_pretrained_gguf(OUTPUT_DIR + "/gguf", tokenizer, quantization_method="q4_k_m")

print("✅ Training complete!")
'''
        return script

    def _generate_requirements(self) -> str:
        """Generate requirements file for training environment."""
        reqs = [
            "torch>=2.0.0",
            "transformers>=4.40.0",
            "datasets>=2.18.0",
            "accelerate>=0.28.0",
            "peft>=0.10.0",
            "trl>=0.8.0",
        ]

        if self.method == "qlora":
            reqs.append("bitsandbytes>=0.43.0")

        if self.backend == "unsloth":
            reqs.append("unsloth>=2024.3")

        reqs.extend(
            [
                "wandb  # optional: for experiment tracking",
                "tensorboard  # optional: for local logging",
            ]
        )

        return "\n".join(reqs) + "\n"

    def print_summary(self) -> None:
        """Print a formatted config summary."""
        print("=" * 60)
        print("FINE-TUNING CONFIGURATION")
        print("=" * 60)
        print(f"\n🤖 Model: {self.model_name}")
        print(f"🔧 Method: {self.method}")
        print(f"⚙️  Backend: {self.backend}")

        if self.method in ("lora", "qlora"):
            print("\n📐 LoRA Config:")
            print(f"   • Rank (r): {self.lora_config['r']}")
            print(f"   • Alpha: {self.lora_config['lora_alpha']}")
            print(f"   • Dropout: {self.lora_config['lora_dropout']}")
            print(f"   • Target: {self.lora_config['target_modules']}")

        t = self.training_config
        print("\n🏋️  Training Config:")
        print(f"   • Epochs: {t['num_train_epochs']}")
        print(f"   • Batch size: {t['per_device_train_batch_size']}")
        print(f"   • Learning rate: {t['learning_rate']}")
        print(f"   • Max sequence length: {t['max_seq_length']}")

        print("=" * 60)
