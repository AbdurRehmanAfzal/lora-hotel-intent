"""LoRA fine-tune: hotel guest intent classification (25 intents).

Base model : Qwen/Qwen2.5-0.5B-Instruct
Dataset    : bitext/Bitext-hospitality-llm-chatbot-training-dataset (25k rows, CDLA-Sharing-1.0)
Method     : LoRA (PEFT) on attention + MLP projections, loss only on the answer tokens
Metric     : exact-match intent accuracy on a held-out test split, BEFORE vs AFTER fine-tuning

Why this task: a hotel voice agent (like the Mojeeb hotel pilot) must route each guest request
(check-in, late checkout, parking, refund...) to the right tool. A small fine-tuned model does this
fast and cheaply, without sending every turn to a large hosted LLM.

Run on Google Colab (Runtime -> Change runtime type -> T4 GPU):
    !pip uninstall -y -q torchao   # Colab's old torchao breaks new peft; not used here
    !pip install -q "transformers>=4.56" peft datasets accelerate
    !python lora_hotel_intent.py
Quick CPU smoke test (a few steps, tiny eval, just checks the code runs):
    SMOKE=1 python lora_hotel_intent.py
"""
import json
import os
import random
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)

SMOKE = os.getenv("SMOKE") == "1"
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DATASET_ID = "bitext/Bitext-hospitality-llm-chatbot-training-dataset"
OUT_DIR = os.getenv("OUT_DIR", "qwen2.5-0.5b-hotel-intent-lora")
SEED = 42

N_TRAIN = 40 if SMOKE else 4000      # training examples (balanced across intents)
N_TEST = 8 if SMOKE else 500         # held-out examples for accuracy
MAX_STEPS = 3 if SMOKE else -1       # -1 = use EPOCHS
EPOCHS = 2

# LoRA hyper-parameters (be ready to explain these in an interview)
LORA_R = 16          # rank of the low-rank update matrices
LORA_ALPHA = 32      # scaling = alpha / r = 2
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device} smoke={SMOKE}")

# ----------------------------------------------------------------------------- data
ds = load_dataset(DATASET_ID, split="train").shuffle(seed=SEED)
INTENTS = sorted(set(ds["intent"]))
print(f"{len(ds)} rows, {len(INTENTS)} intents")

split = ds.train_test_split(test_size=0.1, seed=SEED)
test_rows = split["test"].select(range(N_TEST))

# Balanced training sample: the same number of examples per intent.
per_intent = max(1, N_TRAIN // len(INTENTS))
by_intent = {}
for row in split["train"]:
    bucket = by_intent.setdefault(row["intent"], [])
    if len(bucket) < per_intent:
        bucket.append(row)
train_rows = [r for rows in by_intent.values() for r in rows]
random.shuffle(train_rows)
print(f"train={len(train_rows)} test={len(test_rows)}")

SYSTEM = ("You classify hotel guest messages. Reply with exactly one intent label from this list and "
          "nothing else:\n" + ", ".join(INTENTS))

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"  # left padding for batched generation
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def prompt_text(message):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode(row):
    """Tokenise prompt + answer; mask the prompt with -100 so loss is only on the answer."""
    p_ids = tok(prompt_text(row["instruction"]), add_special_tokens=False)["input_ids"]
    a_ids = tok(row["intent"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    return {"input_ids": p_ids + a_ids, "attention_mask": [1] * (len(p_ids) + len(a_ids)),
            "labels": [-100] * len(p_ids) + a_ids}


train_data = [encode(r) for r in train_rows]

# ----------------------------------------------------------------------------- model
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
    torch.float16 if device == "cuda" else torch.float32)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)


@torch.no_grad()
def evaluate(m, rows, label, batch_size=16):
    m.eval()
    correct, preds = 0, []
    t0 = time.time()
    for i in range(0, len(rows), batch_size):
        batch = rows.select(range(i, min(i + batch_size, len(rows))))
        enc = tok([prompt_text(x) for x in batch["instruction"]], return_tensors="pt", padding=True).to(device)
        out = m.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
        for gold, seq in zip(batch["intent"], out[:, enc["input_ids"].shape[1]:]):
            pred = tok.decode(seq, skip_special_tokens=True).strip().split()[0] if seq.numel() else ""
            pred = pred.strip(".,'\"")
            correct += pred == gold
            preds.append({"gold": gold, "pred": pred})
    acc = correct / len(rows)
    print(f"[{label}] accuracy = {acc:.1%} ({correct}/{len(rows)}) in {time.time() - t0:.0f}s")
    return acc, preds


base_acc, _ = evaluate(model, test_rows, "BASE (zero-shot)")

# ----------------------------------------------------------------------------- LoRA
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                         target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
model.print_trainable_parameters()

BATCH, GRAD_ACC = (1, 1) if SMOKE else (16, 2)
total_steps = MAX_STEPS if MAX_STEPS > 0 else EPOCHS * -(-len(train_data) // (BATCH * GRAD_ACC))
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=2e-4,
    num_train_epochs=EPOCHS,
    max_steps=MAX_STEPS,
    lr_scheduler_type="cosine",
    warmup_steps=max(1, int(0.05 * total_steps)),  # 5% warm-up (int works in transformers 4.x and 5.x)
    logging_steps=1 if SMOKE else 20,
    save_strategy="no",
    bf16=dtype == torch.bfloat16,
    fp16=dtype == torch.float16,
    report_to="none",
    seed=SEED,
)
tok.padding_side = "right"  # right padding for training
trainer = Trainer(model=model, args=args, train_dataset=train_data,
                  data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
t0 = time.time()
train_result = trainer.train()
train_minutes = (time.time() - t0) / 60
tok.padding_side = "left"

model.config.use_cache = True
lora_acc, lora_preds = evaluate(model, test_rows, "LoRA fine-tuned")

# ----------------------------------------------------------------------------- save
model.save_pretrained(OUT_DIR)       # saves only the LoRA adapter (a few MB), not the full model
tok.save_pretrained(OUT_DIR)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
results = {
    "base_model": MODEL_ID, "dataset": DATASET_ID, "intents": len(INTENTS),
    "train_examples": len(train_rows), "test_examples": len(test_rows),
    "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "target_modules": TARGET_MODULES},
    "trainable_params": trainable, "total_params": total, "trainable_pct": round(100 * trainable / total, 3),
    "epochs": EPOCHS if MAX_STEPS < 0 else None, "steps": train_result.global_step,
    "final_train_loss": round(train_result.training_loss, 4), "train_minutes": round(train_minutes, 1),
    "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    "accuracy_base": round(base_acc, 4), "accuracy_lora": round(lora_acc, 4),
    "smoke_test": SMOKE,
}
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
errors = [p for p in lora_preds if p["gold"] != p["pred"]][:30]
with open(os.path.join(OUT_DIR, "errors_sample.json"), "w") as f:
    json.dump(errors, f, indent=2)
print(json.dumps(results, indent=2))
print(f"\nAccuracy: {base_acc:.1%} (base) -> {lora_acc:.1%} (LoRA). Adapter saved to {OUT_DIR}/")
