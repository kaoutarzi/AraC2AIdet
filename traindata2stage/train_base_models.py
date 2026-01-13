import time
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from preprocessing import load_and_preprocess
from metrics import compute_metrics
from utils import tokenize_fn

MODELS = {
    "arabert": "aubmindlab/bert-base-arabertv02",
    "araelectra": "aubmindlab/araelectra-base-discriminator",
    "xlmr": "xlm-roberta-base",
    "deberta": "microsoft/deberta-v3-base"
}

dataset = load_and_preprocess("data/raw/allam_human_machine_dataset.csv")
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset, val_dataset = dataset["train"], dataset["test"]

results = []

for key, model_name in MODELS.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_t = train_dataset.map(lambda x: tokenize_fn(tokenizer, x), batched=True)
    val_t = val_dataset.map(lambda x: tokenize_fn(tokenizer, x), batched=True)

    train_t.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_t.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=f"results/{key}",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        fp16=True,
        report_to="none",
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_t,
        eval_dataset=val_t,
        compute_metrics=compute_metrics
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    metrics = trainer.evaluate()

    model.save_pretrained(f"saved_models/{key}")
    tokenizer.save_pretrained(f"saved_models/{key}")

    results.append({
        "model": key,
        "train_time_sec": train_time,
        **metrics
    })

pd.DataFrame(results).to_csv("results/training_times.csv", index=False)
