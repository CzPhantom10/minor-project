import os
import time

import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from utils import load_label_map, plot_confusion_matrix, save_json, set_seed


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "lora")
ADAPTER_DIR = os.path.join(RESULTS_DIR, "adapter")

MAX_LENGTH = 128
BATCH_SIZE = 1
GRAD_ACCUM = 8
SEED = 42


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
    }


def main() -> None:
    set_seed(SEED)

    datasets = load_from_disk(DATA_DIR)
    id2label, label2id = load_label_map(DATA_DIR)

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = datasets.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text", "label"],
    )
    if "label_id" in tokenized["train"].column_names:
        tokenized = tokenized.rename_column("label_id", "labels")

    num_labels = len(id2label)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: l for i, l in id2label.items()},
        label2id={l: i for i, l in id2label.items()},
    )

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
    )
    model = get_peft_model(base_model, config)

    os.makedirs(ADAPTER_DIR, exist_ok=True)

    args = TrainingArguments(
        output_dir=RESULTS_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_mem = 0.0

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    test_output = trainer.predict(tokenized["test"])
    test_labels = test_output.label_ids
    test_preds = np.argmax(test_output.predictions, axis=-1)

    acc = accuracy_score(test_labels, test_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="macro", zero_division=0
    )

    metrics = {
        "model_name": model_name + "_lora",
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "train_time_sec": train_time,
        "peak_gpu_memory_mb": peak_mem,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_json(os.path.join(RESULTS_DIR, "metrics.json"), metrics)

    with open(os.path.join(RESULTS_DIR, "train_runtime.txt"), "w", encoding="utf-8") as f:
        f.write(str(train_time))

    with open(os.path.join(RESULTS_DIR, "gpu_memory.txt"), "w", encoding="utf-8") as f:
        f.write(str(peak_mem))

    labels_present = sorted(set(test_labels) | set(test_preds))
    target_names = [id2label[i] for i in labels_present]
    report = classification_report(
        test_labels,
        test_preds,
        labels=labels_present,
        target_names=target_names,
        zero_division=0,
    )
    with open(
        os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(report)

    plot_confusion_matrix(
        test_labels,
        test_preds,
        labels=labels_present,
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    )


if __name__ == "__main__":
    main()
