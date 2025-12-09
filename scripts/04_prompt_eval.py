import os
import time
import difflib

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_label_map, plot_confusion_matrix, save_json, set_seed


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "prompt")

MAX_LENGTH = 128
SEED = 42


def build_few_shot_prompt(description: str, examples, label_names):
    header = "Disease classification based on description.\n\n"
    lines = []
    for ex in examples:
        lines.append(f"Description: {ex['text']}")
        lines.append(f"Disease: {ex['label']}")
        lines.append("")
    lines.append("Possible diseases: " + ", ".join(label_names))
    lines.append("")
    lines.append(f"Description: {description}")
    lines.append("Disease:")
    return header + "\n".join(lines)


def fuzzy_match_label(text, label_names):
    text = text.lower()
    candidates = []
    for label in label_names:
        label_low = label.lower()
        if label_low in text:
            return label
        score = difflib.SequenceMatcher(None, text, label_low).ratio()
        candidates.append((score, label))
    if not candidates:
        return label_names[0]
    candidates.sort(reverse=True)
    return candidates[0][1]


def main() -> None:
    set_seed(SEED)

    datasets = load_from_disk(DATA_DIR)
    id2label, label2id = load_label_map(DATA_DIR)
    label_names = [id2label[i] for i in sorted(id2label.keys())]

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    train_ds = datasets["train"]
    few_shot_examples = [
        {"text": train_ds[i]["text"], "label": train_ds[i]["label"]}
        for i in range(min(5, len(train_ds)))
    ]

    y_true = []
    y_pred = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    for example in datasets["test"]:
        desc = example["text"]
        true_label = example["label"]

        prompt = build_few_shot_prompt(desc, few_shot_examples, label_names)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 10,
                do_sample=False,
            )

        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :])
        predicted_label = fuzzy_match_label(generated, label_names)

        y_true.append(label2id[true_label])
        y_pred.append(label2id[predicted_label])

    total_time = time.time() - start_time

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_mem = 0.0

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )

    metrics = {
        "model_name": model_name + "_prompt",
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "train_time_sec": total_time,
        "peak_gpu_memory_mb": peak_mem,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_json(os.path.join(RESULTS_DIR, "metrics.json"), metrics)

    with open(os.path.join(RESULTS_DIR, "train_runtime.txt"), "w", encoding="utf-8") as f:
        f.write(str(total_time))

    with open(os.path.join(RESULTS_DIR, "gpu_memory.txt"), "w", encoding="utf-8") as f:
        f.write(str(peak_mem))

    labels_present = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))
    target_names = [id2label[i] for i in labels_present]
    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels_present,
        target_names=target_names,
        zero_division=0,
    )
    with open(
        os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(report)

    plot_confusion_matrix(
        y_true_arr,
        y_pred_arr,
        labels=labels_present,
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    )


if __name__ == "__main__":
    main()
