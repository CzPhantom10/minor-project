import json
import os

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    paths = {
        "full_finetune": os.path.join(RESULTS_DIR, "full_finetune", "metrics.json"),
        "lora": os.path.join(RESULTS_DIR, "lora", "metrics.json"),
        "prompt": os.path.join(RESULTS_DIR, "prompt", "metrics.json"),
    }

    rows = []
    for name, path in paths.items():
        if not os.path.exists(path):
            continue
        m = load_metrics(path)
        rows.append(
            {
                "method": name,
                "accuracy": m.get("accuracy", 0.0),
                "f1_macro": m.get("f1_macro", 0.0),
                "train_time_sec": m.get("train_time_sec", 0.0),
                "peak_gpu_memory_mb": m.get("peak_gpu_memory_mb", 0.0),
            }
        )

    if not rows:
        raise SystemExit("No metrics found to compare.")

    df = pd.DataFrame(rows)

    comp_dir = os.path.join(RESULTS_DIR, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    df.to_csv(os.path.join(comp_dir, "summary.csv"), index=False)

    def plot_bar(metric, ylabel, filename):
        plt.figure(figsize=(6, 4))
        plt.bar(df["method"], df[metric])
        plt.ylabel(ylabel)
        plt.xlabel("Method")
        if metric in {"accuracy", "f1_macro"}:
            plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, filename))
        plt.close()

    plot_bar("accuracy", "Accuracy", "accuracy_comparison.png")
    plot_bar("f1_macro", "F1 (macro)", "f1_comparison.png")
    plot_bar("train_time_sec", "Training / inference time (s)", "training_time_comparison.png")
    plot_bar(
        "peak_gpu_memory_mb",
        "Peak GPU memory (MB)",
        "gpu_memory_usage_comparison.png",
    )


if __name__ == "__main__":
    main()
