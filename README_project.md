Disease–Symptom Classification: Full Fine-Tuning vs LoRA vs Prompting
====================================================================

This project compares three approaches for disease classification from symptom descriptions using a small GPT-2 family model:

- Full fine-tuning
- LoRA fine-tuning
- Few-shot prompt-based classification

All experiments run on the local dataset already extracted in `data/`.

Project structure
-----------------

- `data/` – raw CSV files from your dataset
  - `symptom_Description.csv` (main training file, columns: `Disease`, `Description`)
  - `dataset.csv`
  - `symptom_precaution.csv`
  - `Symptom-severity.csv`
- `data/processed/` – preprocessed Hugging Face dataset and `label_map.csv`
- `scripts/` – pipeline scripts
  - `01_preprocess.py`
  - `02_train_ft.py`
  - `03_train_lora.py`
  - `04_prompt_eval.py`
  - `05_evaluate.py`
  - `utils.py`
- `notebooks/eda.ipynb` – quick exploratory data analysis
- `results/` – experiment outputs
  - `full_finetune/`
  - `lora/`
  - `prompt/`
  - `comparisons/`

Setup
-----

From the `minor project` folder:

```powershell
# (optional) activate your existing virtual environment
# .\myenv\Scripts\Activate.ps1

pip install -r requirements.txt
```

1. Preprocessing
----------------

Preprocess `symptom_Description.csv`, rename columns, encode labels, and create train/validation/test splits.

```powershell
python .\scripts\01_preprocess.py
```

Outputs:

- `data/processed/` (Hugging Face dataset saved with `save_to_disk`)
- `data/processed/label_map.csv`

2. Full fine-tuning
-------------------

Run full fine-tuning with OOM fallbacks:

1. Try `gpt2`
2. On CUDA OOM, retry with `distilgpt2`
3. On OOM again, freeze all transformer layers except the last two blocks and retry once

```powershell
python .\scripts\02_train_ft.py
```

Outputs under `results/full_finetune/`:

- `model/` (fine-tuned model and tokenizer)
- `metrics.json` (accuracy, precision_macro, recall_macro, f1_macro, training time, peak GPU memory)
- `train_runtime.txt`
- `gpu_memory.txt`
- `confusion_matrix.png`
- `classification_report.txt`

3. LoRA fine-tuning
-------------------

Run LoRA fine-tuning on GPT-2 using PEFT.

```powershell
python .\scripts\03_train_lora.py
```

Outputs under `results/lora/`:

- `adapter/` (LoRA adapter and tokenizer)
- `metrics.json`
- `train_runtime.txt`
- `gpu_memory.txt`
- `confusion_matrix.png`
- `classification_report.txt`

4. Prompt-based evaluation
--------------------------

Run few-shot prompting with GPT-2 using greedy decoding and fuzzy label matching.

```powershell
python .\scripts\04_prompt_eval.py
```

Outputs under `results/prompt/`:

- `metrics.json`
- `train_runtime.txt` (end-to-end inference time)
- `gpu_memory.txt`
- `confusion_matrix.png`
- `classification_report.txt`

5. Aggregate comparison
-----------------------

Gather metrics from all three approaches and produce comparison plots.

```powershell
python .\scripts\05_evaluate.py
```

Outputs under `results/comparisons/`:

- `summary.csv`
- `accuracy_comparison.png`
- `f1_comparison.png`
- `training_time_comparison.png`
- `gpu_memory_usage_comparison.png`

Notes
-----

- All scripts fix the random seed at 42.
- Tokenizers use `pad_token = eos_token`.
- Training uses `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`, `fp16=True` (if CUDA is available), and `max_length=128`.
- Peak GPU memory is measured with `torch.cuda.max_memory_allocated()` when CUDA is available.
