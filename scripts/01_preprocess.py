import os

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from utils import set_seed


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MIN_SAMPLES_PER_CLASS = 3


def main() -> None:
    set_seed(42)

    data_path = os.path.join(DATA_DIR, "symptom_Description.csv")
    df = pd.read_csv(data_path)

    df = df.rename(columns={"Description": "text", "Disease": "label"})
    df = df[["text", "label"]].dropna().reset_index(drop=True)

    counts = df["label"].value_counts()
    frequent_labels = counts[counts >= MIN_SAMPLES_PER_CLASS].index

    if len(frequent_labels) > 0:
        filtered_df = df[df["label"].isin(frequent_labels)].reset_index(drop=True)
        if len(filtered_df) > 0 and filtered_df["label"].nunique() > 1:
            df = filtered_df

    labels = sorted(df["label"].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    df["label_id"] = df["label"].map(label2id)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    label_map_df = pd.DataFrame(
        {"label_id": list(label2id.values()), "label": list(label2id.keys())}
    )
    label_map_df.to_csv(os.path.join(PROCESSED_DIR, "label_map.csv"), index=False)

    label_counts = df["label_id"].value_counts()
    stratify_first = df["label_id"] if label_counts.min() >= 2 else None

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=stratify_first,
        random_state=42,
    )

    temp_counts = temp_df["label_id"].value_counts()
    stratify_second = temp_df["label_id"] if temp_counts.min() >= 2 else None

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=stratify_second,
        random_state=42,
    )

    train_ds = Dataset.from_pandas(
        train_df[["text", "label", "label_id"]], preserve_index=False
    )
    val_ds = Dataset.from_pandas(
        val_df[["text", "label", "label_id"]], preserve_index=False
    )
    test_ds = Dataset.from_pandas(
        test_df[["text", "label", "label_id"]], preserve_index=False
    )

    dataset_dict = DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
    )

    dataset_dict.save_to_disk(PROCESSED_DIR)


if __name__ == "__main__":
    main()
