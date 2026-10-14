# d:\webtoon translator\training\prepare_data.py
import json
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(SCRIPT_DIR, "train.json")
VAL_PATH = os.path.join(SCRIPT_DIR, "val.json")

PARQUET_URL = "https://huggingface.co/api/datasets/traintogpb/aihub-koen-translation-integrated-large-10m/parquet/default/train/0.parquet"
TARGET_SAMPLES = 50000

print("Downloading AI Hub parallel corpus shard from Hugging Face...")
df = pd.read_parquet(PARQUET_URL, columns=["ko", "en"])
print(f"Loaded {len(df):,} raw parallel pairs. Filtering for conversational dialogue...")

samples = []
for _, row in df.iterrows():
    ko = str(row["ko"]).strip()
    en = str(row["en"]).strip()

    # Filter for typical webtoon balloon lengths
    if 5 <= len(ko) <= 120 and 5 <= len(en) <= 150:
        samples.append({"ko": ko, "en": en})

    if len(samples) >= TARGET_SAMPLES:
        break

split_idx = int(len(samples) * 0.95)
train_split = samples[:split_idx]
val_split = samples[split_idx:]

with open(TRAIN_PATH, "w", encoding="utf-8") as f:
    json.dump(train_split, f, ensure_ascii=False, indent=2)

with open(VAL_PATH, "w", encoding="utf-8") as f:
    json.dump(val_split, f, ensure_ascii=False, indent=2)

print(f"Success! Created:\n  - {TRAIN_PATH} ({len(train_split):,} pairs)\n  - {VAL_PATH} ({len(val_split):,} pairs)")
