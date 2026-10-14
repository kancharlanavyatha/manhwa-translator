# d:\webtoon translator\training\train_translator.py
import json
import os
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import evaluate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(SCRIPT_DIR, "train.json")
VAL_PATH = os.path.join(SCRIPT_DIR, "val.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_checkpoints")
FINAL_MODEL_DIR = os.path.join(SCRIPT_DIR, "final_manhwa_translator")

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "kor_Hang"
TGT_LANG = "eng_Latn"

print("Loading dataset splits...")
with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open(VAL_PATH, "r", encoding="utf-8") as f:
    val_data = json.load(f)

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print(f"Loaded {len(train_dataset)} training pairs and {len(val_dataset)} validation pairs.")

print(f"Loading base model and tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def preprocess(batch):
    inputs = [ex for ex in batch["ko"]]
    targets = [ex for ex in batch["en"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing dataset...")
tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=["ko", "en"])
tokenized_val = val_dataset.map(preprocess, batched=True, remove_columns=["ko", "en"])

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=1500,
    save_steps=1500,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    optim="adafactor",
    gradient_checkpointing=True,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    report_to="none"
)

metric = evaluate.load("sacrebleu")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": round(result["score"], 2)}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
)

print("Starting model fine-tuning...")
checkpoint_to_resume = True if os.path.exists(OUTPUT_DIR) and any(d.startswith("checkpoint-") for d in os.listdir(OUTPUT_DIR)) else None
trainer.train(resume_from_checkpoint=checkpoint_to_resume)

print(f"Saving final model to {FINAL_MODEL_DIR}...")
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print("Model fine-tuning complete and saved!")
