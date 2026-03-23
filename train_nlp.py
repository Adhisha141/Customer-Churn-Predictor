"""
train_nlp.py
Fine-tune DistilBERT on customer support chat logs to detect churn signals.
Dataset: GoEmotions or a custom CSV with (text, churn_label) columns.
"""

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

MODEL_NAME = "distilbert-base-uncased"
SAVE_PATH = "models/nlp_churn_model"
MAX_LEN = 128
EPOCHS = 3
BATCH_SIZE = 16


def load_sample_data():
    """
    Load synthetic or real customer chat data.
    Replace this with your own CSV:
        df = pd.read_csv("data/customer_churn_chat.csv")
        # Must have columns: 'text', 'label'  (0 = no churn, 1 = churn)
    """
    # Minimal synthetic example to get started
    texts = [
        "I am really frustrated with the service, thinking of cancelling",
        "Please cancel my subscription immediately",
        "I've been waiting for support for 3 days, this is unacceptable",
        "Your prices are too high, I'm switching to a competitor",
        "I love your service, it works great for me",
        "Thanks for resolving my issue so quickly!",
        "Everything is working fine, no complaints",
        "Renewing my plan, very happy with the product",
        "Can you explain my bill? I'll cancel if this keeps up",
        "Great value, will continue using this service",
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
    return pd.DataFrame({"text": texts, "label": labels})


class ChurnDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }


def train():
    print("Loading data...")
    df = load_sample_data()

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    print("Tokenizing...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_enc = tokenizer(train_df["text"].tolist(), truncation=True, padding=True, max_length=MAX_LEN)
    val_enc = tokenizer(val_df["text"].tolist(), truncation=True, padding=True, max_length=MAX_LEN)

    train_dataset = ChurnDataset(train_enc, train_df["label"].tolist())
    val_dataset = ChurnDataset(val_enc, val_df["label"].tolist())

    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir="models/nlp_checkpoints",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="models/logs",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print(f"Saving model to {SAVE_PATH}")
    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("Done!")


if __name__ == "__main__":
    train()
