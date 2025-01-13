from sklearn.model_selection import RepeatedStratifiedKFold
import os
import numpy as np
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import f1_score
from scipy.special import expit
from datasets import Dataset
import json

# --- Prepare Subnarrative Labels ---
def prepare_labels_for_subnarratives(training_data, all_labels):
    subnarratives_only = [label for label in all_labels if label["type"] == "S"]
    label_to_idx = {label["label"]: idx for idx, label in enumerate(subnarratives_only)}

    num_classes = len(label_to_idx)
    binary_labels = np.zeros((len(training_data), num_classes))

    for i, article in enumerate(training_data):
        subnarratives = article["subnarratives"]
        indices = [label_to_idx[label] for label in subnarratives if label in label_to_idx]
        binary_labels[i, indices] = 1

    texts = [article["content"] for article in training_data]
    return texts, binary_labels, label_to_idx

# --- Tokenization ---
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

# --- Metrics ---
def compute_metrics(pred):
    logits, labels = pred
    probabilities = expit(logits)
    predictions = (probabilities > 0.5).astype(int)
    f1 = f1_score(labels, predictions, average="macro", zero_division=1)
    return {"f1_macro": f1}

# --- Training with Repeated KFold ---
def train_with_repeated_kfold(texts, labels):
    dataset = Dataset.from_dict({"text": texts, "label": labels.tolist()})
    dataset = dataset.map(tokenize, batched=True)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
    labels_flat = labels.argmax(axis=1)

    all_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(rskf.split(np.zeros(len(labels)), labels_flat)):
        print(f"\n=== Fold {fold+1} ===")
        train_dataset = dataset.select(train_idx)
        val_dataset = dataset.select(val_idx)

        model = DebertaV2ForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base", num_labels=labels.shape[1]
        )

        training_args = TrainingArguments(
            output_dir=f"./results_fold_{fold}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"./logs_fold_{fold}",
            per_device_train_batch_size=16,  # Adjust for available GPU memory
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,  # Simulate larger batch sizes
            num_train_epochs=10,
            warmup_steps=1000,
            weight_decay=0.01,
            learning_rate=3e-5,
            fp16=True,  # Enable mixed precision
            logging_steps=50,
            load_best_model_at_end=True,
            save_total_limit=2,
        )

        model.gradient_checkpointing_enable()  # Save memory

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )

        trainer.train()

        # Compute F1 on validation set
        predictions = trainer.predict(val_dataset)
        logits = predictions.predictions
        probabilities = expit(logits)
        predicted_labels = (probabilities > 0.5).astype(int)

        f1 = f1_score(val_dataset["label"], predicted_labels, average="macro", zero_division=1)
        all_f1_scores.append(f1)
        print(f"F1 Score for fold {fold+1}: {f1}")

    mean_f1 = np.mean(all_f1_scores)
    print(f"\n=== Mean F1 Score (RepeatedStratifiedKFold): {mean_f1} ===")

    # Save the model and tokenizer
    output_dir = "/content/drive/MyDrive/final_subnarrative_model"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Subnarrative model and tokenizer saved to {output_dir}.")

    return mean_f1

# --- Main Script ---
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "data", "final_training_dataset.json")
    labels_path = os.path.join(current_dir, "data", "all_labels.json")

    print("Loading data...")
    with open(data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        all_labels = json.load(f)["labels"]

    print("Preparing subnarrative labels...")
    texts, labels, label_to_idx = prepare_labels_for_subnarratives(training_data, all_labels)

    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")

    print("Training model...")
    mean_f1 = train_with_repeated_kfold(texts, labels)
    print(f"Final Mean F1 Score: {mean_f1}")
