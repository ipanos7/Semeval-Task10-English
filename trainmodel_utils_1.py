from sklearn.model_selection import RepeatedStratifiedKFold
import os
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import f1_score
from scipy.special import expit
from datasets import Dataset
import json
import torch

# --- Prepare Combined Labels ---
def prepare_combined_labels(training_data, all_labels):
    combined_labels = [label for label in all_labels if label["type"] in ["N", "S"]]
    label_to_idx = {label["label"]: idx for idx, label in enumerate(combined_labels)}

    num_classes = len(label_to_idx)
    binary_labels = np.zeros((len(training_data), num_classes))

    for i, article in enumerate(training_data):
        combined = article["narratives"] + article["subnarratives"]
        indices = [label_to_idx[label] for label in combined if label in label_to_idx]
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
def train_with_repeated_kfold_and_save(texts, labels):
    dataset = Dataset.from_dict({"text": texts, "label": labels.tolist()})
    dataset = dataset.map(tokenize, batched=True)

    labels_flat = labels.argmax(axis=1)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
    all_f1_scores = []
    best_f1 = 0  # Track the best F1 score
    best_fold = None

    for fold, (train_idx, val_idx) in enumerate(rskf.split(np.zeros(len(labels)), labels_flat)):
        print(f"\n=== Fold {fold + 1} ===")
        train_dataset = dataset.select(train_idx)
        val_dataset = dataset.select(val_idx)

        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=labels.shape[1]
        )

        training_args = TrainingArguments(
            output_dir=f"./results_fold_{fold}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"./logs_fold_{fold}",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=20,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            save_total_limit=1,
            learning_rate=5e-5,
            lr_scheduler_type="linear",
            fp16=True
        )

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
        print(f"F1 Score for fold {fold + 1}: {f1}")

        # Save the best model
        if f1 > best_f1:
            best_f1 = f1
            best_fold = fold
            model.save_pretrained("./final_model")
            tokenizer.save_pretrained("./final_model")
            print(f"Best model saved from fold {fold + 1} with F1: {f1}")

    mean_f1 = np.mean(all_f1_scores)
    print(f"\n=== Mean F1 Score (RepeatedStratifiedKFold): {mean_f1} ===")
    print(f"Best F1 Score: {best_f1} from fold {best_fold + 1}")

    return mean_f1, best_f1, best_fold

# --- Main Script ---
if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available.")

    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "data", "training_dataset.json")
    labels_path = os.path.join(current_dir, "data", "all_labels.json")

    print("Loading data...")
    with open(data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        all_labels = json.load(f)["labels"]

    print("Preparing combined labels...")
    texts, labels, label_to_idx = prepare_combined_labels(training_data, all_labels)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print("Training with Repeated Stratified K-Fold and saving the best model...")
    mean_f1, best_f1, best_fold = train_with_repeated_kfold_and_save(texts, labels)
    print(f"Final Mean F1 Score: {mean_f1}")
    print(f"Best Model from Fold {best_fold + 1} with F1 Score: {best_f1}")
