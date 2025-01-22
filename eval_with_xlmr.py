from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import json
import os

# Load models
narrative_model = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/final_narrative_model_xlmr")
narrative_tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/final_narrative_model_xlmr")
subnarrative_model = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/final_subnarrative_model_xlmr")
subnarrative_tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/final_subnarrative_model_xlmr")

# Load all labels
with open("/content/Semeval-Task10-English/data/all_labels.json", "r") as f:
    all_labels = json.load(f)["labels"]

# Separate narratives and subnarratives
narrative_labels = {label["label"]: label["idx"] for label in all_labels if label["type"] == "N"}
subnarrative_labels = {label["label"]: label["idx"] for label in all_labels if label["type"] == "S"}

# Save narrative labels
with open("narrative_labels.json", "w") as f:
    json.dump({"label_to_idx": narrative_labels}, f, indent=4)

# Save subnarrative labels
with open("subnarrative_labels.json", "w") as f:
    json.dump({"label_to_idx": subnarrative_labels}, f, indent=4)

print("Labels saved to narrative_labels.json and subnarrative_labels.json")


# Function to make predictions
def predict_labels(model, tokenizer, texts, label_to_idx):
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**tokenized)
    logits = outputs.logits.detach().numpy()
    probabilities = 1 / (1 + np.exp(-logits))  # Sigmoid
    predictions = (probabilities > 0.5).astype(int)  # Threshold = 0.5
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    return [[idx_to_label[idx] for idx, val in enumerate(pred) if val == 1] for pred in predictions]

# Load development set
dev_path = "/content/Semeval-Task10-English/data/subtask-2-documents"
texts = []
article_ids = []

for file_name in os.listdir(dev_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(dev_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            texts.append(content)
            article_ids.append(file_name.replace(".txt", ""))  # Use the file name as article ID

# Predict narratives
with open("narrative_labels.json", "r") as f:
    narrative_label_to_idx = json.load(f)["label_to_idx"]
narrative_predictions = predict_labels(narrative_model, narrative_tokenizer, texts, narrative_label_to_idx)

# Predict subnarratives
with open("subnarrative_labels.json", "r") as f:
    subnarrative_label_to_idx = json.load(f)["label_to_idx"]
subnarrative_predictions = predict_labels(subnarrative_model, subnarrative_tokenizer, texts, subnarrative_label_to_idx)

# Format predictions
submission = []
for article_id, narratives, subnarratives in zip(article_ids, narrative_predictions, subnarrative_predictions):
    narrative_str = ";".join(narratives) if narratives else "Other"
    subnarrative_str = ";".join(subnarratives) if subnarratives else "Other"
    submission.append(f"{article_id}.txt\t{narrative_str}\t{subnarrative_str}")

# Save to submission file
with open("submission.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(submission))

print("Predictions saved to submission.txt")
