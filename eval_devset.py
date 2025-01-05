from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
import json

# Load models
narrative_model = RobertaForSequenceClassification.from_pretrained("./narrative_model")
narrative_tokenizer = RobertaTokenizer.from_pretrained("./narrative_model")
subnarrative_model = RobertaForSequenceClassification.from_pretrained("./subnarrative_model")
subnarrative_tokenizer = RobertaTokenizer.from_pretrained("./subnarrative_model")

# Load all labels
with open("all_labels.json", "r") as f:
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
    dataset = [{"text": text} for text in texts]
    tokenized = tokenizer([d["text"] for d in dataset], padding=True, truncation=True, return_tensors="pt")
    outputs = model(**tokenized)
    logits = outputs.logits.detach().numpy()
    probabilities = 1 / (1 + np.exp(-logits))  # Sigmoid
    predictions = (probabilities > 0.5).astype(int)  # Threshold = 0.5
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    return [[idx_to_label[idx] for idx, val in enumerate(pred) if val == 1] for pred in predictions]

# Load development set
dev_path = "data/subtask-2-documents"
with open(dev_path, "r") as f:
    dev_data = json.load(f)

texts = [article["content"] for article in dev_data]

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
for article, narratives, subnarratives in zip(dev_data, narrative_predictions, subnarrative_predictions):
    article_id = article["id"]
    narrative_str = ";".join(narratives) if narratives else "Other"
    subnarrative_str = ";".join(subnarratives) if subnarratives else "Other"
    submission.append(f"{article_id}\t{narrative_str}\t{subnarrative_str}")

# Save to submission file
with open("submission.tsv", "w") as f:
    f.write("\n".join(submission))

print("Predictions saved to submission.tsv")
