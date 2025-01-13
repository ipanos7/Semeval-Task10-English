import os
import json

def create_all_labels_from_files(narratives_file, subnarratives_file):
    """
    Δημιουργεί δύο ξεχωριστές λίστες labels για narratives και subnarratives.
    """
    narratives = []
    subnarratives = []

    with open(narratives_file, "r", encoding="utf-8") as nf:
        narratives = [line.strip() for line in nf if line.strip()]

    with open(subnarratives_file, "r", encoding="utf-8") as sf:
        subnarratives = [line.strip() for line in sf if line.strip()]

    return sorted(narratives), sorted(subnarratives)

def process_annotations(annotations_file):
    """
    Επεξεργάζεται τα annotations και τα επιστρέφει ως λεξικό ανά άρθρο.
    """
    annotations = {}
    with open(annotations_file, "r", encoding="utf-8") as af:
        for line in af:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                print(f"Skipped invalid line: {line.strip()}")
                continue
            article_id, narratives, subnarratives = parts

            narratives_list = list(set(
                narrative.strip() for narrative in narratives.split(";") if narrative.strip()
            ))
            subnarratives_list = list(set(
                subnarrative.strip() for subnarrative in subnarratives.split(";")
                if subnarrative.strip() and subnarrative.strip() != "Other"
            ))

            annotations[article_id] = {
                "narratives": narratives_list,
                "subnarratives": subnarratives_list
            }
    return annotations

def process_gold_annotations(gold_annotations_file):
    """
    Process gold annotations into a dictionary mapping article IDs to labels.
    """
    annotations = {}
    with open(gold_annotations_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                print(f"Invalid line skipped: {line}")
                continue
            article_id, narratives, subnarratives = parts

            narratives_list = list(set(
                narrative.strip() for narrative in narratives.split(";") if narrative.strip()
            ))
            subnarratives_list = list(set(
                subnarrative.strip() for subnarrative in subnarratives.split(";")
                if subnarrative.strip() and subnarrative.strip() != "Other"
            ))

            annotations[article_id] = {
                "narratives": narratives_list,
                "subnarratives": subnarratives_list
            }
    return annotations


def load_and_merge_raw_data(original_raw_folder, dev_raw_folder):
    """
    Load raw text data from both training and development folders.
    """
    raw_data = {}

    # Load training raw data
    for filename in os.listdir(original_raw_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(original_raw_folder, filename), "r", encoding="utf-8") as f:
                raw_data[filename] = f.read()

    # Load development raw data
    for filename in os.listdir(dev_raw_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(dev_raw_folder, filename), "r", encoding="utf-8") as f:
                raw_data[filename] = f.read()

    return raw_data


def create_combined_dataset(original_raw_folder, dev_raw_folder, original_annotations, gold_annotations):
    """
    Combine raw data from training and development sets with annotations.
    """
    # Load raw data from both training and development sets
    raw_data = load_and_merge_raw_data(original_raw_folder, dev_raw_folder)

    dataset = []
    for article_id, content in raw_data.items():
        # Combine narratives and subnarratives from both annotations
        narratives = list(set(original_annotations.get(article_id, {}).get("narratives", [])) |
                          set(gold_annotations.get(article_id, {}).get("narratives", [])))
        subnarratives = list(set(original_annotations.get(article_id, {}).get("subnarratives", [])) |
                             set(gold_annotations.get(article_id, {}).get("subnarratives", [])))

        dataset.append({
            "article_id": article_id,
            "content": content,
            "narratives": narratives,
            "subnarratives": subnarratives
        })

    return dataset


def save_all_labels_to_json(narratives, subnarratives, output_file):
    """
    Αποθηκεύει τα labels σε JSON με ξεχωριστούς δείκτες για narratives και subnarratives.
    """
    all_labels = []

    for idx, narrative in enumerate(narratives):
        all_labels.append({"label": narrative, "type": "N", "idx": idx})

    for idx, subnarrative in enumerate(subnarratives):
        all_labels.append({"label": subnarrative, "type": "S", "idx": idx})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"labels": all_labels}, f, ensure_ascii=False, indent=4)

def save_dataset_to_json(dataset, output_file):
    """
    Αποθηκεύει το dataset σε αρχείο JSON.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

def main():
    
    current_dir = os.path.dirname(__file__)

    # Paths
    original_raw_folder = os.path.join(current_dir, "data", "raw-documents")  # Original training raw data
    dev_raw_folder = os.path.join(current_dir, "data", "subtask-2-documents")  # Development set raw data
    annotations_file = os.path.join(current_dir, "data", "subtask-2-annotations.txt")
    gold_annotations_file = os.path.join(current_dir, "data", "subtask-2-dev_annotations.txt")
    narratives_file = os.path.join(current_dir, "data", "subtask2_narratives.txt")
    subnarratives_file = os.path.join(current_dir, "data", "subtask2_subnarratives.txt")
    all_labels_file = os.path.join(current_dir, "data", "all_labels.json")
    final_training_dataset_file = os.path.join(current_dir, "data", "final_training_dataset.json")

    # Step 1: Load and process labels
    print("Loading and processing labels...")
    narratives, subnarratives = create_all_labels_from_files(narratives_file, subnarratives_file)
    save_all_labels_to_json(narratives, subnarratives, all_labels_file)
    print(f"Labels saved to {all_labels_file}")

    # Step 2: Process original and gold annotations
    print("Processing annotations...")
    original_annotations = process_annotations(annotations_file)
    gold_annotations = process_gold_annotations(gold_annotations_file)

    # Step 3: Combine raw data and annotations
    print("Combining datasets...")
    final_dataset = create_combined_dataset(original_raw_folder, dev_raw_folder, original_annotations, gold_annotations)
    save_dataset_to_json(final_dataset, final_training_dataset_file)
    print(f"Final dataset saved to {final_training_dataset_file}")


if __name__ == "__main__":
    main()
