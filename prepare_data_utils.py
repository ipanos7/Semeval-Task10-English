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

def load_raw_data(raw_data_folder):
    """
    Φορτώνει τα raw documents από τον φάκελο και τα επιστρέφει ως λεξικό.
    """
    raw_data = {}
    for filename in os.listdir(raw_data_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(raw_data_folder, filename), "r", encoding="utf-8") as f:
                raw_data[filename] = f.read()
    return raw_data

def create_dataset(raw_data, annotations):
    """
    Δημιουργεί το dataset από τα raw documents και τα annotations.
    """
    dataset = []
    for article_id, content in raw_data.items():
        narratives = list(set(annotations.get(article_id, {}).get("narratives", [])))
        subnarratives = list(set(annotations.get(article_id, {}).get("subnarratives", [])))
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
    raw_data_folder = os.path.join(current_dir, "data", "raw-documents")
    annotations_file = os.path.join(current_dir, "data", "subtask-2-annotations.txt")
    narratives_file = os.path.join(current_dir, "data", "subtask2_narratives.txt")
    subnarratives_file = os.path.join(current_dir, "data", "subtask2_subnarratives.txt")
    all_labels_file = os.path.join(current_dir, "data", "all_labels.json")
    processed_annotations_file = os.path.join(current_dir, "data", "processed_annotations.json")
    training_dataset_file = os.path.join(current_dir, "data", "training_dataset.json")

    # Step 1: Δημιουργία labels
    print("Creating all labels...")
    narratives, subnarratives = create_all_labels_from_files(narratives_file, subnarratives_file)
    save_all_labels_to_json(narratives, subnarratives, all_labels_file)
    print(f"All labels saved to {all_labels_file}")

    # Step 2: Επεξεργασία annotations
    print("Processing annotations...")
    annotations = process_annotations(annotations_file)
    with open(processed_annotations_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)
    print(f"Annotations saved to {processed_annotations_file}")

    # Step 3: Δημιουργία dataset
    print("Creating dataset...")
    raw_data = load_raw_data(raw_data_folder)
    dataset = create_dataset(raw_data, annotations)
    save_dataset_to_json(dataset, training_dataset_file)
    print(f"Training dataset saved to {training_dataset_file}")

    print("\nSample from training dataset:")
    print(json.dumps(dataset[:2], indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
