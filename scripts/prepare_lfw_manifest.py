"""Create reproducible identity-disjoint LFW manifests without changing images."""

import argparse
import csv
import json
import random
import statistics
from collections import Counter
from pathlib import Path

SPLITS = ("train", "validation", "test")
FIELDS = ("image_path", "identity_id", "identity_name", "split", "train_label", "landmarks_available")


def discover(workspace):
    root = workspace / "data/lfw/lfw_home/lfw_funneled"
    if not root.is_dir():
        raise ValueError(f"Dataset directory missing: {root}")
    people = {}
    for directory in sorted(root.iterdir()):
        if directory.is_dir():
            paths = sorted(directory.glob("*.jpg"))
            if paths:
                people[directory.name] = paths
    return people


def build(workspace, seed=42, minimum=5):
    people = discover(workspace)
    # Global IDs follow the sorted full dataset, independent of split assignment.
    global_ids = {name: index for index, name in enumerate(sorted(people))}
    selected = sorted(name for name, paths in people.items() if len(paths) >= minimum)
    if len(selected) < 3:
        raise ValueError("At least three eligible identities are required")
    shuffled = selected.copy()
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(selected) * 0.70)
    validation_count = int(len(selected) * 0.15)
    assignments = {
        "train": sorted(shuffled[:train_count]),
        "validation": sorted(shuffled[train_count:train_count + validation_count]),
        "test": sorted(shuffled[train_count + validation_count:]),
    }
    labels = {name: index for index, name in enumerate(assignments["train"])}
    rows = {}
    for split, names in assignments.items():
        rows[split] = [
            {
                "image_path": path.relative_to(workspace).as_posix(),
                "identity_id": str(global_ids[name]),
                "identity_name": name,
                "split": split,
                "train_label": str(labels[name]) if split == "train" else "",
                "landmarks_available": "false",
            }
            for name in names for path in people[name]
        ]
    validate(workspace, rows, minimum)
    mapping = [
        {"identity_id": str(global_ids[name]), "identity_name": name, "train_label": str(label)}
        for name, label in labels.items()
    ]
    summary = {
        "seed": seed,
        "minimum_images_per_identity": minimum,
        "algorithm": "Sort identity directory names; shuffle with Python random.Random(seed); floor 70% train, floor 15% validation, remainder test; sort identities within splits.",
        "path_base": "workspace root; forward-slash relative paths",
        "identity_id": "Zero-based position in sorted full LFW identity directory list; not a classifier label",
        "landmarks": "Unavailable; no coordinates generated",
        "source_identities": len(people),
        "source_images": sum(map(len, people.values())),
        "selected_identities": len(selected),
        "selected_images": sum(len(people[name]) for name in selected),
        "splits": {},
    }
    for split in SPLITS:
        counts = list(Counter(row["identity_id"] for row in rows[split]).values())
        summary["splits"][split] = {
            "identities": len(counts), "images": len(rows[split]),
            "min_images": min(counts), "max_images": max(counts),
            "mean_images": statistics.mean(counts), "median_images": statistics.median(counts),
            "images_per_identity_histogram": dict(sorted(Counter(counts).items())),
        }
    return rows, mapping, summary


def validate(workspace, rows, minimum=5):
    paths, owners, identities = set(), {}, {}
    labels = {}
    for split in SPLITS:
        counts = Counter()
        for row in rows[split]:
            path = row["image_path"]
            if path in paths:
                raise ValueError(f"Duplicate image path: {path}")
            paths.add(path)
            resolved = (workspace / path).resolve()
            if not resolved.is_relative_to(workspace.resolve()) or not resolved.is_file():
                raise ValueError(f"Missing or invalid image path: {path}")
            identity = row["identity_id"]
            name = row["identity_name"]
            if identity in owners and owners[identity] != split:
                raise ValueError(f"Identity leakage: {identity}")
            if identity in identities and identities[identity] != name:
                raise ValueError("Identity ID/name mismatch")
            owners[identity], identities[identity] = split, name
            if row["split"] != split or row["landmarks_available"] != "false":
                raise ValueError("Invalid split or landmark status")
            if split == "train":
                label = int(row["train_label"])
                if identity in labels and labels[identity] != label:
                    raise ValueError("Inconsistent training label")
                labels[identity] = label
            elif row["train_label"] != "":
                raise ValueError("Evaluation identity has training label")
            counts[identity] += 1
        if not counts or min(counts.values()) < minimum:
            raise ValueError("Empty split or identity below minimum image count")
    if sorted(labels.values()) != list(range(len(labels))):
        raise ValueError("Training labels are not unique and contiguous")


def read_manifests(output):
    rows = {}
    for split in SPLITS:
        with (output / f"{split}.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(FIELDS):
                raise ValueError("Unexpected manifest columns")
            rows[split] = list(reader)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output or workspace / "data/lfw/manifests"
    targets = [output / f"{split}.csv" for split in SPLITS]
    targets += [output / "train_label_mapping.csv", output / "split_summary.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite existing outputs; choose a new --output directory")
    rows, mapping, summary = build(workspace, seed=args.seed)
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with (output / f"{split}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows[split])
    with (output / "train_label_mapping.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("identity_id", "identity_name", "train_label"))
        writer.writeheader()
        writer.writerows(mapping)
    with (output / "split_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    validate(workspace, read_manifests(output))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
