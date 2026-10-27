"""Validate full LFW annotations without rerunning detector inference.

Synthetic detector rows are parser test fixtures only; they are never written
as annotations. Integration checks require the completed extraction artifacts.
"""

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import unittest

import numpy as np
from PIL import Image

from scripts import extract_lfw_landmarks as pipeline

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/yunet/annotations/yunet_2026may"
SPLITS = ("train", "validation", "test")
COUNTS = {"train": 4161, "validation": 836, "test": 988}
POINT_FIELDS = tuple(f"{name}_{axis}" for name in
                     ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth")
                     for axis in ("x", "y"))
MODEL_HASH = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"


def valid_face():
    # bbox, anatomical right eye, left eye, nose, right mouth, left mouth, score
    return np.array([10, 20, 200, 200, 80, 90, 160, 91, 120, 130,
                     85, 170, 155, 171, 0.97], dtype=float)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class LFWLandmarkContractTests(unittest.TestCase):
    def test_explicit_anatomical_mapping(self):
        result = pipeline.analyze_detections(np.array([valid_face()]))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["landmarks"],
                         [[160, 91], [80, 90], [120, 130], [155, 171], [85, 170]])

    def test_no_detection_has_no_fabricated_points(self):
        for faces in (None, np.empty((0, 15))):
            with self.subTest(faces=faces):
                result = pipeline.analyze_detections(faces)
                self.assertEqual(result["status"], "no_detection")
                self.assertEqual(result["detection_count"], 0)
                self.assertIsNone(result["landmarks"])
                self.assertIsNone(result["selected_detection_index"])
                self.assertIsNone(result["confidence"])

    def test_invalid_landmarks_rejected_without_clamping(self):
        for value in (-0.01, 250.0, float("nan"), float("inf")):
            for offset in (4, 5):
                face = valid_face()
                face[offset] = value
                with self.subTest(value=value, offset=offset):
                    result = pipeline.analyze_detections(np.array([face]))
                    self.assertEqual(result["status"], "invalid_coordinates")
                    self.assertIsNone(result["landmarks"])
        face = valid_face()
        face[4:6] = [0, 249.999]
        self.assertEqual(pipeline.analyze_detections(np.array([face]))["status"], "ok")

    def test_invalid_detector_shape(self):
        for faces in (np.zeros((1, 14)), np.zeros(15), np.zeros((2, 3, 15))):
            with self.subTest(shape=faces.shape):
                result = pipeline.analyze_detections(faces)
                self.assertEqual(result["status"], "invalid_detector_output")
                self.assertIsNone(result["landmarks"])

    def test_invalid_bbox_and_score(self):
        for offset, value in ((2, 0), (3, -1), (0, float("nan")),
                              (14, 1.01), (14, -0.01), (14, float("inf"))):
            face = valid_face()
            face[offset] = value
            with self.subTest(offset=offset, value=value):
                result = pipeline.analyze_detections(np.array([face]))
                self.assertEqual(result["status"], "invalid_detection")
                self.assertIsNone(result["landmarks"])

    def test_largest_face_then_confidence_then_first_index(self):
        faces = np.stack([valid_face()] * 3)
        faces[:, 2:4] = [[20, 20], [100, 100], [100, 100]]
        faces[:, 14] = [0.99, 0.94, 0.96]
        result = pipeline.analyze_detections(faces)
        self.assertEqual(result["selected_detection_index"], 2)
        self.assertEqual(result["detection_count"], 3)
        self.assertIn("multiple_faces", result["warnings"])
        faces[1, 14] = 0.96
        self.assertEqual(pipeline.analyze_detections(faces)["selected_detection_index"], 1)

    def test_duplicate_groups_and_cross_split_pair_counts(self):
        records = []
        for index, (digest, split, identity) in enumerate([
            ("a", "train", "1"), ("a", "train", "1"),
            ("a", "validation", "2"), ("a", "test", "3"),
            ("b", "train", "4"), ("b", "train", "4"),
            ("c", "test", "5"), ("", "test", "6"),
        ]):
            records.append({"image_path": f"fixture_{index}.jpg", "split": split,
                            "identity_id": identity, "image_sha256": digest})
        result = pipeline.duplicate_report(records)
        self.assertEqual(result["hashed_images"], 7)
        self.assertEqual(result["unique_hashes"], 3)
        self.assertEqual(result["duplicate_groups"], 2)
        self.assertEqual(result["duplicate_images"], 6)
        self.assertEqual(result["extra_copies"], 4)
        self.assertEqual(result["cross_split_groups"], 1)
        self.assertEqual(result["cross_split_images"], 4)
        self.assertEqual(result["within_split"]["train"],
                         {"groups": 2, "images": 4, "extra_copies": 2})
        self.assertEqual(result["split_pairs"], {
            "train__validation": {"groups": 1, "image_pairs": 2},
            "train__test": {"groups": 1, "image_pairs": 2},
            "validation__test": {"groups": 1, "image_pairs": 1},
        })
        self.assertEqual({g["sha256"]: g["identity_conflict"] for g in result["groups"]},
                         {"a": True, "b": False})


class LFWLandmarkArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.annotations = {split: read_csv(OUTPUT / f"{split}.csv") for split in SPLITS}
        cls.rows = [row for split in SPLITS for row in cls.annotations[split]]
        cls.manifests = {split: read_csv(ROOT / f"data/lfw/manifests/{split}.csv") for split in SPLITS}
        cls.raw_rows = [json.loads(line) for line in (OUTPUT / "detections.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.raw = {row["image_path"]: row for row in cls.raw_rows}
        cls.summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
        cls.duplicates = json.loads((OUTPUT / "duplicates.json").read_text(encoding="utf-8"))

    def test_exact_manifest_join_without_changed_identity_labels(self):
        self.assertEqual(len(self.rows), 5985)
        self.assertEqual(len({r["image_path"] for r in self.rows}), 5985)
        self.assertEqual(len(self.raw_rows), 5985)
        self.assertEqual(len(self.raw), 5985)
        identities = {}
        for split in SPLITS:
            self.assertEqual(len(self.annotations[split]), COUNTS[split])
            self.assertEqual([r["image_path"] for r in self.annotations[split]],
                             [r["image_path"] for r in self.manifests[split]])
            for annotation, source in zip(self.annotations[split], self.manifests[split]):
                for field in ("image_path", "identity_id", "identity_name", "split", "train_label"):
                    self.assertEqual(annotation[field], source[field])
                identity = annotation["identity_id"]
                self.assertEqual(identities.setdefault(identity, split), split)
                self.assertIn(annotation["image_path"], self.raw)

    def test_original_content_hash_size_and_model_integrity(self):
        model = ROOT / "models/yunet/face_detection_yunet_2026may.onnx"
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), MODEL_HASH)
        for row in self.rows:
            with self.subTest(image=row["image_path"]):
                path = ROOT / row["image_path"]
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["image_sha256"])
                with Image.open(path) as image:
                    self.assertEqual(image.size, (250, 250))
                    self.assertEqual(image.mode, "RGB")
                self.assertEqual((int(row["width"]), int(row["height"])), (250, 250))

    def test_actual_mapping_bounds_and_failure_nullability(self):
        for row in self.rows:
            with self.subTest(image=row["image_path"]):
                raw = self.raw[row["image_path"]]
                self.assertEqual(raw["status"], row["status"])
                faces = raw["raw_detections"]
                self.assertEqual(len(faces), int(row["detection_count"]))
                if row["status"] != "ok":
                    self.assertEqual(row["landmarks_available"], "false")
                    self.assertTrue(all(row[field] == "" for field in POINT_FIELDS))
                    continue
                self.assertEqual(row["landmarks_available"], "true")
                chosen = int(row["selected_detection_index"])
                self.assertEqual(chosen, raw["selected_detection_index"])
                face = faces[chosen]
                self.assertEqual(len(face), 15)
                expected = face[6:8] + face[4:6] + face[8:10] + face[12:14] + face[10:12]
                for field, value in zip(POINT_FIELDS, expected):
                    self.assertTrue(math.isfinite(float(row[field])))
                    self.assertGreaterEqual(float(row[field]), 0)
                    self.assertLess(float(row[field]), 250)
                    self.assertAlmostEqual(float(row[field]), value)
                self.assertAlmostEqual(float(row["confidence"]), face[14])
                self.assertGreaterEqual(float(row["confidence"]), 0.9)
                self.assertLessEqual(float(row["confidence"]), 1)
                for field, value in zip(("bbox_x", "bbox_y", "bbox_width", "bbox_height"), face[:4]):
                    self.assertAlmostEqual(float(row[field]), value)

    def test_actual_largest_face_rule(self):
        for row in self.rows:
            faces = self.raw[row["image_path"]]["raw_detections"]
            if row["status"] == "ok":
                best = sorted(range(len(faces)),
                              key=lambda i: (-faces[i][2] * faces[i][3], -faces[i][14], i))[0]
                self.assertEqual(int(row["selected_detection_index"]), best)
            self.assertEqual("multiple_faces" in row["warnings"].split(";"), len(faces) > 1)

    def test_summary_rates_timing_and_confidence(self):
        self.assertEqual(self.summary["model_sha256"], MODEL_HASH)
        self.assertEqual(self.summary["mapping"], [1, 0, 2, 4, 3])
        self.assertEqual(self.summary["target"], "DNN_TARGET_CPU")
        self.assertEqual(self.summary["score_threshold"], 0.9)
        for split in (*SPLITS, "all"):
            rows = self.rows if split == "all" else self.annotations[split]
            summary = self.summary["overall"] if split == "all" else self.summary["splits"][split]
            statuses = Counter(row["status"] for row in rows)
            successful = sum(row["status"] == "ok" for row in rows)
            detected = sum(int(row["detection_count"]) > 0 for row in rows)
            self.assertEqual(summary["images"], len(rows))
            self.assertEqual(summary["identities"], len({row["identity_id"] for row in rows}))
            self.assertEqual(summary["status_counts"], dict(statuses))
            self.assertEqual(summary["success_images"], successful)
            self.assertEqual(summary["detected_images"], detected)
            self.assertEqual(summary["failed_images"], len(rows) - successful)
            self.assertAlmostEqual(summary["success_rate"], successful / len(rows))
            self.assertAlmostEqual(summary["detection_rate"], detected / len(rows))
            self.assertEqual(summary["multi_face_images"],
                             sum(int(row["detection_count"]) > 1 for row in rows))
            expected_warnings = Counter(w for row in rows for w in row["warnings"].split(";") if w)
            self.assertEqual(summary["warning_counts"], dict(expected_warnings))
            for output_key, row_key in (("confidence", "confidence"), ("detection_ms", "detection_ms")):
                values = [float(row[row_key]) for row in rows if row[row_key] != ""]
                self.assertTrue(all(math.isfinite(v) and v >= 0 for v in values))
                if not values:
                    self.assertIsNone(summary[output_key])
                    continue
                for key, value in {"min": min(values), "max": max(values),
                                   "mean": statistics.mean(values), "median": statistics.median(values)}.items():
                    self.assertAlmostEqual(summary[output_key][key], value)
            self.assertAlmostEqual(summary["total_detection_ms"],
                                   sum(float(row["detection_ms"]) for row in rows if row["detection_ms"] != ""))

    def test_duplicate_report_independently_matches_content_groups(self):
        groups = defaultdict(list)
        for row in self.rows:
            if row["image_sha256"]:
                groups[row["image_sha256"]].append(row)
        expected = {digest: members for digest, members in groups.items() if len(members) > 1}
        report = self.duplicates
        self.assertEqual(report["hashed_images"], sum(len(v) for v in groups.values()))
        self.assertEqual(report["unique_hashes"], len(groups))
        self.assertEqual(report["duplicate_groups"], len(expected))
        self.assertEqual(report["duplicate_images"], sum(len(v) for v in expected.values()))
        self.assertEqual(report["extra_copies"], sum(len(v) - 1 for v in expected.values()))
        self.assertEqual({g["sha256"] for g in report["groups"]}, set(expected))
        cross_split = [members for members in expected.values() if len({m["split"] for m in members}) > 1]
        self.assertEqual(report["cross_split_groups"], len(cross_split))
        self.assertEqual(report["cross_split_images"], sum(len(v) for v in cross_split))
        for group in report["groups"]:
            members = expected[group["sha256"]]
            self.assertEqual(group["members"], [{key: row[key] for key in ("image_path", "split", "identity_id")}
                                                for row in members])
            self.assertEqual(group["splits"], sorted({row["split"] for row in members}))
            self.assertEqual(group["cross_split"], len(group["splits"]) > 1)
            self.assertEqual(group["identity_conflict"], len({row["identity_id"] for row in members}) > 1)
        for split in SPLITS:
            counts = [sum(row["split"] == split for row in members) for members in expected.values()]
            counts = [count for count in counts if count > 1]
            self.assertEqual(report["within_split"][split],
                             {"groups": len(counts), "images": sum(counts),
                              "extra_copies": sum(count - 1 for count in counts)})
        for pair, values in report["split_pairs"].items():
            left, right = pair.split("__")
            paired = [(sum(row["split"] == left for row in members),
                       sum(row["split"] == right for row in members)) for members in expected.values()]
            self.assertEqual(values["groups"], sum(a > 0 and b > 0 for a, b in paired))
            self.assertEqual(values["image_pairs"], sum(a * b for a, b in paired))
        self.assertEqual(self.summary["duplicate_statistics"],
                         {key: value for key, value in report.items() if key != "groups"})

    def test_protected_inputs_and_complete_failure_log(self):
        self.assertEqual(self.summary["integrity"]["images_checked"], 5985)
        self.assertEqual(self.summary["integrity"]["changed_files"], [])
        protected = self.summary["protected_input_hashes"]
        for required in ("README.md", "data/lfw/manifests/train.csv",
                         "data/lfw/manifests/validation.csv", "data/lfw/manifests/test.csv"):
            self.assertIn(required, protected)
        for path, digest in protected.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        expected = [row for row in self.rows if row["status"] != "ok" or row["warnings"]]
        self.assertEqual(read_csv(OUTPUT / "failures.csv"), expected)
        log = (OUTPUT / "processing.log").read_text(encoding="utf-8")
        self.assertIn("COMPLETE processed=5985", log)
        self.assertNotIn("ABORTED", log)
        for row in expected:
            self.assertIn(f"path={row['image_path']} status={row['status']}", log)


if __name__ == "__main__":
    unittest.main()
