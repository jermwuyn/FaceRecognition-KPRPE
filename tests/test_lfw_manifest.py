"""Dataset-dependent checks; run after generating manifests."""

import copy
import csv
import importlib.util
import json
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_lfw_manifest", WORKSPACE / "scripts/prepare_lfw_manifest.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LFWManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = WORKSPACE / "data/lfw/manifests"
        cls.rows = module.read_manifests(cls.output)

    def test_validity(self):
        module.validate(WORKSPACE, self.rows)

    def test_exact_source_coverage(self):
        source = module.discover(WORKSPACE)
        eligible = {name: paths for name, paths in source.items() if len(paths) >= 5}
        expected = {p.relative_to(WORKSPACE).as_posix() for paths in eligible.values() for p in paths}
        actual = [row for split in module.SPLITS for row in self.rows[split]]
        self.assertEqual(len(eligible), 423)
        self.assertEqual(len(actual), 5985)
        self.assertEqual({row["image_path"] for row in actual}, expected)
        for row in actual:
            self.assertEqual(Path(row["image_path"]).parent.name, row["identity_name"])

    def test_reproducibility_mapping_and_summary(self):
        expected, mapping, summary = module.build(WORKSPACE, seed=42)
        self.assertEqual(self.rows, expected)
        with (self.output / "train_label_mapping.csv").open(newline="", encoding="utf-8") as stream:
            self.assertEqual(list(csv.DictReader(stream)), mapping)
        self.assertEqual(json.loads((self.output / "split_summary.json").read_text()), json.loads(json.dumps(summary)))
        self.assertEqual([summary["splits"][split]["identities"] for split in module.SPLITS], [296, 63, 64])

    def test_duplicate_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows["train"].append(rows["train"][0].copy())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            module.validate(WORKSPACE, rows)

    def test_missing_image_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows["train"][0]["image_path"] = "data/lfw/nonexistent.jpg"
        with self.assertRaisesRegex(ValueError, "Missing"):
            module.validate(WORKSPACE, rows)

    def test_identity_leakage_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows["validation"][0]["identity_id"] = rows["train"][0]["identity_id"]
        with self.assertRaisesRegex(ValueError, "Identity leakage"):
            module.validate(WORKSPACE, rows)

    def test_evaluation_training_label_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows["test"][0]["train_label"] = "0"
        with self.assertRaisesRegex(ValueError, "Evaluation identity"):
            module.validate(WORKSPACE, rows)

    def test_noncontiguous_label_rejected(self):
        rows = copy.deepcopy(self.rows)
        for row in rows["train"]:
            row["train_label"] = str(int(row["train_label"]) + 1)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            module.validate(WORKSPACE, rows)


if __name__ == "__main__":
    unittest.main()
