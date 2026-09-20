import csv
import json
import tempfile
import unittest
from pathlib import Path

from otto.evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def run_case(self, labels, predictions, omit=(), extra=()):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_file = root / "labels.jsonl"
            labels_file.write_text("".join(json.dumps(x) + "\n" for x in labels))
            prediction_file = root / "predictions.csv"
            with prediction_file.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["session_type", "labels"])
                for record in labels:
                    for kind in ("clicks", "carts", "orders"):
                        key = f'{record["session"]}_{kind}'
                        if key not in omit:
                            writer.writerow([key, " ".join(map(str, predictions.get(key, [])))])
                writer.writerows(extra)
            return evaluate(labels_file, prediction_file)

    def fixture(self):
        return [{"session": 1, "labels": {"clicks": 1, "carts": [2], "orders": [3]}}]

    def test_hand_calculated_micro_recall(self):
        labels = self.fixture() + [
            {"session": 2, "labels": {"clicks": 4, "carts": [4, 5, 6], "orders": [7]}}
        ]
        result = self.run_case(labels, {"1_clicks": [1], "1_carts": [2], "1_orders": [3]})
        self.assertEqual(result["tasks"]["carts"]["recall@20"], 0.25)
        self.assertAlmostEqual(result["score"], 0.425)

    def test_top20_truncated_before_deduplication(self):
        result = self.run_case(self.fixture(), {"1_clicks": [9] * 20 + [1]})
        self.assertEqual(result["tasks"]["clicks"]["hits"], 0)

    def test_denominator_capped_at20_and_truth_deduplicated(self):
        labels = self.fixture()
        labels[0]["labels"]["orders"] = list(range(30)) + [0]
        result = self.run_case(labels, {"1_orders": list(range(20))})
        self.assertEqual(result["tasks"]["orders"]["denominator"], 20)
        self.assertEqual(result["tasks"]["orders"]["recall@20"], 1)

    def test_missing_predictions_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing prediction"):
            self.run_case(self.fixture(), {}, omit=["1_orders"])

    def test_unknown_session_rejected(self):
        with self.assertRaisesRegex(ValueError, "absent from labels"):
            self.run_case(self.fixture(), {}, extra=[("2_clicks", "1")])

    def test_duplicate_prediction_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_case(self.fixture(), {}, extra=[("1_clicks", "1")])

    def test_multiple_future_clicks_rejected(self):
        labels = self.fixture()
        labels[0]["labels"]["clicks"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "next click"):
            self.run_case(labels, {})

    def test_empty_task_not_reported_as_zero(self):
        labels = self.fixture()
        labels[0]["labels"]["orders"] = []
        with self.assertRaisesRegex(ValueError, "no targets for orders"):
            self.run_case(labels, {})

    def test_click_item_zero_is_a_valid_target(self):
        labels = self.fixture()
        labels[0]['labels']['clicks'] = 0
        result = self.run_case(labels, {'1_clicks': [0]})
        self.assertEqual(result['tasks']['clicks']['denominator'], 1)
        self.assertEqual(result['tasks']['clicks']['hits'], 1)


if __name__ == "__main__":
    unittest.main()
