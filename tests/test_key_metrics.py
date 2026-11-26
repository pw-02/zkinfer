import tempfile
import unittest
from pathlib import Path

from zkinfer.profiling.key_metrics import summarize_key_metrics


class KeyMetricsTests(unittest.TestCase):
    def parse(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "halo2_key_metrics.csv"
            path.write_text("component,operation,bytes,duration_s\n" + rows)
            return summarize_key_metrics(path)

    def test_compact(self):
        result = self.parse(
            "fixed_lagrange,serialize,100,2\n"
            "pk_total,serialize_total,106,3\n"
            "fixed_lagrange,deserialize,100,4\n"
            "fixed_coset,reconstruct,,5\n"
            "compact,materialize,,10\n"
            "pk_total,materialize,106,11\n"
        )
        self.assertEqual(result["pk_representation"], "compact")
        self.assertEqual(result["pk_serialization_time(s)"], 3)
        self.assertEqual(result["pk_deserialization_time(s)"], 4)
        self.assertEqual(result["pk_reconstruction_time(s)"], 5)
        self.assertEqual(result["pk_materialization_time(s)"], 11)
        self.assertEqual(result["pk_deserialized_bytes"], 100)
        self.assertEqual(result["pk_materialized_file_bytes"], 106)

    def test_cached_full(self):
        result = self.parse("full,materialize,,4\npk_total,materialize,100,5\n")
        self.assertEqual(result["pk_representation"], "full")
        self.assertEqual(result["pk_reconstruction_time(s)"], 0)
        self.assertNotIn("pk_serialization_time(s)", result)

    def test_historical_or_partial(self):
        result = self.parse("fixed_coset,serialize,100,2\n")
        self.assertEqual(result["pk_component_serialization_time(s)"], 2)
        self.assertNotIn("pk_materialization_time(s)", result)
        self.assertNotIn("pk_reconstruction_time(s)", result)

    def test_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(summarize_key_metrics(Path(tmp) / "missing.csv"), {})
