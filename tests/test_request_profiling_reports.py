import csv
import json
import tempfile
import unittest
from pathlib import Path

from zkinfer.profiling.collect_reports import collect
from zkinfer.profiling.overhead import write_transfer_overhead_report


class RequestProfilingReportsTest(unittest.TestCase):
    def test_reports_are_request_local_and_preserve_features_and_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rid in ("req_a", "req_b"):
                job = root / "requests" / rid / "artifacts" / "jobs" / "job_1"
                job.mkdir(parents=True)
                (job / "circuit_features.json").write_text(json.dumps({
                    "setup_key": "key",
                    "settings": {"run_args": {"logrows": 12}, "required_lookups": ["x"]},
                    "halo2": {"num_advice_columns": 4},
                }))
                (job / "resource_usage.csv").write_text(
                    "timestamp,type,memory_gb,cpu_machine_percent\n"
                    "1970-01-01T00:00:01.100000+00:00,process_tree,2,10\n"
                )
                (job / "execution_trace.csv").write_text(
                    "timestamp_unix_us,source,stage,event,duration_s,elapsed_s\n"
                    "1000000,halo2,PROVING::a,start,,\n"
                    "1200000,halo2,PROVING::a,end,0.2,\n"
                )
            self.assertEqual(len(collect(root)), 2)
            self.assertFalse((root / "reports").exists())
            for rid in ("req_a", "req_b"):
                reports = root / "requests" / rid / "reports"
                with (reports / "phase_report.csv").open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["request_id"], rid)
                self.assertEqual(rows[0]["job_id"], "job_1")
                self.assertEqual(float(rows[0]["duration_s"]), 0.2)
                self.assertEqual(float(rows[0]["rss_peak_gib"]), 2.0)
                with (reports / "circuit_report.csv").open() as f:
                    circuit = next(csv.DictReader(f))
                self.assertEqual(circuit["settings.run_args.logrows"], "12")
                self.assertEqual(circuit["halo2.num_advice_columns"], "4")
                write_transfer_overhead_report(str(reports))
                self.assertTrue((reports / "transfer_overhead_report.csv").exists())
                self.assertFalse((reports / "transfer_overhead_report.json").exists())
