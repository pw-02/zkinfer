import csv
import tempfile
import unittest
from pathlib import Path

from zkinfer.profiling.halo2_phases import phase_rows, write_phase_report


class PhaseResourceTest(unittest.TestCase):
    def test_exact_intervals_and_missing_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "resource_usage.csv"
            events = root / "halo2_stage_transitions.csv"
            samples.write_text(
                "timestamp,type,stage,memory_gb,cpu_machine_percent\n"
                "1970-01-01T00:00:01.100000+00:00,process_tree,PROVING,2,10\n"
                "1970-01-01T00:00:01.200000+00:00,system,PROVING,99,80\n"
                "1970-01-01T00:00:01.300000+00:00,process_tree,PROVING,4,30\n"
                "1970-01-01T00:00:01.500000+00:00,process_tree,PROVING,8,40\n"
            )
            events.write_text(
                "timestamp_unix_us,event,stage,duration_us,duration_s\n"
                "1000000,start,PROVING::a,,\n"
                "1500000,end,PROVING::a,500000,0.5\n"
                "1500000,start,PROVING::b,,\n"
                "1600000,end,PROVING::b,100000,0.1\n"
                "1700000,start,PROVING::a,,\n"
                "1800000,end,PROVING::a,100000,0.1\n"
                "1900000,start,PROVING::c,,\n"
            )
            rows = phase_rows(samples, events)
            self.assertEqual(rows[0]["rss_samples"], 2)
            self.assertEqual(rows[0]["rss_peak_gib"], 4)
            self.assertEqual(rows[0]["rss_mean_gib"], 3)
            self.assertEqual(rows[1]["rss_peak_gib"], 8)
            self.assertEqual(rows[2]["occurrence"], 2)
            self.assertIsNone(rows[2]["rss_peak_gib"])
            self.assertEqual(rows[3]["status"], "incomplete")
            summary = write_phase_report(root)
            self.assertEqual(summary["halo2_phase_complete_count"], 3)
            self.assertEqual(summary["halo2_phase_sampled_count"], 2)
            with (root / "halo2_phase_resources.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 4)

    def test_old_run_without_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(write_phase_report(tmp),
                             {"halo2_phase_trace_available": False})
