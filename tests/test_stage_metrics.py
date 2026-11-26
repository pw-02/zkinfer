import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from zkinfer.backends.ezkl.ezkl_prover import EZKLProofStages
from zkinfer.profiling.reports import _safe_max, write_request_report
from zkinfer.profiling.resource_parser import parse_resource_usage_file
from zkinfer.runtime.models import InferenceRequest
from zkinfer.runtime.node.prover_task import ProverTaskRunner


class StageMetricsTests(unittest.TestCase):
    def test_request_max_ignores_missing_and_nonfinite_values_in_any_order(self):
        for values in ([None, math.nan, 'NaN', math.inf, '-inf', '', 1.5, 3.0],
                       [3.0, '', None, math.nan, 1.5]):
            self.assertEqual(_safe_max({'memory': values}, 'memory'), 3.0)
        self.assertIsNone(_safe_max({'memory': [None, math.nan, '', math.inf]}, 'memory'))
        self.assertEqual(_safe_max({'memory': [None, 0]}, 'memory'), 0)

    def test_request_csv_max_uses_valid_jobs_and_reports_coverage(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / 'job_report.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['stage_witness_max_memory(GB)',
                                                          'stage_compile_max_memory(GB)'])
                writer.writeheader()
                for value in ('', 2, 'nan', 4, 'inf'):
                    writer.writerow({'stage_witness_max_memory(GB)': value,
                                     'stage_compile_max_memory(GB)': ''})
            request = InferenceRequest('model', 'model.onnx', 'input.json', 'none', 1, 'fifo')
            report = write_request_report(request, str(root))
            self.assertEqual(report['max_stage_witness_max_memory(GB)'], 4)
            self.assertEqual(report['num_jobs_with_stage_witness_memory_samples'], 2)
            self.assertIsNone(report['max_stage_compile_max_memory(GB)'])
            self.assertEqual(report['num_jobs_with_stage_compile_memory_samples'], 0)
            with (root / 'request_report.csv').open() as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row['max_stage_compile_max_memory(GB)'], '')
            self.assertNotIn('nan', row.values())

    def test_unsampled_stage_is_unknown_while_observed_zero_remains_zero(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'resource_usage.csv').write_text(
                'type,stage,memory_gb,cpu_percent,cpu_machine_percent\n'
                'process_tree,PROVING,0,0,0\n'
                'process_tree,KEY_GEN,,25,\n'
                'process_tree,KEY_GEN,nan,,inf\n'
                'system,,nan,,\n'
            )
            (root / 'stage_transitions.csv').write_text(
                'stage,event\nKEY_GEN,enter\nGENERATING_WITNESS,enter\nPROVING,enter\n'
            )
            metrics = parse_resource_usage_file(str(root / 'resource_usage.csv'))
            self.assertIsNone(metrics['stage_generating_witness_max_memory(GB)'])
            self.assertEqual(metrics['stage_generating_witness_memory_samples'], 0)
            self.assertEqual(metrics['stage_generating_witness_samples'], 0)
            self.assertIsNone(metrics['stage_key_gen_max_memory(GB)'])
            self.assertEqual(metrics['stage_key_gen_memory_samples'], 0)
            self.assertEqual(metrics['stage_key_gen_avg_cpu_raw(%)'], 25)
            self.assertIsNone(metrics['stage_key_gen_avg_cpu_machine(%)'])
            self.assertEqual(metrics['stage_proving_max_memory(GB)'], 0)
            self.assertEqual(metrics['stage_proving_memory_samples'], 1)
            self.assertEqual(metrics['stage_proving_avg_cpu_machine(%)'], 0)
            self.assertIsNone(metrics['max_system_memory(GB)'])

    def test_empty_resource_trace_has_unknown_peak_and_cpu(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'resource_usage.csv'
            path.write_text('type,stage,memory_gb,cpu_percent,cpu_machine_percent\n')
            metrics = parse_resource_usage_file(str(path))
            self.assertTrue(all(value is None for value in metrics.values()))

    def test_prediction_error_does_not_invent_unsampled_measurements(self):
        estimates = {'stage_admission_witness_predicted_memory(GB)': 5}
        metrics = ProverTaskRunner._prediction_error_metrics({}, estimates)
        self.assertIsNone(metrics['stage_admission_witness_actual_memory(GB)'])
        self.assertNotIn('stage_admission_witness_memory_error(GB)', metrics)
        metrics = ProverTaskRunner._prediction_error_metrics(
            {'stage_witness_max_memory(GB)': 0}, estimates)
        self.assertEqual(metrics['stage_admission_witness_actual_memory(GB)'], 0)
        self.assertEqual(metrics['stage_admission_witness_memory_error(GB)'], 5)
        self.assertNotIn('stage_admission_witness_memory_abs_pct_error', metrics)

    def run_path(self, warm=False, setup_only=False):
        stages = EZKLProofStages.__new__(EZKLProofStages)
        events = []
        stages._update_status = events.append
        stages.srs_path = 'srs'
        stages.ezkl_settings_dict = {}
        stages.calibrate_settings_with_fallback = Mock(return_value=(False, 0, 0))
        stages.compile_circuit = Mock(return_value=(False, 0, 0))
        stages.get_srs = Mock()
        stages.gen_witness = Mock()
        def keys():
            events.append('KEY_CACHE_HIT' if warm else 'KEY_GEN')
            return warm, 0, 0
        stages.gen_keys = Mock(side_effect=keys)
        stages.compute_proof = Mock(side_effect=lambda: events.append('PROVING'))
        metrics, _ = stages.run_all(setup_only=setup_only)
        return stages, events, metrics

    def test_cold_run_generates_keys_before_proving(self):
        stages, events, metrics = self.run_path()
        self.assertEqual(events, ['KEY_GEN', 'PROVING', 'DONE'])
        self.assertFalse(metrics['ezkl_key_gen_used_cache'])

    def test_warm_run_reuses_keys_and_setup_only_does_not_prove(self):
        stages, events, metrics = self.run_path(warm=True)
        self.assertEqual(events, ['KEY_CACHE_HIT', 'PROVING', 'DONE'])
        self.assertTrue(metrics['ezkl_key_gen_used_cache'])
        stages, events, _ = self.run_path(setup_only=True)
        stages.gen_keys.assert_called_once()
        stages.compute_proof.assert_not_called()
