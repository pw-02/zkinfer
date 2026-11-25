import unittest

from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.models import InferenceRequest, JobStatus, NodeResources, ProofJob
from zkinfer.runtime.resource_profiles import update_resource_profile
from zkinfer.runtime.scheduler import JobScheduler


def make_job(request_id: str, name: str, duration: float, peak: float, artifact_key: str = "") -> ProofJob:
    empty = ArtifactRef.empty()
    job = ProofJob(
        inference_request_name=request_id,
        model_name=name,
        inference_request_id=request_id,
        model_ref=empty,
        input_ref=empty,
        setup_cache_ref=empty,
        output_prefix_ref=empty,
        predicted_duration_s=duration,
        predicted_peak_memory_gb=peak,
        artifact_key=artifact_key,
    )
    return job


class CoordinatorPlacementPolicyTests(unittest.TestCase):
    def test_resource_aware_leases_before_eventual_peak_is_currently_free(self):
        job = make_job("request-a", "a1", 60.0, 40.0)
        job.profiling_data = update_resource_profile(
            {},
            {
                "job_runtime(s)": 60.0,
                "max_process_memory(GB)": 40.0,
                "stage_generating_witness_max_memory(GB)": 6.0,
                "stage_proving_max_memory(GB)": 40.0,
            },
            setup_key="setup-a",
        )
        scheduler = JobScheduler("resource_aware")
        scheduler.add_jobs([job])

        leased = scheduler.next_job(NodeResources(
            node_id="node",
            memory_total_gb=64.0,
            memory_available_gb=10.0,
        ))
        self.assertIs(leased, job)

    def test_resource_aware_rejects_machine_infeasible_jobs(self):
        too_large = make_job("request-a", "large", 100.0, 80.0, artifact_key="cached")
        fits = make_job("request-a", "small", 10.0, 8.0, artifact_key="uncached")
        scheduler = JobScheduler("resource_aware")
        scheduler.add_jobs([too_large, fits])

        leased = scheduler.next_job(NodeResources(
            node_id="node",
            memory_total_gb=64.0,
            memory_available_gb=32.0,
        ))
        self.assertIs(leased, fits)

    def test_request_round_robin_precedes_duration_preference(self):
        cached = make_job("request-a", "a1", 10.0, 4.0, artifact_key="cached")
        other = make_job("request-b", "b1", 10.0, 4.0, artifact_key="uncached")
        scheduler = JobScheduler("resource_aware")
        scheduler.add_jobs([cached, other])
        node = NodeResources(
            node_id="node",
            memory_total_gb=64.0,
            memory_available_gb=32.0,
        )
        self.assertEqual(scheduler.next_job(node).inference_request_id, "request-a")
        self.assertEqual(scheduler.next_job(node).inference_request_id, "request-b")


class RequestProgressTests(unittest.TestCase):
    def test_work_progress_does_not_treat_cheap_jobs_as_equal_to_bottleneck(self):
        request = InferenceRequest(
            name="request",
            onnx_model_path="model.onnx",
            input_data_path="input.json",
            split_mode="fixed",
            ops_per_chunk=1,
            scheduler="resource_aware",
        )
        cheap = make_job(request.request_id, "cheap", 10.0, 1.0)
        bottleneck = make_job(request.request_id, "bottleneck", 90.0, 1.0)
        cheap.job_status = JobStatus.COMPLETED
        bottleneck.job_status = JobStatus.QUEUED
        request.proof_jobs = [cheap, bottleneck]

        self.assertEqual(request.compute_progress(), 50.0)
        self.assertEqual(request.compute_work_progress(), 10.0)
        self.assertEqual(request.predicted_remaining_work_s(), 90.0)
        self.assertEqual(request.predicted_remaining_bottleneck_s(), 90.0)

    def test_work_progress_falls_back_to_job_count_without_duration_history(self):
        request = InferenceRequest(
            name="request",
            onnx_model_path="model.onnx",
            input_data_path="input.json",
            split_mode="fixed",
            ops_per_chunk=1,
            scheduler="fifo",
        )
        first = make_job(request.request_id, "a", 0.0, 1.0)
        second = make_job(request.request_id, "b", 0.0, 1.0)
        first.job_status = JobStatus.COMPLETED
        second.job_status = JobStatus.QUEUED
        request.proof_jobs = [first, second]

        self.assertEqual(request.compute_work_progress(), 50.0)


if __name__ == "__main__":
    unittest.main()
