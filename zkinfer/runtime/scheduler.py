from collections import deque
from typing import Iterable, List, Optional

from zkinfer.runtime.models import NodeResources, ProofJob
from zkinfer.runtime.resource_profiles import max_stage_memory_gb


class JobScheduler:
    """Request-fair coordinator scheduler for whole-job placement.

    The coordinator decides which queued proof job should be leased to a polling
    node. It deliberately does not perform stage-level packing. ``resource_aware``
    uses historical memory only to reject jobs whose predicted peak cannot fit on
    the machine at all. Once a job is leased, the node-local admission controller
    decides when memory-intensive stages may execute using live node state.

    Cache locality is a preference among already feasible jobs, never a reason to
    override a memory-safety check.
    """

    def __init__(self, policy: str = "fifo"):
        self.policy = (policy or "fifo").lower()
        self.policy = {"artifact_aware": "cache_aware"}.get(self.policy, self.policy)
        if self.policy not in {"fifo", "lpt", "resource_aware", "cache_aware"}:
            raise ValueError(f"Unsupported scheduling policy: {self.policy}")
        self._queue: List[ProofJob] = []
        self._request_order: deque[str] = deque()

    def add_jobs(self, jobs: Iterable[ProofJob]) -> None:
        for job in jobs:
            self.requeue(job)

    def next_job(self, node: Optional[NodeResources] = None) -> Optional[ProofJob]:
        if not self._queue:
            return None
        node = node or NodeResources(node_id="unknown")
        candidate_index = self._select_index(node)
        if candidate_index is None:
            return None
        job = self._queue.pop(candidate_index)
        request_id = job.inference_request_id
        self._request_order.remove(request_id)
        if any(item.inference_request_id == request_id for item in self._queue):
            self._request_order.append(request_id)
        return job

    def requeue(self, job: ProofJob) -> None:
        if job.inference_request_id not in self._request_order:
            self._request_order.append(job.inference_request_id)
        self._queue.append(job)

    def _select_index(self, node: NodeResources) -> Optional[int]:
        feasible = [
            (idx, job)
            for idx, job in enumerate(self._queue)
            if self._fits_node(job, node)
        ]
        if not feasible:
            return None

        request_ids = {job.inference_request_id for _, job in feasible}
        chosen_request = next(
            request_id for request_id in self._request_order if request_id in request_ids
        )
        candidates = [
            item for item in feasible if item[1].inference_request_id == chosen_request
        ]

        if self.policy == "fifo":
            return min(candidates, key=lambda item: item[0])[0]

        if self.policy == "lpt":
            return max(
                candidates,
                key=lambda item: (item[1].predicted_duration_s or 0.0, -item[0]),
            )[0]

        cached = set(node.cached_setup_keys or []) if self.policy == "cache_aware" else set()

        def score(item):
            idx, job = item
            cache_hit = 1 if job.artifact_key in cached else 0
            duration = max(0.0, float(job.predicted_duration_s or 0.0))
            return (cache_hit, duration, -idx)

        return max(candidates, key=score)[0]

    @staticmethod
    def _profiled_peak_memory_gb(job: ProofJob) -> float:
        predicted_peak = max_stage_memory_gb(job.profiling_data)
        if predicted_peak <= 0.0:
            predicted_peak = max(0.0, float(job.predicted_peak_memory_gb or 0.0))
        return predicted_peak

    def _fits_node(self, job: ProofJob, node: NodeResources) -> bool:
        if self.policy != "resource_aware":
            # Baseline policies retain their original conservative whole-job
            # admission using the coordinator's attached peak estimate.
            required_memory = max(0.0, float(job.predicted_peak_memory_gb or 0.0))
            if required_memory > 0.0:
                return required_memory <= node.memory_available_gb
            return True

        predicted_peak = self._profiled_peak_memory_gb(job)

        # Do not lease known memory-consuming work to a node reporting no usable
        # memory. This also closes the short interval before local provisional
        # reservations become visible at the coordinator.
        if predicted_peak > 0.0 and node.memory_available_gb <= 0.0:
            return False

        # The coordinator performs only the hard machine-feasibility check. It
        # may lease a job whose eventual peak is larger than currently free RAM;
        # the node-local controller will hold that stage until it is safe.
        if predicted_peak > 0.0 and node.memory_total_gb > 0.0:
            return predicted_peak <= node.memory_total_gb

        # Older/partial node reports may omit total memory. In that case avoid
        # guessing that a known peak fits unless it is currently available.
        if predicted_peak > 0.0:
            return predicted_peak <= node.memory_available_gb
        return True

    def clear(self) -> None:
        self._queue.clear()
        self._request_order.clear()

    def empty(self) -> bool:
        return not self._queue

    def __len__(self) -> int:
        return len(self._queue)
