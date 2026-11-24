import logging
import sys
import time

import grpc
import hydra
from omegaconf import DictConfig, OmegaConf

from zkinfer.client.api import ZKInferenceClient


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] submit: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("submit")


def runtime_partition_mode(cfg: DictConfig) -> str:
    return "fixed" if bool(cfg.execution.partitioning_enabled) else "none"


def wait_for_request(
    client: ZKInferenceClient,
    request_id: str,
    poll_interval_sec: float,
    logger: logging.Logger,
) -> None:
    """Print request-level progress until the request reaches a terminal state."""
    previous = None

    while True:
        status = client.get_request_status(request_id)
        snapshot = (
            status.status,
            status.total_jobs,
            status.completed_jobs,
            status.in_progress_jobs,
            status.queued_jobs,
            status.failed_jobs,
        )
        if snapshot != previous:
            logger.info(
                "Request %s | status=%s | completed=%d/%d | running=%d | queued=%d | failed=%d",
                request_id,
                status.status,
                status.completed_jobs,
                status.total_jobs,
                status.in_progress_jobs,
                status.queued_jobs,
                status.failed_jobs,
            )
            previous = snapshot

        if status.status == "completed":
            return
        if status.status in {"failed", "unknown"}:
            raise RuntimeError(
                f"Request {request_id} {status.status}: {status.message}"
            )

        time.sleep(max(0.1, poll_interval_sec))


@hydra.main(config_path="./config", config_name="runtime_experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = setup_logger()
    target = f"{cfg.launch.coordinator_host}:{cfg.launch.coordinator_port}"

    logger.info("Connecting to coordinator at %s", target)
    logger.info("Submitting workload:\n%s", OmegaConf.to_yaml(cfg.workload))
    logger.info("Execution config:\n%s", OmegaConf.to_yaml(cfg.execution))

    partition_size = int(cfg.execution.partition_size)
    if partition_size < 1:
        raise ValueError("execution.partition_size must be at least one")

    client = ZKInferenceClient(target=target, logger=logger)

    try:
        request_id = client.submit_inference_request(
            name=cfg.workload.name,
            onnx_model_path=cfg.workload.onnx_file,
            input_data_path=cfg.workload.input_file,
            split_mode=runtime_partition_mode(cfg),
            ops_per_chunk=partition_size,
            simplify_model=cfg.execution.get("simplify_model", False),
            input_shapes=cfg.workload.get("input_shapes", None),
        )
        logger.info("Submitted request %s", request_id)
        wait_for_request(
            client=client,
            request_id=request_id,
            poll_interval_sec=float(cfg.launch.poll_completion_interval_sec),
            logger=logger,
        )
        logger.info("Request %s completed", request_id)
    except grpc.RpcError as exc:
        logger.error("gRPC error: %s code=%s", exc.details(), exc.code())
        raise


if __name__ == "__main__":
    main()
