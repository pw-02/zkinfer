"""Compatibility entrypoint for running a node supervisor.

The runtime no longer treats a worker process as the distributed scheduling unit.
Run this module or zkinfer.runtime.node.supervisor to start one node-level agent
per machine/instance. The node agent manages artifact transfers, resource admission,
and stateless prover task subprocesses.
"""

from zkinfer.runtime.node.supervisor import main


if __name__ == "__main__":
    main()
