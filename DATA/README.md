# Synthetic Workloads

This artifact requires no external datasets.

All workloads are generated at runtime by the experiment modules:

- `experiments/agent_correctness_eval.py` defines the personal, team, and
  research correctness scenarios inline.
- `experiments/concurrency_throughput_eval.py` defines assistant,
  chat, and research workload profiles in code.
- `experiments/storage_growth_eval.py` defines the assistant, chat, and
  research storage-growth workloads in code.
- `experiments/deletion_pressure_eval.py`, `experiments/restart_recovery_eval.py`,
  and `experiments/provenance_depth_eval.py` generate their own inputs.

No hidden datasets, downloaded corpora, or private prompts are needed.
