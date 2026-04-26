# Paper Result Mapping

The table below maps the paper’s evaluation outputs to artifact commands and
primary generated files.

| Paper result | Artifact command | Primary output |
| --- | --- | --- |
| Transitive deletion under rollback | `bash SCRIPTS/run_agent_eval.sh` | `RESULTS/main/table_agent_rollback.csv` |
| Restart resilience across restarts | `bash SCRIPTS/run_main_software_eval.sh` | `RESULTS/main/table_restart_recovery.csv` |
| Provenance chain length | `bash SCRIPTS/run_main_software_eval.sh` | `RESULTS/main/table_provenance_length.csv` |
| Per-operation microbenchmark | `bash SCRIPTS/run_main_software_eval.sh` | `RESULTS/main/table_microbench.csv` |
| Scope scaling | `bash SCRIPTS/run_main_software_eval.sh` | `RESULTS/main/table_scope_scaling.csv` |
| Deletion-pressure raw data | `bash SCRIPTS/run_main_software_eval.sh` | `RESULTS/main/figure_deletion_pressure.csv` |
| Concurrency-throughput raw data | `bash SCRIPTS/run_main_software_eval.sh` | `RESULTS/main/figure_concurrency_throughput.csv` |
| Storage-growth raw data | `bash SCRIPTS/run_main_software_eval.sh` | `RESULTS/main/figure_storage_growth.csv` |
| KMS-latency raw data | `bash SCRIPTS/run_main_software_eval.sh` | `RESULTS/main/figure_kms_latency.csv` |
| TPM overhead | `bash SCRIPTS/run_tpm_eval.sh --sim` or `bash SCRIPTS/run_tpm_eval.sh --real` | `RESULTS/main/table_tpm_overhead.csv` |
| KMS availability appendix table | `bash SCRIPTS/run_appendix_eval.sh` | `RESULTS/appendix/table_kms_availability.csv` |
| Figures from generated CSVs | `bash SCRIPTS/run_figures.sh` | `RESULTS/figures/*.png` |

Notes:

- `run_main_software_eval.sh` also writes `RESULTS/main/rollback_matrix.csv`,
  which is an auxiliary rollback-check output.
- The wrapper scripts use the parameter snapshots in `CONFIGS/`.
