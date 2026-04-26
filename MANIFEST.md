# Manifest

## Top-Level Documentation

- `README.md`: high-level overview and entry points.
- `QUICKSTART.md`: shortest setup and run sequence.
- `REPRODUCE.md`: step-by-step reproduction instructions.
- `REQUIREMENTS.md`: dependency notes.
- `OPEN-SCIENCE-MAPPING.md`: mapping from paper results to commands and files.
- `requirements.txt`: Python package requirements.

## Code

- `CODE/agent.py`, `agent_team.py`, `agent_research.py`, `llm.py`:
  agent wrappers used by the correctness experiment.
- `CODE/service.py`, `service_tpm.py`, `crypto.py`, `epoch.py`, `sealing.py`,
  `store.py`, `tpm_epoch.py`, `tpm_sealer.py`:
  current SAGE implementation and TPM-backed support code.
- `CODE/baselines/`:
  the five baselines plus the TPM-backed SAGE variant used by the TPM table.
- `CODE/experiments/`:
  experiment entry points and figure-generation code used by the artifact.

## Scripts

- `SCRIPTS/common.sh`: shared path and runner helpers.
- `SCRIPTS/run_smoke_test.sh`: reduced end-to-end software verification run.
- `SCRIPTS/run_main_software_eval.sh`: full software-path evaluation.
- `SCRIPTS/run_agent_eval.sh`: Ollama-backed correctness table.
- `SCRIPTS/run_appendix_eval.sh`: appendix KMS-availability table.
- `SCRIPTS/run_tpm_eval.sh`: TPM-overhead evaluation.
- `SCRIPTS/run_figures.sh`: figure regeneration from generated CSVs.
- `SCRIPTS/start_swtpm.sh`, `SCRIPTS/stop_swtpm.sh`: local TPM simulator helpers.
- `SCRIPTS/clean_artifact_outputs.sh`: cleanup of generated outputs and caches.

## Configs and Data Notes

- `CONFIGS/*.env`: fixed parameter sets used by the wrapper scripts.
- `DATA/README.md`: synthetic-workload and generated-input notes.

## Generated Outputs

- `RESULTS/README.md`: expected output layout.
