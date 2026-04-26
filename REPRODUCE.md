# Reproduction Guide

## Environment Setup

```bash
cd SAGE-Artifact
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Optional dependencies are documented in [`REQUIREMENTS.md`](REQUIREMENTS.md).

## Software-Only Smoke Test

```bash
bash SCRIPTS/run_smoke_test.sh
```

This reduced verification run executes the software-path experiment modules
with reduced parameters and writes outputs under `RESULTS/smoke/`.

## Main Software Evaluation

```bash
bash SCRIPTS/run_main_software_eval.sh
```

This script writes the following primary outputs:

- `RESULTS/main/table_restart_recovery.csv`
- `RESULTS/main/table_provenance_length.csv`
- `RESULTS/main/table_microbench.csv`
- `RESULTS/main/table_scope_scaling.csv`
- `RESULTS/main/figure_deletion_pressure.csv`
- `RESULTS/main/figure_concurrency_throughput.csv`
- `RESULTS/main/figure_storage_growth.csv`
- `RESULTS/main/figure_kms_latency.csv`
- `RESULTS/main/rollback_matrix.csv`

Temporary databases, sealed-key files, and other generated state are written
under `RESULTS/work/`.

## Agent Correctness Evaluation

```bash
bash SCRIPTS/run_agent_eval.sh
```

This produces:

- `RESULTS/main/table_agent_rollback.csv`

Requirements:

- a local Ollama server reachable at `http://localhost:11434`;
- the configured local model, by default `llama3.2`.

## TPM Evaluation

With `swtpm`:

```bash
bash SCRIPTS/run_tpm_eval.sh --sim
```

With a real TPM or fTPM:

```bash
bash SCRIPTS/run_tpm_eval.sh --real
```

This produces:

- `RESULTS/main/table_tpm_overhead.csv`

## Appendix Availability Evaluation

```bash
bash SCRIPTS/run_appendix_eval.sh
```

This produces:

- `RESULTS/appendix/table_kms_availability.csv`

## Figure Regeneration

```bash
bash SCRIPTS/run_figures.sh
```

This reads whichever result CSVs are present and writes plots under
`RESULTS/figures/`. Missing inputs are skipped cleanly.

## Exact Parameters

The wrapper scripts source fixed parameter snapshots from:

- `CONFIGS/paper-defaults.env`
- `CONFIGS/smoke-defaults.env`
- `CONFIGS/appendix-defaults.env`
- `CONFIGS/tpm-defaults.env`

Those files are the authoritative record of the command-line settings used by
the artifact wrappers.

## Cleanup

To remove generated outputs, logs, caches, and Python bytecode:

```bash
bash SCRIPTS/clean_artifact_outputs.sh
```
