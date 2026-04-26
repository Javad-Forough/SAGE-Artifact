# Quickstart

## 1. Set Up Python

```bash
cd CCS-Artifact
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run a Smoke Test

```bash
bash SCRIPTS/run_smoke_test.sh
```

This runs reduced versions of the software-path experiments and writes outputs
under `RESULTS/smoke/`.

## 3. Run the Main Software Evaluation

```bash
bash SCRIPTS/run_main_software_eval.sh
```

This reproduces the software-only evaluation outputs used for the restart,
provenance, microbenchmark, deletion-pressure, concurrency, scope-scaling,
storage-growth, and KMS-latency results.

## 4. Optional: Run the Agent Correctness Table

This step requires a local Ollama server at `http://localhost:11434` and a
pulled model matching the default configuration (`llama3.2`).

```bash
bash SCRIPTS/run_agent_eval.sh
```

## 5. Optional: Run the TPM Evaluation

With `swtpm` simulation:

```bash
bash SCRIPTS/run_tpm_eval.sh --sim
```

With a real TPM or fTPM, make sure `tpm2-tools` can reach it and run:

```bash
bash SCRIPTS/run_tpm_eval.sh --real
```

## 6. Regenerate Figures

```bash
bash SCRIPTS/run_figures.sh
```
