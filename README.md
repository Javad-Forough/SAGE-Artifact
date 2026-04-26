# Anonymous Artifact for the SAGE Evaluation

This bundle contains the code, experiment drivers, configurations, and
documentation needed to reproduce the paper's evaluation.

The bundle contains:

- the SAGE implementation;
- all five baselines used in the paper: `plain`, `static`, `sealed_no_rp`,
  `sqlite_envelope`, and `kms`;
- experiment entry points used to generate the paper’s tables and figures;
- synthetic workload generators embedded in the experiment code;
- wrapper scripts, parameter snapshots, and reproduction notes.

No external datasets, cloud accounts, or private files are required.

Terminology note: in the artifact code and generated CSVs, `forget`
corresponds to the `delete` operation in the paper. For example,
`forget_scope`, `forget_latency`, and `forget_rate_hz` implement or measure
the paper's delete semantics.

## Fast Start

```bash
cd SAGE-Artifact
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
bash SCRIPTS/run_smoke_test.sh
```

## Main Commands

- Software-only main evaluation:
  `bash SCRIPTS/run_main_software_eval.sh`
- LLM-backed correctness table:
  `bash SCRIPTS/run_agent_eval.sh`
- TPM-backed evaluation with the software TPM simulator:
  `bash SCRIPTS/run_tpm_eval.sh --sim`
- TPM-backed evaluation with a real TPM or fTPM:
  `bash SCRIPTS/run_tpm_eval.sh --real`
- Figure regeneration from generated CSVs:
  `bash SCRIPTS/run_figures.sh`

## Directory Guide

- [`CODE/`](CODE): implementation, baselines, and experiment modules.
- [`SCRIPTS/`](SCRIPTS): reviewer-facing runner and cleanup scripts.
- [`CONFIGS/`](CONFIGS): parameter snapshots used by the wrapper scripts.
- [`DATA/`](DATA): notes on synthetic workloads and generated inputs.
- [`RESULTS/`](RESULTS): generated outputs only.
- [`QUICKSTART.md`](QUICKSTART.md): minimal setup and run sequence.
- [`REPRODUCE.md`](REPRODUCE.md): step-by-step reproduction instructions.
- [`OPEN-SCIENCE-MAPPING.md`](OPEN-SCIENCE-MAPPING.md): mapping from paper
  results to commands and output files.
