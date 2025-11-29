# Boltz2_IpSAE

Utilities for running ipSAE on Boltz co-folding predictions, including a
high-level pipeline script that takes binders and a target directly from the
command line.

---

## 1. Overview

This folder contains:

- `ipsae.py` – the ipSAE calculator for AF2/AF3/Boltz structures.
- `run_ipsae_pipeline.py` – **recommended local CLI** that runs Boltz + ipSAE.
- `modal_boltz_ipsae.py` – **cloud version** for serverless GPU execution on Modal.
- `visualise_binder_validation.py` – runs ipSAE on sets of Boltz predictions,
  aggregates results and makes plots/heatmaps.
- `make_binder_validation_scripts.py` – legacy helper for YAML-based workflows.

The examples under `example_yaml/` provide a complete Nipah G use case
including known binders.

---

## 2. Quick Start

**Simplest case** – single binder against a target:

```bash
python run_ipsae_pipeline.py \
  --binder_seq "MKTAYIAKQRQISFVK..." \
  --binder_name my_binder \
  --target_seq "QNYTRS..." \
  --target_name my_target \
  --out_dir ./results
```

**From CSV with multiple binders:**

```bash
python run_ipsae_pipeline.py \
  --binder_csv binders.csv \
  --target_name nipah_g \
  --target_fasta example_yaml/nipah_g.fasta \
  --target_msa example_yaml/nipah.a3m \
  --out_dir ./results
```

For cloud execution with parallel GPUs, see [Section 7: Modal Cloud Deployment](#7-modal-cloud-deployment-modal_boltz_ipsaepy).

---

## 3. Prerequisites

### 3.1. Install Boltz

Create and activate a Python virtualenv (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Boltz from PyPI (GPU: `[cuda]`, CPU-only: drop `[cuda]`):

```bash
pip install --upgrade pip
pip install "boltz[cuda]"
```

Install plotting libraries used by the ipSAE helpers:

```bash
pip install seaborn matplotlib
```

Verify that the `boltz` CLI is available:

```bash
boltz --help
```

### 3.2. Install Boltz2_IpSAE scripts

Clone this repository:

```bash
git clone https://github.com/cytokineking/Boltz2_IpSAE
cd Boltz2_IpSAE
```

From here you can run the helper scripts directly:

```bash
python run_ipsae_pipeline.py --help
```

---

## 4. CLI Options Reference

Both `run_ipsae_pipeline.py` (local) and `modal_boltz_ipsae.py` (cloud) support
the same core options.

> **Note:** Local uses underscores (`--target_name`), Modal uses dashes (`--target-name`).

### 4.1. Binder Inputs (choose one)

| Option | Description |
|--------|-------------|
| `--binder_csv` | CSV file with binder name and sequence columns |
| `--binder_fasta_dir` | Directory of FASTA files (one binder per file) |
| `--binder_fasta` + `--binder_name` | Single FASTA file |
| `--binder_seq` + `--binder_name` | Single sequence from CLI |

> **Multi-chain binders:** Separate chains with `:` (e.g., `"HEAVYSEQ:LIGHTSEQ"` for antibodies).

**CSV options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--binder_name_col` | `name` | Column name for binder names |
| `--binder_seq_col` | `sequence` | Column name for sequences |

**Binder tweaks:**

| Option | Description |
|--------|-------------|
| `--add_n_terminal_lysine` | Prepend 'K' to each binder chain if missing |

### 4.2. Target & Antitarget

| Option | Description |
|--------|-------------|
| `--target_name` | Name of target protein (required) |
| `--target_fasta` / `--target_seq` | Target sequence (one required) |
| `--target_msa` | MSA file for target (**recommended** – significantly improves quality) |
| `--antitarget_name` | Name of off-target protein |
| `--antitarget_fasta` / `--antitarget_seq` | Antitarget sequence |
| `--antitarget_msa` | Optional MSA file for antitarget |
| `--include_self` | Run each binder against itself |

> **Important:** Providing `--target_msa` with a precomputed MSA (e.g., from ColabFold
> or MMseqs2) significantly improves prediction quality. Without it, Boltz uses the
> MSA server which may produce different or lower-quality results.

### 4.3. Boltz Parameters

| Local Option | Modal Option | Default | Description |
|--------------|--------------|---------|-------------|
| `--recycling_steps` | `--recycling-steps` | 10 | Boltz recycling iterations |
| `--diffusion_samples` | `--diffusion-samples` | 5 | Number of structure samples |
| `--use_msa_server` | `--use-msa-server` | `auto` | MSA server usage (`auto`/`true`/`false`) |

### 4.4. ipSAE Parameters

| Local Option | Modal Option | Default | Description |
|--------------|--------------|---------|-------------|
| `--ipsae_pae_cutoff` | `--pae-cutoff` | 15 | PAE cutoff in Å |
| `--ipsae_dist_cutoff` | `--dist-cutoff` | 15 | Distance cutoff in Å |

---

## 5. Output Format

Both pipelines produce identical output:

```
output_dir/
├── binder_pair_summary.csv           # Summary CSV with all metrics
├── binder_MyBinder1/
│   ├── metrics.json                  # Detailed metrics for all partners
│   ├── structures_target/            # CIF files for target predictions
│   │   ├── MyBinder1_vs_target_model_0.cif
│   │   ├── MyBinder1_vs_target_model_1.cif
│   │   └── ...
│   ├── structures_antitarget/        # CIF files for antitarget (if used)
│   └── structures_self/              # CIF files for self-binding (if used)
├── binder_MyBinder2/
│   └── ...
```

### 5.1. Output Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **`ipSAE`** | Primary metric from `max` row (best direction) | **Primary ranking metric** |
| **`ipSAE_min`** | Minimum across asymmetric directions | Conservative filtering |
| **`ipSAE_max`** | Maximum across asymmetric directions | Optimistic estimate |
| **`ipTM_af`** | Native Boltz/AlphaFold interface confidence | Cross-validation |
| **`pDockQ2`** | Alternative interface score (Elofsson lab) | Additional validation |
| **`ipSAE_d0chn`** | ipSAE with d0 = sum of chain lengths | Alternative variant |
| **`ipSAE_d0dom`** | ipSAE with d0 = interface residue count | Alternative variant |

For each metric, the pipeline reports mean ± std across all models (typically 5).

---

## 6. Local Pipeline (`run_ipsae_pipeline.py`)

### 6.1. Local-Only Options

In addition to the common options above:

| Option | Description |
|--------|-------------|
| `--out_dir` | Root output directory |
| `--gpus` | GPU selection: omit for single GPU, `all` for all GPUs, or `0,1,2` for specific IDs |
| `--resume` | Reuse existing output dir, skip completed binders |
| `--overwrite` | Delete output dir and start fresh (mutually exclusive with `--resume`) |
| `--use_best_model` | Affects global heatmap aggregation only |
| `--num_cpu` | CPUs for final global stage (default: 4) |
| `--verbose` | Stream full Boltz/ipSAE output to terminal |

### 6.2. Example: Nipah G with Known Binders

```bash
python run_ipsae_pipeline.py \
  --binder_fasta_dir example_yaml/known_binders \
  --target_name nipah_g \
  --target_fasta example_yaml/nipah_g.fasta \
  --target_msa example_yaml/nipah.a3m \
  --antitarget_name Sialidase_2F29 \
  --antitarget_fasta example_yaml/sialidase_2F29.fasta \
  --include_self \
  --out_dir ./boltz_ipsae_nipah \
  --recycling_steps 10 \
  --diffusion_samples 5 \
  --ipsae_pae_cutoff 15 \
  --ipsae_dist_cutoff 15
```

---

## 7. Modal Cloud Deployment (`modal_boltz_ipsae.py`)

For serverless GPU execution on Modal's cloud infrastructure.

### 7.1. Setup

Install Modal and authenticate:

```bash
pip install modal
modal token new
```

Initialize the Boltz model cache (run once):

```bash
modal run modal_boltz_ipsae.py::init_cache
```

### 7.2. Modal-Only Options

In addition to the common options (using dashes instead of underscores):

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | None | Local directory to save results |
| `--gpu` | `A100-80GB` | GPU type (see table below) |
| `--max-parallel` | 10 | Max concurrent containers |
| `--verbose` | off | Show detailed Boltz/ipSAE output |
| `--no-stream` | off | Disable real-time streaming |
| `--run-id` | auto | Custom run ID for streaming |
| `--sync-interval` | 5.0 | Seconds between sync polls |

### 7.3. GPU Options

| GPU | VRAM | Cost |
|-----|------|------|
| `T4` | 16GB | $0.59/h |
| `L4` | 24GB | $0.80/h |
| `A10G` | 24GB | $1.10/h |
| `L40S` | 48GB | $1.95/h |
| `A100-40GB` | 40GB | $2.10/h |
| `A100-80GB` | 80GB | **$2.50/h (default)** |
| `H100` | 80GB | $3.95/h |
| `H200` | 141GB | $4.54/h |
| `B200` | 192GB | $6.25/h |

List available GPUs: `modal run modal_boltz_ipsae.py::list_gpus`

### 7.4. Real-Time Streaming

**Streaming is enabled by default.** Results are saved to your local filesystem
as each binder completes, rather than waiting for all predictions to finish.

| Behavior | Streaming ON (default) | Streaming OFF (`--no-stream`) |
|----------|------------------------|-------------------------------|
| Results saved | As each binder completes | After all binders complete |
| Network interruption | Partial results preserved | All results may be lost |
| Memory usage | Lower (results streamed out) | Higher (all in memory) |

If interrupted, resume syncing with: `modal run modal_boltz_ipsae.py::sync_results`

### 7.5. Example: Nipah G on Modal

The equivalent of the local Nipah example:

```bash
modal run modal_boltz_ipsae.py::run_pipeline \
  --binder-fasta-dir example_yaml/known_binders \
  --target-name nipah_g \
  --target-fasta example_yaml/nipah_g.fasta \
  --target-msa example_yaml/nipah.a3m \
  --antitarget-name sialidase \
  --antitarget-fasta example_yaml/sialidase_2F29.fasta \
  --include-self \
  --recycling-steps 10 \
  --diffusion-samples 5 \
  --pae-cutoff 15 \
  --dist-cutoff 15 \
  --gpu H100 \
  --max-parallel 8 \
  --output-dir ./nipah_modal_results
```

### 7.6. Utility Commands

| Command | Description |
|---------|-------------|
| `modal run modal_boltz_ipsae.py::test_connection` | Test Modal connection |
| `modal run modal_boltz_ipsae.py::list_gpus` | List available GPU types |
| `modal run modal_boltz_ipsae.py::convert_fasta_to_csv --fasta-file binders.fasta` | Convert FASTA to CSV |

---

## 8. Legacy Workflow

The older `make_binder_validation_scripts.py` + `run_all_cofolding.sh` +
`visualise_binder_validation.py` workflow is still present for compatibility,
but `run_ipsae_pipeline.py` should usually be more convenient.
