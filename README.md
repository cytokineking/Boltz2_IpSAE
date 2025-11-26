# Boltz2_IpSAE

Utilities for running ipSAE on Boltz co-folding predictions, including a
high-level pipeline script that takes binders and a target directly from the
command line (no manual YAML editing).

---

## 1. Overview

This folder contains:

- `ipsae.py` – the ipSAE calculator for AF2/AF3/Boltz structures.
- `visualise_binder_validation.py` – runs ipSAE on sets of Boltz predictions,
  aggregates results and makes plots/heatmaps.
- `make_binder_validation_scripts.py` – legacy helper that reads a config YAML
  and generates Boltz YAMLs and run scripts.
- `run_ipsae_pipeline.py` – **recommended** CLI wrapper that:
  - takes binders (CSV/FASTA/sequence) and a target (± antitarget, self),
  - runs Boltz predictions,
  - runs ipSAE per binder,
  - streams a compact summary CSV and prints per-binder ipSAE numbers,
  - produces the same global ipSAE heatmaps and `ipsae_summary_all_binders.csv`
    as the legacy flow.

The examples under `example_yaml/` provide a complete Nipah G use case
including known binders.

---

## 2. Prerequisites

You can install and use these scripts anywhere, as long as the `boltz` Python
package is installed and importable in your environment.

### 2.1. Install Boltz in your environment

Create and activate a Python virtualenv (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Boltz from PyPI (GPU: `[cuda]`, CPU-only: drop `[cuda]`):

```bash
python -m pip install --upgrade pip
python -m pip install "boltz[cuda]"
```

or from the main Boltz repo if you prefer:

```bash
git clone https://github.com/jwohlwend/boltz.git
cd boltz
python -m pip install -e .[cuda]
```

Install plotting libraries used by the ipSAE helpers:

```bash
python -m pip install seaborn matplotlib
```

Verify that the `boltz` CLI is available:

```bash
boltz --help
```

All commands below assume your virtualenv is activated and `boltz` is
installed in that environment.

### 2.2. Install Boltz2_IpSAE scripts

Clone this repository:

```bash
git clone https://github.com/cytokineking/Boltz2_IpSAE
cd Boltz2_IpSAE
```

From here you can run the helper scripts directly, e.g.:

```bash
python run_ipsae_pipeline.py --help
```

---

## 3. `run_ipsae_pipeline.py` – binder-by-binder pipeline

Run from the cloned `Boltz2_IpSAE` directory (or pass absolute/relative paths
to it); it does not depend on a specific folder layout beyond the paths you
provide.

### 3.1. Inputs

**Binders** (choose exactly one source):

- CSV:
  - `--binder_csv path/to/binders.csv`
  - `--binder_name_col binder_name` (default: `name`)
  - `--binder_seq_col binder_sequence` (default: `sequence`)
  - The CSV must have a header row; at minimum:

    ```text
    binder_name,binder_sequence
    2vsm,IVLEPIYWNSSN...
    6vy5,HEAVYSEQ:LIGHTSEQ
    ```

  - Sequences can be multi-chain by separating chains with `:` (e.g. heavy:light).
  - An example CSV using the bundled Nipah binders is provided:
    - `Boltz2_IpSAE/example_yaml/known_binders.csv`

- FASTA directory:
  - `--binder_fasta_dir path/to/binder_fastas/`
  - One FASTA file per binder; binder name from filename.

- Single binder from FASTA:
  - `--binder_fasta path/to/binder.fasta`
  - `--binder_name Binder1`

- Single binder from CLI sequence:
  - `--binder_seq "AA...AA:BB...BB"`
  - `--binder_name Binder1`

Optional binder tweak:

- `--add_n_terminal_lysine` – prepend `K` at N-terminus of each chain if missing.

**Target (required, single)**:

- `--target_name nipah_g`
- Either:
  - `--target_seq "QNYTRS..."` (chains separated by `:`), or
  - `--target_fasta Boltz2_IpSAE/example_yaml/nipah_g.fasta`
- Optional:
  - `--target_msa Boltz2_IpSAE/example_yaml/nipah.a3m` (applied to chain 0).

**Antitarget (optional, single)**:

- `--antitarget_name Sialidase_2F29`
- Either:
  - `--antitarget_seq "GSMASL..."`, or
  - `--antitarget_fasta Boltz2_IpSAE/example_yaml/sialidase_2F29.fasta`
- Optional:
  - `--antitarget_msa path/to/msa.a3m`

**Self-binding control (optional)**:

- `--include_self` – also run each binder against itself.

**Boltz options**:

- `--out_dir ./boltz_ipsae` – root output directory.
- `--recycling_steps 10`
- `--diffusion_samples 5`
- `--use_msa_server {auto,true,false}` (default `auto`).

**ipSAE options**:

- `--ipsae_pae_cutoff 15` (Å)
- `--ipsae_dist_cutoff 15` (Å)
- `--use_best_model` (affects global heatmap aggregation only)
- `--num_cpu 4` (used for the final global stage in `visualise_binder_validation`).

**Multi-GPU / resume options**:

- `--gpus`:
  - Omitted → sequential, single-worker behavior (one binder at a time).
  - `--gpus all` → use all available CUDA GPUs on the machine.
  - `--gpus 0,1,2` → use the listed GPU IDs, one worker process per GPU.
- `--resume`:
  - Reuses an existing `--out_dir` and detects which binders still need Boltz/ipSAE/summary work.
  - Fully completed binders (with predictions, `ipsae_summary.csv`, and a summary row) are skipped.
- `--overwrite`:
  - Deletes `--out_dir` if it exists and starts from scratch.
  - Mutually exclusive with `--resume`.

Logging:

- Default: clean stage/progress messages, logs written under `out_dir/logs/`.
- `--verbose`: stream full Boltz/ipSAE output to the terminal as well.

---

### 3.2. Nipah G example with bundled known binders

The repo includes:

- Nipah G FASTA: `Boltz2_IpSAE/example_yaml/nipah_g.fasta`
- Nipah G MSA: `Boltz2_IpSAE/example_yaml/nipah.a3m`
- Known binders: `Boltz2_IpSAE/example_yaml/known_binders/`
- Sialidase off-target FASTA: `Boltz2_IpSAE/example_yaml/sialidase_2F29.fasta`

Run the pipeline from the `Boltz2_IpSAE` directory:

```bash
python run_ipsae_pipeline.py \
  --binder_fasta_dir example_yaml/known_binders \
  --target_name nipah_g \
  --target_fasta example_yaml/nipah_g.fasta \
  --target_msa example_yaml/nipah.a3m \
  --antitarget_name Sialidase_2F29 \
  --antitarget_fasta example_yaml/sialidase_2F29.fasta \
  --include_self \
  --out_dir example_yaml/boltz_ipsae_nipah \
  --recycling_steps 10 \
  --diffusion_samples 5 \
  --use_msa_server auto \
  --ipsae_pae_cutoff 15 \
  --ipsae_dist_cutoff 15 \
  --num_cpu 4
```

Alternatively, you can drive the same example from the **CSV** of binders:

```bash
python run_ipsae_pipeline.py \
  --binder_csv example_yaml/known_binders.csv \
  --binder_name_col binder_name \
  --binder_seq_col binder_sequence \
  --target_name nipah_g \
  --target_fasta example_yaml/nipah_g.fasta \
  --target_msa example_yaml/nipah.a3m \
  --antitarget_name Sialidase_2F29 \
  --antitarget_fasta example_yaml/sialidase_2F29.fasta \
  --include_self \
  --out_dir example_yaml/boltz_ipsae_nipah_from_csv \
  --recycling_steps 10 \
  --diffusion_samples 5 \
  --use_msa_server auto \
  --ipsae_pae_cutoff 15 \
  --ipsae_dist_cutoff 15 \
  --num_cpu 4
```

For each binder the script:

1. Runs Boltz for binder vs Nipah G / antitarget / self.
2. Runs ipSAE on those predictions.
3. Prints binder‑level ipSAE summaries (target and antitarget).
4. Appends one row to:
   - `example_yaml/boltz_ipsae_nipah/summary/binder_pair_summary.csv`

After all binders are processed, it also writes:

- Per-binder CSVs/plots:
  - `boltz_ipsae_nipah/binder_*/plots/ipsae_summary.csv`
- Global data/heatmaps:
  - `boltz_ipsae_nipah/summary/ipsae_summary_all_binders.csv`
  - `boltz_ipsae_nipah/summary/ipSAE_min_heatmap.csv`
  - corresponding PNG/SVG heatmaps.

---

## 4. Modal Cloud Deployment

For serverless GPU execution on Modal's cloud infrastructure, use
`modal_boltz_ipsae.py`. This provides the same functionality as
`run_ipsae_pipeline.py` but runs on Modal's cloud GPUs with automatic
parallel processing.

### 4.1. Setup

Install Modal and authenticate:

```bash
pip install modal
modal token new
```

Initialize the Boltz model cache (run once):

```bash
modal run modal_boltz_ipsae.py::init_cache
```

### 4.2. Basic Usage

**Single binder from sequence:**

```bash
modal run modal_boltz_ipsae.py::run_pipeline \
  --binder-name "my_binder" \
  --binder-seq "MKTAYIAKQRQISFVK..." \
  --target-name nipah_g \
  --target-fasta example_yaml/nipah_g.fasta \
  --output-dir ./results
```

**Multiple binders from CSV:**

```bash
modal run modal_boltz_ipsae.py::run_pipeline \
  --binder-csv binders.csv \
  --binder-name-col binder_name \
  --binder-seq-col binder_sequence \
  --target-name nipah_g \
  --target-fasta example_yaml/nipah_g.fasta \
  --target-msa example_yaml/nipah.a3m \
  --output-dir ./results
```

**Full pipeline with antitarget and self-binding:**

```bash
modal run modal_boltz_ipsae.py::run_pipeline \
  --binder-csv binders.csv \
  --target-name nipah_g \
  --target-fasta example_yaml/nipah_g.fasta \
  --target-msa example_yaml/nipah.a3m \
  --antitarget-name sialidase \
  --antitarget-fasta example_yaml/sialidase_2F29.fasta \
  --include-self \
  --output-dir ./results
```

### 4.3. CLI Options

The Modal pipeline supports all options from `run_ipsae_pipeline.py`:

**Binder inputs** (choose one):

| Option | Description |
|--------|-------------|
| `--binder-csv` | CSV file with binder name and sequence columns |
| `--binder-fasta` | Single FASTA file with multiple binders |
| `--binder-fasta-dir` | Directory of FASTA files (one binder per file) |
| `--binder-name` + `--binder-seq` | Single binder from CLI |

**Binder options:**

| Option | Description |
|--------|-------------|
| `--binder-name-col` | Column name for binder names in CSV (default: `name`) |
| `--binder-seq-col` | Column name for sequences in CSV (default: `sequence`) |
| `--add-n-terminal-lysine` | Prepend 'K' to each binder chain if missing |

**Target/Antitarget:**

| Option | Description |
|--------|-------------|
| `--target-name` | Name of target protein (required) |
| `--target-fasta` / `--target-seq` | Target sequence (one required) |
| `--target-msa` | Optional MSA file for target |
| `--antitarget-name` | Name of off-target protein |
| `--antitarget-fasta` / `--antitarget-seq` | Antitarget sequence |
| `--antitarget-msa` | Optional MSA file for antitarget |
| `--include-self` | Run each binder against itself |

**Boltz parameters:**

| Option | Default | Description |
|--------|---------|-------------|
| `--recycling-steps` | 10 | Boltz recycling iterations |
| `--diffusion-samples` | 5 | Number of structure samples |
| `--use-msa-server` | auto | MSA server usage (`auto`/`true`/`false`) |

**ipSAE parameters:**

| Option | Default | Description |
|--------|---------|-------------|
| `--pae-cutoff` | 15 | PAE cutoff in Å |
| `--dist-cutoff` | 15 | Distance cutoff in Å |

**GPU and parallelization:**

| Option | Default | Description |
|--------|---------|-------------|
| `--gpu` | A100-80GB | GPU type (see table below) |
| `--max-parallel` | 10 | Max concurrent containers |

**Output:**

| Option | Description |
|--------|-------------|
| `--output-dir` | Local directory to save results |
| `--verbose` | Show detailed Boltz/ipSAE output |

### 4.4. GPU Options

Modal supports various GPU types. **A100-80GB is the default** (good balance of
cost and performance for protein structure prediction).

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

List available GPUs:

```bash
modal run modal_boltz_ipsae.py::list_gpus
```

**GPU specification examples:**

```bash
--gpu H100           # H100 (fast)
--gpu A100-80GB      # A100 80GB (default)
--gpu L40S           # L40S (good cost/performance)
--gpu T4             # T4 (cheapest, for testing)
```

### 4.5. Parallel Processing

The Modal pipeline automatically processes binders in parallel:

- Each binder runs in its own container with dedicated GPU
- `--max-parallel` controls maximum concurrent containers
- Failures are handled gracefully (other binders continue)

Example with 8 parallel B200 GPUs:

```bash
modal run modal_boltz_ipsae.py::run_pipeline \
  --binder-csv binders.csv \
  --target-fasta target.fasta \
  --gpu B200 \
  --max-parallel 8 \
  --output-dir ./results
```

### 4.6. Output Format

The Modal pipeline produces output identical to `run_ipsae_pipeline.py`:

```
output_dir/
├── binder_pair_summary.csv          # Summary CSV (harmonized format)
├── binder_MyBinder1/
│   ├── metrics.json                 # Detailed metrics for all partners
│   ├── structures_target/           # CIF files for target predictions
│   │   ├── MyBinder1_vs_target_model_0.cif
│   │   ├── MyBinder1_vs_target_model_1.cif
│   │   └── ...
│   ├── structures_antitarget/       # CIF files for antitarget (if used)
│   └── structures_self/             # CIF files for self-binding (if used)
├── binder_MyBinder2/
│   └── ...
```

**Summary CSV columns** (matches `run_ipsae_pipeline.py`):

```
binder_name, n_target_models, n_antitarget_models, n_self_models,
target_ipSAE_mean, target_ipSAE_std, target_ipSAE_min_mean, ...,
target_ipTM_af_mean, target_ipTM_af_std, target_pDockQ2_mean, ...,
antitarget_ipSAE_mean, antitarget_ipSAE_std, ...,
self_ipSAE_mean, self_ipSAE_std, ...
```

### 4.7. Utility Commands

**Test Modal connection:**

```bash
modal run modal_boltz_ipsae.py::test_connection
```

**Convert FASTA to CSV:**

```bash
modal run modal_boltz_ipsae.py::convert_fasta_to_csv --fasta-file binders.fasta
```

### 4.8. Example: Nipah G with Known Binders

Run the bundled Nipah example on Modal with parallel processing:

```bash
modal run modal_boltz_ipsae.py::run_pipeline \
  --binder-fasta-dir example_yaml/known_binders \
  --target-name nipah_g \
  --target-fasta example_yaml/nipah_g.fasta \
  --target-msa example_yaml/nipah.a3m \
  --antitarget-name sialidase \
  --antitarget-fasta example_yaml/sialidase_2F29.fasta \
  --include-self \
  --gpu H100 \
  --max-parallel 8 \
  --output-dir ./nipah_modal_results
```

---

## 5. Output Metrics

The pipeline now reports comprehensive metrics for each binder-target pair:

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

## 6. Legacy helpers

The older `make_binder_validation_scripts.py` + `run_all_cofolding.sh` +
`visualise_binder_validation.py` workflow is still present for compatibility,
but `run_ipsae_pipeline.py` should usually be more convenient: it exposes the
same functionality from a single CLI entry point and streams binder‑level
summaries into a small CSV as the script progresses.
