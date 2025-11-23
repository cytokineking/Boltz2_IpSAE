# ipSAE Pipeline Helper (`run_ipsae_pipeline.py`)

This script lets you run the full **Boltz → structure/PAE → ipSAE** pipeline
without manually writing Boltz YAML files. You provide:

- binder sequences (CSV, FASTA directory, or a single sequence),
- a target (sequence and optional MSA),
- optional antitarget and self-binding controls,

and the script:

1. Builds per-binder YAMLs for Boltz co-folding.  
2. Runs `boltz predict` for all binder–partner complexes.  
3. Runs `ipsae.py` via `visualise_binder_validation.py` to score interfaces.  
4. Writes per-binder summaries + plots and global ipSAE heatmaps/CSVs.

By default, it prints clear stage/progress messages and hides noisy Boltz/ipSAE
stdout. Use `--verbose` to see full subprocess output.

---

## 1. Prerequisites

Assume you cloned the main `boltz` repo to `/root/boltz`:

```bash
cd /root
git clone https://github.com/jwohlwend/boltz.git
cd boltz
```

Create and activate a Python virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Boltz (GPU: `[cuda]`, CPU-only: drop `[cuda]`):

```bash
python -m pip install --upgrade pip
python -m pip install -e .[cuda]
```

Install plotting libraries used by the ipSAE helpers:

```bash
python -m pip install seaborn matplotlib
```

Check that the CLI is available:

```bash
boltz --help
```

You should see Boltz usage information. All commands below assume your venv is
activated (`source .venv/bin/activate`).

---

## 2. Script overview

The script lives at:

- `Boltz2_IpSAE/run_ipsae_pipeline.py`

Basic help:

```bash
cd /root/boltz
python Boltz2_IpSAE/run_ipsae_pipeline.py --help
```

You must provide:

- **Binders** (one of):
  - `--binder_csv` + `--binder_name_col` + `--binder_seq_col`, or
  - `--binder_fasta_dir`, or
  - `--binder_fasta` + `--binder_name`, or
  - `--binder_seq` + `--binder_name`.
- **Target** (required):
  - `--target_name`, and
  - either `--target_seq` or `--target_fasta`,
  - optional `--target_msa`.
- **Optional antitarget**:
  - `--antitarget_name`, plus `--antitarget_seq` or `--antitarget_fasta`,
  - optional `--antitarget_msa`.
- **Optional self-binding control**:
  - `--include_self` to run binder vs binder.

Boltz/ipSAE knobs:

- `--out_dir` (root output directory; default `./boltz_ipsae`),
- `--recycling_steps`, `--diffusion_samples`,
- `--use_msa_server {auto,true,false}` (default `auto`),
- `--ipsae_pae_cutoff`, `--ipsae_dist_cutoff` (default `15 Å`),
- `--num_cpu` (for ipSAE), `--use_best_model` (heatmap aggregation),
- `--verbose` (stream full subprocess output).

Sequences can be multi-chain by separating chains with `:` (e.g.
`"CHAINA:CHAINB"`). The script assigns chain IDs:

- binder chains → `A`, `B`, `C`, …  
- target chains → `TA`, `TB`, …  
- antitarget chains → `AA`, `AB`, …  
- self chains → `SA`, `SB`, …

---

## 3. Nipah G example (using bundled known binders)

The repo already includes:

- Nipah G sequence and MSA:
  - sequence in `Boltz2_IpSAE/example_yaml/config.yaml` (under `targets: nipah_g`)
  - MSA: `Boltz2_IpSAE/example_yaml/nipah.a3m`
- Known binders:
  - per-binder FASTAs in `Boltz2_IpSAE/example_yaml/known_binders/`
- A sialidase off-target:
  - sequence in `Boltz2_IpSAE/example_yaml/config.yaml` (name `Sialidase_2F29`)

Below is a concrete Nipah G use case using those assets.

### 3.1. Create Nipah G and antitarget FASTA files (once)

For convenience, we’ll write short FASTA files from the sequences in the
example config. You can copy-paste the sequences from
`Boltz2_IpSAE/example_yaml/config.yaml` or reuse this snippet:

```bash
cd /root/boltz

cat > Boltz2_IpSAE/example_yaml/nipah_g.fasta << 'EOF'
>nipah_g
QNYTRSTDNQAVIKDALQGIQQQIKGLADKIGTEIGPKVSLIDTSSTITIPANIGLLGSKISQSTASINENVNEKCKFTLPPLKIHECNISCPNPLPFREYRPQTEGVSNLVGLPNNICLQKTSNQILKPKLISYTLPVVGQSGTCITDPLLAMDEGYFAYSHLERIGSCSRGVSKQRIIGVGEVLDRGDEVPSLFMTNVWTPPNPNTVYHCSAVYNNEFYYVLCAVSTVGDPILNSTYWSGSLMMTRLAVKPKSNGGGYNQHQLALRSIEKGRYDKVMPYGPSGIKQGDTLYFPAVGFLVRTEFKYNDSNCPITKCQYSKPENCRLSMGIRPNSHYILRSGLLKYNLSDGENPKVVFIEISDQRLSIGSPSKIYDSLGQPVFYQASFSWDTMIKFGDVLTVNPLVVNWRNNTVISRPGQSQCPRFNTCPEICWEGVYNDAFLIDRINWISAGVFLDSNQTAENPVFTVFKDNEILYRAQLASEDTNAQKTITNCFLLKNKIWCISLVEIYDTGDNVIRPKLFAVKIPEQCT
EOF

cat > Boltz2_IpSAE/example_yaml/sialidase_2F29.fasta << 'EOF'
>Sialidase_2F29
GSMASLPVLQKESVFQSGAHAYRIPALLYLPGQQSLLAFAEQRASKKDEHAELIVLRRGDYDAPTHQVQWQAQEVVAQARLDGHRSMNPCPLYDAQTGTLFLFFIAIPGQVTEQQQLETRANVTRLCQVTSTDHGRTWSSPRDLTDAAIGPAYREWSTFAVGPGHCLQLNDRARSLVVPAYAYRKLHPIQRPIPSAFCFLSHDHGRTWARGHFVAQDTLECQVAEVETGEQRVVTLNARSHLRARVQAQSTNDGLDFQESQLVKKLVEPPPQGCQGSVISFPSPRSGPGSPAQWLLYTHPTHSWQRADLGAYLNPRPPAPEAWSEPVLLAKGSCAYSDLQSMGTGPDGSPLFGCLYEANDYEEIVFLMFTLKQAFPAEYLPQ
EOF
```

This mirrors the sequences already present in the example YAML config.

### 3.2. Run the pipeline on known Nipah binders

Use the per-binder FASTAs in `example_yaml/known_binders` as binders, Nipah G
as the target, sialidase as an antitarget, and include self-binding:

```bash
cd /root/boltz

python Boltz2_IpSAE/run_ipsae_pipeline.py \
  --binder_fasta_dir Boltz2_IpSAE/example_yaml/known_binders \
  --target_name nipah_g \
  --target_fasta Boltz2_IpSAE/example_yaml/nipah_g.fasta \
  --target_msa Boltz2_IpSAE/example_yaml/nipah.a3m \
  --antitarget_name Sialidase_2F29 \
  --antitarget_fasta Boltz2_IpSAE/example_yaml/sialidase_2F29.fasta \
  --include_self \
  --out_dir Boltz2_IpSAE/example_yaml/boltz_ipsae_nipah \
  --recycling_steps 10 \
  --diffusion_samples 5 \
  --use_msa_server auto \
  --ipsae_pae_cutoff 15 \
  --ipsae_dist_cutoff 15 \
  --num_cpu 4
```

What happens:

1. **Stage 1/3** – inputs parsed and YAMLs written under:
   - `Boltz2_IpSAE/example_yaml/boltz_ipsae_nipah/binder_*/binder_*_vs_*.yaml`
2. **Stage 2/3** – Boltz predictions:
   - For each binder vs target/antitarget/self the script runs
     `python -m boltz.main predict ...` inside the binder directory.
   - Logs go to `boltz_ipsae_nipah/logs/boltz_binder_*_vs_*.log`.
3. **Stage 3/3** – ipSAE:
   - `visualise_binder_validation.py` is called with your ipSAE cutoffs and
     `--num_cpu`.
   - It runs `ipsae.py` on all predicted complexes and writes:
     - per-binder summaries + stripplots under
       `boltz_ipsae_nipah/binder_*/plots/`,
     - global ipSAE heatmaps and CSVs under
       `boltz_ipsae_nipah/summary/`,
     - detailed logs under `boltz_ipsae_nipah/logs/ipsae_pipeline.log`.

Use `--verbose` if you want to see full Boltz/ipSAE output instead of just
high-level progress.

---

## 4. Using CSV binders instead of FASTAs

You can supply binders from a CSV file, e.g.:

```text
binder_name,binder_sequence
Ab1,EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMS...
Ab2,QLVVTQSGAEVKKPGASVKVSCKASGYTFTNYWM...
```

Run:

```bash
python Boltz2_IpSAE/run_ipsae_pipeline.py \
  --binder_csv my_binders.csv \
  --binder_name_col binder_name \
  --binder_seq_col binder_sequence \
  --target_name nipah_g \
  --target_fasta Boltz2_IpSAE/example_yaml/nipah_g.fasta \
  --target_msa Boltz2_IpSAE/example_yaml/nipah.a3m \
  --out_dir boltz_ipsae_my_binders_nipah \
  --include_self
```

Multi-chain binders can be expressed as `chainA:chainB` in the sequence
column; the script splits on `:` and assigns chain IDs `A`, `B`, `C`, …

---

## 5. Outputs and logs

For a given `--out_dir`, you should see:

- Per-binder directories:
  - `binder_<binder_name>/outputs/boltz_results_.../`
  - `binder_<binder_name>/plots/ipsae_summary.csv`, stripplots, etc.
- Global summaries:
  - `summary/ipSAE_min_heatmap.csv`, `ipSAE_max_heatmap.csv`
  - corresponding `.png`/`.svg` heatmaps.
- Logs:
  - `logs/boltz_binder_*_vs_*.log` – raw Boltz outputs.
  - `logs/ipsae_pipeline.log` – ipSAE & visualisation script output.

These CSVs and plots give you ipSAE-based metrics for each binder vs target,
antitarget, and self, suitable for ranking and downstream selection. If you
want to further filter binders (e.g. strong target binding, weak antitarget
and self binding), see `make_fastas_from_heatmap_csv.py` in this folder.

