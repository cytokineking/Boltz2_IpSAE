#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generalized binder validation script.

✅ Binder is always chain A (appears first in YAML)
✅ Targets / off-targets follow as B, C, ...
✅ Supports multi-chain binders and targets
✅ Allows --msa only when exactly one --target is specified
✅ MSA applies only to target chains
✅ Generates:
   - YAMLs for all binder–target/off-target pairs
   - Per-binder run.sh
   - Global run_all_cofolding.sh
   - Visualization helper script
"""

import argparse
import os
import re
import sys
from pathlib import Path

SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_name(name: str) -> str:
    cleaned = SANITIZE_RE.sub("_", name.strip())
    if not cleaned:
        raise ValueError(f"Invalid name: {name!r}")
    return cleaned


def write_text(path: Path, text: str) -> None:
    """Write text to a file, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_entity_arg(items):
    """Parse command-line entries like name=SEQ[:SEQ2:SEQ3]."""
    entities = []
    for item in items:
        if "=" not in item:
            sys.exit(f"ERROR: Expected name=SEQS, got {item!r}")
        name, seqs = item.split("=", 1)
        name = sanitize_name(name)
        chains = [s.strip().replace("\\n", "").replace("\n", "") for s in seqs.split(":") if s.strip()]
        if not chains:
            sys.exit(f"ERROR: No sequences found for {name!r}")
        entities.append((name, chains))
    return entities


def read_fasta_dir_entities(fasta_dir: Path):
    """Read FASTA files from a directory -> list[(name, [seqs])]."""
    entities = []
    for fasta_path in sorted(fasta_dir.glob("*")):
        if not fasta_path.is_file() or not fasta_path.suffix.lower() in {".fasta", ".fa", ".fna", ".faa", ".txt"}:
            continue
        name = sanitize_name(fasta_path.stem)
        seqs, seq = [], []
        with open(fasta_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith(">"):
                    if seq:
                        seqs.append("".join(seq).replace(" ", "").upper())
                        seq = []
                else:
                    seq.append(line)
            if seq:
                seqs.append("".join(seq).replace(" ", "").upper())
        if seqs:
            entities.append((name, seqs))
    return entities


def make_master_run_sh(output_root: Path):
    """Generate top-level script to run all binder run.sh scripts."""
    lines = [
        "#!/bin/bash",
        "set -e",
        "",
        "# Run all binder_* run.sh scripts",
        'for f in $(find . -type f -name "run.sh" | sort); do',
        '  echo "Running $f..."',
        '  (cd \"$(dirname \"$f\")\" && bash run.sh)',
        "done",
        "",
    ]
    run_all_path = output_root / "run_all_cofolding.sh"
    write_text(run_all_path, "\n".join(lines))
    os.chmod(run_all_path, 0o755)
    print(f"✅ Created {run_all_path}")


def make_visualisation_sh(output_root: Path):
    """Create visualization helper script."""
    lines = [
        f"python visualise_binder_validation.py --root_dir {output_root} --generate_data --plot"
    ]
    sh_path = output_root / "visualise_cofolding_results.sh"
    write_text(sh_path, "\n".join(lines) + "\n")
    os.chmod(sh_path, 0o755)
    print(f"✅ Created {sh_path}")

def yaml_for_pair(binder_seqs, target_seqs, msa_path=None):
    """
    ✅ Binder always first in YAML (chain A)
    ✅ If msa_path is given → all chains must have msa (binder: empty, target: msa)
    """
    lines = ["version: 1", "sequences:"]

    # --- Binder first (chain A, etc.) ---
    for i, seq in enumerate(binder_seqs):
        cid = chr(ord("A") + i)
        lines.append("  - protein:")
        lines.append(f"      id: {cid}")
        lines.append(f"      sequence: {seq}")
        if msa_path:
            lines.append("      msa: empty")

    # --- Then target/off-targets (chains B, C, …) ---
    start_idx = len(binder_seqs)
    for i, seq in enumerate(target_seqs):
        cid = chr(ord("A") + start_idx + i)
        lines.append("  - protein:")
        lines.append(f"      id: {cid}")
        lines.append(f"      sequence: {seq}")
        if msa_path:
            lines.append(f"      msa: {msa_path}")

    return "\n".join(lines) + "\n"


def make_run_sh(dir_path: Path, yaml_paths, use_msa_server=True):
    """Create per-binder run.sh."""
    lines = [
        f"boltz predict {p.name} --recycling_steps 10 {'--use_msa_server' if use_msa_server else ''}"
        f"--diffusion_samples 5 --out_dir {os.path.join(dir_path, 'outputs')}"
        for p in yaml_paths
    ]
    run_path = dir_path / "run.sh"
    write_text(run_path, "\n".join(lines) + "\n")
    os.chmod(run_path, 0o755)


def parse_args():
    ap = argparse.ArgumentParser(description="Generate YAMLs and run scripts for binder validation.")
    ap.add_argument("--output_dir", required=True, help="Output directory.")
    ap.add_argument("--binder", action="append", help="Binder as name=SEQ[:SEQ2]")
    ap.add_argument("--target", action="append", help="Target as name=SEQ[:SEQ2]")
    ap.add_argument("--off_target", action="append", help="Off-target as name=SEQ[:SEQ2]")
    ap.add_argument("--binder_dir", help="Directory with binder FASTA files.")
    ap.add_argument("--target_dir", help="Directory with target FASTA files.")
    ap.add_argument("--off_target_dir", help="Directory with off-target FASTA files.")
    ap.add_argument("--msa", help="Optional MSA file (only allowed when exactly one --target).")
    ap.add_argument("--add_n_terminal_lysine", action="store_true",
                    help="Prepend 'K' to each binder chain if missing.")
    return ap.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    binders, targets, off_targets = [], [], []
    if args.binder:
        binders.extend(parse_entity_arg(args.binder))
    if args.binder_dir:
        binders.extend(read_fasta_dir_entities(Path(args.binder_dir)))

    if args.target:
        targets.extend(parse_entity_arg(args.target))
    if args.target_dir:
        targets.extend(read_fasta_dir_entities(Path(args.target_dir)))

    if args.off_target:
        off_targets.extend(parse_entity_arg(args.off_target))
    if args.off_target_dir:
        off_targets.extend(read_fasta_dir_entities(Path(args.off_target_dir)))

    if not binders:
        sys.exit("ERROR: Must provide at least one binder (--binder or --binder_dir).")
    if not (targets or off_targets):
        sys.exit("ERROR: Must provide at least one target or off-target (--target or --target_dir).")

    # --- Validate MSA ---
    msa_path = args.msa
    if msa_path:
        if len(targets) != 1 or args.target_dir:
            sys.exit("ERROR: --msa can only be used when exactly one --target is provided.")
        msa_path = Path(msa_path).resolve()
        if not msa_path.exists():
            sys.exit(f"ERROR: MSA file not found: {msa_path}")

    # --- Optional N-terminal K ---
    if args.add_n_terminal_lysine:
        binders = [
            (n, ["K" + s if not s.startswith("K") else s for s in seqs])
            for n, seqs in binders
        ]

    # --- Generate YAMLs and run.sh for each binder ---
    for bname, bseqs in binders:
        binder_dir = output_root / f"binder_{bname}"
        binder_dir.mkdir(parents=True, exist_ok=True)
        yaml_paths = []

        # Binder vs targets
        for tname, tseqs in targets:
            ypath = binder_dir / f"binder_{bname}_vs_{sanitize_name(tname)}.yaml"
            write_text(ypath, yaml_for_pair(bseqs, tseqs, msa_path))
            yaml_paths.append(ypath)

        # Binder vs off-targets
        for oname, oseqs in off_targets:
            ypath = binder_dir / f"binder_{bname}_vs_{sanitize_name(oname)}.yaml"
            write_text(ypath, yaml_for_pair(bseqs, oseqs))
            yaml_paths.append(ypath)

        make_run_sh(binder_dir, yaml_paths, use_msa_server= not bool(msa_path))

    make_master_run_sh(output_root)
    make_visualisation_sh(output_root)
    print(f"\n✅ Done. YAMLs and scripts written under: {output_root}\n")


if __name__ == "__main__":
    main()
