#!/usr/bin/env python3
"""
Run Boltz + ipSAE pipeline without manual YAML configuration.

High-level flow:
  1) Parse binder(s), target, optional antitarget and self from CLI arguments.
  2) Generate per-binder YAML files for Boltz (binder vs target / antitarget / self).
  3) Run `boltz predict` on each YAML, writing outputs under:
         <out_dir>/binder_<binder_name>/outputs/...
  4) Run ipSAE over all predictions via visualise_binder_validation.py
     and write summary CSVs and plots under <out_dir>.

By default this script prints clean, high-level progress messages and hides
Boltz/ipSAE internal logs. Use --verbose to surface full subprocess output.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from make_binder_validation_scripts import (
    add_n_terminal_lysine,
    read_fasta_dir_entities,
    read_fasta_multi,
    sanitize_name,
    yaml_for_pair,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run Boltz + ipSAE pipeline for one target and many binders "
            "without manually writing YAML files."
        )
    )

    # Binder sources (exactly one)
    binder_group = ap.add_mutually_exclusive_group(required=True)
    binder_group.add_argument(
        "--binder_csv",
        type=str,
        help="CSV file with binders (binder name + sequence columns).",
    )
    binder_group.add_argument(
        "--binder_fasta_dir",
        type=str,
        help="Directory containing per-binder FASTA files.",
    )
    binder_group.add_argument(
        "--binder_fasta",
        type=str,
        help="Single binder FASTA file (use with --binder_name).",
    )
    binder_group.add_argument(
        "--binder_seq",
        type=str,
        help="Single binder amino acid sequence (use with --binder_name). "
        "For multi-chain binders, separate chains with ':'.",
    )

    ap.add_argument(
        "--binder_name",
        type=str,
        help="Binder name when using --binder_fasta or --binder_seq.",
    )
    ap.add_argument(
        "--binder_name_col",
        type=str,
        default="name",
        help="Column name for binder names in --binder_csv (default: name).",
    )
    ap.add_argument(
        "--binder_seq_col",
        type=str,
        default="sequence",
        help="Column name for binder sequences in --binder_csv (default: sequence).",
    )
    ap.add_argument(
        "--add_n_terminal_lysine",
        action="store_true",
        help="Prepend 'K' to each binder chain if missing.",
    )

    # Target (required)
    ap.add_argument(
        "--target_name",
        required=True,
        type=str,
        help="Name of the target protein (e.g. nipah_g).",
    )
    ap.add_argument(
        "--target_seq",
        type=str,
        help="Target amino acid sequence. For multi-chain targets, separate chains "
        "with ':'. Mutually exclusive with --target_fasta.",
    )
    ap.add_argument(
        "--target_fasta",
        type=str,
        help="FASTA file with target sequence(s). Mutually exclusive with --target_seq.",
    )
    ap.add_argument(
        "--target_msa",
        type=str,
        help="Optional MSA file (e.g. .a3m) for target chain 0.",
    )

    # Antitarget (optional)
    ap.add_argument(
        "--antitarget_name",
        type=str,
        help="Name of an off-target protein to penalize binding (optional).",
    )
    ap.add_argument(
        "--antitarget_seq",
        type=str,
        help="Antitarget amino acid sequence. Multi-chain: chains separated by ':'.",
    )
    ap.add_argument(
        "--antitarget_fasta",
        type=str,
        help="FASTA file with antitarget sequence(s).",
    )
    ap.add_argument(
        "--antitarget_msa",
        type=str,
        help="Optional MSA file for antitarget chain 0.",
    )

    ap.add_argument(
        "--include_self",
        action="store_true",
        help="Also run each binder against itself (self-binding control).",
    )

    # Boltz options
    ap.add_argument(
        "--out_dir",
        type=str,
        default="./boltz_ipsae",
        help="Root output directory for all results (default: ./boltz_ipsae).",
    )
    ap.add_argument(
        "--recycling_steps",
        type=int,
        default=10,
        help="Boltz recycling steps (default: 10).",
    )
    ap.add_argument(
        "--diffusion_samples",
        type=int,
        default=5,
        help="Boltz diffusion samples (default: 5).",
    )
    ap.add_argument(
        "--use_msa_server",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help=(
            "Whether to use the MSA server for Boltz predictions. "
            "'auto' (default) uses the server only when neither binder "
            "nor partner has an explicit MSA."
        ),
    )

    # ipSAE options
    ap.add_argument(
        "--ipsae_pae_cutoff",
        type=int,
        default=15,
        help="ipSAE PAE cutoff in Å (default: 15).",
    )
    ap.add_argument(
        "--ipsae_dist_cutoff",
        type=int,
        default=15,
        help="ipSAE distance cutoff in Å (default: 15).",
    )
    ap.add_argument(
        "--use_best_model",
        action="store_true",
        help=(
            "For summary heatmaps, use only the best model (highest ipSAE_max) "
            "per binder/partner instead of averaging across models."
        ),
    )
    ap.add_argument(
        "--num_cpu",
        type=int,
        default=1,
        help="Number of CPUs to use for ipSAE scoring (default: 1).",
    )

    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Show full Boltz and ipSAE subprocess output.",
    )

    return ap.parse_args()


def _clean_seq_string(seq: str) -> str:
    """Normalize a sequence string: remove whitespace, uppercase."""
    return "".join(seq.split()).upper()


def _split_multi_chain(seq: str) -> List[str]:
    """
    Split a multi-chain sequence string into chains, using ':' as separator.
    """
    seq = seq.strip()
    if not seq:
        return []
    parts = [p for p in seq.split(":") if p.strip()]
    return [_clean_seq_string(p) for p in parts]


def load_binders_from_csv(
    csv_path: Path,
    name_col: str,
    seq_col: str,
    addK: bool,
) -> List[Dict[str, object]]:
    binders: List[Dict[str, object]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")
        if name_col not in reader.fieldnames:
            raise ValueError(
                f"CSV file {csv_path} missing binder name column {name_col!r}."
            )
        if seq_col not in reader.fieldnames:
            raise ValueError(
                f"CSV file {csv_path} missing binder sequence column {seq_col!r}."
            )
        for row in reader:
            raw_name = (row.get(name_col) or "").strip()
            raw_seq = (row.get(seq_col) or "").strip()
            if not raw_name or not raw_seq:
                continue
            name = sanitize_name(raw_name)
            chains = _split_multi_chain(raw_seq)
            if not chains:
                continue
            if addK:
                chains = add_n_terminal_lysine(chains)
            msas = [None] * len(chains)
            binders.append({"name": name, "seqs": chains, "msas": msas})
    if not binders:
        raise ValueError(f"No valid binders found in CSV: {csv_path}")
    return binders


def load_binders_from_dir(
    fasta_dir: Path,
    addK: bool,
) -> List[Dict[str, object]]:
    if not fasta_dir.is_dir():
        raise ValueError(f"Binder FASTA directory not found: {fasta_dir}")
    entities = read_fasta_dir_entities(fasta_dir)
    binders: List[Dict[str, object]] = []
    for name, seqs in entities:
        if not seqs:
            continue
        chains = [_clean_seq_string(s) for s in seqs]
        if addK:
            chains = add_n_terminal_lysine(chains)
        msas = [None] * len(chains)
        binders.append({"name": name, "seqs": chains, "msas": msas})
    if not binders:
        raise ValueError(f"No valid binder sequences found in directory: {fasta_dir}")
    return binders


def load_single_binder_from_fasta(
    name: str,
    fasta_path: Path,
    addK: bool,
) -> List[Dict[str, object]]:
    if not fasta_path.is_file():
        raise ValueError(f"Binder FASTA not found: {fasta_path}")
    seqs = read_fasta_multi(fasta_path)
    if not seqs:
        raise ValueError(f"No sequences found in binder FASTA: {fasta_path}")
    chains = [_clean_seq_string(s) for s in seqs]
    if addK:
        chains = add_n_terminal_lysine(chains)
    msas = [None] * len(chains)
    return [{"name": sanitize_name(name), "seqs": chains, "msas": msas}]


def load_single_binder_from_seq(
    name: str,
    seq: str,
    addK: bool,
) -> List[Dict[str, object]]:
    chains = _split_multi_chain(seq)
    if not chains:
        raise ValueError("Binder sequence is empty.")
    if addK:
        chains = add_n_terminal_lysine(chains)
    msas = [None] * len(chains)
    return [{"name": sanitize_name(name), "seqs": chains, "msas": msas}]


def load_partner(
    name: str,
    seq: Optional[str],
    fasta: Optional[str],
    msa: Optional[str],
    role: str,
) -> Dict[str, object]:
    """
    Load a target or antitarget from CLI args.
    Returns dict: {name, role, seqs, msas}
    """
    if seq and fasta:
        raise ValueError(f"{role} {name!r}: specify only one of sequence or FASTA.")

    if seq:
        seqs = _split_multi_chain(seq)
    elif fasta:
        fasta_path = Path(fasta).resolve()
        if not fasta_path.is_file():
            raise ValueError(f"{role} FASTA not found: {fasta_path}")
        seqs = [_clean_seq_string(s) for s in read_fasta_multi(fasta_path)]
    else:
        raise ValueError(f"{role} {name!r}: must specify sequence or FASTA.")

    if not seqs:
        raise ValueError(f"{role} {name!r}: no sequences found.")

    msas: List[Optional[str]] = [None] * len(seqs)
    if msa:
        # Attach MSA to chain 0 only by default
        msas[0] = str(Path(msa).resolve())

    return {"name": sanitize_name(name), "role": role, "seqs": seqs, "msas": msas}


def decide_use_msa_server(
    global_mode: str,
    binder_msas: List[Optional[str]],
    partner_msas: List[Optional[str]],
) -> bool:
    """
    Decide whether to use MSA server for a given binder-partner pair.
    """
    if global_mode == "true":
        return True
    if global_mode == "false":
        return False
    # auto: use server only if neither binder nor partner has explicit MSAs
    any_msa = any(binder_msas) or any(partner_msas)
    return not any_msa


def run_subprocess(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    verbose: bool,
) -> int:
    """
    Run a subprocess in the given cwd, optionally capturing output to log_path.
    Returns the process return code.
    """
    cwd.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        proc = subprocess.run(cmd, cwd=str(cwd))
        return proc.returncode

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(proc.stdout or "")
    return proc.returncode


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print("Stage 1/3: Parsing inputs and preparing Boltz YAMLs...")

    addK = bool(args.add_n_terminal_lysine)

    # --- Load binders ---
    binders: List[Dict[str, object]]
    if args.binder_csv:
        binders = load_binders_from_csv(
            Path(args.binder_csv).resolve(),
            name_col=args.binder_name_col,
            seq_col=args.binder_seq_col,
            addK=addK,
        )
    elif args.binder_fasta_dir:
        binders = load_binders_from_dir(
            Path(args.binder_fasta_dir).resolve(),
            addK=addK,
        )
    elif args.binder_fasta:
        if not args.binder_name:
            raise SystemExit(
                "ERROR: --binder_name is required when using --binder_fasta."
            )
        binders = load_single_binder_from_fasta(
            args.binder_name,
            Path(args.binder_fasta).resolve(),
            addK=addK,
        )
    elif args.binder_seq:
        if not args.binder_name:
            raise SystemExit(
                "ERROR: --binder_name is required when using --binder_seq."
            )
        binders = load_single_binder_from_seq(
            args.binder_name,
            args.binder_seq,
            addK=addK,
        )
    else:
        raise SystemExit("ERROR: No binder source provided.")

    # --- Load target (required) ---
    target = load_partner(
        name=args.target_name,
        seq=args.target_seq,
        fasta=args.target_fasta,
        msa=args.target_msa,
        role="target",
    )

    # --- Load antitarget (optional) ---
    antitarget = None
    if args.antitarget_name:
        if not (args.antitarget_seq or args.antitarget_fasta):
            raise SystemExit(
                "ERROR: antitarget_name was provided but neither "
                "--antitarget_seq nor --antitarget_fasta was given."
            )
        antitarget = load_partner(
            name=args.antitarget_name,
            seq=args.antitarget_seq,
            fasta=args.antitarget_fasta,
            msa=args.antitarget_msa,
            role="antitarget",
        )

    partners: List[Dict[str, object]] = [target]
    if antitarget is not None:
        partners.append(antitarget)

    if not partners:
        raise SystemExit("ERROR: At least one target/antitarget must be defined.")

    print(
        f"  Loaded {len(binders)} binder(s) and {len(partners)} partner type(s) "
        f"(target / antitarget)."
    )

    # --- Prepare YAMLs and job list ---
    jobs: List[Dict[str, object]] = []

    for binder in binders:
        bname = binder["name"]  # type: ignore[assignment]
        bseqs = binder["seqs"]  # type: ignore[assignment]
        bmsas = binder["msas"]  # type: ignore[assignment]

        binder_dir = out_root / f"binder_{bname}"
        binder_dir.mkdir(parents=True, exist_ok=True)

        # binder vs each partner (target / antitarget)
        for partner in partners:
            role = partner["role"]  # type: ignore[index]
            pname = partner["name"]  # type: ignore[index]
            tseqs = partner["seqs"]  # type: ignore[index]
            tmsas = partner["msas"]  # type: ignore[index]

            if role not in {"target", "antitarget"}:
                raise SystemExit(f"Unexpected partner role: {role!r}")

            yaml_stem = f"binder_{bname}_vs_{role}_{pname}"
            yaml_path = binder_dir / f"{yaml_stem}.yaml"

            yaml_text = yaml_for_pair(
                binder_seqs=bseqs,  # type: ignore[arg-type]
                partner_seqs=tseqs,  # type: ignore[arg-type]
                partner_role=role,  # type: ignore[arg-type]
                binder_msas=bmsas,  # type: ignore[arg-type]
                partner_msas=tmsas,  # type: ignore[arg-type]
            )
            yaml_path.write_text(yaml_text, encoding="utf-8")

            jobs.append(
                {
                    "binder": bname,
                    "role": role,
                    "partner": pname,
                    "yaml_path": yaml_path,
                    "binder_dir": binder_dir,
                    "binder_msas": bmsas,
                    "partner_msas": tmsas,
                }
            )

        # binder vs self (optional)
        if args.include_self:
            role = "self"
            pname = "self"
            tseqs = list(bseqs)
            tmsas = list(bmsas)
            yaml_stem = f"binder_{bname}_vs_self"
            yaml_path = binder_dir / f"{yaml_stem}.yaml"

            yaml_text = yaml_for_pair(
                binder_seqs=bseqs,  # type: ignore[arg-type]
                partner_seqs=tseqs,  # type: ignore[arg-type]
                partner_role=role,
                binder_msas=bmsas,  # type: ignore[arg-type]
                partner_msas=tmsas,  # type: ignore[arg-type]
            )
            yaml_path.write_text(yaml_text, encoding="utf-8")

            jobs.append(
                {
                    "binder": bname,
                    "role": role,
                    "partner": pname,
                    "yaml_path": yaml_path,
                    "binder_dir": binder_dir,
                    "binder_msas": bmsas,
                    "partner_msas": tmsas,
                }
            )

    if not jobs:
        raise SystemExit("ERROR: No binder–partner jobs were created.")

    print(
        f"  Prepared {len(jobs)} Boltz jobs "
        f"({len(binders)} binder(s) × {len(partners) + int(args.include_self)} partner type(s))."
    )

    # ------------------------------------------------------------------
    # Stage 2: Run Boltz predictions
    # ------------------------------------------------------------------
    print(
        f"Stage 2/3: Running Boltz predictions for {len(jobs)} complex(es)..."
    )

    logs_dir = out_root / "logs"
    total = len(jobs)
    for idx, job in enumerate(jobs, start=1):
        binder_name = job["binder"]
        role = job["role"]
        partner_name = job["partner"]
        yaml_path = job["yaml_path"]
        binder_dir = job["binder_dir"]
        bmsas = job["binder_msas"]
        tmsas = job["partner_msas"]

        out_dir = binder_dir / "outputs"
        use_msa = decide_use_msa_server(
            args.use_msa_server,
            binder_msas=bmsas,  # type: ignore[arg-type]
            partner_msas=tmsas,  # type: ignore[arg-type]
        )

        print(
            f"  [{idx}/{total}] Predicting binder='{binder_name}' "
            f"vs {role}='{partner_name}' with Boltz..."
        )

        cmd = [
            sys.executable,
            "-m",
            "boltz.main",
            "predict",
            yaml_path.name,
            "--out_dir",
            str(out_dir),
            "--recycling_steps",
            str(args.recycling_steps),
            "--diffusion_samples",
            str(args.diffusion_samples),
        ]
        if use_msa:
            cmd.append("--use_msa_server")

        log_name = (
            f"boltz_binder_{binder_name}_vs_{role}_{partner_name}.log"
        )
        log_path = logs_dir / sanitize_name(log_name)

        returncode = run_subprocess(
            cmd=cmd,
            cwd=binder_dir,
            log_path=log_path,
            verbose=args.verbose,
        )
        if returncode != 0:
            print(
                f"    !! Boltz failed for binder='{binder_name}' vs {role}='{partner_name}'. "
                f"See log: {log_path}"
            )
        else:
            if not args.verbose:
                print(f"    Done. (log: {log_path})")

    # ------------------------------------------------------------------
    # Stage 3: Run ipSAE on all predictions and summarize
    # ------------------------------------------------------------------
    print(
        "Stage 3/3: Running ipSAE on predicted complexes and generating summaries..."
    )

    visualise_script = script_dir / "visualise_binder_validation.py"
    if not visualise_script.is_file():
        raise SystemExit(
            f"ERROR: visualise_binder_validation.py not found at {visualise_script}"
        )

    cmd_ipsae = [
        sys.executable,
        str(visualise_script),
        "--ipsae_e",
        str(args.ipsae_pae_cutoff),
        "--ipsae_d",
        str(args.ipsae_dist_cutoff),
        "--root_dir",
        str(out_root),
        "--generate_data",
        "--plot",
        "--num_cpu",
        str(args.num_cpu),
    ]
    if args.use_best_model:
        cmd_ipsae.append("--use_best_model")

    ipsae_log = logs_dir / "ipsae_pipeline.log"
    returncode = run_subprocess(
        cmd=cmd_ipsae,
        cwd=out_root,
        log_path=ipsae_log,
        verbose=args.verbose,
    )
    if returncode != 0:
        print(
            f"    !! ipSAE pipeline failed. See log: {ipsae_log}"
        )
        raise SystemExit(returncode)
    else:
        if not args.verbose:
            print(f"    ipSAE summaries written. (log: {ipsae_log})")

    print(
        f"Done. Results are under: {out_root}\n"
        f"  - Per-binder summaries and plots: {out_root}/binder_*/plots\n"
        f"  - Global heatmaps and CSVs:        {out_root}/summary\n"
    )


if __name__ == "__main__":
    main()

