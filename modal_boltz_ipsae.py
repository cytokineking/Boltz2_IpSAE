#!/usr/bin/env python3
"""
Modal deployment for Boltz2 + ipSAE pipeline.

This module provides serverless GPU execution of Boltz2 predictions and ipSAE scoring
on Modal's cloud infrastructure. Harmonized with run_ipsae_pipeline.py for feature parity.

KEY DESIGN PRINCIPLES:
- All inputs are read locally and passed to Modal functions
- All outputs are returned directly (no need to access Modal volumes)
- Model weights and CCD data are cached in a persistent volume for fast startup
- Parallel processing via .map() for multiple binders

Usage:
    # Run full pipeline for a single binder
    modal run modal_boltz_ipsae.py::run_pipeline \
        --binder-name "my_binder" \
        --binder-seq "MKTAYIAK..." \
        --target-name nipah_g \
        --target-fasta target.fasta

    # Run for multiple binders from CSV with antitarget
    modal run modal_boltz_ipsae.py::run_pipeline \
        --binder-csv binders.csv \
        --target-fasta target.fasta \
        --antitarget-fasta antitarget.fasta \
        --include-self \
        --output-dir ./results

    # With custom GPU and parallelism
    modal run modal_boltz_ipsae.py::run_pipeline \
        --binder-csv binders.csv \
        --target-fasta target.fasta \
        --gpu "H100" \
        --max-parallel 10
"""

import csv
import datetime
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

app = modal.App("boltz2-ipsae")

# Volume for caching model weights and CCD data (not for user outputs)
cache_volume = modal.Volume.from_name("boltz-cache", create_if_missing=True)

# Dict for real-time result streaming (immediate writes, 7-day TTL per entry)
results_dict = modal.Dict.from_name("boltz-results", create_if_missing=True)

# Supported GPU types with VRAM and hourly cost
GPU_TYPES = {
    "T4": "16GB - $0.59/h",
    "L4": "24GB - $0.80/h",
    "A10G": "24GB - $1.10/h",
    "L40S": "48GB - $1.95/h",
    "A100-40GB": "40GB - $2.10/h",
    "A100-80GB": "80GB - $2.50/h (DEFAULT)",
    "H100": "80GB - $3.95/h",
    "H200": "141GB - $4.54/h",
    "B200": "192GB - $6.25/h",
}

DEFAULT_GPU = "A100-80GB"

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        "biopython==1.84",
        "numpy==1.26.3",
        "pyyaml>=6.0",
        "pandas>=2.0",
        "seaborn>=0.12",
        "matplotlib>=3.7",
        "torch",
    )
    .pip_install(
        "boltz>=2.2.1",
    )
    .run_commands(
        # cuequivariance can cause issues, handle gracefully
        "pip install cuequivariance-torch || pip install cuequivariance_torch || echo 'Warning: Could not install cuequivariance-torch'",
    )
    # Add all necessary scripts to the image
    .add_local_file("ipsae.py", "/root/ipsae.py")
    .add_local_file("visualise_binder_validation.py", "/root/visualise_binder_validation.py")
    .add_local_file("make_binder_validation_scripts.py", "/root/make_binder_validation_scripts.py")
)


# =============================================================================
# MODAL FUNCTIONS
# =============================================================================

@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=3600,  # 1 hour for downloads
    volumes={"/cache": cache_volume},
)
def initialize_cache() -> str:
    """
    Download and cache Boltz model weights and CCD data.
    Run this once before using the pipeline.
    """
    from boltz.main import download_boltz2

    cache_dir = Path("/cache/boltz")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Boltz model weights and CCD data...")
    print("This may take 10-20 minutes on first run.")
    print(f"Cache directory: {cache_dir}")

    try:
        download_boltz2(cache_dir)
        cache_volume.commit()

        files = list(cache_dir.rglob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        return f"Cache initialized successfully!\nFiles: {len(files)}\nTotal size: {total_size / 1e9:.2f} GB"
    except Exception as e:
        return f"Error initializing cache: {e}"


def _run_boltz_and_ipsae_impl(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core implementation for Boltz prediction and ipSAE scoring.

    This is the actual logic, separated from the Modal decorator so we can
    have multiple GPU-specific wrapper functions.
    
    When stream_to_dict is True, results are written to the Modal Dict
    immediately after each partner prediction completes, enabling real-time
    sync to local filesystem.
    """
    import time as time_module  # Local import to avoid issues
    
    # Reconstruct task from dict
    binder_name = task_dict["binder_name"]
    binder_sequences = task_dict["binder_sequences"]
    partners = task_dict["partners"]
    recycling_steps = task_dict.get("recycling_steps", 10)
    diffusion_samples = task_dict.get("diffusion_samples", 5)
    pae_cutoff = task_dict.get("pae_cutoff", 15.0)
    dist_cutoff = task_dict.get("dist_cutoff", 15.0)
    use_msa_server = task_dict.get("use_msa_server", "auto")
    verbose = task_dict.get("verbose", False)
    
    # Streaming options
    run_id = task_dict.get("run_id")
    stream_to_dict = task_dict.get("stream_to_dict", False)

    work_dir = Path(tempfile.mkdtemp())

    results = {
        "status": "pending",
        "binder_name": binder_name,
        "target": None,
        "antitarget": None,
        "self": None,
        "error": None,
    }

    try:
        # Set up Boltz cache directory
        boltz_cache = Path("/cache/boltz")
        boltz_cache.mkdir(parents=True, exist_ok=True)
        os.environ["BOLTZ_CACHE"] = str(boltz_cache)

        print(f"\n{'='*60}")
        print(f"Processing binder: {binder_name}")
        print(f"  Sequences: {len(binder_sequences)} chain(s)")
        print(f"  Partners: {len(partners)}")
        if stream_to_dict:
            print(f"  Streaming: enabled (run_id={run_id})")
        print(f"{'='*60}")

        # Process each partner
        for partner in partners:
            partner_name = partner["name"]
            partner_role = partner["role"]
            partner_sequences = partner["sequences"]
            partner_msa = partner.get("msa_content")

            print(f"\n  [{partner_role.upper()}] Running vs {partner_name}...")

            partner_result = _run_single_prediction(
                binder_sequences=binder_sequences,
                partner_name=partner_name,
                partner_role=partner_role,
                partner_sequences=partner_sequences,
                partner_msa=partner_msa,
                work_dir=work_dir,
                boltz_cache=boltz_cache,
                recycling_steps=recycling_steps,
                diffusion_samples=diffusion_samples,
                pae_cutoff=pae_cutoff,
                dist_cutoff=dist_cutoff,
                use_msa_server=use_msa_server,
                verbose=verbose,
            )

            results[partner_role] = partner_result

            if partner_result.get("error"):
                print(f"    ⚠ Error: {partner_result['error'][:100]}")
            else:
                m = partner_result.get("metrics", {})
                print(f"    ✓ ipSAE: {m.get('ipSAE_mean', 0):.3f} ± {m.get('ipSAE_std', 0):.3f}")
            
            # Stream result to Dict immediately after each partner completes
            if stream_to_dict and run_id:
                dict_key = f"{run_id}:{binder_name}:{partner_role}"
                try:
                    results_dict[dict_key] = {
                        "run_id": run_id,
                        "binder_name": binder_name,
                        "partner_role": partner_role,
                        "partner_name": partner_result.get("partner_name"),
                        "n_models": partner_result.get("n_models", 0),
                        "metrics": partner_result.get("metrics", {}),
                        "per_model": partner_result.get("per_model", []),
                        "structures": partner_result.get("structures", {}),
                        "error": partner_result.get("error"),
                        "timestamp": time_module.time(),
                    }
                    print(f"    → Streamed to Dict: {dict_key}")
                except Exception as stream_err:
                    print(f"    ⚠ Stream error: {stream_err}")

        results["status"] = "success"
        
        # Mark binder as complete in Dict
        if stream_to_dict and run_id:
            try:
                results_dict[f"{run_id}:{binder_name}:_status"] = {
                    "status": "complete",
                    "timestamp": time_module.time(),
                }
            except Exception:
                pass  # Non-critical

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        import traceback
        print(f"Error processing {binder_name}: {traceback.format_exc()}")
        
        # Stream error status
        if stream_to_dict and run_id:
            try:
                results_dict[f"{run_id}:{binder_name}:_status"] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": time_module.time(),
                }
            except Exception:
                pass

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return results


# GPU-specific Modal functions
# Each GPU type gets its own pre-defined function for clean Modal integration

@app.function(image=image, gpu="T4", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_boltz_T4(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_boltz_and_ipsae_impl(task_dict)

@app.function(image=image, gpu="L4", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_boltz_L4(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_boltz_and_ipsae_impl(task_dict)

@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_boltz_A10G(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_boltz_and_ipsae_impl(task_dict)

@app.function(image=image, gpu="L40S", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_boltz_L40S(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_boltz_and_ipsae_impl(task_dict)

@app.function(image=image, gpu="A100", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_boltz_A100_40GB(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_boltz_and_ipsae_impl(task_dict)

@app.function(image=image, gpu="A100-80GB", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_boltz_A100_80GB(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_boltz_and_ipsae_impl(task_dict)

@app.function(image=image, gpu="H100", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_boltz_H100(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_boltz_and_ipsae_impl(task_dict)

@app.function(image=image, gpu="H200", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_boltz_H200(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_boltz_and_ipsae_impl(task_dict)

@app.function(image=image, gpu="B200", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_boltz_B200(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_boltz_and_ipsae_impl(task_dict)

# Default alias
run_boltz_and_ipsae = run_boltz_A100_80GB

# Map GPU names to functions
GPU_FUNCTIONS = {
    "T4": run_boltz_T4,
    "L4": run_boltz_L4,
    "A10G": run_boltz_A10G,
    "L40S": run_boltz_L40S,
    "A100": run_boltz_A100_40GB,
    "A100-40GB": run_boltz_A100_40GB,
    "A100-80GB": run_boltz_A100_80GB,
    "H100": run_boltz_H100,
    "H200": run_boltz_H200,
    "B200": run_boltz_B200,
}


def _run_single_prediction(
    binder_sequences: List[str],
    partner_name: str,
    partner_role: str,
    partner_sequences: List[str],
    partner_msa: Optional[str],
    work_dir: Path,
    boltz_cache: Path,
    recycling_steps: int,
    diffusion_samples: int,
    pae_cutoff: float,
    dist_cutoff: float,
    use_msa_server: str,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Run a single Boltz + ipSAE prediction for binder vs one partner.

    Returns:
        Dict with partner_name, role, n_models, metrics, per_model, structures, error
    """
    import numpy as np

    result = {
        "partner_name": partner_name,
        "role": partner_role,
        "n_models": 0,
        "metrics": {},
        "per_model": [],
        "structures": {},
        "error": None,
    }

    try:
        # Create prediction-specific work directory
        pred_work_dir = work_dir / f"pred_vs_{partner_role}_{partner_name}"
        pred_work_dir.mkdir(parents=True, exist_ok=True)

        # Create YAML config
        yaml_content = _create_yaml(
            binder_sequences=binder_sequences,
            partner_name=partner_name,
            partner_role=partner_role,
            partner_sequences=partner_sequences,
            partner_msa=partner_msa,
            work_dir=pred_work_dir,
        )

        yaml_stem = f"binder_vs_{partner_role}_{partner_name}"
        yaml_file = pred_work_dir / f"{yaml_stem}.yaml"
        yaml_file.write_text(yaml_content)

        # Run Boltz prediction
        output_dir = pred_work_dir / "boltz_output"
        output_dir.mkdir(exist_ok=True)

        # Determine whether to use MSA server
        should_use_msa = _decide_use_msa_server(use_msa_server, partner_msa)

        cmd = [
            "boltz", "predict",
            str(yaml_file),
            "--recycling_steps", str(recycling_steps),
            "--diffusion_samples", str(diffusion_samples),
            "--out_dir", str(output_dir),
            "--cache", str(boltz_cache),
            "--no_kernels",  # Use pure PyTorch, avoids cuequivariance issues
        ]

        if should_use_msa:
            cmd.append("--use_msa_server")

        if verbose:
            print(f"    Boltz cmd: {' '.join(cmd)}")
            if partner_msa:
                print(f"    MSA: provided ({len(partner_msa)} chars), use_msa_server={should_use_msa}")
            else:
                print(f"    MSA: None, use_msa_server={should_use_msa}")

        boltz_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(pred_work_dir),
        )

        if boltz_result.returncode != 0:
            result["error"] = f"Boltz failed: {boltz_result.stderr[:500]}"
            return result

        # Find prediction outputs
        pred_dir = output_dir / f"boltz_results_{yaml_stem}" / "predictions" / yaml_stem

        if not pred_dir.exists():
            # Try alternative path structures
            for possible_dir in output_dir.rglob("predictions"):
                if possible_dir.is_dir():
                    subdirs = list(possible_dir.iterdir())
                    if subdirs:
                        pred_dir = subdirs[0]
                        break

        if not pred_dir.exists():
            result["error"] = "Prediction directory not found"
            return result

        # Process each model
        all_metrics = []
        pae_files = sorted(pred_dir.glob("pae_*.npz"))

        for pae_file in pae_files:
            match = re.search(r"model_(\d+)", pae_file.name)
            if not match:
                continue

            model_idx = match.group(1)
            cif_file = pred_dir / pae_file.name.replace("pae_", "").replace(".npz", ".cif")

            if not cif_file.exists():
                continue

            # Run ipSAE
            ipsae_metrics = _run_ipsae(
                pae_file=pae_file,
                cif_file=cif_file,
                pae_cutoff=pae_cutoff,
                dist_cutoff=dist_cutoff,
                verbose=verbose,
            )

            if ipsae_metrics:
                ipsae_metrics["model"] = int(model_idx)
                all_metrics.append(ipsae_metrics)

                # Store structure
                result["structures"][f"model_{model_idx}"] = cif_file.read_text()

        if not all_metrics:
            result["error"] = "No valid ipSAE metrics computed"
            return result

        result["n_models"] = len(all_metrics)
        result["per_model"] = all_metrics
        result["metrics"] = _compute_summary_stats(all_metrics)

    except Exception as e:
        result["error"] = str(e)

    return result


def _decide_use_msa_server(mode: str, partner_msa: Optional[str]) -> bool:
    """Decide whether to use MSA server based on mode and available MSAs."""
    if mode == "true":
        return True
    if mode == "false":
        return False
    # auto: use server only if no explicit MSA provided
    return partner_msa is None


def _create_yaml(
    binder_sequences: List[str],
    partner_name: str,
    partner_role: str,
    partner_sequences: List[str],
    partner_msa: Optional[str],
    work_dir: Path,
) -> str:
    """
    Create YAML config for Boltz prediction.

    Chain ID assignment:
    - Binder chains: A, B, C, ...
    - Partner chains: TA, TB, TC, ... (for target/antitarget)
                  or SA, SB, SC, ... (for self)
    """
    lines = ["version: 1", "sequences:"]

    # Add binder chains
    for i, seq in enumerate(binder_sequences):
        chain_id = chr(ord("A") + i)
        lines.append("  - protein:")
        lines.append(f"      id: {chain_id}")
        lines.append(f"      sequence: {seq}")
        lines.append("      msa: empty")

    # Partner chain ID prefix based on role
    if partner_role == "self":
        prefix = "S"
    elif partner_role == "antitarget":
        prefix = "X"  # X for antitarget to avoid confusion
    else:
        prefix = "T"  # Target

    # Add partner chains
    msa_file = None
    if partner_msa:
        msa_file = work_dir / f"{partner_name}.a3m"
        msa_file.write_text(partner_msa)

    for i, seq in enumerate(partner_sequences):
        chain_id = f"{prefix}{chr(ord('A') + i)}"
        lines.append("  - protein:")
        lines.append(f"      id: {chain_id}")
        lines.append(f"      sequence: {seq}")

        # Only first partner chain gets MSA
        if i == 0 and msa_file:
            lines.append(f"      msa: {msa_file}")
        else:
            lines.append("      msa: empty")

    return "\n".join(lines) + "\n"


def _run_ipsae(
    pae_file: Path,
    cif_file: Path,
    pae_cutoff: float,
    dist_cutoff: float,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run ipSAE calculation and parse results."""
    cmd = [
        "python", "/root/ipsae.py",
        str(pae_file),
        str(cif_file),
        str(int(pae_cutoff)),
        str(int(dist_cutoff)),
    ]

    if verbose:
        print(f"      Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if verbose:
            print(f"      ipSAE failed: {result.stderr[:200]}")
        return None

    # Find output file
    pae_str = f"{int(pae_cutoff):02d}" if pae_cutoff < 10 else str(int(pae_cutoff))
    dist_str = f"{int(dist_cutoff):02d}" if dist_cutoff < 10 else str(int(dist_cutoff))
    out_txt = str(cif_file).replace(".cif", f"_{pae_str}_{dist_str}.txt")

    if not Path(out_txt).exists():
        # Try alternative patterns
        cif_dir = cif_file.parent
        txt_files = list(cif_dir.glob(f"*_{pae_str}_{dist_str}.txt"))
        if txt_files:
            out_txt = str(txt_files[0])
        else:
            return None

    with open(out_txt) as f:
        output_content = f.read()

    return _parse_ipsae_output(output_content)


def _parse_ipsae_output(output: str) -> Dict[str, Any]:
    """
    Parse ipSAE output file to extract all metrics.

    The output format is:
    Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ...
    A    TA    15  15    asym   0.1234   ...
    A    TA    15  15    max    0.1234   ...
    """
    metrics = {}

    lines = output.strip().split("\n")
    asym_values = []
    max_row = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith(("Chn1", "#")):
            continue

        parts = line.split()
        if len(parts) < 13:
            continue

        try:
            row_type = parts[4].lower()

            if row_type == "max":
                max_row = {
                    "ipSAE": float(parts[5]),
                    "ipSAE_d0chn": float(parts[6]),
                    "ipSAE_d0dom": float(parts[7]),
                    "ipTM_af": float(parts[8]),
                    "pDockQ": float(parts[10]),
                    "pDockQ2": float(parts[11]),
                    "LIS": float(parts[12]),
                }
                if len(parts) > 13:
                    max_row["n0res"] = int(parts[13])
                if len(parts) > 14:
                    max_row["n0chn"] = int(parts[14])
            elif row_type == "asym":
                asym_values.append(float(parts[5]))

        except (ValueError, IndexError):
            continue

    if max_row:
        metrics.update(max_row)

    if asym_values:
        metrics["ipSAE_min"] = min(asym_values)
        metrics["ipSAE_max"] = max(asym_values)

    return metrics


def _compute_summary_stats(all_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute mean and std for all metrics across models."""
    import numpy as np

    summary = {}

    keys = ["ipSAE", "ipSAE_min", "ipSAE_max", "ipSAE_d0chn", "ipSAE_d0dom",
            "ipTM_af", "pDockQ", "pDockQ2", "LIS"]

    for key in keys:
        values = [m.get(key) for m in all_metrics if m.get(key) is not None]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))

    return summary


# =============================================================================
# LOCAL ENTRYPOINTS
# =============================================================================

@app.local_entrypoint()
def run_pipeline(
    # Binder inputs (one required)
    binder_csv: Optional[str] = None,
    binder_fasta: Optional[str] = None,
    binder_fasta_dir: Optional[str] = None,
    binder_name: Optional[str] = None,
    binder_seq: Optional[str] = None,
    # CSV column configuration
    binder_name_col: str = "name",
    binder_seq_col: str = "sequence",
    # Binder options
    add_n_terminal_lysine: bool = False,
    # Target (required)
    target_name: str = "target",
    target_fasta: Optional[str] = None,
    target_seq: Optional[str] = None,
    target_msa: Optional[str] = None,
    # Antitarget (optional)
    antitarget_name: Optional[str] = None,
    antitarget_fasta: Optional[str] = None,
    antitarget_seq: Optional[str] = None,
    antitarget_msa: Optional[str] = None,
    # Controls
    include_self: bool = False,
    # Boltz parameters
    recycling_steps: int = 10,
    diffusion_samples: int = 5,
    use_msa_server: str = "auto",
    # ipSAE parameters
    pae_cutoff: float = 15.0,
    dist_cutoff: float = 15.0,
    # GPU configuration
    gpu: str = DEFAULT_GPU,
    max_parallel: int = 10,
    # Output
    output_dir: Optional[str] = None,
    verbose: bool = False,
    # Streaming options (streaming is ON by default)
    no_stream: bool = False,
    run_id: Optional[str] = None,
    sync_interval: float = 5.0,
):
    """
    Run the full Boltz2 + ipSAE pipeline with parallel processing.

    All inputs are read locally and passed to Modal.
    All outputs are returned and saved locally.

    Examples:
        # Single binder with target only
        modal run modal_boltz_ipsae.py::run_pipeline \\
            --binder-name "my_binder" \\
            --binder-seq "MKTAYIAK..." \\
            --target-fasta target.fasta

        # Multiple binders with antitarget and self-binding control
        modal run modal_boltz_ipsae.py::run_pipeline \\
            --binder-csv binders.csv \\
            --target-fasta target.fasta \\
            --antitarget-fasta antitarget.fasta \\
            --include-self \\
            --output-dir ./results

        # With custom GPU and high parallelism
        modal run modal_boltz_ipsae.py::run_pipeline \\
            --binder-csv binders.csv \\
            --target-fasta target.fasta \\
            --gpu "H100" \\
            --max-parallel 20

        # From FASTA directory with N-terminal lysine
        modal run modal_boltz_ipsae.py::run_pipeline \\
            --binder-fasta-dir ./binders/ \\
            --target-fasta target.fasta \\
            --add-n-terminal-lysine
            
        # Disable real-time streaming (results only saved at end)
        modal run modal_boltz_ipsae.py::run_pipeline \\
            --binder-csv binders.csv \\
            --target-fasta target.fasta \\
            --output-dir ./results \\
            --no-stream
    """
    
    # Convert no_stream flag to stream boolean for internal use
    stream = not no_stream

    # Parse binders
    binders = _parse_binder_inputs(
        csv_path=binder_csv,
        fasta_path=binder_fasta,
        fasta_dir=binder_fasta_dir,
        name=binder_name,
        seq=binder_seq,
        name_col=binder_name_col,
        seq_col=binder_seq_col,
        add_k=add_n_terminal_lysine,
    )

    if not binders:
        print("Error: No binders provided.")
        print("Use one of: --binder-csv, --binder-fasta, --binder-fasta-dir, or --binder-name/--binder-seq")
        return

    # Parse target
    target_sequences = _read_sequences(target_fasta, target_seq, "target")
    if not target_sequences:
        print("Error: No target provided. Use --target-fasta or --target-seq")
        return

    target_msa_content = None
    if target_msa:
        target_msa_content = Path(target_msa).read_text()

    # Build partners list
    partners = [
        {
            "name": target_name,
            "role": "target",
            "sequences": target_sequences,
            "msa_content": target_msa_content,
        }
    ]

    # Parse antitarget (optional)
    if antitarget_fasta or antitarget_seq:
        antitarget_sequences = _read_sequences(antitarget_fasta, antitarget_seq, "antitarget")
        if antitarget_sequences:
            antitarget_msa_content = None
            if antitarget_msa:
                antitarget_msa_content = Path(antitarget_msa).read_text()

            partners.append({
                "name": antitarget_name or "antitarget",
                "role": "antitarget",
                "sequences": antitarget_sequences,
                "msa_content": antitarget_msa_content,
            })

    # Setup output directory
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup streaming
    effective_run_id = None
    sync_thread = None
    stop_sync = None
    
    if stream:
        effective_run_id = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_path:
            output_path = Path(f"./results_{effective_run_id}")
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Streaming enabled but no --output-dir specified. Using: {output_path}")

    # Print configuration
    print(f"\n{'='*70}")
    print("BOLTZ2 + ipSAE PIPELINE (Modal)")
    print(f"{'='*70}")
    print(f"Binders: {len(binders)}")
    print(f"Target: {target_name} ({len(target_sequences)} chain(s))")
    if len(partners) > 1:
        print(f"Antitarget: {partners[1]['name']} ({len(partners[1]['sequences'])} chain(s))")
    if include_self:
        print("Self-binding: enabled")
    print(f"GPU: {gpu}")
    print(f"Max parallel: {max_parallel}")
    print(f"Output: {output_path or 'stdout only'}")
    if stream:
        print(f"Streaming: ENABLED (run_id={effective_run_id})")
        print(f"  Results will be saved to {output_path} as they complete")
    print(f"{'='*70}\n")

    # Build task list
    tasks = []
    for binder in binders:
        # Build partner list for this binder
        binder_partners = list(partners)  # Copy base partners

        # Add self if requested
        if include_self:
            binder_partners.append({
                "name": binder["name"],
                "role": "self",
                "sequences": binder["sequences"],
                "msa_content": None,
            })

        task = {
            "binder_name": binder["name"],
            "binder_sequences": binder["sequences"],
            "partners": binder_partners,
            "recycling_steps": recycling_steps,
            "diffusion_samples": diffusion_samples,
            "pae_cutoff": pae_cutoff,
            "dist_cutoff": dist_cutoff,
            "use_msa_server": use_msa_server,
            "verbose": verbose,
            # Streaming options
            "run_id": effective_run_id,
            "stream_to_dict": stream,
        }
        tasks.append(task)

    # Run in parallel using .map() with selected GPU
    print(f"Submitting {len(tasks)} binder task(s) to Modal...")
    print(f"Processing will run in parallel (up to {max_parallel} concurrent).\n")

    # Select function based on GPU
    if gpu not in GPU_FUNCTIONS:
        print(f"Error: Unknown GPU type '{gpu}'. Available: {', '.join(GPU_FUNCTIONS.keys())}")
        return
    
    configured_fn = GPU_FUNCTIONS[gpu]
    
    # Start background sync thread if streaming enabled
    if stream and output_path:
        stop_sync = threading.Event()
        sync_thread = threading.Thread(
            target=_sync_worker,
            args=(effective_run_id, output_path, stop_sync, sync_interval),
            daemon=True,
        )
        sync_thread.start()
        print(f"Background sync started (polling every {sync_interval}s)\n")

    # Execute tasks with concurrency limit
    all_results = []
    
    def _run_task_safe(task_input):
        """Run a single task safely, capturing exceptions."""
        try:
            return configured_fn.remote(task_input)
        except Exception as e:
            return e

    print(f"Executing {len(tasks)} tasks (concurrency limit: {max_parallel})...")
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(_run_task_safe, t): t for t in tasks}
        
        # Wait for results as they complete
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            all_results.append(result)
            
            # Print progress
            if not isinstance(result, Exception):
                name = result.get("binder_name", "unknown")
                print(f"[{i+1}/{len(tasks)}] Finished: {name}")
            else:
                name = futures[future]["binder_name"]
                print(f"[{i+1}/{len(tasks)}] Failed: {name}")

    # Stop sync thread
    if sync_thread is not None:
        print("\nStopping background sync...")
        stop_sync.set()
        sync_thread.join(timeout=30)
        print("Background sync stopped.")

    # Process results
    successful = []
    failed = []

    for i, result in enumerate(all_results):
        binder_name_i = tasks[i]["binder_name"]

        if isinstance(result, Exception):
            failed.append({"binder_name": binder_name_i, "error": str(result)})
            print(f"✗ {binder_name_i}: Exception - {str(result)[:100]}")
        elif result.get("status") == "success":
            successful.append(result)
            # Print summary
            target_m = result.get("target", {}).get("metrics", {})
            print(f"✓ {binder_name_i}: target ipSAE={target_m.get('ipSAE_mean', 0):.3f}")
        else:
            failed.append({"binder_name": binder_name_i, "error": result.get("error", "Unknown")})
            print(f"✗ {binder_name_i}: {result.get('error', 'Unknown')[:100]}")

    # Print final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Successful: {len(successful)}/{len(all_results)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed[:5]:  # Show first 5 failures
            print(f"  - {f['binder_name']}: {f.get('error', 'Unknown')[:50]}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    # Save results
    if output_path and successful:
        print(f"\nSaving results to: {output_path}")

        for result in successful:
            _save_binder_results(output_path, result)

        _save_harmonized_summary_csv(output_path, successful)
        print(f"  - Summary CSV: {output_path / 'binder_pair_summary.csv'}")

    return all_results


def _parse_binder_inputs(
    csv_path: Optional[str],
    fasta_path: Optional[str],
    fasta_dir: Optional[str],
    name: Optional[str],
    seq: Optional[str],
    name_col: str,
    seq_col: str,
    add_k: bool,
) -> List[Dict[str, Any]]:
    """Parse binder inputs from various sources."""
    binders = []

    if csv_path:
        binders.extend(_load_binders_from_csv(Path(csv_path), name_col, seq_col))

    if fasta_path:
        binders.extend(_load_binders_from_fasta(Path(fasta_path)))

    if fasta_dir:
        binders.extend(_load_binders_from_fasta_dir(Path(fasta_dir)))

    if name and seq:
        sequences = [s.strip() for s in seq.split(":") if s.strip()]
        binders.append({"name": name, "sequences": sequences})

    # Add N-terminal lysine if requested
    if add_k:
        for binder in binders:
            binder["sequences"] = [
                seq if seq.startswith("K") else "K" + seq
                for seq in binder["sequences"]
            ]

    return binders


def _load_binders_from_csv(csv_path: Path, name_col: str, seq_col: str) -> List[Dict[str, Any]]:
    """Load binders from CSV file."""
    binders = []

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]

        # Find name and sequence columns
        effective_name_col = name_col
        effective_seq_col = seq_col

        # Try exact match first
        if name_col not in fieldnames:
            # Try case-insensitive match
            for fn in fieldnames:
                if fn.lower() == name_col.lower():
                    effective_name_col = fn
                    break
            else:
                # Fallback to first column
                effective_name_col = fieldnames[0] if fieldnames else None

        if seq_col not in fieldnames:
            for fn in fieldnames:
                if fn.lower() == seq_col.lower() or "seq" in fn.lower():
                    effective_seq_col = fn
                    break
            else:
                # Fallback to second column
                effective_seq_col = fieldnames[1] if len(fieldnames) > 1 else None

        if not effective_name_col or not effective_seq_col:
            raise ValueError(f"Could not identify name/sequence columns in {csv_path}")

        for row in reader:
            name = row.get(effective_name_col, "").strip()
            seq = row.get(effective_seq_col, "").strip()
            if name and seq:
                sequences = [s.strip().upper() for s in seq.split(":") if s.strip()]
                binders.append({"name": _sanitize_name(name), "sequences": sequences})

    return binders


def _load_binders_from_fasta(fasta_path: Path) -> List[Dict[str, Any]]:
    """Load binders from FASTA file (multi-entry FASTA, one binder per entry)."""
    binders = []
    current_name = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name and current_seq:
                    binders.append({
                        "name": _sanitize_name(current_name),
                        "sequences": ["".join(current_seq).upper()]
                    })
                current_name = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)

        if current_name and current_seq:
            binders.append({
                "name": _sanitize_name(current_name),
                "sequences": ["".join(current_seq).upper()]
            })

    return binders


def _load_binders_from_fasta_dir(fasta_dir: Path) -> List[Dict[str, Any]]:
    """Load binders from directory of FASTA files (one binder per file)."""
    binders = []

    if not fasta_dir.is_dir():
        print(f"Warning: FASTA directory not found: {fasta_dir}")
        return binders

    fasta_files = sorted(fasta_dir.glob("*.fasta")) + sorted(fasta_dir.glob("*.fa"))

    for fasta_file in fasta_files:
        name = fasta_file.stem
        sequences = []
        current_seq = []

        with open(fasta_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq).upper())
                        current_seq = []
                elif line:
                    current_seq.append(line)

            if current_seq:
                sequences.append("".join(current_seq).upper())

        if sequences:
            binders.append({"name": _sanitize_name(name), "sequences": sequences})

    return binders


def _read_sequences(fasta_path: Optional[str], seq: Optional[str], name: str) -> Optional[List[str]]:
    """Read sequence(s) from FASTA file or direct string."""
    if seq:
        return [s.strip().upper() for s in seq.split(":") if s.strip()]

    if fasta_path:
        path = Path(fasta_path)
        if not path.exists():
            print(f"Warning: {name} FASTA not found: {fasta_path}")
            return None

        sequences = []
        current_seq = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq).upper())
                        current_seq = []
                elif line:
                    current_seq.append(line)

            if current_seq:
                sequences.append("".join(current_seq).upper())

        return sequences if sequences else None

    return None


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use in filenames."""
    # Replace problematic characters
    name = re.sub(r"[^\w\-.]", "_", name)
    # Remove consecutive underscores
    name = re.sub(r"_+", "_", name)
    # Strip leading/trailing underscores
    name = name.strip("_")
    return name


def _save_binder_results(output_path: Path, result: Dict[str, Any]):
    """Save results for a single binder."""
    binder_name = result["binder_name"]
    binder_dir = output_path / f"binder_{binder_name}"
    binder_dir.mkdir(parents=True, exist_ok=True)

    # Save comprehensive metrics JSON
    metrics_data = {
        "binder_name": binder_name,
        "status": result["status"],
    }

    for role in ["target", "antitarget", "self"]:
        partner_result = result.get(role)
        if partner_result:
            metrics_data[role] = {
                "partner_name": partner_result.get("partner_name"),
                "n_models": partner_result.get("n_models", 0),
                "metrics": partner_result.get("metrics", {}),
                "per_model": partner_result.get("per_model", []),
            }

            # Save structures
            structures = partner_result.get("structures", {})
            if structures:
                struct_dir = binder_dir / f"structures_{role}"
                struct_dir.mkdir(exist_ok=True)
                for model_name, cif_content in structures.items():
                    cif_file = struct_dir / f"{binder_name}_vs_{role}_{model_name}.cif"
                    cif_file.write_text(cif_content)

    metrics_file = binder_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)


def _save_harmonized_summary_csv(output_path: Path, results: List[Dict[str, Any]]):
    """
    Save summary CSV in the harmonized format matching run_ipsae_pipeline.py.

    Format: binder_name, n_target_models, n_antitarget_models, n_self_models,
            target_ipSAE_mean, target_ipSAE_std, ... (all metrics for each partner type)
    """
    csv_file = output_path / "binder_pair_summary.csv"

    # Define header to match run_ipsae_pipeline.py
    header = [
        "binder_name",
        "n_target_models", "n_antitarget_models", "n_self_models",
        # Target metrics
        "target_ipSAE_mean", "target_ipSAE_std",
        "target_ipSAE_min_mean", "target_ipSAE_min_std",
        "target_ipSAE_max_mean", "target_ipSAE_max_std",
        "target_ipTM_af_mean", "target_ipTM_af_std",
        "target_pDockQ2_mean", "target_pDockQ2_std",
        # Antitarget metrics
        "antitarget_ipSAE_mean", "antitarget_ipSAE_std",
        "antitarget_ipSAE_min_mean", "antitarget_ipSAE_min_std",
        "antitarget_ipSAE_max_mean", "antitarget_ipSAE_max_std",
        "antitarget_ipTM_af_mean", "antitarget_ipTM_af_std",
        "antitarget_pDockQ2_mean", "antitarget_pDockQ2_std",
        # Self metrics
        "self_ipSAE_mean", "self_ipSAE_std",
        "self_ipSAE_min_mean", "self_ipSAE_min_std",
        "self_ipSAE_max_mean", "self_ipSAE_max_std",
        "self_ipTM_af_mean", "self_ipTM_af_std",
        "self_pDockQ2_mean", "self_pDockQ2_std",
    ]

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for result in results:
            row = [result["binder_name"]]

            # Model counts
            target_r = result.get("target", {})
            antitarget_r = result.get("antitarget", {})
            self_r = result.get("self", {})

            row.append(target_r.get("n_models", 0))
            row.append(antitarget_r.get("n_models", 0) if antitarget_r else 0)
            row.append(self_r.get("n_models", 0) if self_r else 0)

            # Target metrics
            tm = target_r.get("metrics", {})
            row.extend([
                tm.get("ipSAE_mean"), tm.get("ipSAE_std"),
                tm.get("ipSAE_min_mean"), tm.get("ipSAE_min_std"),
                tm.get("ipSAE_max_mean"), tm.get("ipSAE_max_std"),
                tm.get("ipTM_af_mean"), tm.get("ipTM_af_std"),
                tm.get("pDockQ2_mean"), tm.get("pDockQ2_std"),
            ])

            # Antitarget metrics
            am = antitarget_r.get("metrics", {}) if antitarget_r else {}
            row.extend([
                am.get("ipSAE_mean"), am.get("ipSAE_std"),
                am.get("ipSAE_min_mean"), am.get("ipSAE_min_std"),
                am.get("ipSAE_max_mean"), am.get("ipSAE_max_std"),
                am.get("ipTM_af_mean"), am.get("ipTM_af_std"),
                am.get("pDockQ2_mean"), am.get("pDockQ2_std"),
            ])

            # Self metrics
            sm = self_r.get("metrics", {}) if self_r else {}
            row.extend([
                sm.get("ipSAE_mean"), sm.get("ipSAE_std"),
                sm.get("ipSAE_min_mean"), sm.get("ipSAE_min_std"),
                sm.get("ipSAE_max_mean"), sm.get("ipSAE_max_std"),
                sm.get("ipTM_af_mean"), sm.get("ipTM_af_std"),
                sm.get("pDockQ2_mean"), sm.get("pDockQ2_std"),
            ])

            writer.writerow(row)

    print(f"Summary CSV saved: {csv_file}")


# =============================================================================
# STREAMING SYNC FUNCTIONS
# =============================================================================

def _sync_worker(
    run_id: str,
    output_path: Path,
    stop_event: threading.Event,
    interval: float = 5.0,
):
    """
    Background worker that polls the Modal Dict and saves results locally.
    
    Runs in a separate thread, polling every `interval` seconds until
    `stop_event` is set.
    """
    synced_keys = set()
    sync_count = 0
    
    while not stop_event.is_set():
        try:
            # Get all keys for this run
            all_keys = [k for k in results_dict.keys() if k.startswith(f"{run_id}:")]
            
            # Find new keys (exclude status markers)
            new_keys = [k for k in all_keys if k not in synced_keys and not k.endswith(":_status")]
            
            for key in new_keys:
                try:
                    result = results_dict[key]
                    _save_streamed_result(output_path, result)
                    synced_keys.add(key)
                    sync_count += 1
                    binder = result.get("binder_name", "?")
                    role = result.get("partner_role", "?")
                    print(f"  [SYNC] ✓ {binder}:{role} saved ({sync_count} total)")
                except Exception as e:
                    print(f"  [SYNC] ✗ Error syncing {key}: {e}")
            
        except Exception as e:
            # Dict access can fail if not yet created, etc.
            pass
        
        # Wait for next poll interval or stop signal
        stop_event.wait(timeout=interval)
    
    # Final sync after stop signal
    try:
        all_keys = [k for k in results_dict.keys() if k.startswith(f"{run_id}:")]
        new_keys = [k for k in all_keys if k not in synced_keys and not k.endswith(":_status")]
        
        for key in new_keys:
            try:
                result = results_dict[key]
                _save_streamed_result(output_path, result)
                synced_keys.add(key)
                sync_count += 1
            except Exception:
                pass
    except Exception:
        pass
    
    print(f"  [SYNC] Final count: {sync_count} results synced")


def _save_streamed_result(output_path: Path, result: Dict[str, Any]):
    """Save a single streamed result from Dict to local filesystem."""
    binder_name = result.get("binder_name", "unknown")
    partner_role = result.get("partner_role", "unknown")
    
    binder_dir = output_path / f"binder_{binder_name}"
    binder_dir.mkdir(parents=True, exist_ok=True)
    
    # Save structures
    structures = result.get("structures", {})
    if structures:
        struct_dir = binder_dir / f"structures_{partner_role}"
        struct_dir.mkdir(exist_ok=True)
        for model_name, cif_content in structures.items():
            cif_file = struct_dir / f"{binder_name}_vs_{partner_role}_{model_name}.cif"
            cif_file.write_text(cif_content)
    
    # Update metrics JSON (append to existing or create new)
    metrics_file = binder_dir / "metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = {"binder_name": binder_name, "status": "in_progress"}
    else:
        metrics_data = {"binder_name": binder_name, "status": "in_progress"}
    
    metrics_data[partner_role] = {
        "partner_name": result.get("partner_name"),
        "n_models": result.get("n_models", 0),
        "metrics": result.get("metrics", {}),
        "per_model": result.get("per_model", []),
        "error": result.get("error"),
    }
    
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)


# =============================================================================
# UTILITY ENTRYPOINTS
# =============================================================================

@app.local_entrypoint()
def init_cache():
    """
    Initialize the Boltz cache (download model weights and CCD data).
    Run this ONCE before using the pipeline.

    Usage:
        modal run modal_boltz_ipsae.py::init_cache
    """
    print("Initializing Boltz cache...")
    print("This downloads ~2GB of model weights and CCD data.")
    print("It only needs to be done once.\n")

    result = initialize_cache.remote()
    print(result)


@app.local_entrypoint()
def test_connection(gpu: str = DEFAULT_GPU):
    """
    Test Modal connection and GPU availability.

    Usage:
        modal run modal_boltz_ipsae.py::test_connection
        modal run modal_boltz_ipsae.py::test_connection --gpu H100
    """
    print(f"Testing Modal connection with GPU: {gpu}...")
    
    if gpu not in GPU_FUNCTIONS:
        print(f"Error: Unknown GPU type '{gpu}'. Available: {', '.join(GPU_FUNCTIONS.keys())}")
        return
    
    # Use the GPU-specific function to test (just run with empty task to verify GPU)
    result = _test_gpu.remote()
    print(f"\nGPU Info:\n{result}")
    print(f"\nNote: Test used default GPU. Your pipeline will use: {gpu}")


@app.function(image=image, gpu=DEFAULT_GPU, timeout=60)
def _test_gpu() -> str:
    """Test GPU availability."""
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    return result.stdout


@app.local_entrypoint()
def list_gpus():
    """
    List available GPU types and their specifications.

    Usage:
        modal run modal_boltz_ipsae.py::list_gpus
    """
    print("\nSupported GPU Types:")
    print("=" * 60)
    for gpu_type, description in GPU_TYPES.items():
        print(f"  {gpu_type:15s} - {description}")
    print(f"\nDefault: {DEFAULT_GPU}")
    print("\nUsage examples:")
    print("  --gpu H100           # Single H100 (fastest)")
    print("  --gpu A100-80GB      # A100 80GB (default, good balance)")
    print("  --gpu L40S           # L40S (cost-effective)")
    print("  --gpu B200           # B200 (largest memory)")


@app.local_entrypoint()
def convert_fasta_to_csv(
    fasta_file: str,
    output_csv: Optional[str] = None,
):
    """
    Convert a FASTA file to CSV format.

    Usage:
        modal run modal_boltz_ipsae.py::convert_fasta_to_csv --fasta-file binders.fasta
    """
    fasta_path = Path(fasta_file)
    if not fasta_path.exists():
        print(f"Error: File not found: {fasta_file}")
        return

    binders = _load_binders_from_fasta(fasta_path)

    csv_path = Path(output_csv) if output_csv else fasta_path.with_suffix(".csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "sequence"])
        for binder in binders:
            seq = ":".join(binder["sequences"])
            writer.writerow([binder["name"], seq])

    print(f"Converted {len(binders)} sequences to: {csv_path}")


@app.local_entrypoint()
def sync_results(
    run_id: str,
    output_dir: str = "./results",
    poll: bool = False,
    interval: float = 5.0,
    timeout: float = 3600.0,
):
    """
    Sync results from Modal Dict to local filesystem.
    
    Use this to manually sync results or as a backup if streaming was interrupted.
    
    Usage:
        # One-shot sync (get current results for a run)
        modal run modal_boltz_ipsae.py::sync_results --run-id 20241129_143022
        
        # Continuous polling (run alongside pipeline if using --no-stream)
        modal run modal_boltz_ipsae.py::sync_results --run-id 20241129_143022 --poll
        
        # Custom output directory and interval
        modal run modal_boltz_ipsae.py::sync_results --run-id 20241129_143022 \\
            --output-dir ./my_results --poll --interval 10
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    synced_keys = set()
    start_time = time.time()
    
    print(f"Syncing results for run: {run_id}")
    print(f"Output directory: {output_path}")
    if poll:
        print(f"Polling every {interval}s (timeout: {timeout}s)")
        print("Press Ctrl+C to stop\n")
    
    while True:
        try:
            all_keys = [k for k in results_dict.keys() if k.startswith(f"{run_id}:")]
        except Exception as e:
            print(f"Error listing keys: {e}")
            if not poll:
                break
            time.sleep(interval)
            continue
        
        new_keys = [k for k in all_keys if k not in synced_keys and not k.endswith(":_status")]
        
        for key in new_keys:
            try:
                result = results_dict[key]
                _save_streamed_result(output_path, result)
                synced_keys.add(key)
                binder = result.get("binder_name", "?")
                role = result.get("partner_role", "?")
                print(f"✓ {binder}:{role}")
            except Exception as e:
                print(f"✗ Error syncing {key}: {e}")
        
        if not poll:
            break
        
        if time.time() - start_time > timeout:
            print(f"\nTimeout reached ({timeout}s)")
            break
        
        time.sleep(interval)
    
    print(f"\nSynced {len(synced_keys)} results to {output_path}")


@app.local_entrypoint()
def list_runs():
    """
    List all run IDs in the results Dict.
    
    Usage:
        modal run modal_boltz_ipsae.py::list_runs
    """
    try:
        all_keys = list(results_dict.keys())
    except Exception as e:
        print(f"Error: {e}")
        return
    
    if not all_keys:
        print("No runs found in results Dict.")
        return
    
    # Extract unique run IDs and count entries
    run_counts = {}
    for key in all_keys:
        parts = key.split(":")
        if len(parts) >= 1:
            rid = parts[0]
            run_counts[rid] = run_counts.get(rid, 0) + 1
    
    print("Available runs in Dict:")
    print("=" * 50)
    for rid in sorted(run_counts.keys()):
        print(f"  {rid}: {run_counts[rid]} entries")
    print(f"\nTotal: {len(run_counts)} runs, {len(all_keys)} entries")


@app.local_entrypoint()
def clear_results(
    run_id: Optional[str] = None,
    confirm: bool = False,
):
    """
    Clear results from the Dict.
    
    Usage:
        # Clear specific run
        modal run modal_boltz_ipsae.py::clear_results --run-id 20241129_143022 --confirm
        
        # Clear ALL results (use with caution!)
        modal run modal_boltz_ipsae.py::clear_results --confirm
    """
    if not confirm:
        print("This will delete results from the Dict.")
        print("Add --confirm to proceed.")
        if run_id:
            print(f"Target: run_id={run_id}")
        else:
            print("Target: ALL runs (dangerous!)")
        return
    
    try:
        all_keys = list(results_dict.keys())
    except Exception as e:
        print(f"Error: {e}")
        return
    
    if run_id:
        keys_to_delete = [k for k in all_keys if k.startswith(f"{run_id}:")]
    else:
        keys_to_delete = all_keys
    
    if not keys_to_delete:
        print("No matching entries found.")
        return
    
    print(f"Deleting {len(keys_to_delete)} entries...")
    for key in keys_to_delete:
        try:
            del results_dict[key]
        except Exception:
            pass
    
    print(f"Deleted {len(keys_to_delete)} entries.")
