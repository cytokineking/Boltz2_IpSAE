#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize Boltz/ipSAE validation outputs.

Key assumptions / behaviour:

- Boltz configs use chain IDs:
    * Binder chains: A, B, C, ...
    * Target chains: TA, TB, TC, ...
    * Antitarget chains: AA, AB, AC, ...
- Binder is always the A-chain (chain_of_focus = "A").
- Boltz outputs live under:
      binder_<binder_name>/outputs/boltz_results_<yaml_stem>/
- YAML stems look like:
      binder_<binder>_vs_target_<target>
      binder_<binder>_vs_antitarget_<name>

This script:
  * runs ipSAE on all models for each binder–(anti)target pair
  * extracts metrics for chain A vs its best partner
  * stores:
        - binder
        - vs (full name)
        - partner (e.g. Spike, HA, ...)
        - target_type (target / antitarget / unknown)
        - model_idx
        - numeric ipSAE metrics (_min, _max)
  * makes per-binder stripplots (all targets & antitargets together)
  * makes global heatmaps (ipSAE_min, ipSAE_max) averaged across models
"""

import argparse
from fileinput import filename
import subprocess
import re
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def classify(chain_id):
    # Binder chains: A, B, C ...
    if len(chain_id) == 1 and chain_id.isupper():
        return "binder"

    # Self-chains: SA, SB, SC ...
    if chain_id.startswith("S"):
        return "self"

    # Targets: TA, TB, ...
    if chain_id.startswith("T"):
        return "target"

    # Antitarget: AA, AB, ...
    if chain_id.startswith("A") and len(chain_id) > 1:
        return "antitarget"

    return "other"



def run_ipsae(
    pae_file,
    cif_file,
    pae_cutoff=15,
    dist_cutoff=15,
    chain_of_focus="A"
):
    """
    Run ipsae.py and extract comprehensive metrics for chain_of_focus.

    Extracts from both 'asym' and 'max' rows:
    - ipSAE (primary metric from max row)
    - ipSAE_min, ipSAE_max (from asym rows)
    - ipTM (ipTM_af from max row)
    - pDockQ2 (from max row)
    - ipSAE_d0chn, ipSAE_d0dom (alternative ipSAE variants)

    If multiple partner chains exist, choose the one with the highest ipSAE
    from the max row.
    """

    import pandas as pd
    import numpy as np
    import os
    import subprocess

    # ---------------------------
    # Run ipSAE
    # ---------------------------
    cmd = [
        "python", f"{os.path.dirname(os.path.abspath(__file__))}/ipsae.py",
        str(pae_file),
        str(cif_file),
        str(pae_cutoff),
        str(dist_cutoff)
    ]
    subprocess.run(cmd, check=True)

    out_txt = str(cif_file).replace(".cif", f"_{pae_cutoff}_{dist_cutoff}.txt")
    if not os.path.exists(out_txt):
        raise FileNotFoundError(f"Missing ipSAE output: {out_txt}.\nCommand was: {' '.join(cmd)}")

    # ---------------------------------------------------
    # Convert fixed-width text to CSV by collapsing spaces
    # ---------------------------------------------------
    with open(out_txt, "r") as f:
        raw_lines = f.readlines()

    clean_lines = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = re.sub(r"\s+", ",", stripped)
        clean_lines.append(cleaned)

    csv_tmp = out_txt + ".csv"
    with open(csv_tmp, "w") as f:
        for cl in clean_lines:
            f.write(cl + "\n")

    # Load CSV normally
    df_full = pd.read_csv(csv_tmp)
    df_full["Type"] = df_full["Type"].astype(str).str.lower().str.strip()

    # Required columns
    chn1_col = "Chn1"
    chn2_col = "Chn2"

    # Compute category for each chain
    df_full["cat1"] = df_full[chn1_col].apply(classify)
    df_full["cat2"] = df_full[chn2_col].apply(classify)

    # Keep only binder–target/antitarget/self pairs
    df_full = df_full[
        ((df_full["cat1"] == "binder") & (df_full["cat2"].isin(["target", "antitarget", "self"]))) |
        ((df_full["cat2"] == "binder") & (df_full["cat1"].isin(["target", "antitarget", "self"])))
    ]

    if df_full.empty:
        raise ValueError(f"No valid binder–target/antitarget rows for chain {chain_of_focus}")

    # Keep only rows where focus chain appears
    df_full = df_full[(df_full[chn1_col] == chain_of_focus) | (df_full[chn2_col] == chain_of_focus)]
    if df_full.empty:
        raise ValueError(f"No rows involving chain {chain_of_focus}")

    # Split into asym and max rows
    df_asym = df_full[df_full["Type"] == "asym"].copy()
    df_max = df_full[df_full["Type"] == "max"].copy()

    # ---------------------------
    # Identify partner chains and select best partner
    # ---------------------------
    partners = set()
    for _, row in df_full.iterrows():
        partner = row[chn2_col] if row[chn1_col] == chain_of_focus else row[chn1_col]
        partners.add(partner)
    partners = sorted(partners)

    # Choose partner with highest ipSAE from max row (or asym if no max rows)
    partner_best = None
    partner_best_score = -np.inf
    best_asym_df = None
    best_max_df = None

    for p in partners:
        # Get asym rows for this partner
        asym_sub = df_asym[
            ((df_asym[chn1_col] == chain_of_focus) & (df_asym[chn2_col] == p)) |
            ((df_asym[chn2_col] == chain_of_focus) & (df_asym[chn1_col] == p))
        ]
        # Get max rows for this partner
        max_sub = df_max[
            ((df_max[chn1_col] == chain_of_focus) & (df_max[chn2_col] == p)) |
            ((df_max[chn2_col] == chain_of_focus) & (df_max[chn1_col] == p)) |
            ((df_max[chn1_col] == p) & (df_max[chn2_col] == chain_of_focus)) |
            ((df_max[chn2_col] == p) & (df_max[chn1_col] == chain_of_focus))
        ]

        # Determine score for this partner (prefer max row, fallback to asym)
        if not max_sub.empty and "ipSAE" in max_sub.columns:
            score = max_sub["ipSAE"].max()
        elif not asym_sub.empty and "ipSAE" in asym_sub.columns:
            score = asym_sub["ipSAE"].max()
        else:
            continue

        if score > partner_best_score:
            partner_best_score = score
            partner_best = p
            best_asym_df = asym_sub.copy() if not asym_sub.empty else None
            best_max_df = max_sub.copy() if not max_sub.empty else None

    if partner_best is None:
        raise ValueError(f"No valid partner rows found for {chain_of_focus}")

    # ---------------------------
    # Extract comprehensive metrics
    # ---------------------------
    output = {
        "chain_of_focus": chain_of_focus,
        "involved_chains": partner_best
    }

    # Key metrics to extract (in order of importance)
    key_metrics = ["ipSAE", "ipSAE_d0chn", "ipSAE_d0dom", "ipTM_af", "ipTM_d0chn", "pDockQ", "pDockQ2", "LIS"]

    # From MAX row: get the primary/best values
    if best_max_df is not None and not best_max_df.empty:
        max_row = best_max_df.iloc[0]  # Should be only one max row per partner pair
        for metric in key_metrics:
            if metric in max_row.index:
                try:
                    output[metric] = float(max_row[metric])
                except (ValueError, TypeError):
                    pass

    # From ASYM rows: get min/max across both directions
    if best_asym_df is not None and not best_asym_df.empty:
        for metric in key_metrics:
            if metric in best_asym_df.columns:
                try:
                    output[f"{metric}_min"] = float(best_asym_df[metric].min())
                    output[f"{metric}_max"] = float(best_asym_df[metric].max())
                except (ValueError, TypeError):
                    pass

    # If no max row exists, use asym max as the primary ipSAE
    if "ipSAE" not in output and "ipSAE_max" in output:
        output["ipSAE"] = output["ipSAE_max"]

    # Also extract n0res, n0chn, n0dom for context (from max row if available)
    context_cols = ["n0res", "n0chn", "n0dom", "d0res", "d0chn", "d0dom", "nres1", "nres2"]
    if best_max_df is not None and not best_max_df.empty:
        max_row = best_max_df.iloc[0]
        for col in context_cols:
            if col in max_row.index:
                try:
                    output[col] = float(max_row[col])
                except (ValueError, TypeError):
                    pass

    return output


def parse_vs_name(vs_name: str):
    """
    Parse:
        binder_<binder>_vs_target_<name>
        binder_<binder>_vs_antitarget_<name>
        binder_<binder>_vs_self
    """
    m = re.search(r"_vs_(target|antitarget|self)_(.*)$", vs_name)
    if m:
        role = m.group(1)
        partner = m.group(2)
        return partner, role

    # self without a partner name (binder_X_vs_self)
    m2 = re.search(r"_vs_(self)$", vs_name)
    if m2:
        return "self", "self"

    return vs_name, "unknown"



def analyse_binder(binder_dir: Path, args):
    """
    Analyse a binder directory: compute ipSAE for all vs_* pairs, save plots.

    Now extracts comprehensive metrics:
    - ipSAE (primary, from max row)
    - ipSAE_min, ipSAE_max (from asym rows)
    - ipTM_af (AlphaFold/Boltz ipTM)
    - pDockQ2
    - And more context metrics
    """
    plots_dir = binder_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    binder_records = []

    for vs_dir in (binder_dir / "outputs").glob("boltz_results_*vs*"):
        vs_name = vs_dir.name.replace("boltz_results_", "")
        pred_root = vs_dir / "predictions" / vs_name

        partner_name, target_type = parse_vs_name(vs_name)

        for pae_file in pred_root.glob("pae_*_model_*.npz"):
            m = re.search(r"model_(\d+)", pae_file.name)
            if not m:
                continue
            model_idx = m.group(1)
            cif_file = pred_root / pae_file.name.replace("pae_", "").replace(".npz", ".cif")
            if not cif_file.exists():
                continue

            try:
                rec = run_ipsae(
                    pae_file,
                    cif_file,
                    chain_of_focus="A",
                    pae_cutoff=int(args.ipsae_e),
                    dist_cutoff=int(args.ipsae_d)
                )
            except Exception as e:
                print(f"ipSAE failed for {pae_file} ({e}). Skipping.")
                continue

            # -----------------------------------------------------------
            # Add sequences from YAML (binder1, binder2, target1, ...)
            # -----------------------------------------------------------
            yaml_path = binder_dir / f"{vs_name}.yaml"

            binder_seqs, target_seqs, antitarget_seqs, self_seqs = extract_sequences_from_yaml(yaml_path)

            # binder may have multiple chains
            if len(binder_seqs) <= 1:
                rec["binder_sequence"] = binder_seqs[0] if binder_seqs else ""
            else:
                rec["binder_sequence"] = ":".join(binder_seqs)

            # partner (target/antitarget/self) is mutually exclusive
            partner_list = target_seqs or antitarget_seqs or self_seqs

            if len(partner_list) <= 1:
                rec["target_sequence"] = partner_list[0] if partner_list else ""
            else:
                rec["target_sequence"] = ":".join(partner_list)

            rec.update({
                "binder": binder_dir.name,
                "vs": vs_name,
                "model_idx": int(model_idx),
                "partner": partner_name,
                "target_type": target_type,
            })
            binder_records.append(rec)

    if not binder_records:
        print(f"No valid ipSAE data for {binder_dir.name}")
        return

    df = pd.DataFrame(binder_records)

    # Reorder columns for clarity: identifiers first, then key metrics, then context
    id_cols = ["binder", "vs", "partner", "target_type", "model_idx"]
    key_metric_cols = [
        "ipSAE", "ipSAE_min", "ipSAE_max",
        "ipTM_af", "ipTM_af_min", "ipTM_af_max",
        "pDockQ2", "pDockQ2_min", "pDockQ2_max",
        "ipSAE_d0chn", "ipSAE_d0dom",
        "pDockQ", "LIS"
    ]
    seq_cols = ["binder_sequence", "target_sequence"]
    context_cols = ["chain_of_focus", "involved_chains", "n0res", "n0chn", "n0dom"]

    # Build ordered column list (only include columns that exist)
    ordered_cols = []
    for col in id_cols + key_metric_cols + seq_cols + context_cols:
        if col in df.columns:
            ordered_cols.append(col)
    # Add any remaining columns not in our predefined list
    for col in df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)

    df = df[ordered_cols]

    csv_path = plots_dir / "ipsae_summary.csv"
    df.to_csv(csv_path, index=False)

    # Generate stripplots for key metrics
    plot_metrics = ["ipSAE", "ipSAE_min", "ipSAE_max", "ipTM_af", "pDockQ2"]
    partner_order = sorted(df["partner"].dropna().unique().tolist())

    for metric in plot_metrics:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(6, 3.5))
        sns.stripplot(
            data=df,
            x="partner",
            y=metric,
            hue="model_idx",
            alpha=0.7,
            order=partner_order,
        )
        short_title = re.sub(r"^binder_", "", binder_dir.name)
        plt.title(f"{metric} for {short_title}")
        plt.ylabel(metric)
        plt.xlabel("Target / Antitarget / Self")
        plt.xticks(rotation=30)
        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            order_idx = sorted(range(len(labels)), key=lambda i: int(labels[i]))
            plt.legend(
                [handles[i] for i in order_idx],
                [labels[i] for i in order_idx],
                title="model_idx",
                loc="best",
            )

        plt.tight_layout()
        for ext in ["png"]:
            plt.savefig(plots_dir / f"{metric}_stripplot.{ext}", dpi=200)
        plt.close()

    print(f"Saved: {csv_path}")

import yaml

def extract_sequences_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f)

    binder_seqs = []
    target_seqs = []
    antitarget_seqs = []
    self_seqs = []

    for entry in y.get("sequences", []):
        prot = entry.get("protein", {})
        cid = prot.get("id", "")
        seq = prot.get("sequence", "")

        # Binder: A, B, C...
        if len(cid) == 1 and cid.isupper():
            binder_seqs.append(seq)

        # Target: TA, TB, ...
        elif cid.startswith("T") and len(cid) > 1:
            target_seqs.append(seq)

        # Antitarget: AA, AB, ...
        elif cid.startswith("A") and len(cid) > 1:
            antitarget_seqs.append(seq)

        # Self: SA, SB, ...
        elif cid.startswith("S") and len(cid) > 1:
            self_seqs.append(seq)

    return binder_seqs, target_seqs, antitarget_seqs, self_seqs


def plot_overall(root_dir: Path, use_best_model: bool = False):
    """
    Combine all per-binder CSVs and plot heatmaps for multiple metrics.

    Now generates heatmaps for:
    - ipSAE (primary metric from max row)
    - ipSAE_min, ipSAE_max (from asym rows)
    - ipTM_af (AlphaFold/Boltz ipTM)
    - pDockQ2

    Also generates a comprehensive summary CSV with mean/std for all metrics.
    """
    csvs = list(root_dir.glob("binder_*/plots/ipsae_summary.csv"))
    if not csvs:
        print("No binder CSVs found.")
        return

    dfs = []
    for csv_file in csvs:
        df = pd.read_csv(csv_file)

        # Backwards compatibility: older CSVs may not have 'partner' or 'target_type'
        if "partner" not in df.columns and "vs" in df.columns:
            df["partner"] = df["vs"].str.extract(r"_vs_(.*)$")
        if "target_type" not in df.columns and "vs" in df.columns:
            df["target_type"] = "unknown"

        df["binder_short"] = df["binder"].str.replace(r"^binder_", "", regex=True)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    binder_sequences = all_df.groupby("binder_short")["binder_sequence"].first().to_dict()

    # Add the binder_sequence column
    all_df["binder_sequence"] = all_df["binder_short"].map(binder_sequences)

    # Define metrics to process (in order of importance)
    # Primary metrics for heatmaps
    heatmap_metrics = ["ipSAE", "ipSAE_min", "ipSAE_max", "ipTM_af", "pDockQ2"]
    # All metrics for summary statistics
    all_metrics = [
        "ipSAE", "ipSAE_min", "ipSAE_max",
        "ipSAE_d0chn", "ipSAE_d0dom",
        "ipTM_af", "ipTM_af_min", "ipTM_af_max",
        "pDockQ", "pDockQ2", "pDockQ2_min", "pDockQ2_max",
        "LIS"
    ]

    # Filter to metrics that actually exist in the data
    available_heatmap_metrics = [m for m in heatmap_metrics if m in all_df.columns]
    available_all_metrics = [m for m in all_metrics if m in all_df.columns]

    if not available_heatmap_metrics:
        print("No metrics found for heatmap plotting.")
        return

    # Create summary directory
    summary_dir = root_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    # Save combined raw data
    all_df_path = summary_dir / "ipsae_summary_all_binders.csv"
    all_df.to_csv(all_df_path, index=False)
    print(f"Saved combined ipSAE data at {all_df_path}")

    # ------------------------------------------------------
    # AGGREGATION ACROSS MODELS
    # ------------------------------------------------------
    # Determine primary metric for ordering (prefer ipSAE, fallback to ipSAE_max)
    primary_metric = "ipSAE" if "ipSAE" in all_df.columns else "ipSAE_max" if "ipSAE_max" in all_df.columns else available_heatmap_metrics[0]

    if use_best_model and primary_metric in all_df.columns:
        # Pick the row (model) with highest primary metric per binder/partner
        idx = all_df.groupby(["binder_short", "partner"])[primary_metric].idxmax()
        agg_base = all_df.loc[idx].copy()
    else:
        agg_base = all_df.copy()

    # Compute mean across models for all available metrics
    agg = agg_base.groupby(["binder_short", "partner"])[available_heatmap_metrics].mean().reset_index()

    # Also compute std for comprehensive summary
    agg_std = agg_base.groupby(["binder_short", "partner"])[available_all_metrics].std().reset_index()
    agg_std.columns = ["binder_short", "partner"] + [f"{m}_std" for m in available_all_metrics]

    agg_mean = agg_base.groupby(["binder_short", "partner"])[available_all_metrics].mean().reset_index()
    agg_mean.columns = ["binder_short", "partner"] + [f"{m}_mean" for m in available_all_metrics]

    agg_count = agg_base.groupby(["binder_short", "partner"]).size().reset_index(name="n_models")

    # Merge mean, std, and count
    agg_full = agg_mean.merge(agg_std, on=["binder_short", "partner"])
    agg_full = agg_full.merge(agg_count, on=["binder_short", "partner"])

    # Add target_type back
    type_map = agg_base.groupby("partner")["target_type"].first().to_dict()
    agg_full["target_type"] = agg_full["partner"].map(type_map)

    # Add binder_sequence
    agg_full["binder_sequence"] = agg_full["binder_short"].map(binder_sequences)

    # Reorder columns for clarity
    id_cols = ["binder_short", "binder_sequence", "partner", "target_type", "n_models"]
    metric_cols = []
    for m in available_all_metrics:
        if f"{m}_mean" in agg_full.columns:
            metric_cols.append(f"{m}_mean")
        if f"{m}_std" in agg_full.columns:
            metric_cols.append(f"{m}_std")

    ordered_cols = id_cols + metric_cols
    ordered_cols = [c for c in ordered_cols if c in agg_full.columns]
    agg_full = agg_full[ordered_cols]

    # Save comprehensive summary
    summary_csv_path = summary_dir / "ipsae_comprehensive_summary.csv"
    agg_full.to_csv(summary_csv_path, index=False, float_format="%.6f")
    print(f"Saved comprehensive summary at {summary_csv_path}")

    # ------------------------------------------------------
    # ORDER BINDERS BY PRIMARY METRIC
    # ------------------------------------------------------
    binder_order = (
        agg.groupby("binder_short")[primary_metric]
        .max()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    partner_order = ["self", "target", "antitarget"]

    print(f"Primary metric for ordering: {primary_metric}")
    print(f"Partner order: {partner_order}")
    print(f"Binder order (best→worst by {primary_metric}): {binder_order}\n")

    # ------------------------------------------------------
    # PLOT HEATMAPS FOR EACH METRIC
    # ------------------------------------------------------
    for metric in available_heatmap_metrics:
        if metric not in agg.columns:
            continue

        # Replace partner names with their class
        agg["partner_class"] = agg["partner"].map(type_map)

        pivot = agg.pivot(index="partner_class", columns="binder_short", values=metric)
        pivot = pivot.reindex(index=partner_order, columns=binder_order)

        plt.figure(figsize=(max(7, len(binder_order) * 0.7),
                            max(5, len(partner_order) * 0.4)))

        # Determine appropriate vmax based on metric
        vmax = 1.0 if metric.startswith("ipSAE") or metric.startswith("ipTM") or metric.startswith("pDockQ") else None

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={"label": metric},
            linewidths=0.5,
            vmin=0,
            vmax=vmax,
        )
        plt.title(f"{metric} (mean across models)")
        plt.ylabel("Self / Target / Antitarget", rotation=90)
        plt.xlabel("Binder")
        plt.yticks(rotation=0)

        for ext in ["png", "svg"]:
            path = summary_dir / f"{metric}_heatmap.{ext}"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved heatmap for {metric} at {path}")
        plt.close()

        # Save CSV (rows = binders, columns = targets)
        csv_out = summary_dir / f"{metric}_heatmap.csv"
        pivot_out = pivot.T.copy()  # rows = binders
        pivot_out.insert(0, "binder_sequence", pivot_out.index.map(binder_sequences))
        pivot_out.to_csv(csv_out, float_format="%.5f")
        print(f"Saved heatmap data for {metric} as {csv_out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipsae_e", type=int, default=15,
                    help="ipSAE PAE cutoff (default: 15 Å)")
    ap.add_argument("--ipsae_d", type=int, default=15,
                    help="ipSAE distance cutoff (default: 15 Å)")
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--generate_data", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument(
        "--use_best_model",
        action="store_true",
        help="Use only the best model (highest ipSAE_max) per binder/partner instead of averaging"
    )
    ap.add_argument("--num_cpu", type=int, default=1,
                    help="Number of CPUs for parallel processing")
    args = ap.parse_args()


    root = Path(args.root_dir)
    if args.generate_data:
        binder_dirs = [d for d in sorted(root.glob("binder_*")) if d.is_dir()]
        if args.num_cpu == 1:
            # sequential
            for d in binder_dirs:
                analyse_binder(d,args)
        else:
            # parallel
            from multiprocessing import Pool
            with Pool(processes=args.num_cpu) as pool:
                pool.starmap(analyse_binder, [(d, args) for d in binder_dirs])

    if args.plot:
        plot_overall(root, use_best_model=args.use_best_model)


if __name__ == "__main__":
    main()
