#!/usr/bin/env python3
"""
run_ssgsea_clusters.py
======================
Compute ssGSEA scores for every cluster in one or more snRNA-seq .h5ad files.
GMT collections can be supplied as local files (--gmt) OR fetched directly
from MSigDB (--msigdb).  Both flags can be mixed in the same run.

No plots are produced — outputs are raw TSV score matrices and a per-cluster
top-activity summary.

──────────────────────────────────────────────────────────────────────────────
Quick-start examples
──────────────────────────────────────────────────────────────────────────────

# Hallmarks, auto-downloaded from MSigDB:
python run_ssgsea_clusters.py \\
    --h5ad   "Z:/data/SAR01026141.h5ad" \\
    --msigdb H \\
    --out    "Z:/results/hallmark"

# Several MSigDB collections + a local GMT in one run:
python run_ssgsea_clusters.py \\
    --h5ad   "Z:/data/cellbender_objects" \\
    --msigdb H C2:CP C5:GO:BP C7:IMMUNESIGDB C8 \\
    --gmt    "Z:/custom/my_signatures.gmt" \\
    --out    "Z:/results/multi" \\
    --cluster_col seurat_clusters \\
    --sample_col  Sample_name \\
    --top_n 25 --threads 8

# List every available MSigDB collection alias and exit:
python run_ssgsea_clusters.py --list_collections

──────────────────────────────────────────────────────────────────────────────
MSigDB collection aliases  (case-insensitive, colon-separated sub-collections)
──────────────────────────────────────────────────────────────────────────────
  H                     Hallmarks
  C1                    Positional gene sets
  C2 / C2:CGP / C2:CP   Chemical/genetic perturbations & canonical pathways
  C2:CP:BIOCARTA|KEGG|KEGG_MEDICUS|PID|REACTOME|WIKIPATHWAYS
  C3 / C3:MIR / C3:TFT  Regulatory targets
  C4 / C4:CGN|CM|3CA    Computational gene sets
  C5 / C5:GO / C5:GO:BP|CC|MF / C5:HPO   Ontology gene sets
  C6                    Oncogenic signatures
  C7 / C7:IMMUNESIGDB / C7:VAX   Immunologic signatures
  C8                    Cell type signatures (scRNA-seq markers)

──────────────────────────────────────────────────────────────────────────────
Outputs  (written to --out / <label>/)
──────────────────────────────────────────────────────────────────────────────
  <label>_ssgsea_scores.tsv          Full NES matrix [gene_sets x sample|clN]
  <label>_metadata.tsv               One row per column (sample, cluster, n_cells)
  <label>_top<N>_per_cluster.tsv     Long-format top-N activities per cluster
  <label>_bottom<N>_per_cluster.tsv  Long-format bottom-N activities per cluster
  run_summary.tsv                    One row per collection with run stats
"""

import argparse
import re
import sys
import time
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# MSigDB collection catalogue
# Base URL pattern:
#   https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{version}/
#           {stem}.v{version}.symbols.gmt
# ──────────────────────────────────────────────────────────────────────────────

_MSIGDB_COLLECTIONS: dict = {
    "H"                  : "h.all",
    "C1"                 : "c1.all",
    "C2"                 : "c2.all",
    "C2:CGP"             : "c2.cgp",
    "C2:CP"              : "c2.cp",
    "C2:CP:BIOCARTA"     : "c2.cp.biocarta",
    "C2:CP:KEGG"         : "c2.cp.kegg_legacy",
    "C2:CP:KEGG_LEGACY"  : "c2.cp.kegg_legacy",
    "C2:CP:KEGG_MEDICUS" : "c2.cp.kegg_medicus",
    "C2:CP:PID"          : "c2.cp.pid",
    "C2:CP:REACTOME"     : "c2.cp.reactome",
    "C2:CP:WIKIPATHWAYS" : "c2.cp.wikipathways",
    "C3"                 : "c3.all",
    "C3:MIR"             : "c3.mir",
    "C3:MIR:MIRDB"       : "c3.mir.mirdb",
    "C3:MIR:MIR_LEGACY"  : "c3.mir.mir_legacy",
    "C3:TFT"             : "c3.tft",
    "C3:TFT:GTRD"        : "c3.tft.gtrd",
    "C3:TFT:TFT_LEGACY"  : "c3.tft.tft_legacy",
    "C4"                 : "c4.all",
    "C4:CGN"             : "c4.cgn",
    "C4:CM"              : "c4.cm",
    "C4:3CA"             : "c4.3ca",
    "C5"                 : "c5.all",
    "C5:GO"              : "c5.go",
    "C5:GO:BP"           : "c5.go.bp",
    "C5:GO:CC"           : "c5.go.cc",
    "C5:GO:MF"           : "c5.go.mf",
    "C5:HPO"             : "c5.hpo",
    "C6"                 : "c6.all",
    "C7"                 : "c7.all",
    "C7:IMMUNESIGDB"     : "c7.immunesigdb",
    "C7:VAX"             : "c7.vax",
    "C8"                 : "c8.all",
}

_MSIGDB_BASE = (
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release"
    "/{version}/{stem}.v{version}.symbols.gmt"
)
_MSIGDB_DEFAULT_VERSION = "2025.1.Hs"


def _msigdb_url(stem: str, version: str) -> str:
    return _MSIGDB_BASE.format(version=version, stem=stem)


def _safe_label(alias: str) -> str:
    """'C5:GO:BP' -> 'C5_GO_BP'"""
    return alias.upper().replace(":", "_")


# ──────────────────────────────────────────────────────────────────────────────
# Lazy import guard
# ──────────────────────────────────────────────────────────────────────────────

def _check_imports() -> None:
    missing = [
        pkg for pkg in ["anndata", "gseapy", "numpy", "pandas", "scipy", "tqdm"]
        if __import__("importlib").util.find_spec(pkg) is None
    ]
    if missing:
        sys.exit(
            "ERROR: The following required packages are not installed:\n"
            "  " + ", ".join(missing)
            + "\nActivate your environment and retry."
        )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_ssgsea_clusters",
        description=(
            "ssGSEA on per-cluster mean expression from snRNA-seq .h5ad files.\n"
            "GMT collections can be fetched automatically from MSigDB (--msigdb)\n"
            "and/or supplied as local files (--gmt).\n"
            "Use --list_collections to print all recognised MSigDB aliases."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io = p.add_argument_group("Input / Output")
    io.add_argument(
        "--h5ad",
        help="Path to a single .h5ad file OR a directory of .h5ad files.",
    )
    io.add_argument(
        "--out",
        help="Root output directory.  A sub-folder is created per collection.",
    )

    gs = p.add_argument_group(
        "Gene-set source  (at least one of --msigdb / --gmt required)"
    )
    gs.add_argument(
        "--msigdb", nargs="+", metavar="COLLECTION",
        help=(
            "One or more MSigDB collection aliases.\n"
            "Examples: H  C2:CP  C5:GO:BP  C7:IMMUNESIGDB  C8\n"
            "Run --list_collections to see all aliases."
        ),
    )
    gs.add_argument(
        "--gmt", nargs="+", metavar="FILE",
        help="One or more local .gmt file paths.",
    )
    gs.add_argument(
        "--msigdb_version", default=_MSIGDB_DEFAULT_VERSION, metavar="VERSION",
        help=f"MSigDB release version string (default: {_MSIGDB_DEFAULT_VERSION}).",
    )
    gs.add_argument(
        "--gmt_cache", default=None, metavar="DIR",
        help="Directory for caching downloaded GMT files.\n"
             "Defaults to  <--out>/msigdb_cache/.",
    )

    sc = p.add_argument_group("Single-cell options")
    sc.add_argument(
        "--cluster_col", default="seurat_clusters",
        help="obs column with cluster assignments (default: seurat_clusters).",
    )
    sc.add_argument(
        "--sample_col", default="Sample_name",
        help="obs column for the sample/library name.  "
             "If absent, the file stem is used.",
    )

    ss = p.add_argument_group("ssGSEA parameters")
    ss.add_argument(
        "--top_n", type=int, default=20,
        help="Top- and bottom-N gene sets reported per cluster (default: 20).",
    )
    ss.add_argument(
        "--min_size", type=int, default=5,
        help="Min gene-set size after overlap filtering (default: 5).",
    )
    ss.add_argument(
        "--max_size", type=int, default=5000,
        help="Max gene-set size (default: 5000).",
    )
    ss.add_argument(
        "--threads", type=int, default=8,
        help="CPU threads for ssGSEA (default: 8).",
    )
    ss.add_argument(
        "--weight", type=float, default=0.25,
        help="ssGSEA enrichment score weight (default: 0.25).",
    )
    ss.add_argument(
        "--reuse", action="store_true",
        help="Reload saved score TSVs if they already exist.",
    )

    p.add_argument(
        "--list_collections", action="store_true",
        help="Print all available MSigDB collection aliases and exit.",
    )
    return p


# ──────────────────────────────────────────────────────────────────────────────
# MSigDB download
# ──────────────────────────────────────────────────────────────────────────────

def resolve_msigdb_gmts(
    aliases: list,
    version: str,
    cache_dir: Path,
) -> list:
    """
    Resolve MSigDB aliases -> list of (label, local_path).
    Downloads any missing GMT files into cache_dir.
    """
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for raw_alias in aliases:
        alias = raw_alias.strip().upper()
        if alias not in _MSIGDB_COLLECTIONS:
            suggestions = sorted(
                k for k in _MSIGDB_COLLECTIONS if k.startswith(alias[:2])
            )[:8]
            msg = f"ERROR: Unknown MSigDB collection alias '{raw_alias}'."
            if suggestions:
                msg += f"\n  Possible matches: {', '.join(suggestions)}"
            msg += "\n  Run  --list_collections  to see all valid aliases."
            sys.exit(msg)

        stem  = _MSIGDB_COLLECTIONS[alias]
        label = _safe_label(alias)
        url   = _msigdb_url(stem, version)
        fname = f"{stem}.v{version}.symbols.gmt"
        local = cache_dir / fname

        if local.exists():
            size_kb = local.stat().st_size // 1024
            print(f"  [{label}] Cached GMT found: {fname}  ({size_kb:,} KB)")
        else:
            print(f"  [{label}] Downloading {fname} ...")
            print(f"    URL: {url}")
            try:
                urllib.request.urlretrieve(url, local)
                size_kb = local.stat().st_size // 1024
                print(f"    Saved: {fname}  ({size_kb:,} KB)")
            except Exception as exc:
                local.unlink(missing_ok=True)
                sys.exit(
                    f"ERROR: Could not download MSigDB collection '{alias}'.\n"
                    f"  URL   : {url}\n"
                    f"  Reason: {exc}\n"
                    "  Check your internet connection or use a local --gmt file."
                )

        results.append((label, local))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# GMT parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_gmt(path: Path) -> dict:
    gs: dict = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                gs[parts[0]] = [g for g in parts[2:] if g]
    if not gs:
        raise ValueError(f"No gene sets found in {path}")
    return gs


def gene_universe(gs: dict) -> set:
    out: set = set()
    for genes in gs.values():
        out.update(genes)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def infer_patient_id(text: str) -> str:
    m = re.search(r"(\d{5,8})", str(text))
    return m.group(1) if m else str(text)


def run_ssgsea(expr, gs, min_size, max_size, threads, weight):
    res = gp.ssgsea(
        data=expr,
        gene_sets=gs,
        sample_norm_method="rank",
        correl_norm_type="rank",
        outdir=None,
        min_size=min_size,
        max_size=max_size,
        threads=threads,
        weight=weight,
        ascending=False,
        permutation_num=0,
        no_plot=True,
        verbose=True,
    )
    r = res.res2d.copy()
    score_col = "NES" if "NES" in r.columns else ("ES" if "ES" in r.columns else None)
    if score_col is None:
        raise RuntimeError("gseapy result has neither NES nor ES column.")
    pivot = r.pivot(index="Term", columns="Name", values=score_col)
    return (
        pivot
        .apply(pd.to_numeric, errors="coerce")
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load .h5ad files -> per-cluster mean-expression matrix
# ──────────────────────────────────────────────────────────────────────────────

def load_cluster_expr(h5ad_paths, cluster_col, sample_col):
    """
    Returns
    -------
    expr : DataFrame  [genes x "sample|clN" columns]
    meta : DataFrame  index = column labels,
                      cols : sample_id, cluster_id, n_cells, source_file, patient_id
    """
    vecs:      list = []
    meta_rows: list = []

    for fp in tqdm(h5ad_paths, desc="Reading .h5ad files"):
        adata = ad.read_h5ad(fp)

        if cluster_col not in adata.obs.columns:
            print(f"  WARNING: '{cluster_col}' not found in {fp.name} — skipping.")
            continue

        obs = adata.obs.copy()
        obs[cluster_col] = obs[cluster_col].astype(str)

        if sample_col in obs.columns:
            uniq      = pd.Series(obs[sample_col].astype(str)).dropna().unique().tolist()
            sample_id = uniq[0] if uniq else fp.stem
        else:
            sample_id = fp.stem

        X     = adata.X
        genes = pd.Index(adata.var_names.astype(str), name="gene")

        clusters_sorted = sorted(
            obs[cluster_col].unique(),
            key=lambda x: (len(str(x)), str(x)),
        )

        for cl in clusters_sorted:
            idx = np.where(obs[cluster_col].values == cl)[0]
            if idx.size == 0:
                continue
            block    = X[idx, :]
            mean_vec = (
                np.asarray(block.mean(axis=0)).ravel()
                if sparse.issparse(block)
                else np.asarray(block).mean(axis=0).ravel()
            )
            col_id = f"{sample_id}|cl{cl}"
            vecs.append(pd.Series(mean_vec, index=genes, name=col_id))
            meta_rows.append({
                "column_id":   col_id,
                "sample_id":   str(sample_id),
                "cluster_id":  str(cl),
                "n_cells":     int(idx.size),
                "source_file": fp.name,
                "patient_id":  infer_patient_id(sample_id),
            })

    if not vecs:
        sys.exit(
            f"ERROR: No usable clusters found. "
            f"Check that '{cluster_col}' exists in your .h5ad obs columns."
        )

    expr = (
        pd.concat(vecs, axis=1)
        .fillna(0.0)
        .groupby(level=0)
        .mean()
    )
    meta = (
        pd.DataFrame(meta_rows)
        .drop_duplicates(subset=["column_id"])
        .set_index("column_id")
    )
    print(
        f"\nExpression matrix: {expr.shape[0]:,} genes x {expr.shape[1]:,} cluster-columns"
        f"  ({meta['sample_id'].nunique()} sample(s), {len(meta):,} clusters total)"
    )
    return expr, meta


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Run ssGSEA for one GMT collection
# ──────────────────────────────────────────────────────────────────────────────

def run_collection(label, gmt_path, expr, meta, out_dir, args):
    out_dir.mkdir(parents=True, exist_ok=True)

    score_path  = out_dir / f"{label}_ssgsea_scores.tsv"
    meta_path   = out_dir / f"{label}_metadata.tsv"
    top_path    = out_dir / f"{label}_top{args.top_n}_per_cluster.tsv"
    bottom_path = out_dir / f"{label}_bottom{args.top_n}_per_cluster.tsv"

    # ── load or compute ──────────────────────────────────────────────────────
    if args.reuse and score_path.exists() and meta_path.exists():
        print(f"[{label}] Reusing saved scores: {score_path}")
        scores = pd.read_csv(score_path, sep="\t", index_col=0).apply(
            pd.to_numeric, errors="coerce"
        )
        scores.columns = scores.columns.astype(str)
    else:
        print(f"[{label}] Parsing GMT: {gmt_path.name}")
        gs = parse_gmt(gmt_path)

        raw_ov = len(set(expr.index) & gene_universe(gs))
        up_ov  = len(
            set(expr.index.str.upper())
            & {g.upper() for g in gene_universe(gs)}
        )
        if up_ov > raw_ov:
            expr_run       = expr.copy()
            expr_run.index = expr_run.index.str.upper()
            expr_run       = expr_run.groupby(expr_run.index).mean()
            gs             = {k: [g.upper() for g in v] for k, v in gs.items()}
            final_ov       = len(set(expr_run.index) & gene_universe(gs))
            print(
                f"[{label}] Gene case uppercased -> overlap "
                f"{raw_ov} -> {final_ov} / {expr_run.shape[0]:,}"
            )
        else:
            expr_run = expr
            final_ov = raw_ov
            print(
                f"[{label}] Gene overlap: {final_ov:,} / {expr_run.shape[0]:,} "
                f"({100*final_ov/max(1, expr_run.shape[0]):.1f}%)"
            )

        t0 = time.perf_counter()
        scores = run_ssgsea(
            expr_run, gs,
            min_size=args.min_size,
            max_size=args.max_size,
            threads=args.threads,
            weight=args.weight,
        )
        elapsed = time.perf_counter() - t0
        print(
            f"[{label}] ssGSEA done in {elapsed/60:.1f} min  "
            f"-> {scores.shape[0]:,} gene sets x {scores.shape[1]:,} clusters"
        )

        valid_cols = meta.index.intersection(scores.columns)
        scores     = scores[valid_cols]
        scores.to_csv(score_path, sep="\t")

    meta.to_csv(meta_path, sep="\t")
    print(f"[{label}] Scores  -> {score_path}")

    _write_top_summary(scores, meta, top_path, bottom_path, args.top_n, label)

    return {
        "collection":   label,
        "gmt":          gmt_path.name,
        "gene_sets":    int(scores.shape[0]),
        "cluster_cols": int(scores.shape[1]),
        "samples":      int(meta["sample_id"].nunique()),
        "score_file":   str(score_path),
        "top_n_file":   str(top_path),
    }


def _write_top_summary(scores, meta, top_path, bottom_path, top_n, label):
    top_rows:    list = []
    bottom_rows: list = []

    for col in scores.columns:
        col_scores = scores[col].dropna().sort_values(ascending=False)
        sample_id  = meta.at[col, "sample_id"]  if col in meta.index else col
        cluster_id = meta.at[col, "cluster_id"] if col in meta.index else "?"
        n_cells    = int(meta.at[col, "n_cells"]) if col in meta.index else None
        base = {"sample_id": sample_id, "cluster_id": cluster_id,
                "column_id": col, "n_cells": n_cells}

        for rank, (gs_name, nes) in enumerate(col_scores.head(top_n).items(), start=1):
            top_rows.append(
                {**base, "rank": rank, "gene_set": gs_name, "NES": round(float(nes), 4)}
            )
        for rank, (gs_name, nes) in enumerate(
            col_scores.tail(top_n).iloc[::-1].items(), start=1
        ):
            bottom_rows.append(
                {**base, "rank": rank, "gene_set": gs_name, "NES": round(float(nes), 4)}
            )

    def _sort(df):
        d = df.copy()
        d["_cl_num"] = pd.to_numeric(d["cluster_id"], errors="coerce")
        return d.sort_values(
            ["sample_id", "_cl_num", "cluster_id", "rank"]
        ).drop(columns="_cl_num")

    _sort(pd.DataFrame(top_rows)).to_csv(top_path,    sep="\t", index=False)
    _sort(pd.DataFrame(bottom_rows)).to_csv(bottom_path, sep="\t", index=False)
    print(
        f"[{label}] Top-{top_n}    -> {top_path.name}\n"
        f"[{label}] Bottom-{top_n} -> {bottom_path.name}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── list collections and exit ────────────────────────────────────────────
    if args.list_collections:
        col_w = max(len(k) for k in _MSIGDB_COLLECTIONS)
        print(f"\n  {'Alias':<{col_w+2}}  GMT file stem")
        print("  " + "-" * 68)
        for alias, stem in sorted(_MSIGDB_COLLECTIONS.items()):
            print(f"  {alias:<{col_w+2}}  {stem}")
        print(
            f"\n  Default version : {_MSIGDB_DEFAULT_VERSION}\n"
            "  Override with   : --msigdb_version <VERSION>\n"
            "  Download base   : https://data.broadinstitute.org/"
            "gsea-msigdb/msigdb/release/\n"
        )
        sys.exit(0)

    # ── validate required flags ──────────────────────────────────────────────
    if not args.h5ad:
        parser.error("--h5ad is required.")
    if not args.out:
        parser.error("--out is required.")
    if not args.msigdb and not args.gmt:
        parser.error("Provide at least one of  --msigdb  or  --gmt.")

    # ── lazy-import heavy deps now that we know the user is running for real ─
    _check_imports()
    global ad, gp, np, pd, sparse, tqdm
    import anndata as ad
    import gseapy as gp
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from tqdm.auto import tqdm

    # ── resolve input .h5ad files ────────────────────────────────────────────
    h5ad_input = Path(args.h5ad)
    if h5ad_input.is_dir():
        h5ad_paths = sorted(h5ad_input.glob("*.h5ad"))
        if not h5ad_paths:
            sys.exit(f"ERROR: No .h5ad files found in: {h5ad_input}")
        print(f"Found {len(h5ad_paths)} .h5ad file(s) in {h5ad_input}")
    elif h5ad_input.is_file():
        h5ad_paths = [h5ad_input]
    else:
        sys.exit(f"ERROR: --h5ad path does not exist: {h5ad_input}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── assemble the GMT queue: (label, Path) ────────────────────────────────
    gmt_queue: list = []

    if args.msigdb:
        cache_dir = (
            Path(args.gmt_cache) if args.gmt_cache
            else out_root / "msigdb_cache"
        )
        print(
            f"\nResolving {len(args.msigdb)} MSigDB collection(s)"
            f"  [version {args.msigdb_version}]"
            f"  cache -> {cache_dir}"
        )
        gmt_queue += resolve_msigdb_gmts(args.msigdb, args.msigdb_version, cache_dir)

    if args.gmt:
        for gmt_str in args.gmt:
            gmt_path = Path(gmt_str)
            if not gmt_path.exists():
                sys.exit(f"ERROR: Local GMT file not found: {gmt_path}")
            # Strip MSigDB-style version suffix from the stem to get a clean label
            label = re.sub(r"\.v\d{4}\.\d+\.\w+\.symbols$", "", gmt_path.stem)
            label = label.replace(".", "_")
            gmt_queue.append((label, gmt_path))

    # ── STEP 1: build cluster expression matrix ──────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1  Building per-cluster mean-expression matrix")
    print("=" * 70)
    expr, meta = load_cluster_expr(h5ad_paths, args.cluster_col, args.sample_col)

    # ── STEP 2: ssGSEA per collection ────────────────────────────────────────
    summary_rows: list = []
    for label, gmt_path in gmt_queue:
        print("\n" + "=" * 70)
        print(f"STEP 2  ssGSEA  [{label}]  ({gmt_path.name})")
        print("=" * 70)
        row = run_collection(
            label    = label,
            gmt_path = gmt_path,
            expr     = expr,
            meta     = meta,
            out_dir  = out_root / label,
            args     = args,
        )
        summary_rows.append(row)

    # ── run summary ──────────────────────────────────────────────────────────
    summary_df   = pd.DataFrame(summary_rows)
    summary_path = out_root / "run_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nRun summary -> {summary_path}")


if __name__ == "__main__":
    main()
