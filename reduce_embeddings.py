"""Project 1536-D embeddings to 2-D with PCA -> UMAP, plus HDBSCAN clusters.

Why PCA first: UMAP's cost is dominated by approximate nearest-neighbour
search, which scales with dimensionality. Squeezing 1536 -> 50 dims keeps
~50% of total variance (measured on this corpus) while removing most of the
noise directions, and cuts runtime substantially. Neighbourhoods are
essentially unchanged.

Reproducibility note: passing random_state to UMAP forces single-threaded
execution -- it is the single biggest lever on runtime here. Use
--no-seed for a fast, non-reproducible run.

Read the output with care: in a UMAP plot, distances BETWEEN clusters and
the relative SIZE of clusters are artefacts. Only local neighbourhood
membership is meaningful.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA

INPUT_DIR = Path("output")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    p.add_argument("--pca-dims", type=int, default=50,
                   help="0 disables PCA pre-reduction")
    p.add_argument("--n-neighbors", type=int, default=30,
                   help="Higher = more global structure, less fragmentation")
    p.add_argument("--min-dist", type=float, default=0.05)
    p.add_argument("--no-seed", action="store_true",
                   help="Drop random_state to enable multi-threaded UMAP")
    p.add_argument("--cluster", action="store_true",
                   help="Also run HDBSCAN and label each cluster by tag lift")
    p.add_argument("--min-cluster-size", type=int, default=40,
                   help="Smallest group HDBSCAN will call a cluster")
    p.add_argument("--min-samples", type=int, default=5,
                   help="Lower = fewer points dumped into 'unclustered'")
    return p


def add_clusters(vectors: np.ndarray, meta: pd.DataFrame,
                 min_cluster_size: int = 40, min_samples: int = 5) -> None:
    """Assign density-based clusters and name them by over-represented tags.

    Clustering runs on the PCA space, not the 2-D UMAP output -- UMAP
    coordinates distort density and would produce misleading clusters.
    """
    import json
    from collections import Counter

    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        print("HDBSCAN needs scikit-learn >= 1.3; skipping --cluster.")
        return

    print(f"Running HDBSCAN (min_cluster_size={min_cluster_size}, "
          f"min_samples={min_samples})...")
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    ).fit_predict(vectors)
    meta["cluster"] = labels

    tags = meta["tags"].fillna("[]").apply(json.loads)
    base = Counter(t for row in tags for t in row)
    total = sum(base.values()) or 1

    names: dict[int, str] = {-1: "unclustered"}
    for cid in sorted(set(labels) - {-1}):
        mask = labels == cid
        local = Counter(t for row in tags[mask] for t in row)
        n_local = sum(local.values()) or 1
        # Lift = how much more common a tag is here than corpus-wide.
        # Require >=5 occurrences so tiny counts cannot dominate.
        lift = {
            t: (c / n_local) / (base[t] / total)
            for t, c in local.items() if c >= 5
        }
        top = sorted(lift, key=lift.get, reverse=True)[:3]
        names[cid] = " / ".join(top) if top else f"cluster {cid}"

    meta["cluster_name"] = [names[c] for c in labels]
    n_clusters = len(set(labels) - {-1})
    noise = float((labels == -1).mean())
    print(f"  {n_clusters} clusters, {noise:.1%} unclustered")
    if noise > 0.5:
        print("  Most points are unclustered -- lower --min-samples or "
              "--min-cluster-size, or accept that the space is a smooth "
              "continuum rather than discrete groups.")
    for cid in sorted(set(labels) - {-1})[:12]:
        print(f"    [{(labels == cid).sum():5d}] {names[cid]}")


def main(args: argparse.Namespace) -> None:
    vectors = np.load(args.input_dir / "embeddings.npy").astype(np.float32)
    meta = pd.read_csv(args.input_dir / "movies_metadata.csv")

    if len(vectors) != len(meta):
        raise SystemExit(
            f"Row mismatch: {len(vectors)} vectors vs {len(meta)} metadata rows. "
            "Re-run extract_embeddings.py -- the two files are positionally aligned."
        )
    print(f"Loaded {vectors.shape[0]:,} x {vectors.shape[1]}")

    reduced = vectors
    if args.pca_dims:
        t0 = time.perf_counter()
        pca = PCA(n_components=args.pca_dims, random_state=42)
        reduced = pca.fit_transform(vectors).astype(np.float32)
        kept = pca.explained_variance_ratio_.sum()
        print(f"PCA -> {args.pca_dims}D in {time.perf_counter() - t0:.1f}s "
              f"({kept:.1%} variance retained)")

    print(f"Running UMAP (n_neighbors={args.n_neighbors}, "
          f"min_dist={args.min_dist})...")
    t0 = time.perf_counter()
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=None if args.no_seed else 42,
        verbose=True,
    )
    coords = reducer.fit_transform(reduced)
    print(f"UMAP finished in {time.perf_counter() - t0:.1f}s")

    meta["x"], meta["y"] = coords[:, 0], coords[:, 1]

    if args.cluster:
        add_clusters(reduced, meta, args.min_cluster_size, args.min_samples)

    out = args.input_dir / "movies_2d.csv"
    meta.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main(build_parser().parse_args())
