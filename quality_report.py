"""Measure embedding quality. Run after every re-ingest; diff the numbers.

    python quality_report.py

Each check answers a specific failure mode:

  integrity     duplicate / zero / NaN vectors from retried ingestion jobs
  norms         unnormalised vectors, which silently break L2 distance
  anisotropy    "cone collapse" -- if random pairs already score high, the
                model has little room left to express similarity
  dimensionality a 1536-D vector that really lives in 20 dims is wasted spend
  hubness       a few vectsmors that are everyone's nearest neighbour, which
                poisons top-k retrieval
  alignment     does cosine similarity actually track semantic overlap?
  sequels       end-to-end retrieval on pairs with known ground truth
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_DIR = Path("output")
K = 15
RNG = np.random.default_rng(42)


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _knn(m: np.ndarray, queries: np.ndarray, k: int, batch: int = 2000) -> np.ndarray:
    """Exact top-k by cosine, batched so the n x n matrix is never realised."""
    out = []
    for i in range(0, len(queries), batch):
        q = queries[i:i + batch]
        sims = m[q] @ m.T
        np.put_along_axis(sims, q[:, None], -np.inf, axis=1)
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, idx, axis=1), axis=1)
        out.append(np.take_along_axis(idx, order, axis=1))
    return np.vstack(out)


def _header(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def integrity(v: np.ndarray, meta: pd.DataFrame) -> None:
    _header("INTEGRITY")
    norms = np.linalg.norm(v, axis=1)
    _, counts = np.unique(v, axis=0, return_counts=True)
    print(f"  rows                 {len(v):,} x {v.shape[1]}")
    print(f"  duplicate vectors    {(counts > 1).sum()}")
    print(f"  duplicate titles     {meta['title'].duplicated().sum()}")
    print(f"  zero / NaN vectors   {(norms < 1e-6).sum()} / {np.isnan(v).any(1).sum()}")
    print(f"  L2 norm              mean {norms.mean():.4f}  std {norms.std():.5f}"
          f"  {'OK (pre-normalised)' if norms.std() < 1e-3 else 'NOT normalised'}")


def anisotropy(v: np.ndarray) -> None:
    _header("ANISOTROPY  (the similarity floor)")
    a, b = RNG.choice(len(v), 40_000), RNG.choice(len(v), 40_000)
    keep = a != b
    sims = np.einsum("ij,ij->i", v[a[keep]], v[b[keep]])
    p1, p50, p99 = np.percentile(sims, [1, 50, 99])
    print(f"  random-pair cosine   mean {sims.mean():.3f}  p1 {p1:.3f} "
          f"p50 {p50:.3f}  p99 {p99:.3f}")
    print(f"  ||mean vector||      {np.linalg.norm(v.mean(0)):.3f} "
          "(0 = isotropic, 1 = fully collapsed)")
    print(f"  => a score of {p50:.2f} means UNRELATED, not 'somewhat similar'.")
    print(f"  => set retrieval thresholds relative to {p99:.2f}, not to 0.")


def dimensionality(v: np.ndarray) -> None:
    _header("EFFECTIVE DIMENSIONALITY")
    sample = v[RNG.choice(len(v), min(8000, len(v)), replace=False)]
    s = np.linalg.svd(sample - sample.mean(0), compute_uv=False)
    ev = s ** 2 / (s ** 2).sum()
    cum = np.cumsum(ev)
    for t in (0.5, 0.9, 0.99):
        print(f"  {t:.0%} variance in      {np.searchsorted(cum, t) + 1:5d} "
              f"of {v.shape[1]} dims")
    print(f"  top PC alone         {ev[0]:.1%}  (>20% = a rogue dimension)")
    print(f"  participation ratio  {1 / (ev ** 2).sum():.0f}")


def hubness(v: np.ndarray) -> None:
    from scipy.stats import skew
    _header("HUBNESS")
    occ = np.bincount(_knn(v, np.arange(len(v)), 10).ravel(), minlength=len(v))
    print(f"  k-occurrence skew    {skew(occ):.2f}  (> 3 distorts top-k)")
    print(f"  biggest hub          appears in {occ.max()} top-10 lists")
    print(f"  never a neighbour    {(occ == 0).sum():,} ({(occ == 0).mean():.1%})")


def alignment(v: np.ndarray, meta: pd.DataFrame) -> None:
    from scipy.stats import spearmanr
    _header("SEMANTIC ALIGNMENT  (does geometry match meaning?)")
    tags = meta["tags"].fillna("[]").apply(lambda s: set(json.loads(s))).values

    def jac(i, j):
        u = tags[i] | tags[j]
        return len(tags[i] & tags[j]) / len(u) if u else 0.0

    q = RNG.choice(len(v), min(2500, len(v)), replace=False)
    nn = _knn(v, q, K)
    obs = np.mean([np.mean([jac(n, qi) for n in row]) for row, qi in zip(nn, q)])

    a, b = RNG.choice(len(v), 6000), RNG.choice(len(v), 6000)
    keep = a != b
    a, b = a[keep], b[keep]
    base = np.mean([jac(i, j) for i, j in zip(a, b)])
    cos = np.einsum("ij,ij->i", v[a], v[b])
    js = np.array([jac(i, j) for i, j in zip(a, b)])

    print(f"  tag-Jaccard @{K}      {obs:.3f}  vs random baseline {base:.3f} "
          f"({obs / base:.1f}x)")
    print(f"  spearman(cos, tags)  {spearmanr(cos, js).statistic:.3f} "
          "(> 0.3 is healthy)")
    rating = pd.to_numeric(meta["rating"], errors="coerce").values
    leak = spearmanr(cos, -np.abs(rating[a] - rating[b])).statistic
    print(f"  spearman(cos, rating) {leak:.3f}  "
          "(want ~0; high = rating leaked into the text)")


def sequels(v: np.ndarray, meta: pd.DataFrame) -> None:
    """Ground-truth retrieval: can a film find its own sequel?"""
    _header("SEQUEL RETRIEVAL  (ground-truth end-to-end check)")
    titles = meta["title"].values
    index = {t.lower(): i for i, t in enumerate(titles)}
    pairs = []
    for t, i in index.items():
        stem = re.sub(r"\s*\(.*\)$", "", t)
        for suffix in (" 2", " ii", " part 2"):
            j = index.get(stem + suffix) or index.get(f"{stem}{suffix} (film)")
            if j is not None:
                pairs.append((i, j))
    if not pairs:
        print("  no sequel pairs detected")
        return

    ranks = []
    for i, j in pairs:
        sims = v @ v[i]
        sims[i] = -np.inf
        ranks.append(int((sims > sims[j]).sum()) + 1)
    ranks = np.array(ranks)
    n = len(v)
    print(f"  pairs tested         {len(pairs)}")
    print(f"  median rank          {np.median(ranks):.0f} of {n:,}")
    for k in (1, 10, 100):
        print(f"  recall@{k:<4}          {(ranks <= k).mean():.1%}"
              f"   (chance {k / n:.2%})")


def main(input_dir: Path) -> None:
    v = np.load(input_dir / "embeddings.npy").astype(np.float32)
    meta_file = input_dir / "movies_2d.csv"
    if not meta_file.exists():
        meta_file = input_dir / "movies_metadata.csv"
    meta = pd.read_csv(meta_file)
    if len(v) != len(meta):
        raise SystemExit(f"Row mismatch: {len(v)} vectors vs {len(meta)} rows.")

    integrity(v, meta)
    vn = _normalise(v)
    anisotropy(vn)
    dimensionality(v)
    hubness(vn)
    alignment(vn, meta)
    sequels(vn, meta)
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    main(p.parse_args().input_dir)
