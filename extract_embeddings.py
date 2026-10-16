"""Stream embeddings + metadata out of pgvector into output/.

Writes:
    output/embeddings.npy        float32 (n, dim)
    output/movies_metadata.csv   id, title, rating, tags

Row alignment between the two files is positional, so the ORDER BY in the
query is load-bearing. Do not remove it.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from db import get_connection

OUTPUT_DIR = Path("output")
BATCH_SIZE = 1_000

QUERY = """
    SELECT
        id,
        metadata->>'title'       AS title,
        metadata->>'movieRating' AS rating,
        metadata->>'tags'        AS tags,
        embedding
    FROM wiki_vector_store
    WHERE embedding IS NOT NULL
    ORDER BY id
"""


def _to_vector(value) -> np.ndarray:
    """Coerce a pgvector column to float32.

    With register_vector() this is already a numpy array, so the fast path
    hits. The text fallback exists only so a missing adapter degrades to
    'slow' rather than 'crash'.
    """
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str):
        return np.fromstring(value.strip("[]"), sep=",", dtype=np.float32)
    raise TypeError(f"Unsupported embedding type: {type(value)!r}")


def main(output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "movies_metadata.csv"
    vec_path = output_dir / "embeddings.npy"

    vectors: list[np.ndarray] = []

    print("Connecting to database...")
    with get_connection() as conn:
        # Named cursor => server-side, streams instead of buffering the whole
        # result set (~100 MB at current corpus size) in client memory.
        with conn.cursor(name="embedding_export") as cur, \
                meta_path.open("w", newline="", encoding="utf-8") as fh:
            cur.itersize = BATCH_SIZE
            cur.execute(QUERY)

            writer = csv.writer(fh)
            writer.writerow(["id", "title", "rating", "tags"])

            for n, (row_id, title, rating, tags, embedding) in enumerate(cur, 1):
                vectors.append(_to_vector(embedding))
                writer.writerow([row_id, title, rating, tags])
                if n % 5_000 == 0:
                    print(f"  {n:,} rows...")

    if not vectors:
        raise SystemExit("No rows returned -- check the table name and filters.")

    matrix = np.vstack(vectors)
    np.save(vec_path, matrix)

    norms = np.linalg.norm(matrix, axis=1)
    print(f"\nRows     : {len(matrix):,}")
    print(f"Shape    : {matrix.shape}")
    print(f"L2 norms : mean {norms.mean():.4f}  std {norms.std():.5f}")
    if norms.std() > 1e-3:
        print("  WARNING: norms are not uniform. Normalise before using "
              "euclidean or inner-product distance.")
    print(f"Wrote    : {vec_path}\n           {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    main(parser.parse_args().output_dir)
