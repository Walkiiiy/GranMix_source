"""Build the fixed hierarchy from training-only Jina-v2 embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DOMAINS = ("arxiv", "book", "c4", "cc", "github", "stackexchange", "wikipedia")
CONNECTORS = {
    "root": ["arxiv", "c1", "c2"],
    "c1": ["book", "github"],
    "c2": ["c4", "c3"],
    "c3": ["cc", "wikipedia", "stackexchange"],
}
MODEL = "jinaai/jina-embeddings-v2-small-en"
DIMENSION = 512
SEQUENCE_LENGTH = 2048
DEPTH = 5
CHUNK_SIZE = 8192


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def similarities(vectors, indices, direction):
    values = np.empty(len(indices), dtype=np.float64)
    for start in range(0, len(indices), CHUNK_SIZE):
        selected = indices[start:start + CHUNK_SIZE]
        values[start:start + len(selected)] = np.asarray(vectors[selected], dtype=np.float64) @ direction
    return values


def centroid(vectors, indices):
    total = np.zeros(vectors.shape[1], dtype=np.float64)
    for start in range(0, len(indices), CHUNK_SIZE):
        total += np.asarray(vectors[indices[start:start + CHUNK_SIZE]], dtype=np.float64).sum(axis=0)
    norm = np.linalg.norm(total)
    if norm == 0:
        return np.asarray(vectors[indices[0]], dtype=np.float64).copy()
    return total / norm


def balanced_split(vectors, indices, rng, max_iterations=100):
    """Alternate spherical centroid updates and exact two-cluster capacity assignment."""
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    if len(indices) < 2:
        raise ValueError("A binary split needs at least two training rows")
    first = np.asarray(vectors[indices[rng.integers(len(indices))]], dtype=np.float64)
    second = np.asarray(vectors[indices[np.argmin(similarities(vectors, indices, first))]], dtype=np.float64)
    centers = np.stack([first, second])
    capacity = (len(indices) + 1) // 2
    previous = None
    for _ in range(max_iterations):
        differences = similarities(vectors, indices, centers[0] - centers[1])
        # Fixed capacities reduce two-means assignment to sorting the cosine advantage.
        order = np.argsort(-differences, kind="stable")
        left = np.zeros(len(indices), dtype=bool)
        left[order[:capacity]] = True
        if previous is not None and np.array_equal(left, previous):
            break
        previous = left
        centers[0] = centroid(vectors, indices[left])
        centers[1] = centroid(vectors, indices[~left])
    return indices[left], indices[~left]


def load_inputs(embedding_dir, data_dir):
    metadata_path = embedding_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get("model") != MODEL or metadata.get("split") != "train"
            or metadata.get("sequence_length") != SEQUENCE_LENGTH):
        raise ValueError(f"metadata.json must declare model={MODEL}, split=train, sequence_length=2048")
    expected = {f"{domain}/train.npy" for domain in DOMAINS}
    supplied = {path.relative_to(embedding_dir).as_posix() for path in embedding_dir.rglob("*.npy")}
    if supplied != expected:
        raise ValueError(f"Expected exactly seven training embedding files; missing={sorted(expected-supplied)}, extra={sorted(supplied-expected)}")
    embeddings, counts, data_files, embedding_files = {}, {}, [], []
    for domain in DOMAINS:
        relative = f"{domain}/train.bin"
        path = data_dir / relative
        byte_count = path.stat().st_size
        if byte_count % 2:
            raise ValueError(f"{relative}: uint16 token data has an odd byte count")
        rows = byte_count // 2 // SEQUENCE_LENGTH
        if rows < 2 ** DEPTH:
            raise ValueError(f"{domain}: at least 32 complete training sequences are required")
        embedding_path = embedding_dir / domain / "train.npy"
        vectors = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        if vectors.shape != (rows, DIMENSION) or vectors.dtype.kind != "f":
            raise ValueError(f"{embedding_path}: expected floating-point shape ({rows}, 512)")
        for start in range(0, rows, CHUNK_SIZE):
            block = np.asarray(vectors[start:start + CHUNK_SIZE], dtype=np.float64)
            if (not np.isfinite(block).all()
                    or not np.allclose(np.linalg.norm(block, axis=1), 1., rtol=1e-3, atol=1e-6)):
                raise ValueError(f"{embedding_path}: expected finite L2-normalized vectors")
        embeddings[domain], counts[domain] = vectors, rows
        data_files.append({"path": relative, "bytes": byte_count, "sha256": sha256(path)})
        embedding_files.append({
            "path": f"{domain}/train.npy", "bytes": embedding_path.stat().st_size,
            "sha256": sha256(embedding_path), "rows": rows,
        })
    provenance = {
        "model": MODEL, "dimension": DIMENSION, "normalization": "l2", "split": "train",
        "sequence_length": SEQUENCE_LENGTH, "source_metadata": metadata,
        "metadata_sha256": sha256(metadata_path), "files": embedding_files,
    }
    return embeddings, counts, data_files, provenance


def build_tree(embedding_dir, data_dir, output_dir, *, seed=0, max_iterations=100):
    embedding_dir, data_dir, output_dir = map(lambda path: Path(path).resolve(),
                                            (embedding_dir, data_dir, output_dir))
    repository = Path(__file__).resolve().parents[1]
    if output_dir == repository or repository in output_dir.parents:
        raise ValueError("Choose an output directory outside the source repository")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise ValueError("The output parent directory must already exist")
    if seed < 0 or max_iterations < 1:
        raise ValueError("seed must be nonnegative and max_iterations must be positive")
    embeddings, counts, data_files, provenance = load_inputs(embedding_dir, data_dir)
    offsets, total = {}, 0
    for domain in DOMAINS:
        offsets[domain] = total
        total += counts[domain]
    labels = np.full(total, -1, dtype=np.int32)
    nodes, leaf_ids = {}, []
    rng = np.random.default_rng(seed)

    def partition(domain, path, indices):
        name = domain + ("/" + "/".join(map(str, path)) if path else "")
        if len(path) == DEPTH:
            leaf = len(leaf_ids)
            leaf_ids.append(name)
            labels[offsets[domain] + indices] = leaf
            nodes[name] = {"id": name, "children": [], "leaves": [leaf], "roots": [domain]}
        else:
            children = balanced_split(embeddings[domain], indices, rng, max_iterations)
            names = [partition(domain, path + [side], rows) for side, rows in enumerate(children)]
            nodes[name] = {"id": name, "children": names}
        return name

    for domain in DOMAINS:
        partition(domain, [], np.arange(counts[domain], dtype=np.int64))
        print(f"{domain}: {counts[domain]} training rows, 32 leaves", flush=True)
    for name, children in CONNECTORS.items():
        nodes[name] = {"id": name, "children": list(children)}
    if np.any(labels < 0):
        raise ArithmeticError("Some training rows were not assigned to a leaf")
    leaf_count = np.bincount(labels, minlength=len(leaf_ids))
    if len(nodes) != 445 or len(leaf_ids) != 224 or np.any(leaf_count == 0):
        raise ArithmeticError("Hierarchy must contain 445 nodes and 224 nonempty leaves")
    leaf_mass = np.array([leaf_count[i] / (len(DOMAINS) * counts[name.split("/")[0]])
                          for i, name in enumerate(leaf_ids)], dtype=np.float64)

    def finalize(name, parent, depth):
        node = nodes[name]
        node.update(parent=parent, depth=depth)
        if node["children"]:
            for child in node["children"]:
                finalize(child, name, depth + 1)
            node["leaves"] = [leaf for child in node["children"] for leaf in nodes[child]["leaves"]]
            node["roots"] = sorted({domain for child in node["children"] for domain in nodes[child]["roots"]})
        node["size"] = int(leaf_count[node["leaves"]].sum())
        node["mass"] = float(leaf_mass[node["leaves"]].sum())

    finalize("root", None, 0)
    tree = {
        "root": "root", "depth": DEPTH, "branch": 2, "mode": "capacity-constrained-spherical",
        "n_leaves": len(leaf_ids), "leaf_ids": leaf_ids, "nodes": nodes,
        "dom_offset": offsets, "dom_n": counts, "leaf_count": leaf_count.tolist(),
        "leaf_mass": leaf_mass.tolist(),
    }
    output_dir.mkdir()
    (output_dir / "nodes.json").write_text(json.dumps(tree, indent=2) + "\n")
    np.save(output_dir / "train_leaf.npy", labels, allow_pickle=False)
    identity = {
        "version": 1,
        "tree": {name: sha256(output_dir / name) for name in ("nodes.json", "train_leaf.npy")},
        "files": data_files, "embeddings": provenance,
        "construction": {
            "method": "capacity-constrained spherical 2-means", "depth": DEPTH, "seed": seed,
            "max_iterations": max_iterations, "node_count": len(nodes), "leaf_count": len(leaf_ids),
        },
    }
    (output_dir / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def main():
    parser = argparse.ArgumentParser(
        description="Build 445 nodes and 224 leaves from seven sources' training-only Jina-v2 embeddings.",
        epilog=("Input: <embeddings>/<domain>/train.npy, in the exact packed train.bin row order, "
                "plus metadata.json declaring model, split=train and sequence_length=2048. "
                "The input must exclude probe/dev/test rows. Shape and normalization are checked; "
                "encoder identity, row alignment and split provenance remain input responsibilities."))
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing <domain>/train.bin (uint16)")
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the source repository")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=100)
    args = parser.parse_args()
    build_tree(args.embeddings, args.data, args.output, seed=args.seed, max_iterations=args.max_iterations)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
