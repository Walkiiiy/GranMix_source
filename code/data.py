from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ReferenceTree:
    """Reference masses on a tree; descendants stores leaf positions, not node IDs."""

    def __init__(self, children, reference_mass):
        self.children = [tuple(int(child) for child in row) for row in children]
        self.mass = np.array(reference_mass, dtype=np.float64, copy=True)
        self.n = len(self.children)
        if self.n == 0 or self.mass.shape != (self.n,):
            raise ValueError("one mass is required for every tree node")
        if not np.isfinite(self.mass).all() or np.any(self.mass <= 0):
            raise ValueError("reference masses must be finite and positive")
        self.parents = np.full(self.n, -1, dtype=np.int64)
        for node, row in enumerate(self.children):
            if len(row) != len(set(row)):
                raise ValueError("duplicate tree child")
            for child in row:
                if not 0 <= child < self.n or child == node:
                    raise ValueError("invalid tree edge")
                if self.parents[child] != -1:
                    raise ValueError("a tree node cannot have two parents")
                self.parents[child] = node
            if row and not np.isclose(self.mass[list(row)].sum(), self.mass[node],
                                       rtol=1e-10, atol=1e-12):
                raise ValueError("child masses must sum to their parent mass")
        roots = np.flatnonzero(self.parents == -1)
        if roots.size != 1:
            raise ValueError("the tree must have exactly one root")
        self.root = int(roots[0])
        if not np.isclose(self.mass[self.root], 1.0, atol=1e-10, rtol=0):
            raise ValueError("root reference mass must be one")
        self.leaves = np.array([i for i, row in enumerate(self.children) if not row], dtype=np.int64)
        self.leaf_position = {int(node): i for i, node in enumerate(self.leaves)}
        self.leaf_mass = self.mass[self.leaves].copy()
        self.descendants = [None] * self.n
        seen = set()

        def collect(node):
            if node in seen:
                raise ValueError("cycle or repeated subtree")
            seen.add(node)
            row = self.children[node]
            positions = (np.concatenate([collect(child) for child in row])
                         if row else np.array([self.leaf_position[node]], dtype=np.int64))
            self.descendants[node] = positions
            return positions

        collect(self.root)
        if len(seen) != self.n:
            raise ValueError("disconnected tree")

    def validate_cut(self, cut):
        raw = list(cut)
        if any(isinstance(x, (bool, np.bool_)) or int(x) != x for x in raw):
            raise ValueError("cut node IDs must be integers")
        nodes = [int(x) for x in raw]
        if not nodes or len(nodes) != len(set(nodes)) or any(node < 0 or node >= self.n for node in nodes):
            raise ValueError("invalid cut")
        coverage = np.zeros(len(self.leaves), dtype=np.int64)
        for node in nodes:
            coverage[self.descendants[node]] += 1
        if not np.all(coverage == 1):
            raise ValueError("cut must cover each leaf exactly once")
        return nodes

    def sample_distribution(self, cut, weights):
        nodes = self.validate_cut(cut)
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(nodes),) or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("cut weights must be finite and positive")
        if not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-10):
            raise ValueError("cut weights must sum to one")
        w = w / w.sum()
        q = np.zeros(len(self.leaves), dtype=np.float64)
        for node, weight in zip(nodes, w):
            positions = self.descendants[node]
            q[positions] = weight * self.leaf_mass[positions] / self.mass[node]
        if np.any(q <= 0) or not np.isclose(q.sum(), 1.0, rtol=0, atol=1e-10):
            raise ArithmeticError("induced sampler is not a probability distribution")
        return q

    def kappa(self, cut, weights):
        """Reference-conditional inherited log sampling preference at every node."""
        log_ratio = np.log(self.sample_distribution(cut, weights) / self.leaf_mass)
        return np.array([float(self.leaf_mass[pos] @ log_ratio[pos]) / self.mass[node]
                         for node, pos in enumerate(self.descendants)])


def integer_vector(value, name):
    array = np.asarray(value)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be an integer vector")
    return array.astype(np.int64, copy=False)


class TreeCorpus:
    def __init__(self, tree_dir, data_dir, domains, sequence_length=2048, seed=0):
        self.tree_dir = Path(tree_dir)
        self.data_dir = Path(data_dir)
        self.domains = tuple(map(str, domains))
        if not self.domains or len(set(self.domains)) != len(self.domains):
            raise ValueError("Select unique nonempty domains")
        if int(sequence_length) != sequence_length or sequence_length < 1:
            raise ValueError("sequence_length must be a positive integer")
        self.sequence_length = int(sequence_length)
        self.domain_to_id = {name: index for index, name in enumerate(self.domains)}
        tree_bytes = (self.tree_dir / "nodes.json").read_bytes()
        raw = json.loads(tree_bytes)
        if any(d not in raw["dom_n"] or d not in raw["nodes"] for d in self.domains):
            raise ValueError("Selected domain is absent from the hierarchy")
        original_names = raw["leaf_ids"]
        if len(set(original_names)) != len(original_names):
            raise ValueError("Duplicate original leaf names")
        labels = np.load(self.tree_dir / "train_leaf.npy", mmap_mode="r", allow_pickle=False)
        expected_total = max(int(raw["dom_offset"][d]) + int(n) for d, n in raw["dom_n"].items())
        if labels.ndim != 1 or labels.dtype.kind not in "iu" or labels.size != expected_total:
            raise ValueError("Leaf assignments do not match domain offsets")
        self._maps = {}
        self._sizes = {}
        self.tail_tokens = {}
        self.n_train = np.empty(len(self.domains), dtype=np.int64)
        self.original_domain_offsets = np.array([raw["dom_offset"][d] for d in self.domains], dtype=np.int64)
        self._domain_labels = []
        original_pools, original_domains = {}, {}
        selected = set()
        label_hash = hashlib.sha256()
        for domain_id, domain in enumerate(self.domains):
            if (self.data_dir / domain / "strat_labels.npy").exists():
                raise ValueError("strat_labels.npy changes the frozen sampling protocol")
            count = self.split_size(domain_id, "train")
            if count < 1 or count != int(raw["dom_n"][domain]):
                raise ValueError(f"{domain}: complete training rows disagree with the hierarchy")
            self.n_train[domain_id] = count
            start = self.original_domain_offsets[domain_id]
            domain_labels = np.asarray(labels[start:start + count])
            self._domain_labels.append(domain_labels)
            allowed = np.asarray(raw["nodes"][domain]["leaves"], dtype=np.int64)
            if (not allowed.size or np.any(allowed < 0) or np.any(allowed >= len(original_names))
                    or len(set(allowed.tolist())) != len(allowed)
                    or not np.isin(domain_labels, allowed).all()):
                raise ValueError(f"{domain}: invalid semantic leaf assignment")
            if selected.intersection(allowed.tolist()):
                raise ValueError("Domains cannot share leaves")
            selected.update(allowed.tolist())
            # Stable sorting preserves original row order inside each leaf.
            order = np.argsort(domain_labels, kind="stable")
            sorted_labels = domain_labels[order]
            for leaf in allowed.tolist():
                lower, upper = np.searchsorted(sorted_labels, leaf, side="left"), np.searchsorted(sorted_labels, leaf, side="right")
                pool = order[int(lower):int(upper)]
                if not len(pool) or len(pool) != int(raw["leaf_count"][leaf]):
                    raise ValueError(f"Leaf {leaf}: row count disagrees with the hierarchy")
                original_pools[leaf] = pool
                original_domains[leaf] = domain_id
            label_hash.update(domain.encode())
            label_hash.update(np.asarray(domain_labels, dtype="<i8").tobytes())

        kept = {}
        active = set()
        selected_names = {original_names[i] for i in selected}

        def prune(name):
            if name in active:
                raise ValueError("Hierarchy contains a cycle")
            active.add(name)
            children = raw["nodes"][name]["children"]
            remaining = [child for child in children if prune(child)]
            survives = bool(remaining) if children else name in selected_names
            if survives:
                kept[name] = remaining
            active.remove(name)
            return survives

        if not prune(raw["root"]):
            raise ValueError("No selected hierarchy leaves")
        self.node_names = []

        def visit(name):
            self.node_names.append(name)
            for child in kept[name]:
                visit(child)

        visit(raw["root"])
        if len(set(self.node_names)) != len(self.node_names):
            raise ValueError("Hierarchy is not a tree")
        self.names = self.node_names
        self.node_to_id = {name: i for i, name in enumerate(self.node_names)}
        self.children = [[self.node_to_id[c] for c in kept[name]] for name in self.node_names]
        self.parents = np.full(len(self.children), -1, dtype=np.int64)
        self.reference_mass = np.zeros(len(self.children), dtype=np.float64)
        original_id = {name: i for i, name in enumerate(original_names)}
        for node in reversed(range(len(self.children))):
            children = self.children[node]
            if children:
                self.parents[children] = node
                self.reference_mass[node] = self.reference_mass[children].sum()
            else:
                leaf = original_id[self.node_names[node]]
                domain = original_domains[leaf]
                self.reference_mass[node] = len(original_pools[leaf]) / (len(self.domains) * self.n_train[domain])
        self.tree = ReferenceTree(self.children, self.reference_mass)
        self.source_cut = [self.node_to_id[d] for d in self.domains]
        self.initial_cut = list(self.source_cut)
        self.tree.validate_cut(self.source_cut)
        self.leaf_nodes = np.asarray(self.tree.leaves, dtype=np.int64).copy()
        self.original_leaf_ids = np.array([original_id[self.node_names[n]] for n in self.leaf_nodes], dtype=np.int64)
        self.leaf_domain_ids = np.array([original_domains[int(n)] for n in self.original_leaf_ids], dtype=np.int64)
        self._leaf_pools = [original_pools[int(n)] for n in self.original_leaf_ids]
        self.leaf_counts = np.array([len(pool) for pool in self._leaf_pools], dtype=np.int64)
        self._original_to_node = dict(zip(map(int, self.original_leaf_ids), map(int, self.leaf_nodes)))
        self._cut_cache = {}
        self.rng = np.random.default_rng(seed)
        identity = {"domains": self.domains, "sequence_length": self.sequence_length,
                    "tree_sha256": hashlib.sha256(tree_bytes).hexdigest(),
                    "selected_labels_sha256": label_hash.hexdigest(), "n_train": self.n_train.tolist()}
        self._identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.manifest = dict(identity, tail_tokens=dict(self.tail_tokens), node_names=self.node_names,
                             source_cut=self.source_cut, sampling="independent replacement within the selected tree cut")

    @property
    def masses(self):
        return self.reference_mass.copy()

    def _domain(self, domain):
        if isinstance(domain, str):
            if domain not in self.domain_to_id:
                raise ValueError(f"Unselected domain: {domain}")
            return self.domain_to_id[domain]
        if not isinstance(domain, (int, np.integer)) or not 0 <= domain < len(self.domains):
            raise ValueError("Invalid domain index")
        return int(domain)

    def _open_split(self, domain, split):
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        key = (int(domain), split)
        if key not in self._maps:
            file = self.data_dir / self.domains[int(domain)] / f"{split}.bin"
            byte_count = file.stat().st_size
            if byte_count % 2:
                raise ValueError(f"{file}: incomplete uint16 token")
            tokens = np.memmap(file, mode="r", dtype=np.uint16) if byte_count else np.empty(0, dtype=np.uint16)
            self._maps[key] = tokens
            self._sizes[key] = len(tokens) // self.sequence_length
            self.tail_tokens[f"{self.domains[int(domain)]}/{split}"] = len(tokens) % self.sequence_length
        return self._maps[key]

    def split_size(self, domain, split="val"):
        domain = self._domain(domain)
        self._open_split(domain, split)
        return self._sizes[(domain, split)]

    def validation(self, domain, indices, *, split="val"):
        domain = self._domain(domain)
        rows = integer_vector(indices, "indices")
        tokens = self._open_split(domain, split)
        if np.any(rows < 0) or np.any(rows >= self._sizes[(domain, split)]):
            raise IndexError("Row outside complete sequences")
        result = np.empty((len(rows), self.sequence_length), dtype=np.int64)
        for i, row in enumerate(rows):
            offset = int(row) * self.sequence_length
            result[i] = tokens[offset:offset + self.sequence_length]
        return result

    def leaf_nodes_for_rows(self, domain_ids, row_ids):
        domains, rows = integer_vector(domain_ids, "domain_ids"), integer_vector(row_ids, "row_ids")
        if domains.shape != rows.shape or np.any(domains < 0) or np.any(domains >= len(self.domains)):
            raise ValueError("Invalid row/domain pairs")
        result = np.empty(rows.size, dtype=np.int64)
        for domain in np.unique(domains):
            mask = domains == domain
            selected = rows[mask]
            if np.any(selected < 0) or np.any(selected >= self.n_train[domain]):
                raise IndexError("Training row outside complete sequences")
            result[mask] = [self._original_to_node[int(n)] for n in self._domain_labels[domain][selected]]
        return result

    def draw(self, cut, weights, batch_size, rng=None):
        cut = self.tree.validate_cut(cut)
        weights = np.asarray(weights, dtype=np.float64)
        if (weights.shape != (len(cut),) or not np.isfinite(weights).all() or np.any(weights <= 0)
                or not np.isclose(weights.sum(), 1.0, atol=1e-10, rtol=0)
                or int(batch_size) != batch_size or batch_size < 1):
            raise ValueError("Invalid mixture or batch size")
        weights = weights / weights.sum()
        rng = self.rng if rng is None else rng
        if not isinstance(rng, np.random.Generator):
            raise TypeError("Use a numpy Generator")
        key = tuple(cut)
        if key not in self._cut_cache:
            membership = np.empty(len(self.leaf_nodes), dtype=np.int64)
            for group, node in enumerate(cut):
                membership[self.tree.descendants[node]] = group
            self._cut_cache[key] = membership
        membership = self._cut_cache[key]
        leaf_q = weights[membership] * self.tree.leaf_mass / self.reference_mass[np.asarray(cut)[membership]]
        leaf_q /= leaf_q.sum()
        positions = rng.choice(len(self.leaf_nodes), size=int(batch_size), p=leaf_q)
        offsets = rng.integers(0, self.leaf_counts[positions])
        rows = np.array([self._leaf_pools[int(p)][int(i)] for p, i in zip(positions, offsets)], dtype=np.int64)
        domains = self.leaf_domain_ids[positions]
        ids = np.empty((int(batch_size), self.sequence_length), dtype=np.int64)
        for domain in np.unique(domains):
            mask = domains == domain
            ids[mask] = self.validation(int(domain), rows[mask], split="train")
        return {"ids": ids, "domain_ids": domains.copy(), "leaf_nodes": self.leaf_nodes[positions].copy(),
                "row_ids": rows, "p_i": 1.0 / (len(self.domains) * self.n_train[domains]),
                "q_i": leaf_q[positions] / self.leaf_counts[positions],
                "global_row_ids": self.original_domain_offsets[domains] + rows,
                "original_leaf_ids": self.original_leaf_ids[positions].copy(),
                "group_ids": membership[positions].copy()}

    def state_dict(self):
        return {"version": 1, "identity": self._identity, "rng": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("identity") != self._identity:
            raise ValueError("Sampler state belongs to a different corpus")
        candidate = np.random.default_rng()
        candidate.bit_generator.state = copy.deepcopy(state["rng"])
        self.rng.bit_generator.state = candidate.bit_generator.state


def batch_sha256(ids):
    return hashlib.sha256(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()


class _JoinedChunks:
    """Concatenate partial token chunks before rechunking, without copying the corpus."""

    def __init__(self, tokens, offsets, sequence_length):
        self.tokens = tokens
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.sequence_length = int(sequence_length)
        self.lengths = np.minimum(self.sequence_length, len(tokens) - self.offsets)
        if np.any(self.offsets < 0) or np.any(self.lengths <= 0):
            raise ValueError("Invalid source chunk offsets")
        self.boundaries = np.concatenate(([0], np.cumsum(self.lengths, dtype=np.int64)))
        self.rows = int(self.boundaries[-1]) // self.sequence_length
        self.has_tail = bool(np.any(self.lengths != self.sequence_length))

    def row(self, index):
        if not 0 <= index < self.rows:
            raise IndexError("Joined row outside token stream")
        start = int(index) * self.sequence_length
        chunk = int(np.searchsorted(self.boundaries, start, side="right") - 1)
        inside = start - int(self.boundaries[chunk])
        result = np.empty(self.sequence_length, dtype=np.int64)
        written = 0
        while written < self.sequence_length:
            take = min(int(self.lengths[chunk]) - inside, self.sequence_length - written)
            offset = int(self.offsets[chunk]) + inside
            result[written:written + take] = self.tokens[offset:offset + take]
            written += take
            chunk += 1
            inside = 0
        return result


@dataclass(frozen=True)
class ProbeSegment:
    batches: tuple
    target: int
    requested_updates: int
    selected_offsets: tuple
    mapped_quota: tuple
    choice_population: tuple
    metadata_supported: bool
    selected_tail_domains: tuple
    source_rows_after_concatenation: tuple

    @property
    def actual_updates(self):
        return len(self.batches)

    @property
    def supported(self):
        return self.metadata_supported

    def __iter__(self):
        return iter(self.batches)

    def audit_record(self):
        digest = hashlib.sha256()
        for offsets in self.selected_offsets:
            digest.update(np.asarray(offsets, dtype="<i8").tobytes())
        return {"target": self.target, "requested_updates": self.requested_updates,
                "actual_updates": self.actual_updates,
                "actual_batch_sizes": [len(batch["ids"]) for batch in self.batches],
                "metadata_supported": self.metadata_supported,
                "selected_tail_domains": list(self.selected_tail_domains),
                "source_rows_after_concatenation": list(self.source_rows_after_concatenation),
                "mapped_quota": list(self.mapped_quota),
                "choice_population": list(self.choice_population),
                "selected_offsets_sha256": digest.hexdigest(),
                "batch_sha256": [batch_sha256(batch["ids"]) for batch in self.batches],
                "normal_window_ht_audit": False,
                "propensity_scope": "original_probe_unconditional_marginal_only"}


class MappedProbeSampler:
    """One persistent source-selection RNG, independent of the normal sampler."""

    def __init__(self, corpus, seed=0):
        self.corpus = corpus
        self.rng = np.random.default_rng(seed)

    def segment(self, target, one_hot_factor, requested_steps, batch_size=8):
        corpus = self.corpus
        domains = len(corpus.domains)
        if domains < 2 or not isinstance(target, (int, np.integer)) or not 0 <= target < domains:
            raise ValueError("Invalid probe target")
        if not 0 < one_hot_factor < 1:
            raise ValueError("one_hot_factor must lie strictly between zero and one")
        if (int(requested_steps) != requested_steps or requested_steps < 1
                or int(batch_size) != batch_size or batch_size < 1):
            raise ValueError("Probe dimensions must be positive integers")
        requested_steps, batch_size = int(requested_steps), int(batch_size)
        probabilities = np.ones(domains) * (1 - one_hot_factor) / (domains - 1)
        probabilities[int(target)] = one_hot_factor
        # Use Python sum before rounding the sample counts and batch quotas.
        probabilities = probabilities / sum(probabilities)
        required = (probabilities * (requested_steps * batch_size)).astype(int)
        required[-1] = requested_steps * batch_size - required[:-1].sum()
        quota = np.array([max(1, int(p * batch_size)) for p in probabilities])
        quota[-1] = batch_size - quota[:-1].sum()
        if np.any(quota < 0) or quota.sum() != batch_size:
            raise ValueError("Invalid mapped batch quotas")
        streams, selected, population, tails = [], [], [], []
        for domain in range(domains):
            tokens = corpus._open_split(domain, "train")
            available = np.arange(0, len(tokens), corpus.sequence_length)
            offsets = self.rng.choice(available, size=int(required[domain]), replace=False)
            stream = _JoinedChunks(tokens, offsets, corpus.sequence_length)
            selected.append(offsets)
            population.append(len(available))
            streams.append(stream)
            if stream.has_tail:
                tails.append(domain)
        supported = not tails
        # Generate all batches before truncating to preserve source-selection RNG state.
        pointers = np.zeros(domains, dtype=np.int64)
        all_pairs = []
        while True:
            pairs = []
            exhausted = False
            for domain, count in enumerate(quota):
                for _ in range(int(count)):
                    if pointers[domain] >= streams[domain].rows:
                        exhausted = True
                        break
                    pairs.append((domain, int(pointers[domain])))
                    pointers[domain] += 1
                if exhausted:
                    break
            if exhausted:
                break
            self.rng.shuffle(pairs)
            all_pairs.append(pairs)
        batches = []
        population = np.asarray(population, dtype=np.int64)
        for pairs in all_pairs[:requested_steps]:
            domain_ids = np.array([d for d, _ in pairs], dtype=np.int64)
            ids = np.stack([streams[d].row(row) for d, row in pairs])
            batch = {"ids": ids, "domain_ids": domain_ids,
                     "metadata_supported": supported,
                     "row_ids": None, "leaf_nodes": None, "p_i": None, "q_i": None}
            if supported:
                rows = np.array([int(selected[d][row]) // corpus.sequence_length for d, row in pairs], dtype=np.int64)
                p_i = 1.0 / (domains * corpus.n_train[domain_ids])
                q_i = quota[domain_ids] / batch_size / population[domain_ids]
                batch.update(row_ids=rows, leaf_nodes=corpus.leaf_nodes_for_rows(domain_ids, rows),
                             global_row_ids=corpus.original_domain_offsets[domain_ids] + rows,
                             p_i=p_i, q_i=q_i, marginal_importance_weights=p_i / q_i)
            batches.append(batch)
        return ProbeSegment(tuple(batches), int(target), requested_steps,
                            tuple(tuple(map(int, offsets)) for offsets in selected),
                            tuple(map(int, quota)), tuple(map(int, population)), supported,
                            tuple(tails), tuple(s.rows for s in streams))

    def window(self, order, one_hot_factor, requested_steps, batch_size=8):
        """Yield segments in an already generated global-numpy permutation order."""
        for target in order:
            yield self.segment(target, one_hot_factor, requested_steps, batch_size)

    def mirror_normal_rng(self):
        """Consume loader shuffles before sampling through TreeCorpus.draw."""
        for domain in range(len(self.corpus.domains)):
            tokens = self.corpus._open_split(domain, "train")
            offsets = np.arange(0, len(tokens), self.corpus.sequence_length)
            self.rng.shuffle(offsets)

    def state_dict(self):
        return {"version": 1, "identity": self.corpus._identity,
                "rng": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("identity") != self.corpus._identity:
            raise ValueError("Probe sampler state belongs to a different corpus")
        candidate = np.random.default_rng()
        candidate.bit_generator.state = copy.deepcopy(state["rng"])
        self.rng.bit_generator.state = candidate.bit_generator.state


class UniformSampler:
    """Uniform sources with a shuffled, sequential, cycling stream per source.

    A partial final source chunk participates in the shuffle, is concatenated
    before rechunking, and can shift all subsequent examples. Uniform does not
    use semantic labels, so those shifted examples remain valid training data.
    """

    def __init__(self, corpus, seed=0, batch_size=8):
        if int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.corpus, self.seed, self.batch_size = corpus, int(seed), int(batch_size)
        self.selection_rng = np.random.default_rng(seed)
        self.interleave_rng = np.random.default_rng(seed)
        self.probabilities = np.ones(len(corpus.domains))
        self.probabilities /= sum(self.probabilities)
        self.streams = []
        for domain in range(len(corpus.domains)):
            tokens = corpus._open_split(domain, "train")
            offsets = np.arange(0, len(tokens), corpus.sequence_length)
            self.selection_rng.shuffle(offsets)
            stream = _JoinedChunks(tokens, offsets, corpus.sequence_length)
            if not stream.rows:
                raise ValueError("Uniform needs nonempty source streams")
            self.streams.append(stream)
        self.cursors = np.zeros(len(self.streams), dtype=np.int64)

    def __iter__(self):
        return self

    def __next__(self):
        # Renormalize in place once per batch, matching the source-index iterator.
        self.probabilities /= sum(self.probabilities)
        domains = self.interleave_rng.choice(len(self.streams), self.batch_size, p=self.probabilities)
        ids = np.empty((self.batch_size, self.corpus.sequence_length), dtype=np.int64)
        stream_rows = np.empty(self.batch_size, dtype=np.int64)
        for i, domain in enumerate(domains):
            row = int(self.cursors[domain])
            stream_rows[i] = row
            ids[i] = self.streams[domain].row(row)
            self.cursors[domain] = (row + 1) % self.streams[domain].rows
        return {"ids": ids, "domain_ids": domains, "stream_row_ids": stream_rows}

    def state_dict(self):
        return {"version": 1, "identity": self.corpus._identity, "seed": self.seed,
                "batch_size": self.batch_size, "cursors": self.cursors.tolist(),
                "probabilities": self.probabilities.tolist(),
                "interleave_rng": copy.deepcopy(self.interleave_rng.bit_generator.state),
                "selection_rng": copy.deepcopy(self.selection_rng.bit_generator.state)}

    def load_state_dict(self, state):
        if (state.get("version") != 1 or state.get("identity") != self.corpus._identity
                or state.get("seed") != self.seed or state.get("batch_size") != self.batch_size):
            raise ValueError("Uniform sampler state belongs to a different stream")
        cursors = np.asarray(state["cursors"], dtype=np.int64)
        if cursors.shape != self.cursors.shape or np.any(cursors < 0) or any(c >= s.rows for c, s in zip(cursors, self.streams)):
            raise ValueError("Invalid Uniform source cursors")
        p = np.asarray(state["probabilities"], dtype=np.float64)
        if p.shape != self.probabilities.shape or not np.isfinite(p).all() or np.any(p <= 0) or not np.isclose(p.sum(), 1):
            raise ValueError("Invalid Uniform source probabilities")
        candidate = np.random.default_rng()
        candidate.bit_generator.state = copy.deepcopy(state["interleave_rng"])
        selection = np.random.default_rng()
        selection.bit_generator.state = copy.deepcopy(state["selection_rng"])
        self.cursors = cursors.copy()
        self.probabilities = p.copy()
        self.interleave_rng.bit_generator.state = candidate.bit_generator.state
        self.selection_rng.bit_generator.state = selection.bit_generator.state


def create_fixture(directory, domains=('arxiv', 'stackexchange'), sequence_length=12):
    root = Path(directory)
    data, tree = root / 'data', root / 'tree'
    tree.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(319)
    leaves, labels, counts, nodes, offsets = [], [], [], {}, {}
    total = 0
    for domain in domains:
        domain_dir = data / domain
        domain_dir.mkdir(parents=True)
        for split, rows in (('train', 32), ('val', 10), ('test', 9)):
            ids = rng.integers(1, 32, size=(rows, sequence_length), dtype=np.uint16)
            ids[:, 4] = 0
            ids.tofile(domain_dir / (split + '.bin'))
        children = [f'{domain}/0', f'{domain}/1']
        leaf_ids = []
        for child in children:
            index = len(leaves)
            leaves.append(child)
            leaf_ids.append(index)
            labels.extend([index] * 16)
            counts.append(16)
            nodes[child] = {'children': [], 'leaves': [index]}
        nodes[domain] = {'children': children, 'leaves': leaf_ids}
        offsets[domain] = total
        total += 32
    nodes['root'] = {'children': list(domains), 'leaves': list(range(len(leaves)))}
    manifest = {'root': 'root', 'nodes': nodes, 'leaf_ids': leaves, 'leaf_count': counts,
                'dom_n': {d: 32 for d in domains}, 'dom_offset': offsets}
    (tree / 'nodes.json').write_text(json.dumps(manifest))
    np.save(tree / 'train_leaf.npy', np.asarray(labels, dtype=np.int32))
    return data, tree
