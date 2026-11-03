"""Preprocessing class for calculating token distributions from datasets."""

import importlib
import os
import tempfile
from multiprocessing import get_context

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from morph.dataset.core.representation import neutral_charge_index
from morph.dataset.core.vocab import Distributions, Vocab
from morph.utils.metrics.tensor import (
    DIST_N_BINS,
    RG_N_BINS,
    accumulate_pairwise_distance_hist,
    accumulate_rog_hist,
    dist_edges,
    rg_edges,
)
from morph.utils.vocab_utils import z_to_atom_types

_PREPROCESSING_WORKER = None


def _init_preprocessing_worker(dataset, vocab):
    global _PREPROCESSING_WORKER  # noqa: PLW0603 (process-local pool state)
    torch.set_num_threads(1)
    _PREPROCESSING_WORKER = _DatasetReduction(dataset, vocab)


def _preprocessing_chunk(task):
    method, indices = task
    return getattr(_PREPROCESSING_WORKER, method)(indices)


class _DatasetReduction:
    """Read raw records and return mergeable vocabulary/statistics reductions.

    This reader has no cache initialization or file-writing behavior, so pool
    workers can construct it directly without bypassing Preprocessing.__init__.
    """

    def __init__(self, dataset: Dataset, vocab: Vocab | None):
        self.train_dataset = dataset
        self.vocab = vocab

    def _collect_tokens(self, indices):
        """Collect vocabulary sets from a range of raw records."""

        # Extract all atom types and edge types from the dataset in one loop
        all_atom_types = set()
        all_edge_type_indices = set()
        all_charge_tokens = set()

        for i in indices:
            data = self.train_dataset.get(i)

            # Extract atom types
            atom_types = z_to_atom_types(data.z.tolist())
            all_atom_types.update(atom_types)

            # Extract edge types — only when the source provides bonds.
            # Datasets without topology (e.g. TMQM) won't have `edge_attr` at all,
            # in which case the canonical edge vocab falls back to ["<NO_BOND>"].
            if (
                hasattr(data, "edge_attr")
                and data.edge_attr is not None
                and data.edge_attr.numel() > 0
            ):
                all_edge_type_indices.update(data.edge_attr.tolist())

            if hasattr(data, "charges") and data.charges is not None:
                all_charge_tokens.update(data.charges.tolist())

        return all_atom_types, all_edge_type_indices, all_charge_tokens

    def _collect_distribution_counts(self, indices):
        """Accumulate mergeable counts and coordinate moments from raw records."""
        # Compute distributions from the provided training dataset only.
        dataset = self.train_dataset

        # Streaming counts and coordinate moments avoid retaining every atom and
        # an O(N^2) dense adjacency for each molecule. The coordinate recurrence
        # is Chan's parallel-variance formula, mathematically identical to the
        # sample standard deviation of the old concatenated coordinate tensor.
        n_atoms_counts: dict[int, int] = {}
        atom_counts = torch.zeros(len(self.vocab.atom_tokens), dtype=torch.long)
        edge_counts = torch.zeros(len(self.vocab.edge_tokens), dtype=torch.long)
        charge_counts = torch.zeros(len(self.vocab.charge_tokens), dtype=torch.long)
        saw_topology = False
        saw_charges = False
        coord_count = 0
        coord_mean = torch.tensor(0.0, dtype=torch.float64)
        coord_m2 = torch.tensor(0.0, dtype=torch.float64)

        atom_to_idx = {token: i for i, token in enumerate(self.vocab.atom_tokens)}
        edge_to_idx = {token: i for i, token in enumerate(self.vocab.edge_tokens)}
        charge_to_idx = {token: i for i, token in enumerate(self.vocab.charge_tokens)}

        # Pointcloud-mode evaluation targets — computed canonically so the same
        # cache serves any representation.
        num_atom_types = len(self.vocab.atom_tokens)
        pair_hist = torch.zeros(
            num_atom_types, num_atom_types, DIST_N_BINS, dtype=torch.float64
        )
        rog_hist = torch.zeros(RG_N_BINS, dtype=torch.float64)
        d_edges = dist_edges()
        r_edges = rg_edges()

        for i in indices:
            data = dataset.get(i)
            num_atoms = data.num_nodes

            atom_types = z_to_atom_types(data.z.tolist())
            atom_type_indices_t = torch.tensor(
                [atom_to_idx[token] for token in atom_types], dtype=torch.long
            )
            atom_counts += atom_type_indices_t.bincount(minlength=atom_counts.numel())

            # Edge histogram: built only if the dataset provides bonds. Sources
            # without topology contribute nothing here (vocab is the singleton
            # canonical [NO_BOND] for them anyway).
            if (
                getattr(data, "edge_attr", None) is not None
                and getattr(data, "edge_index", None) is not None
            ):
                saw_topology = True
                upper = data.edge_index[0] < data.edge_index[1]
                upper_types = data.edge_attr[upper].view(-1)
                # The source datasets store unique symmetric bonds. Their dense
                # upper triangle therefore consists of these bonds plus exactly
                # the remaining <NO_BOND> pairs.
                n_pairs = num_atoms * (num_atoms - 1) // 2
                edge_counts[0] += n_pairs - upper_types.numel()
                mapped_edges = torch.tensor(
                    [edge_to_idx[str(int(t))] for t in upper_types.tolist()],
                    dtype=torch.long,
                )
                edge_counts += mapped_edges.bincount(minlength=edge_counts.numel())

            coord = data.pos - data.pos.mean(dim=0)
            flat_coord = coord.double().reshape(-1)
            batch_count = flat_coord.numel()
            batch_mean = flat_coord.mean()
            batch_m2 = ((flat_coord - batch_mean) ** 2).sum()
            if coord_count == 0:
                coord_mean = batch_mean
                coord_m2 = batch_m2
                coord_count = batch_count
            else:
                delta = batch_mean - coord_mean
                total_count = coord_count + batch_count
                coord_m2 += (
                    batch_m2 + delta.square() * coord_count * batch_count / total_count
                )
                coord_mean += delta * batch_count / total_count
                coord_count = total_count

            accumulate_pairwise_distance_hist(
                pair_hist, coord, atom_type_indices_t, d_edges
            )
            accumulate_rog_hist(rog_hist, coord, r_edges)

            if hasattr(data, "charges") and data.charges is not None:
                saw_charges = True
                charge_type_indices = torch.tensor(
                    [charge_to_idx[str(token)] for token in data.charges.tolist()],
                    dtype=torch.long,
                )
                charge_counts += charge_type_indices.bincount(
                    minlength=charge_counts.numel()
                )

            n_atoms_counts[num_atoms] = n_atoms_counts.get(num_atoms, 0) + 1

        return {
            "n_atoms_counts": n_atoms_counts,
            "atom_counts": atom_counts,
            "edge_counts": edge_counts,
            "charge_counts": charge_counts,
            "saw_topology": saw_topology,
            "saw_charges": saw_charges,
            "coord_count": coord_count,
            "coord_mean": coord_mean,
            "coord_m2": coord_m2,
            "pair_hist": pair_hist,
            "rog_hist": rog_hist,
        }


class Preprocessing:
    """Load or compute a training vocabulary and reusable dataset statistics.

    A dataset config is instantiated only when an artifact is missing. Numerical
    reductions run in _DatasetReduction so workers never initialize cache files.
    """

    NO_BOND_TOKEN = "<NO_BOND>"

    def __init__(
        self,
        root: str,
        train_dataset: Dataset | dict | None = None,
        atom_tokens_path: str = None,
        edge_tokens_path: str = None,
        charge_tokens_path: str = None,
        distributions_path: str = None,
        num_workers: int = 10,
        chunk_size: int = 2048,
    ):
        """Use root-level artifact paths unless explicit paths are supplied."""
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.root = root
        self._train_dataset_cfg = train_dataset
        self.train_dataset = None

        self.atom_tokens_path = (
            os.path.join(root, "atom_tokens.txt")
            if atom_tokens_path is None
            else atom_tokens_path
        )
        self.edge_tokens_path = (
            os.path.join(root, "edge_tokens.txt")
            if edge_tokens_path is None
            else edge_tokens_path
        )
        self.charge_tokens_path = (
            os.path.join(root, "charge_tokens.txt")
            if charge_tokens_path is None
            else charge_tokens_path
        )
        self.distributions_path = (
            os.path.join(root, "distributions.pt")
            if distributions_path is None
            else distributions_path
        )

        has_all_files = (
            os.path.exists(self.atom_tokens_path)
            and os.path.exists(self.edge_tokens_path)
            and os.path.exists(self.charge_tokens_path)
            and os.path.exists(self.distributions_path)
        )

        self._artifacts_were_complete = has_all_files
        if not has_all_files:
            self._ensure_train_dataset()

        self.vocab = None
        self.vocab = self._load_or_compute_tokens()
        self.distributions = self._load_or_compute_distributions(
            force=not self._artifacts_were_complete
        )

        # Release raw records after producing the small artifacts.
        self.train_dataset = None

    def _ensure_train_dataset(self):
        """Instantiate train dataset lazily only when preprocessing needs it."""
        if self.train_dataset is not None:
            return

        if self._train_dataset_cfg is None:
            raise ValueError(
                "train_dataset is required when preprocessing artifacts are missing."
            )

        if isinstance(self._train_dataset_cfg, Dataset):
            self.train_dataset = self._train_dataset_cfg
            return

        try:
            hydra = importlib.import_module("hydra")
        except ImportError as e:
            raise ImportError(
                "Hydra is required to instantiate train_dataset config lazily."
            ) from e

        self.train_dataset = hydra.utils.instantiate(self._train_dataset_cfg)

    def _load_or_compute_tokens(self) -> Vocab:
        """Load complete token files, or discover and publish all three together."""
        atom_tokens_exist = os.path.exists(self.atom_tokens_path)
        edge_tokens_exist = os.path.exists(self.edge_tokens_path)
        charge_tokens_exist = os.path.exists(self.charge_tokens_path)

        if atom_tokens_exist and edge_tokens_exist and charge_tokens_exist:
            atom_tokens = self._load_tokens(self.atom_tokens_path)
            edge_tokens = self._load_tokens(self.edge_tokens_path)
            charge_tokens = self._load_tokens(self.charge_tokens_path)
        else:
            atom_tokens, edge_tokens, charge_tokens = self._compute_tokens_from_data()

            # Publish the token set together. If a previous preprocessing run
            # stopped between files, retaining any old subset can produce a
            # vocabulary/distribution mismatch.
            self._save_tokens(self.atom_tokens_path, atom_tokens)
            self._save_tokens(self.edge_tokens_path, edge_tokens)
            self._save_tokens(self.charge_tokens_path, charge_tokens)

        return Vocab(
            atom_tokens=atom_tokens,
            edge_tokens=edge_tokens,
            charge_tokens=charge_tokens,
        )

    def _compute_tokens_from_data(self) -> tuple[list[str], list[str], list[str]]:
        """Discover sorted atom, bond, and charge labels from training records."""
        all_atom_types, all_edge_type_indices, all_charge_tokens = set(), set(), set()
        for atoms, edges, charges in self._map_chunks("_collect_tokens"):
            all_atom_types.update(atoms)
            all_edge_type_indices.update(edges)
            all_charge_tokens.update(charges)

        tokens = sorted(all_atom_types)
        edge_tokens = [self.NO_BOND_TOKEN] + [
            str(int(idx)) for idx in sorted(all_edge_type_indices)
        ]
        charge_tokens = [str(int(idx)) for idx in sorted(all_charge_tokens)] or ["0"]
        return tokens, edge_tokens, charge_tokens

    def _map_chunks(self, method):
        """Read disjoint ranges in workers, returning only small reductions."""
        length = len(self.train_dataset)
        workers = getattr(self, "num_workers", 0)
        if workers <= 1 or length < 2:
            reader = _DatasetReduction(self.train_dataset, self.vocab)
            yield getattr(reader, method)(range(length))
            return
        workers = min(workers, length)
        chunk_size = min(
            getattr(self, "chunk_size", 2048),
            max(1, (length + workers - 1) // workers),
        )
        tasks = (
            (method, range(start, min(start + chunk_size, length)))
            for start in range(0, length, chunk_size)
        )
        with get_context("spawn").Pool(
            workers,
            initializer=_init_preprocessing_worker,
            initargs=(self.train_dataset, self.vocab),
        ) as pool:
            yield from tqdm(
                pool.imap(_preprocessing_chunk, tasks),
                total=(length + chunk_size - 1) // chunk_size,
                desc=method.removeprefix("_collect_"),
                unit="chunk",
            )

    def _save_tokens(self, path: str, tokens: list[str]):
        """Publish a token file atomically."""
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=parent, delete=False) as f:
                tmp_path = f.name
                for token in tokens:
                    f.write(f"{token}\n")
            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _load_tokens(self, path: str) -> list[str]:
        """Read one token per line."""
        with open(path, "r") as f:
            tokens = [line.strip() for line in f.readlines()]
        return tokens

    def _load_or_compute_distributions(self, force: bool = False) -> Distributions:
        """Load statistics unless absent or invalidated by a changed vocabulary."""
        if os.path.exists(self.distributions_path) and not force:
            distributions = self._load_distributions()
        else:
            distributions = self._compute_distributions()

            self._save_distributions(distributions)

        distributions = Distributions(**distributions)
        return distributions

    def _compute_distributions(self) -> dict[str, torch.Tensor | float]:
        """Reduce training statistics in deterministic chunk order."""
        merged = None
        for part in self._map_chunks("_collect_distribution_counts"):
            if merged is None:
                merged = part
                continue
            for key in (
                "atom_counts",
                "edge_counts",
                "charge_counts",
                "pair_hist",
                "rog_hist",
            ):
                merged[key] += part[key]
            for n_atoms, count in part["n_atoms_counts"].items():
                counts = merged["n_atoms_counts"]
                counts[n_atoms] = counts.get(n_atoms, 0) + count
            merged["saw_topology"] |= part["saw_topology"]
            merged["saw_charges"] |= part["saw_charges"]
            left, right = merged["coord_count"], part["coord_count"]
            if right:
                delta = part["coord_mean"] - merged["coord_mean"]
                total = left + right
                merged["coord_m2"] += (
                    part["coord_m2"] + delta.square() * left * right / total
                )
                merged["coord_mean"] += delta * right / total
                merged["coord_count"] = total

        n_atoms_counts = merged["n_atoms_counts"]
        atom_counts = merged["atom_counts"]
        edge_counts = merged["edge_counts"]
        charge_counts = merged["charge_counts"]
        saw_topology = merged["saw_topology"]
        saw_charges = merged["saw_charges"]
        coord_count = merged["coord_count"]
        coord_m2 = merged["coord_m2"]
        pair_hist = merged["pair_hist"].float()
        rog_hist = merged["rog_hist"].float()

        if not n_atoms_counts or coord_count < 2:
            raise ValueError("Cannot compute distributions from an empty dataset")

        n_atoms_distribution = torch.zeros(max(n_atoms_counts) + 1, dtype=torch.float32)
        for n_atoms, count in n_atoms_counts.items():
            n_atoms_distribution[n_atoms] = count
        n_atoms_distribution = n_atoms_distribution / n_atoms_distribution.sum()

        atom_type_distribution = atom_counts.float()
        atom_type_distribution = atom_type_distribution / atom_type_distribution.sum()

        # Canonical edge distribution. Sources without bonds emit no edge entries;
        # we fall back to a one-hot at NO_BOND (token 0) so the singleton-canonical
        # vocab maps to a proper one-hot prior.
        if saw_topology and edge_counts.sum() > 0:
            edge_type_distribution = edge_counts.float()
            edge_type_distribution = (
                edge_type_distribution / edge_type_distribution.sum().clamp(min=1.0)
            )
        else:
            edge_type_distribution = torch.zeros(len(self.vocab.edge_tokens))
            edge_type_distribution[0] = 1.0

        # Canonical charge distribution. Sources without charges emit no entries;
        # fall back to one-hot at the canonical "0" index when available.
        if saw_charges:
            charge_type_distribution = charge_counts.float()
            charge_type_distribution = (
                charge_type_distribution / charge_type_distribution.sum()
            )
        else:
            charge_type_distribution = torch.zeros(len(self.vocab.charge_tokens))
            charge_type_distribution[neutral_charge_index(self.vocab)] = 1.0

        coordinate_std = torch.sqrt(coord_m2 / (coord_count - 1)).float().item()

        return {
            "atom_type_distribution": atom_type_distribution,
            "edge_type_distribution": edge_type_distribution,
            "charge_type_distribution": charge_type_distribution,
            "n_atoms_distribution": n_atoms_distribution,
            "coordinate_std": coordinate_std,
            "pairwise_distance_histogram": pair_hist,
            "radius_of_gyration_histogram": rog_hist,
        }

    def _save_distributions(
        self, distributions: dict[str, torch.Tensor | float]
    ) -> None:
        """Save distributions to file."""
        parent = os.path.dirname(self.distributions_path) or "."
        os.makedirs(parent, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=parent, delete=False) as f:
            tmp_path = f.name
        try:
            torch.save(distributions, tmp_path)
            os.replace(tmp_path, self.distributions_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _load_distributions(self) -> dict[str, torch.Tensor | float]:
        """Load distributions from file."""
        return torch.load(self.distributions_path, weights_only=False)
