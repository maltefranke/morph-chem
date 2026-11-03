"""Decode stored molecular conformers into Morph's token representation."""

from __future__ import annotations

import torch

from morph.dataset.core.atomic_data import make_chem_atomic
from morph.dataset.core.properties import load_property_array
from morph.dataset.core.representation import Capabilities
from morph.utils.rdkit_utils import IDX_BOND_MAP
from morph.utils.vocab_utils import z_to_atom_types


def _lookup_tokens(
    labels: torch.Tensor, table: torch.Tensor, kind: str
) -> torch.Tensor:
    """Reject unrecognized source labels instead of treating them as categories."""
    if ((labels < 0) | (labels >= table.numel())).any():
        raise ValueError(f"Source {kind} labels are absent from the vocabulary")
    indices = table[labels]
    if (indices < 0).any():
        raise ValueError(f"Source {kind} labels are absent from the vocabulary")
    return indices


class TokenizedDecodeMixin:
    """Stored PyG ``Data`` -> token-index ``ChemAtomicData`` decode, shared by
    molecular dataset adapters.
    """

    def _build_token_maps(self, vocab) -> None:
        """Build tensor lookup tables for atom, bond, and charge labels."""
        atom_to_idx = {token: i for i, token in enumerate(vocab.atom_tokens)}
        z_table = torch.full((119,), -1, dtype=torch.long)
        for z, symbol in enumerate(z_to_atom_types(list(range(1, 119))), start=1):
            if symbol in atom_to_idx:
                z_table[z] = atom_to_idx[symbol]
        self._z_to_atom_idx = z_table

        # Stored labels describe bond orders; vocabulary indices describe model
        # categories. They differ when a bond order is absent from training or
        # the vocabulary has been reordered.
        bond_labels = {
            token: label
            for label, bond in IDX_BOND_MAP.items()
            if label != 0
            for token in (str(label), str(bond))
        }
        bond_table = torch.full((max(IDX_BOND_MAP) + 1,), -1, dtype=torch.long)
        for index, token in enumerate(vocab.edge_tokens):
            label = 0 if token == "<NO_BOND>" else bond_labels[token]
            bond_table[label] = index
        self._bond_to_idx = bond_table

        charges = torch.tensor([int(token) for token in vocab.charge_tokens])
        self._charge_min = int(charges.min())
        self._charge_to_idx = torch.full(
            (int(charges.max()) - self._charge_min + 1,), -1, dtype=torch.long
        )
        self._charge_to_idx[charges - self._charge_min] = torch.arange(len(charges))

    def _load_property_arrays(
        self, compute_properties: bool, property_names, split: str
    ) -> None:
        """Load precomputed per-graph scalar properties (``qed`` / ``logp`` / ...).

        Read by index from ``<processed_dir>/properties/<name>/<split>.npy`` --
        never per-sample RDKit (which leaks host RAM across forked workers). Off
        by default; supply arrays in the processed split's dataset order.
        """
        self.compute_properties = compute_properties
        self.property_names = tuple(property_names)
        self._property_arrays: dict = {}
        if self.compute_properties:
            self._property_arrays = {
                name: load_property_array(self.processed_dir, name, split)
                for name in self.property_names
            }

    def _build_conditions(self, data, index: int) -> dict:
        """Loose conditioning to attach to the graph. Default: the precomputed
        scalar properties when ``compute_properties`` is on, else nothing.
        Subclasses override to carry extra dataset-specific fields.
        """
        if self.compute_properties:
            return self._molecular_properties(index, getattr(data, "smiles", "") or "")
        return {}

    def _molecular_properties(self, index: int, smiles: str) -> dict:
        """Per-graph scalar properties for molecule ``index`` from the precomputed
        arrays, as ``[1]`` float tensors (so they batch as system properties),
        plus the canonical ``smiles``. Missing SMILES fall back to ``""``.
        """
        out: dict = {"smiles": smiles or ""}
        for name, arr in self._property_arrays.items():
            out[name] = torch.tensor([float(arr[index])], dtype=torch.float)
        return out

    def _decode_to_lean(self, data, index: int):
        """Decode one stored, centered conformer into token indices."""
        coord = data.pos - data.pos.mean(dim=0)
        if self.distributions.coordinate_std is not None:
            coord = coord / self.distributions.coordinate_std

        return make_chem_atomic(
            x=coord,
            a=_lookup_tokens(data.z.long(), self._z_to_atom_idx, "atom"),
            e=_lookup_tokens(data.edge_attr.long(), self._bond_to_idx, "bond"),
            c=_lookup_tokens(
                data.charges.long() - self._charge_min, self._charge_to_idx, "charge"
            ),
            edge_index=data.edge_index,
            **self._build_conditions(data, index),
        )


class TokenizedMolecularDatasetMixin(TokenizedDecodeMixin):
    """Adapt a raw molecular reader to flow-matching targets.

    Place this mixin before the reader in the inheritance list. ``get`` always
    returns raw fields for preprocessing; indexing requires vocabulary and
    coordinate statistics. Source-specific options pass through to the reader.
    """

    CAPABILITIES = Capabilities(provides_charges=True, provides_topology=True)

    def __init__(
        self,
        root,
        vocab=None,
        distributions=None,
        split="train",
        compute_properties: bool = False,
        property_names: tuple[str, ...] = ("qed", "logp"),
        **source_options,
    ):
        super().__init__(
            root,
            split=split,
            vocab=vocab,
            distributions=distributions,
            **source_options,
        )
        if vocab is not None:
            self._build_token_maps(vocab)
            self._load_property_arrays(compute_properties, property_names, split)

    def __getitem__(self, index):
        """Decode token labels and normalize the heavy or explicit-H geometry."""
        if self.vocab is None or self.distributions is None:
            raise RuntimeError(
                "A vocab and distributions are required; use .get() for raw data"
            )
        return self._decode_to_lean(super().__getitem__(index), index)

    def get(self, index):
        """Read raw fields without dispatching through tokenized indexing."""
        return super().__getitem__(index)
