import os

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.datasets.qm9 import conversion
from torch_geometric.utils import one_hot, scatter
from tqdm import tqdm

from morph.dataset.core.representation import Capabilities
from morph.dataset.core.source import molecular_source_identity
from morph.dataset.core.tokenized_decode import TokenizedDecodeMixin
from morph.dataset.core.vocab import Distributions, Vocab
from morph.utils.rdkit_utils import (
    BOND_IDX_MAP,
    mol_is_valid,
    sanitize_mol_correctly,
    smiles_from_mol,
)

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]


class RevisedQM9(InMemoryDataset):
    """QM9 with corrected structures from PropMolFlow."""

    mols_url = "https://zenodo.org/records/15700961/files/all_fixed_gdb9.zip"
    props_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
    split_to_index = {"train": 0, "val": 1, "test": 2}

    def source_identity(self) -> dict:
        """Identify the processed split and any selected record indices."""
        return molecular_source_identity(
            self,
            storage=self.processed_paths[self.split_to_index[self.split]],
            split=self.split,
            indices=self._indices,
        )

    @property
    def raw_file_names(self) -> list[str]:
        """Source structure and property files required for preprocessing."""
        return ["all_fixed_gdb9.sdf", "gdb9.sdf.csv"]

    @property
    def processed_file_names(self) -> list[str]:
        """Serialized records for the train, validation, and test splits."""
        return ["train.pt", "val.pt", "test.pt"]

    def _download(self) -> None:
        """Reuse complete processed caches without fetching raw QM9 inputs."""
        if not self.force_reload and all(
            os.path.isfile(path) for path in self.processed_paths
        ):
            return
        super()._download()

    def load(self, split: str) -> None:
        """Load one named split from the processed cache."""
        if split not in self.split_to_index:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Expected one of {tuple(self.split_to_index)}."
            )
        self.split = self._split = split
        super().load(self.processed_paths[self.split_to_index[split]])

    def download(self) -> None:
        """Download corrected structures and QM9 regression properties."""
        file_path = download_url(self.mols_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

        file_path = download_url(self.props_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)

        os.unlink(file_path)
        # also delete the extra data file in the zip
        os.unlink(os.path.join(self.raw_dir, "gdb9.sdf"))

    def process(self) -> None:
        """Validate molecules, encode raw graphs, and write dataset splits."""
        types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
        bonds = BOND_IDX_MAP

        with open(self.raw_paths[1]) as f:
            target = [
                [float(x) for x in line.split(",")[1:20]]
                for line in f.read().split("\n")[1:-1]
            ]
            y = torch.tensor(target, dtype=torch.float)
            y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
            y = y * conversion.view(1, -1)

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        errors = 0
        skipped = 0

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            # if i in skip:
            #    continue
            if mol is None or not mol_is_valid(mol):
                errors += 1
                continue

            mol = sanitize_mol_correctly(mol)
            if mol is None:
                errors += 1
                continue

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            charges = []

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
                charges.append(atom.GetFormalCharge())

            z = torch.tensor(atomic_number, dtype=torch.long)

            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                edge_types += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            # edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            # edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce="sum").tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = (
                torch.tensor(
                    [atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float
                )
                .t()
                .contiguous()
            )
            x = torch.cat([x1, x2], dim=-1)

            charges = torch.tensor(charges, dtype=torch.int64)

            name = mol.GetProp("_Name")
            smiles = smiles_from_mol(mol, canonical=True) or ""

            # TODO exchange with our own MolData object for consistency
            data = Data(
                x=x,
                z=z,
                pos=pos,
                charges=charges,
                edge_index=edge_index,
                smiles=smiles,
                edge_attr=edge_type,
                y=y[i].unsqueeze(0),
                name=name,
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        n_train = 100_000
        n_val = 20_000

        self.save(data_list[:n_train], self.processed_paths[0])
        self.save(data_list[n_train : n_train + n_val], self.processed_paths[1])
        self.save(data_list[n_train + n_val :], self.processed_paths[2])

        for split_name, split_data in [
            ("train", data_list[:n_train]),
            ("val", data_list[n_train : n_train + n_val]),
            ("test", data_list[n_train + n_val :]),
        ]:
            smiles_path = os.path.join(self.processed_dir, f"{split_name}_smiles.txt")
            unique_smiles = sorted({d.smiles for d in split_data})
            with open(smiles_path, "w") as f:
                f.write("\n".join(unique_smiles))

        print(f"Errors: {errors}")
        print(f"Skipped: {skipped}")

    def get_all_smiles(self) -> list[str]:
        """Read SMILES in the loaded split's record order."""
        smiles_path = os.path.join(self.processed_dir, f"{self._split}_smiles.txt")
        with open(smiles_path) as f:
            return f.read().splitlines()


class FlowMatchingQM9Dataset(TokenizedDecodeMixin, RevisedQM9):
    """Decode corrected QM9 records into tokenized training targets."""

    CAPABILITIES = Capabilities(provides_charges=True, provides_topology=True)

    def __init__(
        self,
        root,
        vocab: Vocab,
        distributions: Distributions,
        transform=None,
        pre_transform=None,
        split="train",
        compute_properties: bool = False,
        property_names: tuple[str, ...] = ("qed", "logp"),
    ):
        super().__init__(root, transform, pre_transform)
        self.load(split)

        self.vocab = vocab
        self.distributions = distributions
        # Token/z-index maps + precomputed property arrays (shared decode adapter).
        if vocab is not None:
            self._build_token_maps(vocab)
            self._load_property_arrays(compute_properties, property_names, self._split)

    def __getitem__(self, index):
        """Return a tokenized molecule with its QM9 regression targets."""
        if self.vocab is None:
            raise RuntimeError(
                "A vocab is required for tokenized QM9 samples; use .get() for raw data"
            )
        data = super().__getitem__(index)
        mol = self._decode_to_lean(data, index)

        # Carry through the QM9 regression targets if present (2D so per-graph
        # batching concatenates along dim 0).
        if hasattr(data, "y") and data.y is not None:
            mol.add_system_property(
                "y", data.y if data.y.dim() == 2 else data.y.unsqueeze(0)
            )

        return mol

    def _build_conditions(self, data, index: int) -> dict:
        conditions = super()._build_conditions(data, index)
        # When not computing properties, still keep the cheap ground-truth SMILES
        # string for downstream eval.
        if not self.compute_properties and hasattr(data, "smiles"):
            conditions["smiles"] = data.smiles
        return conditions


if __name__ == "__main__":
    _root = os.path.join(os.path.dirname(__file__), "../../../../data/qm9")
    qm9 = FlowMatchingQM9Dataset(root=_root)
    print(qm9[0])
