import os
import pickle
from functools import partial

from rdkit import Chem
from torch_geometric.data import Data, download_url

from morph._vendor.geom_drugs_preprocessing import process_geom_drugs
from morph.dataset.core.conversion import mol_to_data
from morph.dataset.core.lmdb import MolecularLMDBDataset
from morph.dataset.core.tokenized_decode import TokenizedMolecularDatasetMixin
from morph.utils.rdkit_utils import (
    sanitize_mol_correctly,
)

# Source of the raw GEOM-Drugs pickle files used by the GEOM-Drugs Revisited
# paper (Nikitin et al., 2025, arXiv:2505.00169). Three files live here:
# train_data.pickle, val_data.pickle, test_data.pickle.
GEOM_RAW_URL = "https://bits.csb.pitt.edu/files/geom_raw"

RAW_FILENAMES = {
    "train": "train_data.pickle",
    "val": "val_data.pickle",
    "test": "test_data.pickle",
}


def process_one_conformer(mol: Chem.Mol, **options) -> Data | None:
    """Sanitize a source conformer, then apply the shared processing settings."""
    if mol is None:
        return None
    mol = sanitize_mol_correctly(Chem.Mol(mol))
    return mol_to_data(mol, **options) if mol is not None else None


class GEOM(MolecularLMDBDataset):
    """GEOM-Drugs source processing with shared molecular LMDB storage."""

    def __init__(self, root, source_dir: str | None = None, **options):
        self.source_dir = source_dir
        self.filtered_dir = (
            os.fspath(source_dir)
            if source_dir is not None
            else os.path.join(root, "raw_filtered")
        )
        super().__init__(root, **options)

    def prepare(self) -> None:
        """Download, prefilter and convert a missing GEOM split."""
        if self.source_dir is None:
            self.download()
            self.prefilter()
        self.process()

    def download(self) -> None:
        """Download missing raw split files."""
        os.makedirs(self.raw_dir, exist_ok=True)
        for fname in RAW_FILENAMES.values():
            target = os.path.join(self.raw_dir, fname)
            if os.path.exists(target):
                continue
            print(f"Downloading {fname} from {GEOM_RAW_URL} ...")
            download_url(f"{GEOM_RAW_URL}/{fname}", self.raw_dir)

    def prefilter(self) -> None:
        """Run the upstream GEOM prefilter once per source directory."""
        all_present = all(
            os.path.exists(os.path.join(self.filtered_dir, fname))
            for fname in RAW_FILENAMES.values()
        )
        if all_present:
            return

        print("Applying GEOM-Drugs Revisited pre-filter to all splits ...")
        process_geom_drugs(self.raw_dir, self.filtered_dir)

        # Keep empty download sentinels once the filtered copies exist.
        for fname in RAW_FILENAMES.values():
            raw_path = os.path.join(self.raw_dir, fname)
            filtered_path = os.path.join(self.filtered_dir, fname)
            if (
                os.path.exists(filtered_path)
                and os.path.exists(raw_path)
                and os.path.getsize(raw_path) > 0
            ):
                with open(raw_path, "wb"):
                    pass
                print(f"  Truncated raw {fname} (filtered copy in {self.filtered_dir})")

    def process(self):
        """Process pre-filtered pickle files and write them into LMDB."""
        raw_file = os.path.join(self.filtered_dir, f"{self.split}_data.pickle")

        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"Filtered data file not found: {raw_file}")

        print(f"Loading filtered data from {raw_file}...")
        with open(raw_file, "rb") as f:
            raw_data = pickle.load(f)

        conformers = (mol for _, group in raw_data for mol in group)
        self._write_records(
            conformers,
            partial(process_one_conformer, **self.processing_options),
            total=sum(len(group) for _, group in raw_data),
        )


class FlowMatchingGEOMDataset(TokenizedMolecularDatasetMixin, GEOM):
    """Decode GEOM records into tokenized flow-matching targets."""
