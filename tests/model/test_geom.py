"""GEOM hydrogen policies, configurable filters, and source/cache isolation."""

import pickle
from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from rdkit import Chem
from rdkit.Chem import AllChem

from morph.dataset.core.lmdb import SPLITS
from morph.dataset.unconditional.geom import GEOM, process_one_conformer
from morph.utils.rdkit_utils import mol_is_valid


def _conformer(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


@pytest.mark.parametrize("mode", ["explicit", "implicit"])
def test_conformer_filters_apply_to_source_atoms_and_are_configurable(mode):
    mol = _conformer("B(O)O")
    before = mol.ToBinary()
    count = mol.GetNumAtoms()
    assert process_one_conformer(mol, hydrogen_mode=mode, max_atoms=count) is not None
    assert process_one_conformer(mol, hydrogen_mode=mode, max_atoms=count - 1) is None
    assert process_one_conformer(mol, allowed_atom_symbols=("H", "O")) is None
    explicit = process_one_conformer(mol)
    heavy = process_one_conformer(mol, hydrogen_mode="implicit")
    keep = explicit.z != 1
    torch.testing.assert_close(heavy.pos, explicit.pos[keep], rtol=0, atol=0)
    torch.testing.assert_close(heavy.charges, explicit.charges[keep])
    assert mol.ToBinary() == before


@pytest.mark.parametrize("workers", [0, 2])
def test_implicit_geom_recipe_prepares_statistics_and_decodes_all_splits(
    tmp_path, monkeypatch, workers
):
    source = tmp_path / "filtered"
    source.mkdir()
    molecules = [_conformer(s) for s in ("CCO", "C[NH3+]", "c1cc[nH]c1")]
    for split in SPLITS:
        with (source / f"{split}_data.pickle").open("wb") as stream:
            pickle.dump([(str(i), [mol]) for i, mol in enumerate(molecules)], stream)
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("GEOM_SOURCE_DIR", str(source))
    config_dir = Path(__file__).parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(
            config_name="default",
            overrides=[
                "defaults=geom/implicit_h",
                f"data.preprocess_workers={workers}",
            ],
        )
    assert cfg.data.preprocessing.root == str(tmp_path / "data/geom_implicit_h")
    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)
    assert "H" not in preprocessing.vocab.atom_tokens
    assert preprocessing.vocab.charge_tokens == ["0", "1"]
    assert preprocessing.distributions.coordinate_std > 0
    for split, dataset_cfg in (
        ("train", cfg.data.datamodule.datasets.train),
        ("val", cfg.data.datamodule.datasets.val[0]),
        ("test", cfg.data.datamodule.datasets.test[0]),
    ):
        dataset = hydra.utils.instantiate(
            dataset_cfg,
            split=split,
            vocab=preprocessing.vocab,
            distributions=preprocessing.distributions,
        )
        assert len(dataset) == len(molecules)
        for index, mol in enumerate(molecules):
            graph = dataset[index]
            raw = dataset.get(index)
            assert not (raw.z == 1).any()
            expected = torch.tensor(
                mol.GetConformer().GetPositions(), dtype=torch.float
            )
            keep = torch.tensor([atom.GetAtomicNum() != 1 for atom in mol.GetAtoms()])
            torch.testing.assert_close(raw.pos, expected[keep], rtol=0, atol=0)
            torch.testing.assert_close(
                graph.x.mean(0), torch.zeros(3), atol=1e-6, rtol=0
            )
            assert mol_is_valid(graph.to_rdkit(preprocessing.vocab), allow_charged=True)
        dataset.close()
    with pytest.raises(ValueError, match="cache is implicit"):
        GEOM(cfg.data.preprocessing.root)


def test_failed_geom_conversion_does_not_publish_partial_cache(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    with (source / "train_data.pickle").open("wb") as stream:
        pickle.dump([("hydrogen", [_conformer("[H][H]")])], stream)
    with pytest.raises(ValueError, match="no heavy atoms"):
        GEOM(output, source_dir=source, hydrogen_mode="implicit", preprocess_workers=0)
    assert not list((output / "processed").iterdir())
