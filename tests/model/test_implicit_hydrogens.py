"""Hydrogen-free conformer processing and RDKit reconstruction contracts."""

import pickle
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from rdkit import Chem
from rdkit.Chem import AllChem

from morph.dataset.core.atomic_data import make_chem_atomic
from morph.dataset.core.conversion import mol_to_data
from morph.utils.metrics.chemistry import Validity, calc_valence_table_stabilities
from morph.utils.rdkit_utils import IDX_BOND_MAP, mol_is_valid, tensors_to_rdkit_mol
from morph.utils.vocab_utils import z_to_atom_types


@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",
        "C[NH3+]",
        "c1cc[nH]c1",
        "c1ccncc1",
        "C[N+](=O)[O-]",
        "CS(=O)(=O)C",
        "B(O)O",
        "N[C@@H](C)C(=O)O",
        "[NH3+]CC(=O)[O-]",
        "C[Si](C)(C)C",
        "C[Se]C",
        "CNS(=O)(=O)CCCCN=[N+]=[N-]",
    ],
)
def test_removal_preserves_heavy_geometry_bonds_charge_and_recovers_h(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    record = mol_to_data(Chem.Mol(mol))
    out = mol_to_data(Chem.Mol(mol), hydrogen_mode="implicit")
    keep = record.z != 1
    assert torch.equal(out.pos, record.pos[keep])
    assert torch.equal(out.charges, record.charges[keep])
    assert not (out.z == 1).any()
    assert int(out.edge_index.max()) < len(out.z)
    upper = out.edge_index[0] < out.edge_index[1]
    reconstructed = tensors_to_rdkit_mol(
        z_to_atom_types(out.z.tolist()),
        out.pos.numpy(),
        out.charges.tolist(),
        [IDX_BOND_MAP[int(label)] for label in out.edge_attr[upper]],
        out.edge_index[:, upper].T.tolist(),
        implicit_hydrogens=True,
    )
    assert mol_is_valid(reconstructed, allow_charged=True)
    attached_h = [
        sum(neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors())
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() != 1
    ]
    assert [atom.GetTotalNumHs() for atom in reconstructed.GetAtoms()] == attached_h


def test_vocabulary_selects_hydrogen_policy_even_when_sample_has_no_h():
    graph = make_chem_atomic(
        x=torch.zeros(1, 3),
        a=torch.zeros(1, dtype=torch.long),
        c=torch.zeros(1, dtype=torch.long),
        e=torch.empty(0, dtype=torch.long),
        edge_index=torch.empty(2, 0, dtype=torch.long),
    )
    implicit = graph.to_rdkit_mol(["C"], ["<NO_BOND>", "1"], ["0"])
    explicit = graph.to_rdkit_mol(["C", "H"], ["<NO_BOND>", "1"], ["0"])
    assert mol_is_valid(implicit)
    assert implicit.GetAtomWithIdx(0).GetTotalNumHs() == 4
    assert not mol_is_valid(explicit)
    assert all(calc_valence_table_stabilities(implicit))
    # Matches Lightning's Python-object gather/pickle boundary.
    assert mol_is_valid(pickle.loads(pickle.dumps(implicit)))


def test_validity_handles_implicit_h_without_mutation_and_rejects_bad_chemistry():
    ethanol = Chem.MolFromSmiles("CCO")
    before = ethanol.ToBinary()
    assert mol_is_valid(ethanol)
    assert ethanol.ToBinary() == before
    assert not mol_is_valid(Chem.MolFromSmiles("[CH3]"))
    assert not mol_is_valid(Chem.MolFromSmiles("C.C"))
    assert mol_is_valid(Chem.MolFromSmiles("[NH4+]"))
    assert not mol_is_valid(Chem.MolFromSmiles("[NH4+]"), allow_charged=False)
    assert not mol_is_valid(Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False))
    assert not mol_is_valid(None)
    assert not mol_is_valid(Chem.Mol())
    metric = Validity(allow_charged=True)
    metric.update(
        [ethanol, None, Chem.MolFromSmiles("C.C"), Chem.MolFromSmiles("[NH4+]")]
    )
    assert float(metric.compute()) == 0.5


@pytest.mark.parametrize(
    ("preset", "hydrogen_mode"),
    [("max100", "explicit"), ("max100_implicit_h", "implicit")],
)
def test_max100_recipes_keep_data_and_checkpoint_contracts(
    preset, hydrogen_mode, tmp_path, monkeypatch
):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[2] / "configs"), version_base="1.1"
    ):
        cfg = compose(config_name="default", overrides=[f"defaults=chembl3d/{preset}"])
    assert cfg.data.hydrogen_mode == hydrogen_mode
    assert cfg.data.max_atoms == 100
    assert cfg.data.preprocessing.root == str(tmp_path / "data" / f"chembl3d_{preset}")
    for split in (
        cfg.data.datamodule.datasets.train,
        *cfg.data.datamodule.datasets.val,
        *cfg.data.datamodule.datasets.test,
    ):
        assert split.hydrogen_mode == hydrogen_mode
        assert split.root == cfg.data.preprocessing.root
        assert split.max_atoms == 100
    assert cfg.trainer.trainer.devices == "auto"
    assert cfg.trainer.trainer.check_val_every_n_epoch == 1
    assert cfg.callbacks.model_checkpoints.every_n_epochs == 5
    assert cfg.callbacks.every_n_epochs_checkpoint.every_n_epochs == 5
    assert cfg.trainer.checkpoint_path is None
