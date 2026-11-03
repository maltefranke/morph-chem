"""Dataset-free inference for the public Morph playground."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime"))
os.environ.setdefault("MORPH_COMPILE", "0")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MORPH_FUSED_KERNELS", "0")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

import hydra
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Draw, rdDepictor

import morph.config  # noqa: F401 -- registers the saved configuration resolvers
from morph.dataset.core.atomic_data import chembatch_from_graphs
from morph.dataset.core.flow_matching_wrapper import uniform_n_atoms_range
from morph.flow_matching.prior import sample_prior_graph
from morph.inference.module import InferenceModule, load_inference_weights
from morph.model.build import prepare_model_data
from morph.utils.rdkit_utils import mol_is_valid

REGISTRY = json.loads((ROOT / "models.json").read_text())
MODELS = {}
RDLogger.DisableLog("rdApp.*")
torch.set_num_threads(2)


def load_models(device="cuda", keys=None):
    """Preload weights before ZeroGPU forks its inference worker."""
    local = json.loads(os.environ.get("MORPH_LOCAL_CHECKPOINTS", "{}"))
    for key in keys or REGISTRY:
        spec = REGISTRY[key]
        assets = ROOT / "assets" / key
        cfg = OmegaConf.load(assets / "config.yaml")
        OmegaConf.set_struct(cfg, False)
        cfg.data.preprocessing.root = str(assets)
        cfg.data.preprocessing.train_dataset = None
        cfg.data.preprocessing.num_workers = 0
        for name in ("atom_tokens", "edge_tokens", "charge_tokens", "distributions"):
            suffix = ".pt" if name == "distributions" else ".txt"
            cfg.data.preprocessing[name + "_path"] = str(assets / (name + suffix))
        preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)
        data = prepare_model_data(cfg, preprocessing.vocab, preprocessing.distributions)
        cfg.model.module.integrator.device = device
        module = InferenceModule(cfg, data)
        module.integrator.num_integration_steps = 100
        if spec.get("max_atoms"):
            module.integrator.max_atoms = spec["max_atoms"]
        checkpoint = local.get(key) or hf_hub_download(
            "maltefranke/morph", spec["checkpoint"], revision=spec["revision"]
        )
        with open(checkpoint, "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != spec["sha256"]:
            raise RuntimeError(f"Checkpoint integrity check failed: {key}")
        load_inference_weights(module.model, checkpoint, use_ema=True)
        prior = spec.get("prior") or uniform_n_atoms_range(data.empirical)
        MODELS[key] = (module.eval().to(device), data, tuple(prior))
        print(f"Loaded {key}: EMA, initial count {prior}, {device}", flush=True)


def validate_request(key, target_logp, target_n, count, seed):
    """Validate API inputs independently of the browser controls."""
    if key not in REGISTRY:
        raise ValueError("Choose a listed model.")
    if int(count) != count or not 1 <= count <= 8:
        raise ValueError("Generate between 1 and 8 molecules at a time.")
    if int(seed) != seed or not 0 <= seed <= 2147483647:
        raise ValueError("Seed must be an integer from 0 to 2147483647.")
    kind = REGISTRY[key]["kind"]
    if kind == "logp" and (
        not math.isfinite(target_logp) or not -8 <= target_logp <= 9
    ):
        raise ValueError("Choose a logP target between -8 and 9.")
    if kind == "count" and (int(target_n) != target_n or not 3 <= target_n <= 120):
        raise ValueError("Choose an atom-count target between 3 and 120.")


def generate_graphs(
    key, target_logp, target_n, count, seed, *, trajectory=False, require_gpu=False
):
    """Run the same 100-step EMA sampler used for evaluation."""
    validate_request(key, target_logp, target_n, count, seed)
    torch.set_float32_matmul_precision("high")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    module, data, prior = MODELS[key]
    if require_gpu and module.device.type != "cuda":
        raise RuntimeError("This endpoint requires real CUDA inference.")
    kind = REGISTRY[key]["kind"]
    graphs = []
    for n in torch.randint(prior[0], prior[1] + 1, (int(count),)):
        graph = sample_prior_graph(data.prior, n_atoms=int(n))
        if kind == "logp":
            graph.add_system_property("target_logp", torch.tensor([float(target_logp)]))
        elif kind == "count":
            graph.add_system_property("target_n_atoms", torch.tensor([int(target_n)]))
        graphs.append(graph)
    batch = chembatch_from_graphs(graphs).to_inplace(module.device)
    with (
        torch.inference_mode(),
        torch.autocast(
            module.device.type,
            dtype=torch.bfloat16,
            enabled=module.device.type == "cuda",
        ),
    ):
        generated = module.sample((batch, batch), 0, return_traj=trajectory)
    if trajectory:
        outputs = [frames[-1] for frames in generated]
    else:
        generated.to_inplace("cpu", non_blocking=False)
        outputs = generated.to_data_list()
    assert len(outputs) == int(count)
    assert all(torch.isfinite(graph.positions).all() for graph in outputs)
    if not trajectory:
        return outputs
    times = [0.0, *np.cumsum(module.integrator.get_time_steps()).tolist()]
    assert all(len(frames) == len(times) == 101 for frames in generated)
    runtime = dict(
        device=str(module.device),
        gpu=torch.cuda.get_device_name() if module.device.type == "cuda" else None,
        torch=torch.__version__,
        precision="bfloat16" if module.device.type == "cuda" else "float32",
        model=key,
        checkpoint_sha256=REGISTRY[key]["sha256"],
        initial_count_prior=list(prior),
        integration_steps=100,
        frames_per_molecule=101,
        coordinate_units="angstrom",
        time_grid="log; final integration boundary 0.999",
    )
    return dict(graphs=outputs, trajectories=generated, times=times, runtime=runtime)


def frame_atoms(graph, vocab):
    """Export raw graph geometry, including chemically invalid intermediate states."""
    positions = graph.positions.tolist()
    atom_types = graph.atom_types.tolist()
    charges = graph.charge_types.tolist()
    atoms = [
        dict(
            elem=vocab.atom_tokens[a],
            x=round(p[0], 5),
            y=round(p[1], 5),
            z=round(p[2], 5),
            charge=int(vocab.charge_tokens[c]),
            bonds=[],
            bondOrder=[],
        )
        for a, c, p in zip(atom_types, charges, positions, strict=True)
    ]
    edge_types, edge_index = graph.get_edge_types_triu(
        no_bond_idx=list(vocab.edge_tokens).index("<NO_BOND>")
    )
    for edge, (a, b) in zip(edge_types.tolist(), edge_index.T.tolist(), strict=True):
        token = vocab.edge_tokens[edge]
        if token == "<NO_BOND>":
            continue
        order = 1.5 if str(token) == "4" else float(token)
        atoms[a]["bonds"].append(b)
        atoms[a]["bondOrder"].append(order)
        atoms[b]["bonds"].append(a)
        atoms[b]["bondOrder"].append(order)
    return atoms


def present(
    graphs,
    key,
    target_logp,
    target_n,
    seed,
    *,
    trajectories=None,
    times=None,
    runtime=None,
):
    """Keep every attempt in the table; export sanitized structures to SDF."""
    _, data, _ = MODELS[key]
    vocab = data.vocab
    directory = Path(tempfile.mkdtemp(prefix="morph-"))
    sdf_path, csv_path = directory / "molecules.sdf", directory / "results.csv"
    writer = Chem.SDWriter(str(sdf_path))
    gallery, rows, records = [], [], []
    valid_count = 0
    kind = REGISTRY[key]["kind"]
    for i, graph in enumerate(graphs, 1):
        mol = graph.to_rdkit_mol(
            vocab.atom_tokens, vocab.edge_tokens, vocab.charge_tokens
        )
        valid = mol is not None and mol_is_valid(mol, allow_charged=data.allow_charged)
        valid_count += int(valid)
        logp = float(Crippen.MolLogP(mol)) if mol is not None else None
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol)) if mol is not None else ""
        row = dict(
            sample=i,
            valid=bool(valid),
            atoms=int(graph.num_nodes),
            logP=round(logp, 4) if logp is not None else None,
            target_logP=float(target_logp) if kind == "logp" else None,
            target_N=int(target_n) if kind == "count" else None,
            seed=int(seed),
            model=key,
            smiles=smiles,
        )
        rows.append(row)
        if mol is not None:
            mol.SetProp("_Name", f"Morph sample {i}")
            for name, value in row.items():
                if value is not None:
                    mol.SetProp(name, str(value))
            writer.write(mol)
            block = Chem.MolToMolBlock(mol)
            display = Chem.RemoveHs(Chem.Mol(mol))
            rdDepictor.Compute2DCoords(display)
            thumbnail = Draw.MolToImage(display, size=(440, 320))
        else:
            thumbnail = Image.new("RGB", (440, 320), "#f4f6f8")
            ImageDraw.Draw(thumbnail).text(
                (125, 150), "Could not sanitize this sample", fill="#566477"
            )
        caption = f"{i} · N={graph.num_nodes}"
        if logp is not None:
            caption += f" · logP={logp:.2f}"
        caption += " · valid" if valid else " · invalid"
        gallery.append((thumbnail, caption))
        record = dict(index=i, molblock=block if mol is not None else None)
        if trajectories is not None:
            record["frames"] = [
                dict(time=float(t), atoms=frame_atoms(frame, vocab))
                for t, frame in zip(times, trajectories[i - 1], strict=True)
            ]
        records.append(record)
    writer.close()
    with csv_path.open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=rows[0])
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    status = f"**{valid_count} / {len(graphs)} chemically valid** · Seed {int(seed)}"
    if runtime is not None and runtime["device"].startswith("cuda"):
        status += " · GPU"
    status += (
        "\n\nAll attempts are shown. SDF includes sanitized structures; "
        "CSV includes every attempt."
    )
    if trajectories is None:
        return gallery, rows, records, str(sdf_path), str(csv_path), status
    archive = directory / "trajectories.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(
            "generation.json",
            json.dumps(
                dict(
                    **runtime,
                    seed=int(seed),
                    target_logp=float(target_logp) if kind == "logp" else None,
                    target_n=int(target_n) if kind == "count" else None,
                ),
                indent=2,
            ),
        )
        for record in records:
            zip_file.writestr(
                f"molecule_{record['index']}.json", json.dumps(record["frames"])
            )
            xyz = []
            for frame in record["frames"]:
                xyz.extend(
                    [
                        str(len(frame["atoms"])),
                        f"t={frame['time']:.9f}; coordinates in angstrom",
                    ]
                )
                xyz.extend(
                    f"{a['elem']} {a['x']:.5f} {a['y']:.5f} {a['z']:.5f}"
                    for a in frame["atoms"]
                )
            zip_file.writestr(f"molecule_{record['index']}.xyz", "\n".join(xyz) + "\n")
    return (
        gallery,
        rows,
        records,
        str(sdf_path),
        str(csv_path),
        status,
        str(archive),
        runtime,
    )
