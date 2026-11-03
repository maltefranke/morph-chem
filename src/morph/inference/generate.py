# Inference helpers load the scientific stack only when invoked.
# ruff: noqa: PLC0415

"""In-process de-novo generation for the ``morph`` CLI.

Shared sampling for pretrained bundles and modules built from Hydra overrides.
Priors are generated lazily; predictions can be consumed one batch at a time.
The collecting API preserves every attempted trajectory with its validity flag.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import morph.config  # noqa: F401  (import registers OmegaConf resolvers)

# A wheel carries configs under ``morph/configs``. The source checkout keeps the
# canonical tree at repository level, so editable installs use that as fallback.
_PACKAGED_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_REPO_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
_CONFIG_DIR = (
    _PACKAGED_CONFIG_DIR if _PACKAGED_CONFIG_DIR.is_dir() else _REPO_CONFIG_DIR
)
DEFAULT_BATCH_SIZE = 32


def _load_repo_env() -> None:
    """Load simple repo-local environment assignments for Hydra construction.

    Configs interpolate ``${oc.env:PROJECT_ROOT}`` (e.g. ``data.root_path``) which
    is resolved even for dataset-free inference. We parse ``.env`` (``export
    KEY=VALUE``), expanding environment-variable references in file order so
    later values can reference earlier ones. Existing environment values take
    precedence. Shell commands, escapes and inline comments are not interpreted.
    Default ``PROJECT_ROOT`` to the repo root when it is otherwise unset.
    """
    repo = Path(__file__).resolve().parents[3]
    env_file = repo / ".env"
    if env_file.exists():
        for raw in env_file.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = line.removeprefix("export ")
            key, sep, val = line.partition("=")
            key = key.strip()
            if not sep or not key or key in os.environ:
                continue
            os.environ[key] = os.path.expandvars(val.strip().strip('"').strip("'"))
    os.environ.setdefault("PROJECT_ROOT", str(repo))


def _build_module(cfg):
    """Build from bundle statistics without constructing a dataset."""
    import hydra
    from omegaconf import OmegaConf

    from morph.model.build import build_model_module, prepare_model_data

    OmegaConf.set_struct(cfg, False)
    # A fine-tuned checkpoint already contains its pretrained base weights.
    if "pretrained_ckpt" in cfg.model.module:
        cfg.model.module.pretrained_ckpt = None
    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)
    data = prepare_model_data(cfg, preprocessing.vocab, preprocessing.distributions)
    module = build_model_module(cfg, data)
    return module, data.vocab, preprocessing.distributions


def _uniform_n_atoms_range(distributions, padding: int = 5) -> tuple[int, int]:
    """(min, max) atom-count bounds for the de-novo ``uniform`` n_atoms prior.

    Delegates to the training wrapper's implementation so the two paths cannot
    silently acquire different source-count priors.
    """
    from morph.dataset.core.flow_matching_wrapper import (
        uniform_n_atoms_range,
    )

    return uniform_n_atoms_range(distributions, padding)


def _prior_loader(
    distributions,
    n_mols: int,
    batch_size: int | None = None,
    uniform_n_atoms_padding: int = 5,
    n_atoms_strategy: str = "uniform",
):
    """Lazily draw prior graphs, keeping only the current batches in memory.

    ``batch_size`` defaults to at most 32. A private CPU random stream prevents
    sampling or consumer code from changing subsequent priors. Repeated loader
    iterations restart that stream; no graph or per-attempt count list is stored.

    ``distributions`` is the exact representation-aware type prior installed on
    the sampling module. It matches the t=0 distribution used by
    ``FlowMatchingDatasetWrapper``, which is uniform in two independent ways:

    * represented atom / charge / edge **types** have equal probability per
      token; channels absent from the representation are pinned to their inert
      token;
    * atom **counts** follow the configured dataset-free strategy: uniform over
      the padded support, or the empirical n-atoms marginal.

    Coords come from the Gaussian prior; paired with itself as ``(mol_t, mol_1)``
    since de-novo sampling reads only ``mol_t``.
    """
    import torch

    from morph.dataset.core.flow_matching_wrapper import eval_collate_fn
    from morph.flow_matching.prior import sample_prior_graph

    if n_atoms_strategy == "uniform":
        n_min, n_max = _uniform_n_atoms_range(distributions, uniform_n_atoms_padding)
    elif n_atoms_strategy != "empirical":
        raise ValueError(
            "dataset-free generation supports n_atoms_strategy 'uniform' or "
            f"'empirical', got {n_atoms_strategy!r}"
        )
    prior_seed = int(torch.empty((), dtype=torch.int64).random_().item())

    class PriorDataset(torch.utils.data.IterableDataset):
        def __len__(self):
            return n_mols

        def __iter__(self):
            state = torch.Generator().manual_seed(prior_seed).get_state()
            for _ in range(n_mols):
                # Categorical.sample has no generator argument. Isolate its
                # CPU draws without saving/restoring any accelerator RNG state.
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(state)
                    n = (
                        int(torch.randint(n_min, n_max + 1, ()).item())
                        if n_atoms_strategy == "uniform"
                        else None
                    )
                    prior = sample_prior_graph(distributions, n_atoms=n)
                    state = torch.get_rng_state()
                yield prior, prior

    return torch.utils.data.DataLoader(
        PriorDataset(),
        batch_size=batch_size or min(n_mols, DEFAULT_BATCH_SIZE),
        num_workers=0,
        collate_fn=eval_collate_fn,
    )


def load_checkpoint(module, checkpoint_path) -> None:
    """Strictly load an eager or ``torch.compile`` checkpoint."""
    import torch

    from morph.utils.checkpoints import state_dict_for_module

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    module.load_state_dict(state_dict_for_module(sd, module), strict=True)


def build_sampling_module(overrides, checkpoint=None):
    """Compose ``overrides`` and build a sampling-ready module — no dataset access.

    The reusable entry point for every in-process inference mode (de-novo or
    scaffold): returns ``(module, vocab, distributions, cfg)`` and, when
    ``checkpoint`` is given, loads it. Works for any ``cfg.model.module`` target.
    Compiles both current and EMA models unless ``MORPH_COMPILE=0``.
    """
    from hydra import compose, initialize_config_dir

    from morph.utils.run_config import configure_runtime

    _load_repo_env()
    configure_runtime()

    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.1"):
        cfg = compose(config_name="default", overrides=list(overrides))
    module, vocab, distributions = _build_module(cfg)
    if checkpoint is not None:
        load_checkpoint(module, Path(checkpoint))
    module.compile()
    return module, vocab, distributions, cfg


def make_predict_trainer(cfg, *, callbacks=None):
    """Build a single-device predict Trainer using the configured accelerator."""
    import pytorch_lightning as pl

    tk = dict(cfg.trainer.trainer)
    # ``num_val_samples`` is a Morph setup setting, not a Trainer argument.
    tk.pop("num_val_samples", None)
    if tk.get("limit_val_batches") is None:
        tk.pop("limit_val_batches", None)
    tk["strategy"] = "auto"
    tk["devices"] = 1
    tk["limit_predict_batches"] = 1.0
    return pl.Trainer(
        logger=False, callbacks=callbacks or [], enable_checkpointing=False, **tk
    )


def generate_denovo(
    checkpoint,
    overrides,
    n_mols: int = 16,
    batch_size: int | None = None,
    seed: int = 42,
    *,
    return_traj: bool = False,
):
    """Sample ``n_mols`` de-novo molecules from an unconditional checkpoint.

    No dataset is loaded: priors are drawn directly from the training
    distributions. Returns ``(trajectories, is_valid, vocab, coordinate_std)``.
    By default each trajectory contains only the final frame; pass
    ``return_traj=True`` to retain all integration frames. Current and EMA models
    are compiled by default; set ``MORPH_COMPILE=0`` to disable compilation.
    Trajectories stay in attempt order (frames in Angstrom;
    ``trajectories[i][-1]`` is the generated molecule), including failures, so
    requesting ``n_mols`` never silently changes the denominator.
    """
    import pytorch_lightning as pl

    pl.seed_everything(seed)
    module, vocab, _distributions, cfg = build_sampling_module(overrides, checkpoint)
    return sample_module(
        module,
        vocab,
        cfg,
        n_mols=n_mols,
        batch_size=batch_size,
        seed=None,
        return_traj=return_traj,
    )


def sample_module(
    module,
    vocab,
    cfg,
    n_mols: int = 16,
    batch_size: int | None = None,
    seed: int | None = 42,
    *,
    return_traj: bool = False,
):
    """Collect every attempt; use ``sample_module_batches`` for bounded storage."""
    trajectories, is_valid = [], []

    def collect(batch_mols, batch_valid):
        trajectories.extend(batch_mols)
        is_valid.extend(batch_valid)

    sample_module_batches(
        module,
        cfg,
        collect,
        n_mols=n_mols,
        batch_size=batch_size,
        seed=seed,
        return_traj=return_traj,
    )
    return trajectories, is_valid, vocab, float(module.distributions.coordinate_std)


def sample_module_batches(
    module,
    cfg,
    on_batch: Callable[[list, list[bool]], None],
    n_mols: int = 16,
    batch_size: int | None = None,
    seed: int | None = 42,
    *,
    return_traj: bool = False,
) -> int:
    """Deliver CPU trajectories and validity flags in order; return attempt count.

    ``on_batch(trajectories, is_valid)`` runs synchronously once per batch and
    includes invalid attempts. Lightning retains no predictions. Consumers
    control whether results are written, summarized or retained. Memory is
    bounded by batch size (and integration steps with ``return_traj=True``).
    """
    import pytorch_lightning as pl
    from lightning_fabric.utilities.apply_func import move_data_to_device

    if n_mols < 1 or (batch_size is not None and batch_size < 1):
        raise ValueError("n_mols and batch_size must be positive")
    if not callable(on_batch):
        raise TypeError("on_batch must be callable")
    if seed is not None:
        pl.seed_everything(seed)
    module.predict_return_traj = return_traj

    # Use the module's prior directly: the initial graph and every insertion
    # renoising step therefore share the exact same categorical probabilities.
    loader = _prior_loader(
        module.distributions,
        n_mols,
        batch_size,
        uniform_n_atoms_padding=int(cfg.data.get("uniform_n_atoms_padding", 5)),
        n_atoms_strategy=str(cfg.data.n_atoms_strategy),
    )

    class BatchConsumer(pl.Callback):
        attempted = 0

        def on_predict_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
        ):
            if not isinstance(outputs, dict):
                raise RuntimeError("Prediction output must be a dictionary")
            batch_mols = outputs.get("all_mols")
            batch_valid = outputs.get("is_valid")
            if (
                batch_mols is None
                or batch_valid is None
                or len(batch_mols) != len(batch_valid)
            ):
                raise RuntimeError(
                    "Prediction output has misaligned molecules and validity flags"
                )
            if self.attempted + len(batch_mols) > n_mols:
                raise RuntimeError(f"Prediction output exceeds {n_mols} attempts")
            on_batch(
                move_data_to_device(batch_mols, "cpu"), [bool(v) for v in batch_valid]
            )
            self.attempted += len(batch_mols)

    consumer = BatchConsumer()
    trainer = make_predict_trainer(cfg, callbacks=[consumer])
    try:
        trainer.predict(module, dataloaders=loader, return_predictions=False)
    finally:
        # Lightning leaves the Trainer attached to the module. Do not let its
        # callback list keep a consumer (and any collected output) alive.
        trainer.callbacks.remove(consumer)
    if consumer.attempted != n_mols:
        raise RuntimeError(
            f"Requested {n_mols} molecules, received {consumer.attempted}"
        )
    return consumer.attempted


def final_mol(traj, vocab):
    """RDKit mol of a trajectory's final frame (keeps the 3D conformer), or None."""
    frame = traj[-1] if isinstance(traj, list) else traj
    try:
        return frame.to_rdkit_mol(
            vocab.atom_tokens, vocab.edge_tokens, vocab.charge_tokens
        )
    except Exception:
        return None


def record_mol(traj, vocab) -> dict:
    """Summarize a trajectory's final molecule (SMILES / mol block / N / logP)."""
    from rdkit import Chem
    from rdkit.Chem import Crippen

    mol = final_mol(traj, vocab)
    if mol is None:
        return {
            "valid": False,
            "smiles": None,
            "molblock": None,
            "n_atoms": None,
            "logp": None,
        }
    try:
        lp = float(Crippen.MolLogP(mol))
    except Exception:
        lp = None
    return {
        "valid": True,
        "smiles": Chem.MolToSmiles(mol),
        "molblock": Chem.MolToMolBlock(mol),
        "n_atoms": Chem.AddHs(mol).GetNumAtoms(),
        "logp": lp,
    }


def write_sdf(leads, vocab, path) -> tuple[int, int]:
    """Write each trajectory's final-frame molecule (3D coords + SMILES name) to an
    SDF. Invalid molecules are skipped. Returns ``(n_written, n_total)``.
    """
    from rdkit import Chem

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    n_written = 0
    try:
        for traj in leads:
            mol = final_mol(traj, vocab)
            if mol is None:
                continue
            mol.SetProp("_Name", Chem.MolToSmiles(mol))
            writer.write(mol)
            n_written += 1
    finally:
        writer.close()
    return n_written, len(leads)
