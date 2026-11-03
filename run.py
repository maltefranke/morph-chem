import os
from typing import TYPE_CHECKING

import hydra
import omegaconf
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from rdkit import Chem

import morph.config  # noqa: F401  (import registers OmegaConf resolvers)
from morph.dataset.core.representation import (
    Representation,
    validate_representation,
)
from morph.model.build import (
    build_model_metrics,
    build_model_module,
    prepare_model_data,
)
from morph.utils.run_config import (
    ManualAccumulationWandbLogger,
    bootstrap_run_id,
    build_callbacks,
    build_trainer,
    configure_manual_gradient_accumulation,
    configure_runtime,
    gather_rank_outputs,
)

if TYPE_CHECKING:
    import pytorch_lightning as pl


def setup(cfg: DictConfig, extra_callbacks=None):
    """Prepare data, construct the model, and attach training infrastructure."""
    configure_runtime(seed=int(cfg.get("seed", 42)), diagnostics=True)
    OmegaConf.set_struct(cfg, False)

    # Validate the run's representation against the chosen dataset's class-level
    # CAPABILITIES before doing any data work. Imports the class only — no
    # instantiation, no I/O.
    representation = Representation(cfg.representation)
    dataset_cls = hydra.utils.get_class(cfg.data.datamodule.datasets.train._target_)
    validate_representation(dataset_cls.CAPABILITIES, representation)

    # Instantiate preprocessing to compute distributions from training dataset
    hydra.utils.log.info("Instantiating preprocessing...")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        artifact_keys = (
            "atom_tokens_path",
            "edge_tokens_path",
            "charge_tokens_path",
            "distributions_path",
        )
        missing = [
            str(cfg.data.preprocessing[key])
            for key in artifact_keys
            if not os.path.exists(str(cfg.data.preprocessing[key]))
        ]
        if missing:
            raise RuntimeError(
                "Preprocessing artifacts are missing in a distributed run. "
                "Run `python scripts/preprocess.py "
                f"{cfg.data.dataset_name}` once before torchrun. Missing: {missing}"
            )
    preprocessing = hydra.utils.instantiate(cfg.data.preprocessing)

    model_data = prepare_model_data(
        cfg, preprocessing.vocab, preprocessing.distributions
    )
    vocab = model_data.vocab

    hydra.utils.log.info(
        f"Preprocessing complete.\n"
        f"Found {len(vocab.atom_tokens)} atom tokens: {vocab.atom_tokens}\n"
        f"Found {len(vocab.edge_tokens)} edge tokens: {vocab.edge_tokens}\n"
        f"Found {len(vocab.charge_tokens)} charge tokens: {vocab.charge_tokens}"
    )

    hydra.utils.log.info("Distributions computed from training dataset.")

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule,
        _recursive_=False,
        vocab=vocab,
        distributions=model_data.prior,
        noise_distributions=model_data.empirical,
    )
    # Build the training/validation datasets needed for metrics and fitting.
    # Lightning calls setup("fit") again, so the datamodule keeps this idempotent.
    datamodule.setup("fit")
    datamodule.configure_model(cfg)

    # Novelty uses the canonical SMILES sidecars written during preprocessing.
    base_dataset = datamodule.train_dataset.base_dataset
    train_smiles = (
        base_dataset.get_all_smiles()
        if hasattr(base_dataset, "get_all_smiles")
        else None
    )
    metrics = build_model_metrics(
        model_data, train_smiles=train_smiles, train_smiles_are_canonical=True
    )
    hydra.utils.log.info(f"Instantiating <{cfg.model.module._target_}>")
    module = build_model_module(cfg, model_data, metrics=metrics)

    module.compile()

    # Setup logging and callbacks. A disabled logging config avoids constructing
    # a WandB logger at all (useful for smoke tests and fully local runs).
    logging_kwargs = OmegaConf.to_container(cfg.logging, resolve=True)
    logging_enabled = bool(logging_kwargs.pop("enabled", True))
    requested_accumulation = int(cfg.trainer.trainer.get("accumulate_grad_batches", 1))
    logger_accumulation = (
        requested_accumulation if not module.automatic_optimization else 1
    )
    logger = (
        ManualAccumulationWandbLogger(
            **logging_kwargs, accumulation_steps=logger_accumulation
        )
        if cfg.trainer.do_train and logging_enabled
        else False
    )
    callbacks = build_callbacks(cfg)
    # LearningRateMonitor refuses to run when there is no logger; only
    # attach it when we actually have one (training runs with W&B).
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    trainer_kwargs = OmegaConf.to_container(cfg.trainer.trainer, resolve=True)
    configure_manual_gradient_accumulation(module, trainer_kwargs)
    if not module.automatic_optimization:
        hydra.utils.log.info(
            "Manual optimizer gradient accumulation: "
            f"{requested_accumulation} microbatches per update"
        )

    trainer = build_trainer(
        validation_batch_size=int(cfg.data.datamodule.batch_size.val),
        logger=logger,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    return module, datamodule, trainer


def train(module, datamodule, trainer, ckpt_path=None):
    trainer.fit(
        module,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )


def validate(module, datamodule, trainer, ckpt_path=None):
    trainer.validate(
        module,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )


def predict(module, datamodule, trainer, ckpt_path=None):

    predictions = trainer.predict(
        module,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )

    def _batch_dicts(items):
        for item in items or []:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, (list, tuple)):
                yield from _batch_dicts(item)

    predictions = list(_batch_dicts(predictions))
    gathered_predictions = gather_rank_outputs(predictions)
    if gathered_predictions is None:
        return predictions
    predictions = gathered_predictions

    all_valid_mols = []
    all_invalid_mols = []
    all_invalid_mols_rdkit = []

    for batch_output in predictions:
        all_valid_mols.extend(batch_output["valid_mols"])
        all_invalid_mols.extend(batch_output["invalid_mols"])
        all_invalid_mols_rdkit.extend(batch_output["invalid_mols_rdkit"])

    if not trainer.is_global_zero:
        return predictions

    torch.save(all_valid_mols, "valid_mols.pt")
    torch.save(all_invalid_mols, "invalid_mols.pt")

    writer = Chem.SDWriter("invalid_molecules.sdf")

    for mol in all_invalid_mols_rdkit:
        # Optional: Catch sanitization errors during writing just in case
        if mol is None:
            continue

        try:
            writer.write(mol)
        except Exception as e:
            print(f"Could not write molecule to SDF: {e}")

    writer.close()
    return predictions


@hydra.main(
    config_path="configs",
    config_name="default",
    version_base="1.1",
)
def main(cfg: omegaconf.DictConfig):
    ckpt_path = cfg.trainer.checkpoint_path

    module, datamodule, trainer = setup(cfg)

    if cfg.trainer.do_train:
        train(module, datamodule, trainer, ckpt_path)
        final_checkpoint = cfg.trainer.get("final_checkpoint_path")
        if final_checkpoint and not trainer.fast_dev_run:
            trainer.save_checkpoint(str(final_checkpoint))
        # The fitted module already contains the resumed-and-updated weights.
        # Reusing the original resume path here would silently validate/predict
        # the pre-fit checkpoint instead.
        ckpt_path = None
    if cfg.trainer.do_validate:
        validate(module, datamodule, trainer, ckpt_path)
    if cfg.trainer.do_predict:
        predict(module, datamodule, trainer, ckpt_path)


if __name__ == "__main__":
    bootstrap_run_id()
    main()
