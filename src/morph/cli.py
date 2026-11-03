# CLI subcommands import the scientific stack only when executed.
# ruff: noqa: PLC0415

"""``morph`` — molecular generation CLI.

Run in-process (on a GPU node / via ``srun``):

    uv run --active --env-file .env morph generate -b 16
    # or, without (re)installing the console script:
    uv run --active --env-file .env python -m morph generate -b 16

Model bundles are downloaded from Hugging Face on demand.
Use ``morph models list`` to inspect variants and available epochs.
"""

from __future__ import annotations

import click


def _catalog_options(command):
    """Share explicit catalog source options across discovery and loading."""
    command = click.option(
        "--catalog-repo", help="HF repository hosting catalog.json."
    )(command)
    command = click.option(
        "--catalog-revision", help="Catalog commit, tag or branch (default: main)."
    )(command)
    return click.option(
        "--catalog-file",
        type=click.Path(exists=True, dir_okay=False),
        help="Local catalog JSON; skips catalog network access.",
    )(command)


def _get_catalog(repo, revision, path, cache_dir, offline, *, optional=False):
    """Resolve an explicit override, or leave default discovery to the loader."""
    from morph.inference import load_catalog

    if optional and repo is None and revision is None and path is None:
        return None
    return load_catalog(
        repo,
        revision=revision,
        path=path,
        cache_dir=cache_dir,
        local_files_only=offline,
    )


@click.group()
@click.version_option(package_name="morph-chem")
def cli() -> None:
    """Morph — molecular generation CLI."""


@cli.group()
def models() -> None:
    """Discover and download pretrained model bundles."""


@models.command("list")
@_catalog_options
@click.option(
    "--cache-dir", type=click.Path(file_okay=False), help="HF cache directory."
)
@click.option("--offline", is_flag=True, help="Use only the cached catalog.")
def list_models(
    catalog_repo, catalog_revision, catalog_file, cache_dir, offline
) -> None:
    """Discover models from the Hub catalog without downloading weights."""
    try:
        catalog = _get_catalog(
            catalog_repo, catalog_revision, catalog_file, cache_dir, offline
        )
    except (ValueError, OSError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"{'MODEL':24} {'DATASET':12} {'EPOCHS':18} DEFAULT EPOCH")
    for spec in catalog.models.values():
        epochs = ", ".join(map(str, spec.epochs)) or "from bundle"
        latest = spec.select_epoch() or "from bundle"
        click.echo(f"{spec.name:24} {spec.dataset or '-':12} {epochs:18} {latest}")
    click.echo(f"Default model: {catalog.default_model or '(none)'}")
    source = (
        f"{catalog.repo_id}@{catalog.revision}"
        if catalog.repo_id
        else str(catalog.path)
    )
    click.echo(f"Catalog: {source}")


@models.command("download")
@_catalog_options
@click.argument("model_name")
@click.option("--epoch", type=click.IntRange(min=1), help="Completed training epoch.")
@click.option("--subfolder", help="Bundle subfolder inside the repository.")
@click.option(
    "--revision", help="HF commit, tag or branch; catalog entries pin a commit."
)
@click.option(
    "--filename", help="Checkpoint path relative to the bundle, instead of epoch."
)
@click.option(
    "--cache-dir", type=click.Path(file_okay=False), help="HF cache directory."
)
@click.option(
    "--local-dir", type=click.Path(file_okay=False), help="Export root directory."
)
@click.option(
    "--offline", is_flag=True, help="Use cached files without network access."
)
def download_model(
    model_name,
    epoch,
    subfolder,
    revision,
    filename,
    cache_dir,
    local_dir,
    offline,
    catalog_repo,
    catalog_revision,
    catalog_file,
) -> None:
    """Download one epoch with its config, vocabularies and distributions.

    Auth: hf auth login, or set HF_TOKEN. Prints the local bundle directory.
    """
    from morph.inference.hub import download_pretrained

    try:
        catalog = _get_catalog(
            catalog_repo,
            catalog_revision,
            catalog_file,
            cache_dir,
            offline,
            optional=True,
        )
        bundle = download_pretrained(
            model_name,
            epoch=epoch,
            subfolder=subfolder,
            revision=revision,
            filename=filename,
            catalog=catalog,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_files_only=offline,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(str(bundle.directory))


@cli.command()
@_catalog_options
@click.option(
    "-b",
    "--batch",
    "n_mols",
    type=click.IntRange(min=1),
    default=16,
    show_default=True,
    help="Number of molecules to generate.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    show_default=True,
    default="eval_outputs/denovo.sdf",
    help="Output SDF of valid final-frame molecules (3D coordinates + SMILES name).",
)
@click.option(
    "--model",
    "model_name",
    default=None,
    help="Catalog name (morph models list) or Hugging Face owner/repo ID.",
)
@click.option(
    "--epoch", type=click.IntRange(min=1), help="Epoch; defaults to latest published."
)
@click.option(
    "--subfolder", help="Bundle subfolder inside the repository or local root."
)
@click.option(
    "--revision", help="HF commit, tag or branch; catalog entries pin a commit."
)
@click.option(
    "--filename", help="Checkpoint path relative to the bundle, instead of epoch."
)
@click.option(
    "--cache-dir", type=click.Path(file_okay=False), help="HF cache directory."
)
@click.option(
    "--offline", is_flag=True, help="Use cached files without network access."
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    show_default=True,
)
@click.option("--steps", type=click.IntRange(min=1), default=100, show_default=True)
@click.option(
    "--ema/--no-ema",
    default=True,
    show_default=True,
    help="Use EMA or current weights.",
)
@click.option(
    "--weights-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Local variant bundle or download root; skips Hugging Face entirely.",
)
@click.option(
    "--checkpoint",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Override local weights; requires --weights-dir with matching assets.",
)
@click.option(
    "--save-trajs",
    is_flag=True,
    help="Save full trajectories in <output>_trajs/batch-*.pt files.",
)
@click.option("--seed", default=42, show_default=True)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=None,
    help="Molecules per sampling batch (default: up to 32).",
)
def generate(
    n_mols,
    output,
    model_name,
    weights_dir,
    checkpoint,
    save_trajs,
    seed,
    batch_size,
    epoch,
    subfolder,
    revision,
    filename,
    catalog_repo,
    catalog_revision,
    catalog_file,
    cache_dir,
    offline,
    device,
    steps,
    ema,
) -> None:
    """Generate molecules de-novo with an unconditional checkpoint.

    Downloads the selected checkpoint and matching assets on first use.
    Uses the saved model config, eager execution, and EMA weights by default.
    """
    from pathlib import Path

    import torch
    from rdkit import Chem

    from morph.inference.generate import final_mol, record_mol
    from morph.inference.hub import load_pretrained

    try:
        catalog = _get_catalog(
            catalog_repo,
            catalog_revision,
            catalog_file,
            cache_dir,
            offline,
            optional=True,
        )
        model = load_pretrained(
            model_name,
            epoch=epoch,
            subfolder=subfolder,
            revision=revision,
            filename=filename,
            catalog=catalog,
            device=device,
            weights_dir=weights_dir,
            checkpoint=checkpoint,
            cache_dir=cache_dir,
            local_files_only=offline,
            steps=steps,
            use_ema=ema,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"[morph generate] model={model.bundle.spec.name}  "
        f"ckpt={model.bundle.checkpoint}  n={n_mols}"
    )

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    trajectory_dir = out.with_name(out.stem + "_trajs")
    if save_trajs:
        # Reusing a nonempty directory could mix batches from separate runs.
        if trajectory_dir.exists() and any(trajectory_dir.iterdir()):
            raise click.ClickException(
                f"Trajectory directory is not empty: {trajectory_dir}. "
                "Choose a different output filename."
            )
        trajectory_dir.mkdir(parents=True, exist_ok=True)

    attempted = n_written = n_valid = batch_index = 0
    unique_smiles = set()
    with Chem.SDWriter(str(out)) as writer:

        def consume(trajectories, is_valid):
            nonlocal attempted, n_written, n_valid, batch_index
            if save_trajs:
                torch.save(
                    {
                        "start_index": attempted,
                        "trajectories": trajectories,
                        "is_valid": is_valid,
                    },
                    trajectory_dir / f"batch-{batch_index:06d}.pt",
                )
            for traj, valid in zip(trajectories, is_valid, strict=True):
                if valid:
                    mol = final_mol(traj, model.vocab)
                    if mol is not None:
                        mol.SetProp("_Name", Chem.MolToSmiles(mol))
                        writer.write(mol)
                        n_written += 1
                row = record_mol(traj, model.vocab)
                n_valid += int(valid)
                if valid and row["smiles"] is not None:
                    unique_smiles.add(row["smiles"])
                click.echo(
                    f"   [{attempted}] valid={valid} N={row['n_atoms']} "
                    f"logP={row['logp']} {row['smiles']}"
                )
                attempted += 1
            writer.flush()
            batch_index += 1

        model.generate_to(
            consume,
            n_mols=n_mols,
            seed=seed,
            batch_size=batch_size,
            return_traj=save_trajs,
        )
    click.echo(f"wrote {n_written}/{attempted} attempts -> {out}")
    if save_trajs:
        click.echo(f"wrote {batch_index} trajectory batches -> {trajectory_dir}")
    click.echo(f"  valid: {n_valid}/{attempted} | unique SMILES: {len(unique_smiles)}")


if __name__ == "__main__":
    cli()
