"""Assemble a runnable Space directory from the published, pinned model assets.

Run with: python apps/hf_space/prepare.py --output /tmp/morph-space
Then install requirements.txt in an isolated environment and run app.py there.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    registry = json.loads((source / "models.json").read_text())
    runtime_revision = registry["geom-logp"]["revision"]
    runtime = Path(
        snapshot_download(
            "maltefranke/morph",
            revision=runtime_revision,
            allow_patterns=["geom/logp_full/runtime/src/morph/**"],
        )
    )
    shutil.copytree(
        runtime / "geom/logp_full/runtime/src/morph",
        destination / "runtime/morph",
        dirs_exist_ok=True,
    )
    # Use the existing native PyTorch paths on ZeroGPU; avoid eager Triton
    # registration before the shared GPU is attached.
    kernels = destination / "runtime/morph/model/layers/kernels.py"
    code = kernels.read_text().replace(
        "torch.cuda.is_available()",
        '(os.environ.get("MORPH_FUSED_KERNELS", "1") == "1"'
        " and torch.cuda.is_available())",
    )
    kernels.write_text(code)
    for key, spec in registry.items():
        folder = spec["checkpoint"].rsplit("/", 1)[0]
        names = (
            "config.yaml",
            "atom_tokens.txt",
            "edge_tokens.txt",
            "charge_tokens.txt",
            "distributions.pt",
        )
        cache = Path(
            snapshot_download(
                "maltefranke/morph",
                revision=spec["revision"],
                allow_patterns=[f"{folder}/{name}" for name in names],
            )
        )
        assets = destination / "assets" / key
        assets.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(cache / folder / name, assets / name)
    for name in (
        "app.py",
        "backend.py",
        "trajectory_view.py",
        "models.json",
        "requirements.txt",
        "README.md",
    ):
        shutil.copy2(source / name, destination / name)
    print(f"Runnable Space prepared at {destination}")


if __name__ == "__main__":
    main()
