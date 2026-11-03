"""Install the opt-in Linux docking tools pinned in pyproject.toml.

Run with the project virtualenv after ``uv sync --extra crossdocked``.
The legacy MGLTools Python runtime stays isolated inside that virtualenv;
only qvina2 and prepare_receptor4.py wrappers enter its bin directory.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def bootstrap(root: Path, config: dict) -> Path:
    """Fetch a checksum-verified micromamba without modifying the user's shell."""
    archive = root / "micromamba.tar.bz2"
    expected = config["micromamba-sha256"]
    if (
        not archive.exists()
        or hashlib.sha256(archive.read_bytes()).hexdigest() != expected
    ):
        temporary = archive.with_suffix(".download")
        try:
            with (
                urllib.request.urlopen(config["micromamba-url"], timeout=60) as source,
                temporary.open("wb") as target,
            ):
                shutil.copyfileobj(source, target)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise RuntimeError("Micromamba download failed SHA-256 verification")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    executable = root / "micromamba"
    with tarfile.open(archive) as bundle:
        member = bundle.getmember("bin/micromamba")
        if not member.isfile():
            raise RuntimeError(
                "Micromamba archive does not contain a regular executable"
            )
        with bundle.extractfile(member) as source, executable.open("wb") as target:
            shutil.copyfileobj(source, target)
    executable.chmod(0o755)
    return executable


def write_wrappers(prefix: Path, bin_dir: Path) -> None:
    """Expose the tools without putting MGLTools' Python 2 on the parent PATH."""
    marker = "# Managed by Morph's CrossDocked installer."
    receptor = prefix / "MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
    wrappers = {
        "qvina2": f'exec {shlex.quote(str(prefix / "bin/qvina2"))} "$@"\n',
        # Upstream pythonsh expands $@ unquoted; use its environment directly so
        # receptor paths containing spaces and Python arguments remain intact.
        "prepare_receptor4.py": (
            f"export MGL_ROOT={shlex.quote(str(prefix))}\n"
            'export PYTHONHOME="$MGL_ROOT"\n'
            'export PYTHONPATH="$MGL_ROOT/MGLToolsPckgs"\n'
            'export LD_LIBRARY_PATH="$MGL_ROOT/lib'
            '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
            f'exec "$MGL_ROOT/bin/python" {shlex.quote(str(receptor))} "$@"\n'
        ),
    }
    for name, body in wrappers.items():
        target = bin_dir / name
        if target.is_symlink() or (
            target.exists() and marker not in target.read_text()
        ):
            raise RuntimeError(
                f"Refusing to replace an existing unmanaged tool: {target}"
            )
        target.write_text(f"#!/bin/sh\n{marker}\n{body}")
        target.chmod(0o755)


def verify(bin_dir: Path) -> dict[str, str]:
    """Check executable startup, including legacy receptor preparation imports."""
    checks = {
        "qvina2": ["--version"],
        "obabel": ["-V"],
        "prepare_receptor4.py": ["-h"],
    }
    results = {}
    for name, arguments in checks.items():
        result = subprocess.run(
            [str(bin_dir / name), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        results[name] = (result.stdout + result.stderr).strip()
        print(results[name].splitlines()[0])
    return results


def main() -> None:
    """Install native packages only after the CrossDocked Python extra is present."""
    config = tomllib.loads((PROJECT / "pyproject.toml").read_text())["tool"]["morph"][
        "crossdocked"
    ]
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("The pinned native docking toolchain requires Linux x86_64")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run with the project virtualenv, not system Python")
    try:
        python_packages = {
            name: version(name)
            for name in ("gdown", "biopython", "easydict", "openbabel-wheel")
        }
    except PackageNotFoundError as exc:
        raise RuntimeError("First run: uv sync --locked --extra crossdocked") from exc

    root = Path(sys.prefix) / "crossdocked"
    prefix = root / "native"
    bin_dir = Path(sys.prefix) / "bin"
    manifest = root / "installed.json"
    root.mkdir(exist_ok=True)
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    if previous.get("config") != config or not (prefix / "bin/qvina2").exists():
        micromamba = bootstrap(root, config)
        command = [
            str(micromamba),
            "--no-rc",
            "install" if (prefix / "conda-meta/history").exists() else "create",
            "--yes",
            "--root-prefix",
            str(root / "mamba"),
            "--prefix",
            str(prefix),
            "--platform",
            config["platform"],
            "--override-channels",
            "--strict-channel-priority",
        ]
        for channel in config["channels"]:
            command.extend(["--channel", channel])
        subprocess.run([*command, *config["packages"]], check=True, timeout=1800)

    write_wrappers(prefix, bin_dir)
    checks = verify(bin_dir)
    native_packages = [
        json.loads(path.read_text())
        for path in sorted((prefix / "conda-meta").glob("*.json"))
    ]
    manifest.write_text(
        json.dumps(
            {
                "config": config,
                "python_packages": python_packages,
                "native_packages": native_packages,
                "checks": checks,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"CrossDocked tools ready in {bin_dir}; installation record: {manifest}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(f"CrossDocked setup failed: {exc}")
