"""Enable ``python -m morph ...`` (works without installing the console script)."""

from morph.cli import cli

if __name__ == "__main__":
    cli()
