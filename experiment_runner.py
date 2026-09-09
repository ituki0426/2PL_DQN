"""Discover experiment packages and dispatch the shared command-line entry points."""
import argparse
import importlib
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parent


def available_experiments():
    return sorted(
        path.name for path in PROJECT_ROOT.iterdir()
        if re.fullmatch(r"EXP[0-9]{3,}", path.name)
        and (path / "__init__.py").is_file()
    )


def result_directory(experiment, out=None):
    """Keep each experiment isolated, including under a custom output root."""
    return (Path(out) if out is not None else PROJECT_ROOT / "result") / experiment


def dispatch(command, argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--experiment", "--exp", choices=available_experiments(), default="EXP001",
                        help="experiment package (default: EXP001)")
    parser.add_argument("--list-experiments", action="store_true",
                        help="list available experiment packages and exit")
    args, remaining = parser.parse_known_args(argv)
    if args.list_experiments:
        print("\n".join(available_experiments()))
        return
    if "--help" in remaining or "-h" in remaining:
        print(parser.format_help())
    if not (PROJECT_ROOT / args.experiment / f"{command}.py").is_file():
        parser.error(f"{args.experiment} does not provide {command}.py")
    module = importlib.import_module(f"{args.experiment}.{command}")
    module.main(remaining)
