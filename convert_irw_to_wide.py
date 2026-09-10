"""Convert IRW long CSVs (id, item, resp) to separate <name>_wide.csv files.

Run from the repository with:
    uv run python convert_irw_to_wide.py
"""
import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "irw_datasets"


def convert_file(source: Path, output_dir: Path) -> tuple[Path, int, int]:
    """Preserve identifiers, literal responses (including NA), and first-seen order."""
    output = output_dir / f"{source.stem}_wide.csv"
    if output.resolve() == source.resolve():
        raise ValueError(f"Output must differ from input: {source}")

    df = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {"id", "item", "resp"}
    if set(df.columns) != required:
        raise ValueError(f"{source.name}: expected exactly the columns id, item, resp")
    if df.empty:
        raise ValueError(f"{source.name}: no response records")
    if df[["id", "item"]].eq("").any().any():
        raise ValueError(f"{source.name}: id and item must not be empty")
    duplicates = int(df.duplicated(["id", "item"]).sum())
    if duplicates:
        raise ValueError(f"{source.name}: {duplicates} duplicate id/item records; resolve before converting")
    if df["item"].eq("id").any():
        raise ValueError(f"{source.name}: item 'id' conflicts with the output identifier column")

    wide = df.pivot(index="id", columns="item", values="resp")
    wide = wide.reindex(index=df["id"].unique(), columns=df["item"].unique())
    output_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output, index=True, index_label="id", na_rep="", encoding="utf-8")
    return output, wide.shape[0], wide.shape[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="directory of long CSV files (default: repository irw_datasets)")
    parser.add_argument("--output-dir", type=Path,
                        help="output directory (default: input directory); filenames end in _wide.csv")
    args = parser.parse_args(argv)
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    sources = sorted(p for p in args.input_dir.glob("*.csv") if not p.stem.endswith("_wide"))
    if not sources:
        parser.error(f"No long CSV files found in {args.input_dir}")
    output_dir = args.output_dir if args.output_dir is not None else args.input_dir
    for source in sources:
        try:
            output, participants, items = convert_file(source, output_dir)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(f"{source.name} -> {output}: {participants:,} participants x {items:,} items", flush=True)


if __name__ == "__main__":
    main()
