#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def read_filter_list(path: Path) -> set[str]:
    if not path:
        return set()
    items = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    return items


def sanitize_labels(raw: str, fillna: str) -> str:
    if pd.isna(raw) or raw == "":
        return fillna
    labels = [lbl.strip() for lbl in str(raw).replace("|", ",").split(",") if lbl.strip()]
    return ",".join(labels) if labels else fillna


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CSV for NIH ChestX-ray14 ROI pipeline.")
    parser.add_argument("--src_csv", required=True, type=Path, help="Path to Data_Entry_2017.csv")
    parser.add_argument("--image_dir", required=True, type=Path, help="Directory containing images/")
    parser.add_argument("--out_csv", required=True, type=Path, help="Output CSV path")
    parser.add_argument("--label_from", default="Finding Labels", help="Column name for labels")
    parser.add_argument("--filter_list", type=Path, default=None, help="Optional list.txt to filter rows")
    parser.add_argument("--fillna", default="Normal", help="Label to use when empty")
    args = parser.parse_args()

    if not args.src_csv.exists():
        raise FileNotFoundError(f"Missing src_csv: {args.src_csv}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Missing image_dir: {args.image_dir}")

    df = pd.read_csv(args.src_csv)
    if args.label_from not in df.columns:
        raise KeyError(f"Column '{args.label_from}' not found in {args.src_csv}")

    keep = read_filter_list(args.filter_list) if args.filter_list else None
    rows = []
    for _, row in df.iterrows():
        image = str(row.get("Image Index")) if "Image Index" in df.columns else str(row.get("image"))
        if not image:
            continue
        if keep is not None and image not in keep:
            continue
        label_raw = row[args.label_from]
        labels = sanitize_labels(label_raw, args.fillna)
        rows.append({"image": image, "labels": labels})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
