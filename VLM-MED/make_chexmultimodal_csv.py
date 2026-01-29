import argparse
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Build train CSV from chexmultimodal indications + images")
    ap.add_argument('--src_csv', default='chexmultimodal/new_indications_324.csv', help='Path to indications CSV with index column matching image filenames')
    ap.add_argument('--image_dir', default='chexmultimodal/new_patient_images_324', help='Directory containing images named <row_index>.jpg')
    ap.add_argument('--out_csv', default='train_chexmultimodal.csv', help='Output CSV with columns image,labels')
    ap.add_argument('--label', default='', help='Optional single label to assign to every row; ignored when --label_from is set')
    ap.add_argument('--label_from', default='', help='Column name in src_csv to use as per-row label (e.g., indication)')
    ap.add_argument('--fillna', default='', help='Value to use when label_from is NaN/empty')
    args = ap.parse_args()

    df = pd.read_csv(args.src_csv)
    rows = []
    for idx in df.index:
        img_name = f"{idx}.jpg"
        img_path = os.path.join(args.image_dir, img_name)
        if args.label_from:
            if args.label_from not in df.columns:
                raise ValueError(f"Column '{args.label_from}' not found in {args.src_csv}")
            raw_label = df.at[idx, args.label_from]
            label_val = args.fillna if pd.isna(raw_label) else str(raw_label).replace('\n', ' ').replace(',', ';').strip()
        else:
            label_val = args.label
        rows.append({"image": img_path, "labels": label_val})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out_csv}")


if __name__ == '__main__':
    main()
