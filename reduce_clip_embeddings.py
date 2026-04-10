import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import build_split_indices
from load_data import load_data


def reduce_clip_embeddings(
    input_path="experiments/cache/clip_thumbnail_embeddings.csv",
    output_path="experiments/cache/clip_thumbnail_embeddings_pca256.csv",
    n_components=256,
):
    from sklearn.decomposition import PCA

    records = load_data()
    train_idx, _, _ = build_split_indices(records)

    df = pd.read_csv(input_path)
    if "video_id" not in df.columns:
        raise ValueError("Input CLIP embedding CSV must include a `video_id` column.")

    video_ids = [record["video_id"] for record in records]
    aligned = df.set_index("video_id").reindex(video_ids)
    missing_mask = aligned.isna().any(axis=1).to_numpy()
    missing_count = int(missing_mask.sum())
    available_mask = ~missing_mask

    x = aligned.fillna(0.0).to_numpy(dtype="float32")
    train_fit_mask = np.zeros(len(records), dtype=bool)
    train_fit_mask[train_idx] = True
    train_fit_mask &= available_mask
    available_train_count = int(train_fit_mask.sum())
    if available_train_count == 0:
        raise ValueError("No training rows with available CLIP embeddings were found.")

    max_components = min(available_train_count, x.shape[1])
    n_components = min(n_components, max_components)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(x[train_fit_mask])

    x_reduced = np.zeros((len(records), n_components), dtype="float32")
    if available_mask.any():
        x_reduced[available_mask] = pca.transform(x[available_mask]).astype("float32")

    out_df = pd.DataFrame(
        x_reduced,
        columns=[f"clip_pca_{i}" for i in range(x_reduced.shape[1])],
    )
    out_df.insert(0, "video_id", video_ids)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    explained = float(pca.explained_variance_ratio_.sum())
    print(f"Saved reduced CLIP embeddings: {output_path}")
    print(f"Shape: {out_df.shape}")
    print(f"Explained variance ratio sum: {explained:.4f}")
    print(f"Missing thumbnails filled with zero vectors after PCA: {missing_count}")
    return {
        "output_path": str(output_path),
        "shape": out_df.shape,
        "explained_variance_ratio_sum": explained,
        "missing_count": missing_count,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        default="experiments/cache/clip_thumbnail_embeddings.csv",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="experiments/cache/clip_thumbnail_embeddings_pca256.csv",
    )
    parser.add_argument("--n-components", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reduce_clip_embeddings(
        input_path=args.input_path,
        output_path=args.output_path,
        n_components=args.n_components,
    )
