import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from load_data import load_data


def _load_clip_components(model_name):
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, CLIPVisionModelWithProjection
    except ImportError as exc:
        raise RuntimeError(
            "This script requires `torch`, `Pillow`, and `transformers`."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPVisionModelWithProjection.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return {
        "torch": torch,
        "Image": Image,
        "processor": processor,
        "model": model,
        "device": device,
    }


def _iter_valid_records(records):
    for record in records:
        image_path = Path(record["thumbnail_path"])
        if image_path.exists():
            yield record, image_path


def extract_clip_thumbnail_embeddings(
    model_name="openai/clip-vit-base-patch32",
    output_path="experiments/cache/clip_thumbnail_embeddings.csv",
    batch_size=32,
    limit=None,
):
    components = _load_clip_components(model_name)
    torch = components["torch"]
    Image = components["Image"]
    processor = components["processor"]
    model = components["model"]
    device = components["device"]

    records = load_data()
    valid_records = list(_iter_valid_records(records))
    if limit is not None:
        valid_records = valid_records[:limit]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for start_idx in range(0, len(valid_records), batch_size):
            batch = valid_records[start_idx : start_idx + batch_size]
            images = []
            batch_records = []
            for record, image_path in batch:
                try:
                    images.append(Image.open(image_path).convert("RGB"))
                    batch_records.append(record)
                except Exception:
                    continue

            if not batch_records:
                continue

            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            image_embeds = model(pixel_values=pixel_values).image_embeds
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
            image_embeds = image_embeds.detach().cpu().numpy().astype(np.float32)

            for record, embed in zip(batch_records, image_embeds):
                row = {"video_id": record["video_id"]}
                for idx, value in enumerate(embed):
                    row[f"clip_thumb_{idx}"] = float(value)
                rows.append(row)

            if (start_idx // batch_size + 1) % 20 == 0:
                processed = min(start_idx + batch_size, len(valid_records))
                print(f"Processed {processed}/{len(valid_records)} thumbnails")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    elapsed = time.perf_counter() - started
    print(f"Saved CLIP thumbnail embeddings: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Dim: {df.shape[1] - 1 if not df.empty else 0}")
    print(f"Device: {device}")
    print(f"Elapsed: {elapsed:.2f}s")
    return {
        "output_path": str(output_path),
        "rows": len(df),
        "dim": int(df.shape[1] - 1) if not df.empty else 0,
        "device": device,
        "elapsed_sec": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--output-path",
        type=str,
        default="experiments/cache/clip_thumbnail_embeddings.csv",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_clip_thumbnail_embeddings(
        model_name=args.model_name,
        output_path=args.output_path,
        batch_size=args.batch_size,
        limit=args.limit,
    )
