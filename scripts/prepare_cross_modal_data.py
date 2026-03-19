"""Prepare real cross-modal retrieval data from COCO + GPT-4o-mini captions.

Downloads COCO val2017 images and generates fresh captions + hard negatives
using GPT-4o-mini vision to avoid data contamination.

Usage:
    # Step 1: Preview 10 samples
    uv run --extra data --extra openai python scripts/prepare_cross_modal_data.py --preview

    # Step 2: Full generation (200 images)
    uv run --extra data --extra openai python scripts/prepare_cross_modal_data.py --count 200
"""

from __future__ import annotations

import base64
import json
import random
import time
from pathlib import Path
from typing import Any

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cross_modal"
IMAGES_DIR = DATA_DIR / "images"

# COCO 2017 val images base URL
COCO_IMAGE_BASE = "http://images.cocodataset.org/val2017"

# COCO categories for diversity sampling
COCO_SUPERCATEGORIES = [
    "person", "vehicle", "outdoor", "animal", "accessory",
    "sports", "kitchen", "food", "furniture", "electronic",
    "appliance", "indoor",
]


def load_coco_annotations(cache_dir: Path | None = None) -> dict:
    """Download and parse COCO val2017 captions annotations."""
    import io
    import zipfile

    cache = cache_dir or DATA_DIR / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    ann_file = cache / "captions_val2017.json"

    if not ann_file.exists():
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        print(f"Downloading COCO annotations from {url}...")
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # Extract only captions_val2017.json
            with zf.open("annotations/captions_val2017.json") as f:
                ann_file.write_bytes(f.read())
        print(f"Saved annotations to {ann_file}")
    else:
        print(f"Using cached annotations: {ann_file}")

    with open(ann_file, encoding="utf-8") as f:
        return json.load(f)


def select_diverse_images(annotations: dict, count: int, seed: int = 42) -> list[dict]:
    """Select a diverse subset of COCO images with one caption each."""
    rng = random.Random(seed)

    # Build image_id -> info lookup
    images_by_id = {img["id"]: img for img in annotations["images"]}

    # Group captions by image_id, keep first caption per image
    captions_by_image: dict[int, str] = {}
    for ann in annotations["annotations"]:
        img_id = ann["image_id"]
        if img_id not in captions_by_image:
            captions_by_image[img_id] = ann["caption"]

    # Shuffle and select
    image_ids = list(captions_by_image.keys())
    rng.shuffle(image_ids)

    selected = []
    for img_id in image_ids[:count]:
        img_info = images_by_id[img_id]
        selected.append({
            "coco_id": img_id,
            "file_name": img_info["file_name"],
            "original_caption": captions_by_image[img_id],
            "width": img_info["width"],
            "height": img_info["height"],
        })

    return selected


def download_coco_images(entries: list[dict]) -> list[dict]:
    """Download COCO val2017 images and return metadata."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for i, entry in enumerate(entries):
            filename = entry["file_name"]
            save_path = IMAGES_DIR / filename
            url = f"{COCO_IMAGE_BASE}/{filename}"

            if save_path.exists():
                print(f"  [{i+1}/{len(entries)}] Already exists: {filename}")
            else:
                print(f"  [{i+1}/{len(entries)}] Downloading {filename}...", end=" ")
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    save_path.write_bytes(resp.content)
                    print(f"OK ({len(resp.content)//1024}KB)")
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

            results.append({
                "id": i,
                "coco_id": entry["coco_id"],
                "image_path": f"images/{filename}",
                "original_caption": entry["original_caption"],
            })

    return results


def generate_caption_batch(
    image_paths: list[Path],
    model: str = "gpt-4o-mini",
) -> list[str]:
    """Generate fresh captions for images using OpenAI vision API."""
    from openai import OpenAI

    client = OpenAI()
    captions = []

    for i, img_path in enumerate(image_paths):
        img_bytes = img_path.read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in 2-3 detailed sentences. "
                            "Include specific objects, colors, actions, and spatial relationships. "
                            "Be precise and factual — do not speculate about context or emotions."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                    },
                ],
            }],
            max_tokens=200,
            temperature=0.3,
        )

        caption = response.choices[0].message.content.strip()
        captions.append(caption)
        print(f"  Caption {i+1}/{len(image_paths)}: {caption[:80]}...")

        # Rate limit
        if i < len(image_paths) - 1:
            time.sleep(0.5)

    return captions


def generate_hard_negatives_batch(
    captions: list[str],
    model: str = "gpt-4o-mini",
    n_negatives: int = 3,
) -> list[list[str]]:
    """Generate hard negative captions using GPT-4o-mini."""
    from openai import OpenAI

    client = OpenAI()
    all_negatives = []

    for i, caption in enumerate(captions):
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    f"Given this image description:\n\"{caption}\"\n\n"
                    f"Generate exactly {n_negatives} WRONG descriptions that are similar but "
                    "differ in one key detail. Each should sound plausible but describe "
                    "a different scene. Change one of: main object, action, color, count, or location.\n\n"
                    'Return a JSON object with key "negatives" containing an array of strings. Example:\n'
                    '{"negatives": ["A blue car parked on the street", "A red truck in the driveway", "Two red cars in a parking lot"]}'
                ),
            }],
            max_tokens=500,
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                # Extract the list from the first list-valued key
                for v in parsed.values():
                    if isinstance(v, list):
                        negatives = [str(x) for x in v[:n_negatives]]
                        break
                else:
                    negatives = [str(v) for v in list(parsed.values())[:n_negatives]]
            elif isinstance(parsed, list):
                negatives = [str(x) for x in parsed[:n_negatives]]
            else:
                negatives = [raw]
        except json.JSONDecodeError:
            negatives = [raw]

        all_negatives.append(negatives)
        print(f"  Hard negatives {i+1}/{len(captions)}: {len(negatives)} generated")

        if i < len(captions) - 1:
            time.sleep(0.5)

    return all_negatives


def estimate_cost(n_images: int) -> dict[str, float]:
    """Estimate OpenAI API costs."""
    # GPT-4o-mini pricing (as of 2024):
    # Input: $0.15/1M tokens, Output: $0.60/1M tokens
    # Image (low detail): ~85 tokens per image

    # Caption generation: ~85 img tokens + ~30 prompt tokens input, ~100 output tokens
    caption_input_tokens = n_images * (85 + 30)
    caption_output_tokens = n_images * 100
    caption_cost = (caption_input_tokens * 0.15 + caption_output_tokens * 0.60) / 1_000_000

    # Hard negative generation: ~100 input tokens, ~150 output tokens
    neg_input_tokens = n_images * 150
    neg_output_tokens = n_images * 200
    neg_cost = (neg_input_tokens * 0.15 + neg_output_tokens * 0.60) / 1_000_000

    total = caption_cost + neg_cost
    return {
        "caption_generation": caption_cost,
        "hard_negatives": neg_cost,
        "total": total,
    }


def infer_category(caption: str) -> str:
    """Infer a simple category from caption text."""
    caption_lower = caption.lower()
    categories = {
        "person": ["person", "man", "woman", "people", "child", "boy", "girl", "player"],
        "animal": ["dog", "cat", "bird", "horse", "cow", "sheep", "elephant", "bear", "giraffe", "zebra"],
        "vehicle": ["car", "truck", "bus", "train", "motorcycle", "bicycle", "boat", "airplane"],
        "food": ["food", "pizza", "cake", "sandwich", "fruit", "banana", "apple", "meal", "plate"],
        "sports": ["tennis", "baseball", "surfing", "skateboard", "ski", "snowboard", "ball"],
        "outdoor": ["street", "beach", "park", "mountain", "field", "river", "ocean", "sky"],
        "indoor": ["room", "kitchen", "bathroom", "bedroom", "office", "table", "desk"],
    }
    for cat, keywords in categories.items():
        if any(kw in caption_lower for kw in keywords):
            return cat
    return "other"


def save_jsonl(path: Path, data: list[dict]) -> None:
    """Write a list of dicts to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_samples(metadata: list[dict], n: int = 5) -> None:
    """Print sample entries for quality review."""
    print("\n" + "=" * 70)
    print(f"  Sample entries (first {min(n, len(metadata))})")
    print("=" * 70)

    for entry in metadata[:n]:
        print(f"\n  ID: {entry['id']}")
        print(f"  Image: {entry['image_path']}")
        print(f"  Category: {entry.get('category', 'N/A')}")
        print(f"  Caption: {entry['caption']}")
        if entry.get("original_caption"):
            print(f"  Original: {entry['original_caption']}")
        if entry.get("hard_negatives"):
            for j, neg in enumerate(entry["hard_negatives"]):
                print(f"  Negative {j+1}: {neg}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare cross-modal retrieval data")
    parser.add_argument("--count", type=int, default=10, help="Number of images (default: 10)")
    parser.add_argument("--preview", action="store_true", help="Preview mode: generate but don't save")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use existing images")
    parser.add_argument("--caption-model", default="gpt-4o-mini", help="OpenAI model for captioning")
    args = parser.parse_args()

    count = args.count
    print(f"\n--- Cross-Modal Data Preparation ({count} images) ---\n")

    # Cost estimate
    costs = estimate_cost(count)
    print("Estimated OpenAI API costs:")
    print(f"  Caption generation:  ${costs['caption_generation']:.4f}")
    print(f"  Hard negatives:      ${costs['hard_negatives']:.4f}")
    print(f"  Total:               ${costs['total']:.4f}")
    print()

    # Step 1: Load COCO annotations and download images
    if not args.skip_download:
        annotations = load_coco_annotations()
        entries = select_diverse_images(annotations, count)
        print(f"\nDownloading {len(entries)} images from COCO val2017...")
        image_entries = download_coco_images(entries)
        print(f"Successfully downloaded {len(image_entries)} images")
    else:
        # Load existing metadata
        meta_path = DATA_DIR / "metadata.jsonl"
        if meta_path.exists():
            image_entries = []
            with open(meta_path) as f:
                for line in f:
                    image_entries.append(json.loads(line))
            image_entries = image_entries[:count]
        else:
            print("No existing metadata found. Remove --skip-download.")
            return

    # Step 2: Generate fresh captions
    print(f"\nGenerating captions with {args.caption_model}...")
    image_paths = [DATA_DIR / e["image_path"] for e in image_entries]
    captions = generate_caption_batch(image_paths, model=args.caption_model)

    # Step 3: Generate hard negatives
    print(f"\nGenerating hard negatives with {args.caption_model}...")
    hard_negatives = generate_hard_negatives_batch(captions, model=args.caption_model)

    # Build final metadata
    metadata = []
    for entry, caption, negatives in zip(image_entries, captions, hard_negatives):
        entry["caption"] = caption
        entry["hard_negatives"] = negatives
        entry["category"] = infer_category(caption)
        metadata.append(entry)

    # Print samples
    print_samples(metadata)

    # Statistics
    categories = {}
    for m in metadata:
        cat = m["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print("\n" + "=" * 70)
    print("  Statistics")
    print("=" * 70)
    print(f"  Total images:      {len(metadata)}")
    print(f"  Total hard negs:   {sum(len(m['hard_negatives']) for m in metadata)}")
    print(f"  Categories:        {dict(sorted(categories.items()))}")
    print(f"  Estimated cost:    ${costs['total']:.4f}")

    if args.preview:
        print("\n[Preview mode] No files written. Remove --preview to save data.")
        return

    # Save
    save_jsonl(DATA_DIR / "metadata.jsonl", metadata)
    print(f"\nData saved to {DATA_DIR}/")
    print(f"  metadata.jsonl:  {len(metadata)} entries")
    print(f"  images/:         {len(metadata)} JPEG files")


if __name__ == "__main__":
    main()
