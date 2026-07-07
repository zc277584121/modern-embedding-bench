"""Upload an exported benchmark folder to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from mm_embed.hf_publish import upload_hf_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", required=True, choices=["dataset", "space"])
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--space-sdk", default="gradio")
    parser.add_argument("--commit-message", default="Update benchmark artifacts")
    parser.add_argument("--space-dataset-repo-id", default=None)
    parser.add_argument(
        "--resolve-hf-ip",
        default=None,
        help="Resolve huggingface.co to this IP inside the upload process only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = upload_hf_folder(
        folder=args.folder,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        space_sdk=args.space_sdk,
        commit_message=args.commit_message,
        dataset_repo_id_for_space=args.space_dataset_repo_id,
        resolve_hf_ip=args.resolve_hf_ip,
    )
    print(f"Uploaded {args.folder} to {args.repo_type} repo {args.repo_id}")
    print(url)


if __name__ == "__main__":
    main()
