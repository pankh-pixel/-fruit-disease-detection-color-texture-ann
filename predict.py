"""
predict.py
==========
Command-line interface for inference on new images.

Usage
-----
    # Single image
    python predict.py --image path/to/apple.jpg

    # Batch (glob pattern)
    python predict.py --batch "data/test_images/*.jpg" --top-k 3

    # Custom model / encoder paths
    python predict.py --image img.jpg \\
                      --model models/fruit_disease_ann.keras \\
                      --encoder models/label_encoder.pkl
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.inference import Predictor, run_inference_cli


def main() -> None:
    parser = argparse.ArgumentParser(
        description="🍎  Fruit Disease Detection — Inference CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--image",   type=str, help="Path to a single image file.")
    parser.add_argument("--batch",   type=str, help="Glob pattern for batch inference.")
    parser.add_argument("--model",   type=str, default="models/checkpoints/best_model.keras")
    parser.add_argument("--encoder", type=str, default="models/label_encoder.pkl")
    parser.add_argument("--config",  type=str, default="config.yaml")
    parser.add_argument("--top-k",   type=int, default=3)
    parser.add_argument("--output",  type=str, default=None,
                        help="Optional JSON file to write batch results.")
    args = parser.parse_args()

    if not args.image and not args.batch:
        parser.error("Provide either --image or --batch.")

    predictor = Predictor(args.model, args.encoder, args.config)

    if args.image:
        run_inference_cli(
            args.image, args.model, args.encoder, args.config, args.top_k
        )

    elif args.batch:
        paths = sorted(glob.glob(args.batch))
        if not paths:
            print(f"No files matched: {args.batch}")
            sys.exit(1)

        results = predictor.predict_batch(paths, top_k=args.top_k)
        for path, result in zip(paths, results):
            print(f"{path:60s}  →  {result['label']} ({result['confidence']:.4f})")

        if args.output:
            payload = [
                {"file": p, **r} for p, r in zip(paths, results)
            ]
            with open(args.output, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
