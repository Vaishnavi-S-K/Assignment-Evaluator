from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Serialize the notebook-trained DeBERTa JAX model to a .keras file")
    parser.add_argument("--weights", default="kaggle/working/best_model.weights.h5", help="Path to the trained weights checkpoint")
    parser.add_argument("--out", default="exports/aes_deberta.keras", help="Output .keras file")
    parser.add_argument("--preset", default="deberta_v3_extra_small_en", help="KerasNLP preset used during training")
    parser.add_argument("--sequence-length", type=int, default=512, help="Sequence length used during training")
    args = parser.parse_args()

    # Keep the model on JAX so the serialization reflects the training backend.
    os.environ.setdefault("KERAS_BACKEND", "jax")

    from essay_pipeline import build_model  # imported after KERAS_BACKEND is set

    weights_path = Path(args.weights)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building model from preset: {args.preset}")
    model = build_model(preset=args.preset)
    print("Model built")

    print(f"Loading weights: {weights_path}")
    model.load_weights(str(weights_path))
    print("Weights loaded")

    print(f"Saving serialized model to: {out_path}")
    model.save(str(out_path))
    print("Done")


if __name__ == "__main__":
    main()
