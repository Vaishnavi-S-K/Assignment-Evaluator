import os
import sys
from pathlib import Path

# Ensure Keras uses the JAX backend before importing keras/keras_nlp
os.environ.setdefault("KERAS_BACKEND", "jax")

print("Using Python:", sys.executable)
print("KERAS_BACKEND:", os.environ.get("KERAS_BACKEND"))

try:
    import jax
    import numpy as np
    import keras
    import keras_nlp
    from tokenizers import Tokenizer
except Exception as e:
    print("Import error:", e)
    raise

PRESET = "deberta_v3_extra_small_en"
WEIGHTS_PATH = Path("kaggle/working/best_model.weights.h5")
SEQ_LEN = 512

print("keras version:", keras.__version__)
print("keras_nlp version:", keras_nlp.__version__)

def build_model():
    # Build the classifier head using the preset backbone
    classifier = keras_nlp.models.DebertaV3Classifier.from_preset(
        PRESET, preprocessor=None, num_classes=6
    )
    inputs = classifier.input
    logits = classifier(inputs)
    outputs = keras.layers.Activation("sigmoid")(logits)
    model = keras.Model(inputs, outputs)
    return model


def try_preprocessor(sample_texts):
    try:
        preprocessor = keras_nlp.models.DebertaV3Preprocessor.from_preset(
            PRESET, sequence_length=SEQ_LEN
        )
        tokenized = preprocessor(sample_texts)
        return tokenized
    except Exception as e:
        print("Preprocessor.from_preset failed (falling back to tokenizers):", repr(e))
        return None


def tokenize_with_tokenizers(sample_texts):
    # Look for cached tokenizer.json produced by kagglehub/keras_hub
    home = Path.home()
    cache_base = home / '.cache' / 'kagglehub' / 'models' / 'keras' / 'deberta_v3' / 'keras' / PRESET
    tokenizer_files = list(cache_base.rglob('tokenizer.json')) if cache_base.exists() else []
    if not tokenizer_files:
        raise FileNotFoundError(f"tokenizer.json not found under {cache_base}; run preset download first or mount tokenizer.json into container")
    tok_path = tokenizer_files[-1]
    print('Using tokenizer.json at', tok_path)
    # The cached file is a Keras serialization config, not a raw tokenizer model.
    # On Windows, DebertaV3 preprocessing needs tensorflow-text, which is not available here.
    # Keep this path as a clear fallback message instead of failing with a JSON parse error.
    raise RuntimeError(
        f"Cached tokenizer at {tok_path} is a Keras config file, and Windows lacks a working tensorflow-text wheel for DebertaV3 preprocessing in this environment."
    )



def main():
    sample = [
        "This essay provides a clear argument with relevant evidence and good organization. The writer uses examples effectively and maintains coherent paragraphs."
    ]

    model = build_model()
    print('Model built. Summary:')
    model.summary()

    if WEIGHTS_PATH.exists():
        print('Loading weights from', WEIGHTS_PATH)
        try:
            model.load_weights(str(WEIGHTS_PATH))
            print('Weights loaded successfully')
        except Exception as e:
            print('Error loading weights in JAX backend:', repr(e))
            raise
    else:
        print('Weights not found at', WEIGHTS_PATH)

    tokenized = try_preprocessor(sample)
    if tokenized is None:
        tokenized = tokenize_with_tokenizers(sample)

    print('Running inference...')
    preds = model.predict(tokenized)
    print('Raw output shape:', np.array(preds).shape)
    scores = np.sum((preds > 0.5).astype(int), axis=-1).clip(1, 6)
    print('Predicted scores:', scores)


if __name__ == '__main__':
    main()
