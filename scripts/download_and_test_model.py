import os
import sys
from pathlib import Path

print("Running model download + checkpoint test")
import os
import sys
from pathlib import Path

print("Running model download + checkpoint test (tokenizer via tokenizers lib)")
print(f"Python: {sys.executable}")

try:
    import tensorflow as tf
    import keras
    import keras_nlp
    import numpy as np
    from tokenizers import Tokenizer
except Exception as e:
    print("Import failed:", e)
    raise

PRESET = "deberta_v3_extra_small_en"
WEIGHTS_PATH = Path("kaggle/working/best_model.weights.h5")
SEQ_LEN = 512

print("Keras version:", keras.__version__)
print("KerasNLP version:", keras_nlp.__version__)

try:
    print("Downloading base classifier from preset (no preprocessor)...")
    classifier = keras_nlp.models.DebertaV3Classifier.from_preset(
        PRESET, preprocessor=None, num_classes=6
    )
    print("Base classifier ready")

    inputs = classifier.input
    logits = classifier(inputs)
    outputs = keras.layers.Activation("sigmoid")(logits)
    model = keras.Model(inputs, outputs)
    print("Model built")

    if WEIGHTS_PATH.exists():
        print(f"Loading fine-tuned weights from {WEIGHTS_PATH}")
        model.load_weights(str(WEIGHTS_PATH))
        print("Fine-tuned weights loaded successfully")
    else:
        print(f"Fine-tuned weights not found at {WEIGHTS_PATH}, continuing with base weights")

    # Find downloaded tokenizer.json in kagglehub cache
    home = Path.home()
    cache_base = home / '.cache' / 'kagglehub' / 'models' / 'keras' / 'deberta_v3' / 'keras' / PRESET
    print('Looking for tokenizer.json under', cache_base)
    tokenizer_files = list(cache_base.rglob('tokenizer.json')) if cache_base.exists() else []
    if not tokenizer_files:
        print('tokenizer.json not found in cache; preprocessor.from_preset probably failed earlier. You can still use an alternative tokenizer manually.')
    else:
        tok_path = tokenizer_files[-1]
        print('Found tokenizer.json at', tok_path)
        tokenizer = Tokenizer.from_file(str(tok_path))

        sample = "This essay provides a clear argument with relevant evidence and good organization. The writer uses examples effectively and maintains coherent paragraphs."
        enc = tokenizer.encode(sample)
        ids = enc.ids[:SEQ_LEN]
        # pad
        pad_len = SEQ_LEN - len(ids)
        ids = ids + [0] * pad_len if pad_len > 0 else ids[:SEQ_LEN]
        attention_mask = [1] * min(len(enc.ids), SEQ_LEN) + [0] * max(0, pad_len)

        token_ids = np.array([ids], dtype='int32')
        padding_mask = np.array([attention_mask], dtype='int32')

        # KerasNLP DebertaV3 expects inputs named 'token_ids' and 'padding_mask'
        model_input = {'token_ids': token_ids, 'padding_mask': padding_mask}

        print('Running model.predict with tokenizer from tokenizers library...')
        preds = model.predict(model_input)
        print('Raw model output shape:', np.array(preds).shape)
        pred_scores = np.sum((preds > 0.5).astype(int), axis=-1).clip(1, 6)
        print('Predicted score:', pred_scores)

        # Run again for stability
        preds2 = model.predict(model_input)
        pred_scores2 = np.sum((preds2 > 0.5).astype(int), axis=-1).clip(1, 6)
        print('Predicted score (2nd run):', pred_scores2)
        stable = np.array_equal(pred_scores, pred_scores2)
        print('Stable across runs:', stable)

except Exception as e:
    print('Error during preset download or model test:', repr(e))
    raise

print('Done')
