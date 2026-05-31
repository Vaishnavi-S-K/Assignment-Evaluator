# Essay Scoring Studio

This folder keeps the notebook-trained essay scoring project together:

- `aes-2-0-kerasnlp-starter.ipynb`
- `kaggle/working/best_model.weights.h5`

The Streamlit app is in `streamlit_app.py` and the reusable scoring logic is in `essay_pipeline.py`.

## Serialize the trained model

If you run in the same KerasNLP/JAX stack that produced the checkpoint, you can serialize the trained model to a single `.keras` file:

```bash
python scripts/serialize_jax_model.py --weights kaggle/working/best_model.weights.h5 --out exports/aes_deberta.keras
```

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Notes

- The app tries to load the notebook-trained DeBERTa checkpoint first.
- If the local environment cannot deserialize the exact KerasNLP stack, the app falls back to heuristic rubric scoring so the UI still works.
- The generated DOCX report includes score, rubric breakdown, feedback, similarity risk, and recommendations.
