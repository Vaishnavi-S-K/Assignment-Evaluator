I'll inspect essay_pipeline.py to confirm the available functions, then create a concise set of install/run commands and a small `run_inference.py` script that uses the project's API. I'll update the todo list to track steps.

Read [](file:Gen-AI/essay_pipeline.py#1-1), lines 1 to 400

Read [](file:Gen-AI/essay_pipeline.py#400-400), lines 400 to 474

I'll add a small `run_inference.py` CLI script to the repo and update the todo list. Then I'll present the install and run commands.

Apply Patch

Applying the script now.

Generating patch (63 lines) in [](file:Gen-AI/run_inference.py)


```

```


Completed: *Inspect reporting exports* (1/5)

Progress update: I added run_inference.py and updated the todo list; next I’ll give the install + run commands and usage examples.

Commands (conda + pip; include JAX note)

- Create & activate conda environment (recommended)
```bash
conda create -n aes_infer python=3.10 -y
conda activate aes_infer
```

- Install runtime dependencies (basic)
```bash
pip install -r requirements.txt
```
If you don't have a requirements.txt or want a full set:
```bash
pip install --upgrade pip
pip install h5py numpy keras keras-nlp tensorflow transformers tokenizers scikit-learn reportlab python-docx streamlit
```

- If you trained with JAX and want exact parity, run in the original JAX environment (recommended). Install JAX per your platform; for CPU:
```bash
pip install \"jax[cpu]\" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
For CUDA builds, follow https://github.com/google/jax#installation.

Run inference (CLI)

- Inline text example:
```bash
python run_inference.py --text "Schools should limit homework because..." --prompt "Write an argumentative essay about whether schools should limit homework."
```

- From a file and save JSON:
```bash
python run_inference.py --file samples/essay1.txt --weights kaggle/working/best_model.weights.h5 -o result.json
```

- Use your existing environment and exact python path:
```powershell
C:\Users\vaish\miniconda3\envs\aes_jax\python.exe run_inference.py --file samples/essay1.txt --weights kaggle/working/best_model.weights.h5
```

Run the Streamlit app (web UI)
```bash
python -m streamlit run streamlit_app.py --server.port 8501
# or (Windows / conda)
C:\Users\vaish\miniconda3\envs\aes_jax\python.exe -m streamlit run streamlit_app.py --server.headless true --server.port 8501
```
Then open http://localhost:8501

Files I added/used
- run_inference.py — CLI script to call `evaluate_essay` and print/save JSON.
- essay_pipeline.py — (existing) contains `load_model`, `evaluate_essay`.
- streamlit_app.py — (existing) UI.

Example run_inference.py usage summary
- Default weights: best_model.weights.h5
- Entrypoint: `evaluate_essay(text, prompt=None, rubric=None, reference_texts=None, weights_path=..., preset=...)`
- Output: JSON with keys `final_score`, `confidence`, `rubric_breakdown`, `similarity_risk`, `strengths`, `weaknesses`, `feedback`, `recommendations`, `model_status`, `raw_score`.

