# Intelligent Assignment Evaluator (AES)

A unified AI-powered assessment platform that evaluates **essays**, **short answers**, and **code** with transparent rubric-driven feedback, explainable scores, and curated learning resources.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Model & Artifact Download](#model--artifact-download)
6. [Usage](#usage)
   - [Individual Inference](#individual-inference)
   - [Batch Inference](#batch-inference)
   - [Run the Streamlit App](#run-the-streamlit-app)
7. [End-to-End Workflow](#end-to-end-workflow)
8. [Dataset & Folder Descriptions](#dataset--folder-descriptions)
9. [Troubleshooting](#troubleshooting)

---

## Features

- **Multi-type Assessment**: Evaluate essays, short answers, and code with assignment-specific models.
- **Transparent Scoring**: Returns normalized scores, confidence intervals, and criterion-level rubric breakdowns.
- **Explainable Feedback**: Automatically detects strengths, weaknesses, and provides tailored recommendations.
- **Resource Recommendations**: Curated learning resources filtered by assignment type and detected weaknesses.
- **Similarity Detection**: For code submissions, surfaces top matches to detect reuse or common mistakes.
- **Report Export**: Generate formatted DOCX/PDF reports for student feedback.
- **Web UI**: Interactive Streamlit app for submission, evaluation, and result visualization.

---

## Project Structure

```
Gen-AI/
├── aes-2-0-kerasnlp-starter.ipynb      # Training notebook (essay model)
├── streamlit_app.py                    # Main web app (Streamlit)
├── essay_pipeline.py                   # Core evaluation logic for all three types
├── reporting.py                        # Report generation (DOCX/PDF)
├── run_inference.py                    # CLI script for individual inference
├── requirements.txt                    # Python dependencies
│
├── kaggle/
│   └── working/
│       └── best_model.weights.h5       # Essay model weights (Keras/KerasNLP)
│
├── short_answer/
│   ├── vectorizer.pkl                  # TF-IDF vectorizer for short answers
│   ├── assignment_evaluator.pkl        # Trained short-answer classifier
│   ├── data/                           # Training data
│   └── shortans.ipynb                  # Training notebook (short answer)
│
├── code/
│   └── checkpoint_bundle/
│       └── code_index.joblib           # Indexed code corpus (similarity search)
│
├── Essay_data/
│   ├── train.csv                       # Essay training set (text + labels)
│   └── test.csv                        # Essay test set
│
├── Frontend/                           # (Optional) UI assets
└── scripts/
    └── serialize_jax_model.py          # Utility to serialize Keras models
```

---

## Prerequisites

- **Python**: 3.9 or 3.10 (recommended: 3.10)
- **Conda**: Recommended for environment isolation
- **Disk Space**: ~2 GB (for models + dependencies)
- **OS**: Windows, macOS, or Linux

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Gen-AI
```

### 2. Create a Conda Environment (Recommended)

```bash
conda create -n aes311 python=3.10 -y
conda activate aes311
```

Or create a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### What's in `requirements.txt`:

```
streamlit>=1.36
python-docx>=1.1.2
pandas>=2.2
numpy>=1.26
scikit-learn>=1.5
keras==3.14.1
keras-nlp==0.29.0
tensorflow>=2.15
kagglehub>=1.0.1
transformers>=4.46.3
reportlab>=4.0
joblib>=1.4.2
```

---

## Model & Artifact Download

### Essay Model

The essay model weights are **already in the repository** at `kaggle/working/best_model.weights.h5`.

If you need to download or sync from Kaggle:

```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.model_download('path/to/kaggle/model')"
```

Or manually:
1. Go to [Kaggle Models](https://www.kaggle.com/models)
2. Search for the essay scoring model
3. Download the `.h5` weights file to `kaggle/working/best_model.weights.h5`

### Short-Answer Artifacts

The short-answer vectorizer and classifier are **already in the repository** at:
- `short_answer/vectorizer.pkl`
- `short_answer/assignment_evaluator.pkl`

To retrain:

```bash
cd short_answer
jupyter notebook shortans.ipynb
```

### Code Index

The code similarity index is **already in the repository** at `code/checkpoint_bundle/code_index.joblib`.

To rebuild with your own code corpus:

```bash
python scripts/build_code_index.py --input-dir your_code_samples/ --output code/checkpoint_bundle/code_index.joblib
```

---

## Dataset & Folder Descriptions

### `Essay_data/`

Contains annotated essays for training and validation:

- **`train.csv`**: Training set with essay text and rubric labels (fields: `essay`, `score`, `coherence`, `grammar`, etc.)
- **`test.csv`**: Test set for validation

Format: CSV with columns for text and per-criterion scores (0–5 or 0–6 scale).

### `short_answer/`

Contains short-answer evaluation artifacts:

- **`vectorizer.pkl`**: Fitted TF-IDF vectorizer (converts text to numeric features)
- **`assignment_evaluator.pkl`**: Trained classifier (predicts score class and confidence)
- **`data/`**: Raw training data (CSV with short answers + labels)
- **`shortans.ipynb`**: Training notebook (shows TF-IDF + classifier pipeline)

### `code/`

Contains code similarity and correctness evaluation:

- **`checkpoint_bundle/code_index.joblib`**: Indexed corpus of reference code snippets
  - Maps code submissions to similarity scores
  - Detects plagiarism / common patterns
  - Supports nearest-neighbor search for code examples

---

## Usage

### Individual Inference

#### 1. Essay Evaluation (CLI)

```bash
python run_inference.py --text "Your essay text here" --prompt "Essay prompt" -o essay_result.json
```

Or from a file:

```bash
python run_inference.py --file essay.txt --weights kaggle/working/best_model.weights.h5 -o essay_result.json
```

**Output** (`essay_result.json`):
```json
{
  "score": 4.2,
  "confidence": 0.87,
  "rubric": {"content": 4, "coherence": 4, "grammar": 4},
  "strengths": ["Clear argument", "Good evidence"],
  "weaknesses": ["Grammar issues"],
  "resources": [{"title": "UNC Writing Center", "url": "..."}],
  "similarity_risk": 0.05
}
```

#### 2. Short-Answer Evaluation (Python)

```python
from essay_pipeline import evaluate_short_answer

result = evaluate_short_answer(
    text="The mitochondria is the powerhouse of the cell.",
    reference_text="Mitochondria are organelles that produce ATP.",
    rubric="accuracy,completeness,clarity"
)
print(result.to_dict())
```

**Output**:
```json
{
  "score": 3.5,
  "confidence": 0.82,
  "rubric": {"accuracy": 4, "completeness": 3, "clarity": 4},
  "weaknesses": ["Missing key organelle functions"],
  "resources": [{"title": "Biology Fundamentals", "url": "..."}]
}
```

#### 3. Code Evaluation (Python)

```python
from essay_pipeline import evaluate_code

result = evaluate_code(
    code="""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
    rubric="correctness,readability,efficiency,edge_cases"
)
print(result.to_dict())
```

**Output**:
```json
{
  "score": 3.0,
  "confidence": 0.75,
  "similarity_score": 0.42,
  "top_matches": [
    {"file": "reference_1.py", "similarity": 0.85},
    {"file": "reference_2.py", "similarity": 0.72}
  ],
  "rubric": {"correctness": 3, "readability": 4, "efficiency": 2, "edge_cases": 2},
  "weaknesses": ["Exponential time complexity", "No input validation"],
  "resources": [{"title": "Algorithm Optimization", "url": "..."}]
}
```

---

### Batch Inference

#### Evaluate Multiple Essays

```python
from essay_pipeline import evaluate_essay
import csv

with open("essays.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        result = evaluate_essay(
            text=row["essay"],
            prompt=row["prompt"]
        )
        print(f"{row['student_id']}: {result.score}")
```

#### Evaluate Multiple Short Answers

```python
from essay_pipeline import evaluate_short_answer
import pandas as pd

df = pd.read_csv("short_answers.csv")
scores = []

for idx, row in df.iterrows():
    result = evaluate_short_answer(
        text=row["student_answer"],
        reference_text=row["correct_answer"]
    )
    scores.append({
        "student_id": row["student_id"],
        "score": result.score,
        "feedback": result.recommendations
    })

pd.DataFrame(scores).to_csv("short_answer_results.csv", index=False)
```

#### Evaluate Multiple Code Submissions

```python
from essay_pipeline import evaluate_code
import json
from pathlib import Path

code_dir = Path("submissions")
results = []

for code_file in code_dir.glob("*.py"):
    code = code_file.read_text()
    result = evaluate_code(code=code)
    results.append({
        "file": code_file.name,
        "score": result.score,
        "top_match": result.top_matches[0] if result.top_matches else None
    })

with open("code_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

### Run the Streamlit App

#### Launch the Web UI

```bash
streamlit run streamlit_app.py
```

The app opens in your browser at `http://localhost:8501`.

#### Features in the App

1. **Submission Form**:
   - Student name, roll number
   - Select assignment type: Essay / Short Answer / Code
   - Choose or enter rubric criteria
   - Paste or upload submission text

2. **Results Dashboard**:
   - Overall score and confidence
   - Rubric breakdown (chart + table)
   - Strengths and weaknesses
   - Recommended resources (with clickable links)
   - Similarity risk indicator

3. **Export Reports**:
   - Generate DOCX/PDF with full feedback
   - Downloadable student report

4. **Batch Processing** (optional):
   - Upload CSV with multiple submissions
   - Export results as Excel

---

## End-to-End Workflow

### Scenario 1: Single Essay Submission

```bash
# 1. Activate environment
conda activate aes311

# 2. Run inference via CLI
python run_inference.py \
  --file student_essay.txt \
  --prompt "Discuss climate change" \
  -o result.json

# 3. View result
cat result.json
```

### Scenario 2: Batch Evaluation + Report Generation

```bash
# 1. Prepare CSV (columns: student_id, assignment_type, submission_text, prompt)
# See `Essay_data/test.csv` for format

# 2. Run batch evaluation (Python script)
python -c "
from essay_pipeline import evaluate_essay
from reporting import generate_report
import csv

with open('submissions.csv') as f:
    for row in csv.DictReader(f):
        result = evaluate_essay(
            text=row['submission_text'],
            prompt=row['prompt']
        )
        generate_report(result, f\"reports/{row['student_id']}.docx\")
        print(f\"Generated report for {row['student_id']}\")
"

# 3. Reports saved to `reports/` directory
ls reports/
```

### Scenario 3: Full Web App Deployment

```bash
# 1. Start the Streamlit server
streamlit run streamlit_app.py

# 2. In browser (http://localhost:8501):
#    - Select assignment type
#    - Enter student details
#    - Paste/upload submission
#    - Click "Submit evaluation"

# 3. View results on dashboard
#    - Export as DOCX/PDF if needed

# 4. Stop server
#    Ctrl+C in terminal
```

### Scenario 4: Mixed Assessment (All Three Types)

```bash
# 1. Activate environment
conda activate aes311

# 2. Evaluate essay
python run_inference.py --file essay.txt -o essay_result.json

# 3. Evaluate short answer (Python)
python -c "
from essay_pipeline import evaluate_short_answer
import json

result = evaluate_short_answer(text='Student answer here')
print(json.dumps(result.to_dict(), indent=2))
" > short_answer_result.json

# 4. Evaluate code (Python)
python -c "
from essay_pipeline import evaluate_code
import json

result = evaluate_code(code=open('submission.py').read())
print(json.dumps(result.to_dict(), indent=2))
" > code_result.json

# 5. Or use the web app for all three in one place
streamlit run streamlit_app.py
```

---

## Troubleshooting

### 1. Keras/KerasNLP Loading Warnings

```
WARNING: Could not deserialize optimizer ...
```

**Solution**: These are non-fatal. The model weights load successfully with `skip_mismatch=True`. Scores are unaffected.

### 2. Scikit-learn InconsistentVersionWarning

```
InconsistentVersionWarning: Scikit-learn version mismatch ...
```

**Solution**: Your environment has a different scikit-learn version than when the model was pickled. Models still work. To avoid: create a fresh conda environment with the exact `requirements.txt`.

### 3. ModuleNotFoundError: No module named 'torch'

```
ModuleNotFoundError: No module named 'torch'
```

**Solution**: Streamlit's file watcher tries to import optional dependencies. Not critical—the app still runs. Suppress warnings:

```bash
export TF_CPP_MIN_LOG_LEVEL=3
streamlit run streamlit_app.py
```

### 4. Port 8501 Already in Use

```
Port 8501 is already in use.
```

**Solution**: Kill the existing Streamlit process or use a different port:

```bash
streamlit run streamlit_app.py --server.port 8502
```

### 5. Artifact Files Not Found

```
FileNotFoundError: kaggle/working/best_model.weights.h5 not found
```

**Solution**: Ensure all artifact directories exist and paths are correct:

```bash
# Check structure
ls kaggle/working/best_model.weights.h5
ls short_answer/vectorizer.pkl
ls code/checkpoint_bundle/code_index.joblib

# If missing, see "Model & Artifact Download" section
```

---

## Running in the `aes311` Environment (Recommended)

For consistency with the training environment, always activate `aes311`:

```bash
# Create environment (one time)
conda create -n aes311 python=3.10 -y
conda activate aes311

# Install dependencies
pip install -r requirements.txt

# Run any command in aes311
python run_inference.py --file essay.txt -o out.json
streamlit run streamlit_app.py
```

---

## Citation & Acknowledgments

This project combines:
- **Essay Scoring**: KerasNLP DeBERTa model trained on rubric-annotated essays
- **Short-Answer Evaluation**: TF-IDF + scikit-learn classifier
- **Code Evaluation**: Similarity-based indexing with code embeddings
- **UI**: Streamlit + Plotly for interactive dashboard

---
## Contact & Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

## Quick Start (TL;DR)

```bash
# Clone & setup (5 min)
git clone <repo-url>
cd Gen-AI
conda create -n aes311 python=3.10 -y
conda activate aes311
pip install -r requirements.txt

# Try essay inference (1 min)
python run_inference.py --text "My essay here" -o result.json
cat result.json

# Or launch the web app (interactive)
streamlit run streamlit_app.py

# Batch process (see Usage section)
python batch_evaluate.py --input essays.csv --output results.csv
```

---

**Happy Evaluating! 🚀**
