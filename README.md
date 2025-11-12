# Exchange Data Concepts

Curated notebooks, datasets, lecture decks, and reference material for an "Intro to Concepts for Big Data" course covering exploratory analysis through classical machine learning projects in Python.

## Highlights
- Structured end-to-end path: foundation tutorials -> supervised models -> assignments -> capstone prep.
- Side-by-side student + solution notebooks to support both instruction and self-paced study.
- Shared CSVs, lecture PDFs, and HTML/PDF deliverables keep grading and review reproducible.

## Table of Contents
- [Repository Structure](#repository-structure)
- [Learning Roadmap](#learning-roadmap)
- [Quick Start](#quick-start)
- [Working With Data & Notebooks](#working-with-data--notebooks)
- [Deliverables & Supporting Material](#deliverables--supporting-material)
- [Troubleshooting](#troubleshooting)
- [Contributing & Next Steps](#contributing--next-steps)
- [Attribution & License](#attribution--license)

## Repository Structure

| Path | What's inside | Highlights |
| --- | --- | --- |
| `Code/In Class/` | Core tutorial and assignment notebooks plus the source CSV files. | `2.x` fundamentals, `3-10` modeling labs, `Assignment 1-3`, `Midterm.ipynb`. |
| `Code/In Class/Deliverables/` | Rendered HTML/PDF submissions, question sheets, cheat sheets. | Ideal for grading, quick review, and sharing exemplar outputs. |
| `Code/Final Project/` | Starter folders for themed projects (KNN, Loan Defaults). | Use as templates for semester projects or demos. |
| `Lectures/` | Slide decks that mirror the notebook numbering. | Perfect for lesson prep or recap sessions. |
| `Textbooks/` | Curated references (PDF/ePub) covering Python, ML, and data mining. | Pair with weekly lecture topics for deeper dives. |
| `ECON7880 Course Outline_2025Fall_S24.pdf` | Term schedule, grading policy, and logistics. | Keeps pacing aligned with assignments and exams. |

> Tip: keep personal work in new files or branches so the distributed solution notebooks remain untouched.

## Learning Roadmap
1. **Foundations (`2.x` notebooks)** - Refresh Python syntax, NumPy arrays, and pandas DataFrames with guided examples.
2. **Classical models (`3-6`, `8-10`)** - Walk through decision trees, linear/logistic regression, SVMs, KNN/K-means, and model evaluation techniques.
3. **Assignments (`Assignment 1-3`)** - Apply the workflows to business-style prompts that reuse the shared CSVs.
4. **Evaluation focus (`9`, `10`, cheat sheets)** - Deepen understanding of metrics, ROC/PR curves, and exam-style questions.
5. **Capstone prep (`Final Project/*`, `Midterm.*`)** - Leverage the project starters and midterm materials for comprehensive assessments.

Feel free to reorder modules to match your syllabus; the dependencies are intentionally light.

## Quick Start

### Requirements
- Python 3.10 or newer
- Jupyter Lab or Notebook
- Recommended: Git, make (or Task) for scripted workflows

### Environment Setup (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install jupyter numpy pandas matplotlib seaborn scikit-learn
```
Add any extra packages (e.g., `statsmodels`, `plotly`) as you introduce them. Capture the environment once it is stable:
```powershell
pip freeze > requirements.txt
```

### Launch Jupyter
```powershell
jupyter lab "Code/In Class"
# or
jupyter notebook
```
Open a notebook, run cells top-to-bottom, and duplicate the file before experimenting so you keep the upstream solutions intact.

## Working With Data & Notebooks

All datasets live beside the teaching notebooks inside `Code/In Class/`. Load them relative to that directory to keep paths portable:

```python
from pathlib import Path
import pandas as pd

data_dir = Path("Code") / "In Class"
churn = pd.read_csv(data_dir / "churn.csv")
```

| Dataset (path) | Primary topic | Typical exercise |
| --- | --- | --- |
| `Code/In Class/advertising.csv` | Multi-feature regression | Marketing spend vs. sales forecasting. |
| `Code/In Class/churn.csv` | Binary classification | Assignments 1 & 2 (decision trees vs. logistic). |
| `Code/In Class/house_price.csv` | Numeric regression | Feature scaling and interpretation practice. |
| `Code/In Class/mushroom.csv` | Categorical preprocessing | One-hot encoding and multi-class modeling. |
| `Code/In Class/smoking.csv` | Policy analytics | Logistic regression with survey data. |
| `Code/In Class/svm_data1.csv` / `svm_data2.csv` | Margin intuition | Linear vs. kernelized SVM demos. |
| `Code/In Class/Wholesale_customers_v2.csv` | Clustering | K-means segment discovery. |

Export polished work through *File -> Export Notebook As...* to create the HTML/PDF artifacts you see under `Deliverables/`.

## Deliverables & Supporting Material
- `Code/In Class/Deliverables/*.html|pdf` - Reference submissions for assignments and the midterm.
- `Assignment* Questions*.pdf` - Student-facing prompts and rubrics.
- `Midterm (q21+) - Coding Cheat Sheet...docx`, `Econ7880 Midterm Theory Cheat Sheet...docx` - Quick lookup tables for exam prep.
- `Lectures/*.pdf` - Slides that align with each in-class lab.
- `Textbooks/*.pdf|epub` - Optional reading to reinforce weekly topics.

Use these artifacts to set expectations, grade consistently, or offer make-up material.

## Troubleshooting
- **Missing packages** - Reactivate your virtual environment and re-run the `pip install` commands above.
- **Kernel hiccups** - Restart the kernel (Kernel -> Restart & Clear Output) after altering file paths or upgrading libraries.
- **File path errors** - Prefer forward slashes or `Path` objects in Python to stay OS-agnostic.
- **Large CSV performance** - Use chunked reads (`pd.read_csv(..., chunksize=5000)`) or sample rows when demonstrating concepts live.

## Contributing & Next Steps
- Track enhancements via issues (e.g., "add tree-based ensemble lab") and document the rationale in notebook headers.
- Consider adding `environment.yml` or `requirements-lock.txt` once the stack stabilizes for a given semester.
- When sharing publicly, strip solutions into a separate branch and keep student templates in the default branch.
- If you create new datasets or case studies, drop them under `Code/In Class/` and update the table above so future cohorts know how to use them.

## Attribution & License
No explicit open-source license is included. Treat the material as educational content for the ECON7880 course and request permission before redistributing outside the class. If you plan to publish or collaborate broadly, add a LICENSE file (MIT, CC-BY-NC, etc.) and update this section accordingly.
