# Exchange Data Concepts

Course kit for ECON7880 "Intro to Concepts for Big Data": end-to-end labs, shared datasets, lecture decks, graded artifacts, and starter material for assignments and a capstone-style project.

## What you get
- Student + solution-style notebooks that mirror the lecture sequence.
- The exact CSV/TXT/Excel files referenced in each lab and assignment.
- Ready-made HTML/PDF exports for quick grading or self-study review.
- Slide decks and textbook excerpts for deeper reading.

## Table of contents
- [Repository map](#repository-map)
- [Suggested sequence](#suggested-sequence)
- [Local setup](#local-setup)
- [Working with notebooks](#working-with-notebooks)
- [Data catalogue](#data-catalogue)
- [Deliverables and supporting docs](#deliverables-and-supporting-docs)
- [Final project starter](#final-project-starter)
- [Troubleshooting](#troubleshooting)
- [License and sharing](#license-and-sharing)

## Repository map

| Path | What's inside | Highlights |
| --- | --- | --- |
| `Code/In Class/` | Teaching notebooks plus the source CSV/TXT files. | `2.x` foundations; `3` decision trees; `4` linear regression; `5` logistic regression; `6` SVM and regularization; `8` KNN and K-means; `9` model metrics; `10` evaluation curves; `11` Naive Bayes; `12` association rules; `Assignment 1-3`; `Midterm.ipynb`. |
| `Code/In Class/Deliverables/` | Rendered HTML/PDF submissions, question sheets, and cheat sheets. | Handy for grading, exemplars, and offline review. |
| `Code/Final Project/KNN/` | Capstone starter focused on credit card defaults. | Includes the `default of credit card clients.xls` dataset plus HTML/PDF exports. |
| `Lectures/` | Slide decks aligned to the notebook numbering. | Ready-to-teach PDFs. |
| `Textbooks/` | Curated PDFs/ePubs on Python, ML, and data mining. | Pair with weekly topics. |
| `ECON7880 Course Outline_2025Fall_S24.pdf` | Schedule, grading policy, and logistics. | Keeps pacing aligned with assignments and exams. |

Tip: keep personal work in copies or a new branch so upstream solution notebooks stay untouched.

## Suggested sequence
1. Kickoff: `Lectures/1 Intro to Big Data Analytics_S2new.pdf`.
2. Foundations: `2.1` and `2.2` notebooks on types, pandas, and arrays.
3. Trees and regression: `3 Decision Tree`, `4 Linear Regression`, `5 Logistic Regression`.
4. Margins and regularization: `6 SVM and Regularization`.
5. Neighbors and clustering: `8 KNN  Kmeans`.
6. Evaluation deep dive: `9 Model Evaluation Metrics` and `10 Model Evaluation Curves`.
7. Probabilistic and market-basket methods: `11 Naive Bayes`, `12 Frequent Itemsets and Association Rules`.
8. Assignments: `Assignment 1-3.ipynb` paired with their question PDFs and deliverable exports.
9. Midterm prep: `Midterm.ipynb` plus the coding/theory cheat sheets.
10. Capstone: `Code/Final Project/KNN` starter.

Feel free to shuffle modules to match your syllabus; dependencies are intentionally light.

## Local setup

### Requirements
- Python 3.10+
- Jupyter Lab or Notebook
- Recommended: Git; optional `make`/`Invoke`/`Task` for scripting

### Create a virtual environment (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install jupyter numpy pandas matplotlib seaborn scikit-learn
# If you plan to read the Excel project data:
pip install xlrd
```
Capture the environment once stable:
```powershell
pip freeze > requirements.txt
```

## Working with notebooks
- Launch from the repo root so relative paths resolve:
  ```powershell
  jupyter lab "Code/In Class"
  # or
  jupyter notebook
  ```
- Duplicate a notebook before experimenting (e.g., append `_mywork`) to preserve the distributed solution.
- Keep paths relative; avoid hard-coding absolute Windows paths so others can run the same notebooks.
- Export polished work via *File -> Export Notebook As...* to produce the HTML/PDF artifacts mirrored under `Deliverables/`.

## Data catalogue

All teaching datasets sit beside the notebooks in `Code/In Class/`. Load them relative to that directory:

```python
from pathlib import Path
import pandas as pd

data_dir = Path("Code") / "In Class"
churn = pd.read_csv(data_dir / "churn.csv")
```

| Dataset (path) | Primary topic | Typical exercise |
| --- | --- | --- |
| `Code/In Class/advertising.csv` | Multi-feature regression | Marketing spend vs. sales forecasting. |
| `Code/In Class/churn.csv` | Binary classification | Assignments 1 & 2 (trees vs. logistic regression). |
| `Code/In Class/house_price.csv` | Numeric regression | Feature scaling and interpretation practice. |
| `Code/In Class/mushroom.csv` | Categorical preprocessing | One-hot encoding and multi-class modeling. |
| `Code/In Class/smoking.csv` | Policy analytics | Logistic regression with survey data. |
| `Code/In Class/svm_data1.csv` / `svm_data2.csv` | Margin intuition | Linear vs. kernelized SVM demos. |
| `Code/In Class/Wholesale_customers_v2.csv` | Clustering | K-means segment discovery. |
| `Code/In Class/food_drink_emoji_tweets.txt` | Text classification | Tokenization and Naive Bayes-style sentiment lab. |
| `Code/Final Project/KNN/default of credit card clients.xls` | Credit risk | KNN/EDA capstone dataset. |

## Deliverables and supporting docs
- `Code/In Class/Deliverables/*.html|pdf` — Reference submissions for assignments and the midterm.
- `Assignment* Questions*.pdf` — Student-facing prompts and rubrics.
- `Midterm (q21+) - Coding Cheat Sheet...docx` and `Econ7880 Midterm Theory Cheat Sheet...docx` — Quick lookup during review.
- `Lectures/*.pdf` — Slides aligned to each in-class lab.
- `Textbooks/*.pdf|epub` — Optional reading to reinforce weekly topics.
- `ECON7880 Course Outline_2025Fall_S24.pdf` — Administrative details and pacing.

Use these artifacts to set expectations, grade consistently, or share make-up material.

## Final project starter
- Location: `Code/Final Project/KNN`.
- Data: `default of credit card clients.xls` (UCI credit default).
- Deliverables: `KNN_FinalProj.ipynb` with matching HTML/PDF for reference.
- Suggested steps: replicate the starter analysis, then extend with your own features, scaling choices, and model comparisons.

## Troubleshooting
- Missing packages: reactivate `.venv` and rerun the `pip install` commands above.
- Kernel hiccups: restart and clear output after changing file paths or upgrading libraries.
- Path errors: prefer forward slashes or `Path` objects to stay OS-agnostic.
- Large CSV/TXT performance: use chunked reads (`pd.read_csv(..., chunksize=5000)`) or sample rows during live demos.

## License and sharing
No explicit open-source license is included. Treat the material as educational content for ECON7880 and request permission before redistributing outside the class. If you plan to publish or collaborate broadly, add a LICENSE file (MIT, CC-BY-NC, etc.) and update this section accordingly.
