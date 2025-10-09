# Exchange Data Concepts

Introductory notebooks, datasets, and assignment handouts for an "Intro to Concepts for Big Data" course. The material walks through exploratory data analysis, classical machine learning models, and practice assignments using Python, NumPy, pandas, and scikit-learn inside Jupyter notebooks.

## Quick Start

1. Install Python 3.10+ and Jupyter if they are not already available.
2. Create and activate a virtual environment (optional but recommended).
3. Install the core scientific stack:
   ```bash
   pip install jupyter numpy pandas matplotlib seaborn scikit-learn
   ```
4. Launch Jupyter Lab or Notebook from the project root:
   ```bash
   jupyter lab
   ```
5. Open any of the `.ipynb` notebooks and run the cells from top to bottom.

## Repository Map

### Notebooks

| File | Focus | Notes |
| --- | --- | --- |
| `2.1 Data Type and Structure_with solution.ipynb` | Python/numpy data types and array manipulations | Includes solved practice prompts. |
| `2.2 Pandas DataFrame_with solution.ipynb` | Building and slicing pandas DataFrames | Demonstrates indexing, aggregation, and joins. |
| `3 Decision Tree_with solution.ipynb` | Decision tree modeling workflow | Covers train/test splits and model evaluation. |
| `4 Linear Regression_with solution.ipynb` | Linear regression with scikit-learn vs. statsmodels | Highlights interpretation vs. prediction use-cases. |
| `5 Logistic Regression_Student.ipynb` | Logistic regression template | Partially scaffolded; intended for in-class or homework completion. |
| `Assignment 1.ipynb` | Decision-tree case study assignment | Mirrors the PDF prompt and references the churn dataset. |

Companion HTML exports (e.g., `Assignment1 De Isa.html`, `Assignment2 De Isa.html`) capture rendered notebook output for quick review without running code.

### Datasets

| File | Columns (first row) | Typical Use |
| --- | --- | --- |
| `advertising.csv` | TV, Radio, Newspaper, Sales | Multi-feature regression practice. |
| `churn.csv` | Customer demographics, usage, churn flag | Classification exercises (assignment 1). |
| `house_price.csv` | dist, age, room, school, price | Simple regression with numeric features. |
| `mushroom.csv` | Edibility plus 19 categorical attributes | Multi-class preprocessing and classification. |
| `smoking.csv` | Smoker status, bans, demographics | Logistic regression and policy analysis. |

### Reference Material

- `Assignment 1 Questions (1).pdf` - Instructions and rubric for the first assignment.
- `Assignment2 Questions.pdf` - Worksheet for the second assignment.

## Recommended Workflow

- Start with the `2.x` tutorials to refresh Python, NumPy, and pandas fundamentals.
- Progress to the model-specific walkthroughs (`3`, `4`, `5`) before attempting the assignments.
- Use the provided CSVs directly from the repository root (`pd.read_csv('advertising.csv')`, etc.).
- Save personal work under new filenames to preserve the distributed solutions.
- Export notebooks to HTML or PDF for submission using `File > Export Notebook As...`.

## Troubleshooting Tips

- If `ModuleNotFoundError` appears, re-run `pip install` inside the active environment.
- Restart the Jupyter kernel and clear outputs when datasets are reloaded or paths change.
- On Windows, prefer forward slashes (`data/file.csv`) or raw strings (`r"data\\file.csv"`) when referencing paths.

---

This repository is intentionally lightweight and does not track environment locks. For reproducibility across semesters, capture the versions of key libraries (`pip freeze > requirements.txt`) once your environment is stable.
