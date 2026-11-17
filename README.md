# IBM Article Recommendation Systems

A notebook project for the Udacity Data Scientist Nanodegree that explores article recommendations using user interactions from the IBM Watson Studio community. The data records which articles users interacted with rather than explicit ratings.

The main notebook is [`starter/Recommendations_with_IBM.ipynb`](starter/Recommendations_with_IBM.ipynb).

## Project workflow

The notebook covers:

1. **Exploratory data analysis:** inspect missing values, user activity, article popularity, and interaction counts; map user emails to numeric IDs.
2. **Rank-based recommendations:** recommend the most popular articles, including for new users without interaction history.
3. **User-user collaborative filtering:** build a binary user–article matrix and recommend unread articles based on similar users, with activity and popularity used to resolve ties.
4. **Content-based recommendations:** represent article titles with TF-IDF, reduce their dimensions with truncated SVD and normalization, and cluster related titles with K-means.
5. **Matrix factorization:** use truncated SVD to learn latent features, examine reconstruction accuracy, precision, and recall, and find similar articles using cosine similarity.

## Current status

This repository contains a starter notebook with some initial data exploration. Many cells still contain incomplete assignments, function bodies, or written-response prompts. It cannot currently run from start to finish, and no completed recommendation results are claimed here.

The dataset and course-provided test helper are also absent from this repository. Obtain them from the project materials before running the notebook.

## Setup

Use Python 3 and create a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

The requirements include pandas, NumPy, Matplotlib, and scikit-learn, plus Jupyter Notebook and nbconvert for opening and exporting the notebook. Versions are unpinned because the notebook does not specify a validated dependency environment.

`project_tests` is a local course helper, so it is not included as a pip dependency. The `subprocess` import is part of Python's standard library and requires no installation.

## Required files

Arrange the project files as follows:

```text
dsnd-recommendation-systems-project/
├── Readme.md
├── requirements.txt
└── starter/
    ├── Recommendations_with_IBM.ipynb
    ├── project_tests.py                 # Obtain from the course materials
    └── data/
        └── user-item-interactions.csv   # Obtain from the course materials
```

The notebook reads `data/user-item-interactions.csv` relative to its working directory. The CSV must include `article_id`, `title`, and `email` columns. Include any supporting files required by the supplied test helper as well.

## Run and validate

With the virtual environment active, launch Jupyter from `starter/` so the notebook's relative data path resolves correctly:

```bash
cd starter
jupyter notebook Recommendations_with_IBM.ipynb
```

Complete the placeholder code and written responses, then run the cells in order. The notebook includes assertions and calls to `project_tests` for checking the analysis and recommendation functions.

To export a saved notebook to HTML, run this command from `starter/`:

```bash
python -m nbconvert --to html Recommendations_with_IBM.ipynb
```

This creates `Recommendations_with_IBM.html`; it exports the saved notebook without executing its cells.
