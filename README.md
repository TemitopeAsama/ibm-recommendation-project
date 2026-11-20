# IBM Article Recommendation Systems

An article recommendation project for the Udacity Data Scientist Nanodegree, using interaction data from the IBM Watson Studio community. The completed notebook explores popularity-based recommendations, user-user collaborative filtering, content-based recommendations, and SVD matrix factorization.

The dataset records article interactions rather than explicit ratings. The project examines how each method can serve users with different amounts of reading history.

## Dataset and findings

The supplied dataset contains:

- **45,993 interactions** across **714 articles**.
- **5,149 users**, including one shared identity for missing emails.
- **17 interactions with missing emails**, assigned to `unknown_user` before mapping emails to numeric user IDs.
- A median of **3 interactions per user** and a maximum of **364**.
- **937 interactions** with the most popular article, article ID **1429**.

Repeated reads count as separate interactions in the popularity analysis. Collaborative filtering uses a binary user–article matrix that records whether a user has read each article.

## Recommendation methods

| Method | Implementation | Use case |
| --- | --- | --- |
| Popularity-based | Rank articles by interaction frequency. | New users with no reading history. |
| User-user collaborative filtering | Find similar users and recommend unread articles. The basic version uses dot products; the ranked version uses cosine similarity, distinct-article counts to break user ties, and total interaction frequency to rank candidate articles. | Users with an interaction history. |
| Content-based | Represent titles with TF-IDF, reduce to 50 dimensions with truncated SVD, normalize, and group into 50 K-means clusters. Recommend other articles in the same cluster, ranked by unique-reader counts. | Suggestions related to a selected article. |
| Matrix factorization | Factor the user–article matrix with truncated SVD and use 200 latent features to find similar articles through cosine similarity. | Discovering relationships in reading patterns. |

Both user-user recommendation functions fall back to popular articles when the user is unknown.

## Project files

```text
dsnd-recommendation-systems-project/
├── README.md
├── requirements.txt
├── notes.md
└── starter/
    ├── Recommendations_with_IBM.ipynb  # Completed analysis, functions, and responses
    ├── project_tests.py               # Course test helper
    ├── top_5.p                        # Expected popularity results
    ├── top_10.p
    ├── top_20.p
    └── data/
        └── user-item-interactions.csv # Obtain from the course materials
```

The dataset is excluded from Git. The course helper and its three reference files are included. Generated HTML/PDF exports, virtual environments, and Python caches are also excluded from Git.

## Setup

Create and activate a Python 3 virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Obtain `user-item-interactions.csv` from the project materials and place it in `starter/data/`. It must contain the `article_id`, `title`, and `email` columns.

The project uses pandas, NumPy, Matplotlib, scikit-learn, Jupyter Notebook, and nbconvert. Dependency versions are currently unpinned.

## Run and validate

Launch Jupyter from `starter/` so the notebook's relative dataset and test-file paths resolve correctly:

```bash
cd starter
jupyter notebook Recommendations_with_IBM.ipynb
```

Select the project environment as the notebook kernel, restart the kernel, and run all cells in order. Save the notebook after execution to retain the plots and recommendation outputs.

The notebook includes course helper checks and assertions for the dataset statistics, popular articles, binary interaction matrix, article lookups, ranked neighbors, new-user recommendations, content recommendations, and SVD article similarities. The Part III ranked-neighbor implementation matches the course helper's expected results.

## Export to HTML

After running and saving the notebook, run this command from `starter/`:

```bash
python -m nbconvert --to html Recommendations_with_IBM.ipynb
```

This creates `Recommendations_with_IBM.html` from the saved notebook without rerunning its cells. The final notebook cell also provides an HTML export command.

## Evaluation and limitations

The notebook compares reconstruction accuracy, precision, and recall as the number of latent features increases. These are measurements on the interaction matrix, not evidence that one recommendation method performs better on unseen interactions.

The written discussion proposes held-out evaluation with precision@10 and recall@10, followed by an online A/B test. These evaluations have not been implemented. The dataset has no timestamps, so a chronological split cannot be made from the supplied columns.

Other limitations include sparse user histories, title-only content features, popularity bias, and the inability of the fitted SVD model to directly represent unseen users or articles. The notebook discusses combining methods to address different user-history scenarios.

## Acknowledgments

This project uses the IBM Watson Studio community dataset and starter materials supplied through the Udacity Data Scientist Nanodegree.
