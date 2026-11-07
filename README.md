# Anime recommendations with Milvus

In this small vector-search project I take the anime embeddings from my [embedding-based recommender](https://github.com/Amable-Valdes/ai-anime-recommendation-system-example), load them into [Milvus](https://milvus.io/), a vector database, and query them by cosine similarity.

Author: [Amable Valdés](https://github.com/Amable-Valdes)

## Table of contents

- [About the project](#about-the-project)
- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Running the pipeline](#running-the-pipeline)
- [Inspecting the stack](#inspecting-the-stack)
- [Known issues](#known-issues)
- [Related projects](#related-projects)

## About the project

In the previous project I turned about 29,000 anime from the [Anime Data Set for ML](https://www.kaggle.com/datasets/wiltheman/anime-data-set-for-ml) into numeric vectors. Those vectors were already enough to find similar anime by cosine similarity.

This time I wanted to answer a follow-up question: what changes when the vectors live in a database built for them? Milvus stores the embeddings in a collection with a vector index, so a search becomes a single API call. In this project I:

- create a database and a collection with a typed schema,
- insert vectors together with their metadata,
- run nearest-neighbour searches with cosine similarity,
- add new vectors to a collection that already exists.

I walked through the whole process in [`execution_in_markdown.md`](execution_in_markdown.md), with all the outputs. The same steps are also in [`main_notebook.ipynb`](main_notebook.ipynb). Note that the outputs in `execution_in_markdown.md` were produced before the two bug fixes described in [Known issues](#known-issues), so the IDs and row counts there differ slightly from what the scripts print now.

## How it works

```text
dataset/embeddings_df.csv ──┐
                            ├─► Python scripts ──► Milvus standalone ──► etcd (metadata)
dataset/65k_anime_data.csv ─┘   (pymilvus)          port 19530          MinIO (vector storage)
```

- **Dataset.** `embeddings_df.csv` holds one 480-dimensional vector per anime, indexed by an integer ID. `65k_anime_data.csv` holds the metadata, including the title in the `title` column. Row *n* of one file matches row *n* of the other.
- **Milvus.** I run it as a single container (`milvus-standalone`), with etcd for metadata and MinIO for object storage. `docker-compose.yml` starts all three.
- **Collection.** The data goes into `anime_embeddings_collection` in the `anime_database` database:

  | Field | Type | Notes |
  | --- | --- | --- |
  | `anime_index` | `INT64` | Primary key, the anime's row ID |
  | `anime_name` | `VARCHAR(512)` | Title |
  | `embedding` | `FLOAT_VECTOR(480)` | Indexed with `AUTOINDEX`, metric `COSINE` |

  I enabled dynamic fields on the schema, so we can add extra attributes later without changing it.

## Repository structure

```text
.
├── as_python_scripts/
│   ├── 1_populate_bbdd.py               # Create the database and collection, insert the data
│   ├── 2_search_one.py                  # Search the 11 nearest neighbours of one anime
│   ├── 3_pipeline_bring_new_data.py     # Insert anime that were held out of step 1
│   └── 4_search_the_same_again.py       # Same search as step 2, after step 3
├── dataset/
│   ├── 65k_anime_data.csv               # Anime metadata (Git LFS)
│   └── embeddings_df.csv                # 480-dimensional embeddings (Git LFS)
├── docker-compose.yml                   # Milvus, etcd and MinIO
├── requirements.txt                     # Python dependencies
├── main_notebook.ipynb                  # The walkthrough as a notebook
├── execution_in_markdown.md             # The walkthrough exported to Markdown, with outputs
└── README.md
```

## Getting started

### Prerequisites

- **Docker** with the Compose plugin (`docker compose version` should work).
- **Python 3.12 or newer.** My scripts use f-strings with nested quotes, which need 3.12+. It was tested everything with Python 3.14.
- **Git LFS.** I have stored the two CSV files with Git LFS. Without it you get small text pointers instead of data, and the scripts fail.

```bash
git lfs install
git lfs pull
```

### Installation

```bash
git clone https://github.com/Amable-Valdes/milvus-using-anime-example.git
cd milvus-using-anime-example

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` only lists my direct dependencies: `pandas`, `numpy` and `pymilvus`.

### Starting Milvus

```bash
docker compose up -d
```

Give the containers a moment to become healthy. It usually takes under a minute:

```bash
docker compose ps
```

To stop the stack, run `docker compose down`. The data lives in `./volumes/`, so it survives restarts. Delete that folder if you want to start from scratch.

## Running the pipeline

Run the scripts from the project root, because they read `./dataset/...` with relative paths. Run them in this order:

```bash
python as_python_scripts/1_populate_bbdd.py
python as_python_scripts/2_search_one.py
python as_python_scripts/3_pipeline_bring_new_data.py
python as_python_scripts/4_search_the_same_again.py
```

| Script | What it does | Safe to re-run? |
| --- | --- | --- |
| `1_populate_bbdd.py` | Drops `anime_database` if it exists, then creates the database, the collection and the index, and inserts the dataset, holding out four rows (9, 2790, 1896 and 1344). | Yes. It starts from scratch each time. |
| `2_search_one.py` | Searches the 11 nearest neighbours of the vector at row 9 (*Monster*). | Yes, read-only. |
| `3_pipeline_bring_new_data.py` | Upserts the three held-out anime (2790, 1896 and 1344) into the existing collection. | Yes. An upsert replaces rows with the same ID, so nothing is duplicated. |
| `4_search_the_same_again.py` | Repeats the search from step 2 on the updated collection. | Yes, read-only. |

Here is an example of the output from step 4, after step 3 has added the held-out anime back (cosine similarity, higher means more similar):

```text
{'anime_index': 2790, 'distance': 0.9991167783737183, 'entity': {'anime_name': 'Warau Salesman'}}
{'anime_index': 1294, 'distance': 0.9990630149841309, 'entity': {'anime_name': 'Lupin III: Part II'}}
{'anime_index': 1319, 'distance': 0.9990367889404297, 'entity': {'anime_name': 'Maison Ikkoku'}}
...
```

*Monster* does not appear in the results. I hold it out of the database in step 1 so I can use it as the query.

## Inspecting the stack

| Service | URL | Credentials |
| --- | --- | --- |
| Milvus web UI | http://localhost:9091/webui | `root` / `Milvus` |
| MinIO console | http://localhost:9001 | `minioadmin` / `minioadmin` |
| Milvus gRPC (used by the scripts) | `localhost:19530` | token `root:Milvus` |
| MinIO API | http://localhost:9000 | `minioadmin` / `minioadmin` |

These are the defaults I use for a local test deployment. Do not expose these ports to a network or internet if you want to do production things ;)

## Known issues

I fixed two bugs from my original scripts:

- **Step 1 held out the wrong anime.** It used to call `pop()` four times in a row, and each removal shifted the rows after it, so it held out IDs 9, 2791, 1897 and 1345 instead of 9, 2790, 1896 and 1344. Step 1 now filters out the four intended rows in one pass.
- **Step 3 duplicated rows on every run.** It used `insert`, which does not replace existing primary keys. It now uses `upsert`, so I can re-run it safely. I checked this: after two extra runs the collection still holds 29,152 rows, one per anime.

The known issues that remain:

- **`2_search_one.py` and `4_search_the_same_again.py` are identical.** I kept them as two separate steps to show you the same query before and after step 3 :)
- **MinIO image.** My original compose file pinned `minio/minio:RELEASE.2024-12-18T13-15-44Z`, which is no longer available on Docker Hub. The compose file now uses `chainguard/minio:latest`, which is not pinned to a version. I should pin it to a specific tag for reproducible setups.
- **Dependency pins.** I only pin the direct dependencies. Transitive versions, such as `protobuf` and `milvus-lite`, come from `pymilvus` and may change between installs.

## Related projects

- [ai-anime-recommendation-system-example](https://github.com/Amable-Valdes/ai-anime-recommendation-system-example): the notebook where I build the anime embeddings and the cosine-similarity recommender that this project extends.

## Acknowledgements

- [wiltheman](https://www.kaggle.com/wiltheman) for the [Anime Data Set for ML](https://www.kaggle.com/datasets/wiltheman/anime-data-set-for-ml) on Kaggle.
- [MyAnimeList](https://myanimelist.net/) as the original source of the data.
- The [Milvus](https://milvus.io/) team for the vector database and its documentation.
