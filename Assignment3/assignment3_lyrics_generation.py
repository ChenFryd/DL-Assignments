# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Assignment 3 - Lyrics Generation
#
# * Chen Frydman - 208009845
# * Amit Ner-Gaon - 211649801
#
# Ben-Gurion University of the Negev - Faculty of Computer & Information Science
#
#
# This .ipynb (for human readability) or .py (for AI readability) file is sync with the file with the same name and other extension (using `jupytext`).
#
# See `Assignment3/conda_requirement.yaml`

# %% [markdown]
# ##  Imports

# %%
# utils
from pathlib import Path
import re

# Data Manipulation
import numpy as np
import pandas as pd

# DL
import torch

# Preprocessing
import gensim.downloader as gensim_downloader

# %% [markdown]
# ## Parameters

# %%
# Data
DATA_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
TRAIN_CSV_PATH = DATA_DIR / "lyrics_train_set.csv"
TEST_CSV_PATH = DATA_DIR / "lyrics_test_set.csv"

# WORD2VEC
LYRICS_COLUMNS = ["artist", "song_name", "lyrics"]
WORD2VEC_MODEL_NAME = "word2vec-google-news-300"
WORD2VEC_DIM = 300
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_word2vec_model = None


# %% [markdown]
# ## Load Data

# %%
def load_lyrics_dataset(csv_path: Path) -> pd.DataFrame:
    """Load one assignment lyrics CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with artist, song_name, and lyrics columns.
    """
    return pd.read_csv(
        csv_path,
        header=None,
        names=LYRICS_COLUMNS,
        usecols=[0, 1, 2],
        skipinitialspace=True,
    )


def tokenize_lyrics(lyrics: str) -> list[str]:
    """Split lyrics text into lowercase word tokens."""
    return _WORD_RE.findall(str(lyrics).lower())


def lyrics_dataset_statistics(df: pd.DataFrame) -> pd.Series:
    """Compute basic statistics for a lyrics dataframe.

    Args:
        df: DataFrame with a lyrics column.

    Returns:
        Series with number of rows and average song length in words.
    """
    song_lengths = df["lyrics"].apply(lambda lyrics: len(tokenize_lyrics(lyrics)))
    return pd.Series(
        {
            "number_of_rows": len(df),
            "average_song_length_words": song_lengths.mean(),
        }
    )


# %%
train_df = load_lyrics_dataset(TRAIN_CSV_PATH)
test_df = load_lyrics_dataset(TEST_CSV_PATH)

dataset_statistics = pd.DataFrame(
    {
        "train": lyrics_dataset_statistics(train_df),
        "test": lyrics_dataset_statistics(test_df),
    }
).T

print(f"Train dataframe shape: {train_df.shape}")
print(f"Test dataframe shape: {test_df.shape}")
display(dataset_statistics)


# %% [markdown]
# ## Word2vec

# %%
def load_word2vec_model():
    """Load the online 300-dimensional Word2Vec model lazily.

    Returns:
        Gensim keyed vectors for the Google News Word2Vec model.
    """
    global _word2vec_model

    if _word2vec_model is None:
        _word2vec_model = gensim_downloader.load(WORD2VEC_MODEL_NAME)

    if _word2vec_model.vector_size != WORD2VEC_DIM:
        raise ValueError(
            f"Expected {WORD2VEC_DIM}-dimensional embeddings, "
            f"got {_word2vec_model.vector_size}."
        )

    return _word2vec_model


def word_to_word2vec(word: str, model=None) -> np.ndarray:
    """Return the 300-entry Word2Vec representation of a word.

    Args:
        word: Word to convert into a Word2Vec vector.
        model: Optional preloaded Gensim keyed vectors object.

    Returns:
        NumPy array with shape (300,).
    """
    keyed_vectors = model if model is not None else load_word2vec_model()

    for candidate in (word, word.lower(), word.title()):
        if candidate in keyed_vectors:
            return keyed_vectors[candidate]

    raise KeyError(f"Word '{word}' was not found in {WORD2VEC_MODEL_NAME}.")


# %%
# Test
music_vec_shape = word_to_word2vec("music").shape  
print (f"Word2Vec vector shape for 'music': {music_vec_shape}") # Expected output: (300,)
