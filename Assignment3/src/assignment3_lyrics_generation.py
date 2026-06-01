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
import os
from pathlib import Path
import re
import warnings

DATA_DIR = Path("../data")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
Path("/tmp/matplotlib-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

# Data Manipulation
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
from IPython.display import display


# DL
import torch

# Preprocessing
import gensim.downloader as gensim_downloader
from gensim.models import KeyedVectors
import mido
import pretty_midi

# %% [markdown]
# ## Parameters

# %%
# Data
TRAIN_CSV_PATH = DATA_DIR / "lyrics_train_set.csv"
TEST_CSV_PATH = DATA_DIR / "lyrics_test_set.csv"
MIDI_DIR = DATA_DIR / "midi_files"

# WORD2VEC
LYRICS_COLUMNS = ["artist", "song_name", "lyrics"]
WORD2VEC_MODEL_NAME = "word2vec-google-news-300"
WORD2VEC_DIM = 300
WORD2VEC_MODELS_DIR = RESULTS_DIR / "word2vec_models"
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_word2vec_model = None
_word2vec_model_name = None

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
def get_word2vec_model_path(model_name: str = WORD2VEC_MODEL_NAME) -> Path:
    """Return the local path for a cached Word2Vec model.

    Args:
        model_name: Name of the Gensim downloader model.

    Returns:
        Path to the local keyed-vectors file.
    """
    return WORD2VEC_MODELS_DIR / model_name / "model.kv"


def load_word2vec_model(model_name: str = WORD2VEC_MODEL_NAME) -> KeyedVectors:
    """Load a 300-dimensional Word2Vec model lazily.

    The loader first tries to read the model from
    ``results/word2vec_models/<model_name>/model.kv``. If it is missing, the
    model is downloaded with Gensim and saved there for future runs.

    Args:
        model_name: Name of the Gensim downloader model.

    Returns:
        Gensim keyed vectors for the requested Word2Vec model.
    """
    global _word2vec_model, _word2vec_model_name

    if _word2vec_model is None or _word2vec_model_name != model_name:
        model_path = get_word2vec_model_path(model_name)

        if model_path.exists():
            _word2vec_model = KeyedVectors.load(str(model_path), mmap="r")
        else:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            _word2vec_model = gensim_downloader.load(model_name)
            _word2vec_model.save(str(model_path))
        _word2vec_model_name = model_name

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
# The first real call downloads a large model. Uncomment to test manually:
# music_vec_shape = word_to_word2vec("music").shape
# print(f"Word2Vec vector shape for 'music': {music_vec_shape}")  # Expected: (300,)


# %% [markdown]
# ## MIDI Files

# %% [markdown]
# ### Load MIDI

# %%
def load_one_midi_file(midi_path: Path) -> pretty_midi.PrettyMIDI:
    """Load a MIDI file, clipping invalid bytes if needed.

    Args:
        midi_path: Path to one .mid file.

    Returns:
        Loaded PrettyMIDI object.
    """
    try:
        return pretty_midi.PrettyMIDI(str(midi_path))
    except ValueError:
        clipped_midi = mido.MidiFile(str(midi_path), clip=True)
        return pretty_midi.PrettyMIDI(mido_object=clipped_midi)


def load_midi_files(midi_dir: Path) -> tuple[dict[Path, pretty_midi.PrettyMIDI], pd.DataFrame]:
    """Load all MIDI files from a directory.

    Args:
        midi_dir: Directory containing .mid files.

    Returns:
        Dictionary mapping each MIDI path to its loaded PrettyMIDI object, and a
        dataframe describing files that could not be loaded.
    """
    midi_paths = sorted(midi_dir.glob("*.mid"))
    if not midi_paths:
        raise FileNotFoundError(f"No .mid files found in {midi_dir}.")

    loaded_midi_files = {}
    load_errors = []

    for midi_path in midi_paths:
        try:
            loaded_midi_files[midi_path] = load_one_midi_file(midi_path)
        except Exception as error:
            load_errors.append({"file_name": midi_path.name, "error": repr(error)})

    if not loaded_midi_files:
        raise RuntimeError(f"Could not load any MIDI files from {midi_dir}.")

    if load_errors:
        warnings.warn(f"Skipped {len(load_errors)} MIDI files that could not be loaded.")

    return loaded_midi_files, pd.DataFrame(load_errors)


# %% [markdown]
# ### MIDI Statistics

# %%
def midi_file_statistics(midi_path: Path, midi_data: pretty_midi.PrettyMIDI) -> dict:
    """Extract compact statistics from one loaded MIDI file.

    Args:
        midi_path: Path of the MIDI file.
        midi_data: Loaded PrettyMIDI object.

    Returns:
        Dictionary of numeric and categorical MIDI statistics.
    """
    instruments = midi_data.instruments
    notes = [note for instrument in instruments for note in instrument.notes]
    pitches = np.array([note.pitch for note in notes], dtype=float)
    velocities = np.array([note.velocity for note in notes], dtype=float)
    note_durations = np.array([note.end - note.start for note in notes], dtype=float)
    tempo_times, tempi = midi_data.get_tempo_changes()

    duration_seconds = midi_data.get_end_time()
    non_drum_instruments = [instrument for instrument in instruments if not instrument.is_drum]
    drum_instruments = [instrument for instrument in instruments if instrument.is_drum]

    return {
        "file_name": midi_path.name,
        "song_key": midi_path.stem.lower(),
        "duration_seconds": duration_seconds,
        "num_instruments": len(instruments),
        "num_non_drum_instruments": len(non_drum_instruments),
        "num_drum_tracks": len(drum_instruments),
        "num_notes": len(notes),
        "notes_per_second": len(notes) / duration_seconds if duration_seconds > 0 else np.nan,
        "avg_pitch": pitches.mean() if len(pitches) else np.nan,
        "min_pitch": pitches.min() if len(pitches) else np.nan,
        "max_pitch": pitches.max() if len(pitches) else np.nan,
        "avg_velocity": velocities.mean() if len(velocities) else np.nan,
        "avg_note_duration_seconds": note_durations.mean() if len(note_durations) else np.nan,
        "avg_tempo_bpm": np.mean(tempi) if len(tempi) else np.nan,
        "num_tempo_changes": len(tempo_times),
        "num_key_signature_changes": len(midi_data.key_signature_changes),
        "num_time_signature_changes": len(midi_data.time_signature_changes),
    }


def midi_statistics_dataframe(midi_files: dict[Path, pretty_midi.PrettyMIDI]) -> pd.DataFrame:
    """Build a dataframe with one row per MIDI file."""
    rows = [
        midi_file_statistics(midi_path, midi_data)
        for midi_path, midi_data in midi_files.items()
    ]
    return pd.DataFrame(rows).sort_values("file_name").reset_index(drop=True)


def instrument_usage_dataframe(midi_files: dict[Path, pretty_midi.PrettyMIDI]) -> pd.DataFrame:
    """Count General MIDI instrument usage across loaded files."""
    rows = []
    for midi_path, midi_data in midi_files.items():
        for instrument in midi_data.instruments:
            instrument_name = "Drums" if instrument.is_drum else pretty_midi.program_to_instrument_name(
                instrument.program
            )
            rows.append(
                {
                    "file_name": midi_path.name,
                    "instrument_name": instrument_name,
                    "num_notes": len(instrument.notes),
                    "is_drum": instrument.is_drum,
                }
            )

    return pd.DataFrame(rows)


def all_midi_notes_dataframe(midi_files: dict[Path, pretty_midi.PrettyMIDI]) -> pd.DataFrame:
    """Collect all MIDI notes into a tidy dataframe."""
    rows = []
    for midi_path, midi_data in midi_files.items():
        for instrument_index, instrument in enumerate(midi_data.instruments):
            instrument_name = "Drums" if instrument.is_drum else pretty_midi.program_to_instrument_name(
                instrument.program
            )
            for note in instrument.notes:
                rows.append(
                    {
                        "file_name": midi_path.name,
                        "instrument_index": instrument_index,
                        "instrument_name": instrument_name,
                        "pitch": note.pitch,
                        "velocity": note.velocity,
                        "start": note.start,
                        "end": note.end,
                        "duration": note.end - note.start,
                        "is_drum": instrument.is_drum,
                    }
                )

    return pd.DataFrame(rows)


def save_current_figure(file_name: str) -> Path:
    """Save the active Matplotlib figure into the results directory."""
    output_path = RESULTS_DIR / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def format_midi_figure(
    title: str,
    xlabel: str,
    ylabel: str,
    comma_x_axis: bool = False,
    comma_y_axis: bool = False,
) -> None:
    """Apply readable styling and comma-separated tick formatting."""
    axis = plt.gca()
    axis.set_title(title, fontsize=20, pad=16)
    axis.set_xlabel(xlabel, fontsize=16, labelpad=10)
    axis.set_ylabel(ylabel, fontsize=16, labelpad=10)
    axis.tick_params(axis="both", labelsize=13)

    if comma_x_axis:
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    if comma_y_axis:
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def save_midi_figures(
    midi_stats: pd.DataFrame,
    instrument_usage: pd.DataFrame,
    midi_notes: pd.DataFrame,
) -> list[Path]:
    """Create and save aggregate MIDI-analysis figures.

    Args:
        midi_stats: One-row-per-file MIDI statistics dataframe.
        instrument_usage: One-row-per-instrument-track dataframe.
        midi_notes: One-row-per-note dataframe.

    Returns:
        List of saved figure paths.
    """
    saved_paths = []

    plt.figure(figsize=(9, 5))
    sns.histplot(midi_stats["duration_seconds"], bins=35, kde=True, color="#2f6f73")
    format_midi_figure(
        title="MIDI Song Duration Distribution",
        xlabel="Duration (seconds)",
        ylabel="Number of songs",
        comma_x_axis=True,
        comma_y_axis=True,
    )
    saved_paths.append(save_current_figure("midi_duration_distribution.png"))

    plt.figure(figsize=(9, 5))
    sns.histplot(midi_stats["num_notes"], bins=35, kde=True, color="#8a5a44")
    format_midi_figure(
        title="Number of Notes per MIDI File",
        xlabel="Number of notes",
        ylabel="Number of songs",
        comma_x_axis=True,
        comma_y_axis=True,
    )
    saved_paths.append(save_current_figure("midi_note_count_distribution.png"))

    plt.figure(figsize=(9, 5))
    sns.histplot(midi_stats["num_instruments"], bins=range(1, int(midi_stats["num_instruments"].max()) + 2), color="#6c6f93")
    format_midi_figure(
        title="Instrument Tracks per MIDI File",
        xlabel="Number of instrument tracks",
        ylabel="Number of songs",
        comma_y_axis=True,
    )
    saved_paths.append(save_current_figure("midi_instrument_count_distribution.png"))

    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=midi_stats,
        x="duration_seconds",
        y="notes_per_second",
        size="num_instruments",
        hue="num_drum_tracks",
        palette="viridis",
        alpha=0.75,
        sizes=(20, 160),
    )
    format_midi_figure(
        title="MIDI Density: Duration vs. Notes per Second",
        xlabel="Duration (seconds)",
        ylabel="Notes per second",
        comma_x_axis=True,
    )
    plt.legend(title="Tracks", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12, title_fontsize=13)
    saved_paths.append(save_current_figure("midi_density_scatter.png"))

    top_instruments = (
        instrument_usage.groupby("instrument_name")["num_notes"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_instruments, y="instrument_name", x="num_notes", color="#4f7cac")
    format_midi_figure(
        title="Top MIDI Instruments by Number of Notes",
        xlabel="Total notes",
        ylabel="Instrument",
        comma_x_axis=True,
    )
    saved_paths.append(save_current_figure("midi_top_instruments.png"))

    plt.figure(figsize=(10, 5))
    sns.histplot(data=midi_notes, x="pitch", bins=range(0, 129), color="#7f5f9f")
    format_midi_figure(
        title="MIDI Pitch Distribution",
        xlabel="MIDI pitch",
        ylabel="Number of notes",
        comma_y_axis=True,
    )
    saved_paths.append(save_current_figure("midi_pitch_distribution.png"))

    plt.figure(figsize=(9, 5))
    sns.histplot(midi_stats["avg_tempo_bpm"].dropna(), bins=30, kde=True, color="#b85c38")
    format_midi_figure(
        title="Average Tempo Distribution",
        xlabel="Average tempo (BPM)",
        ylabel="Number of songs",
        comma_y_axis=True,
    )
    saved_paths.append(save_current_figure("midi_tempo_distribution.png"))

    return saved_paths

# %%
midi_files, midi_load_errors_df = load_midi_files(MIDI_DIR)
midi_stats_df = midi_statistics_dataframe(midi_files)
midi_instrument_usage_df = instrument_usage_dataframe(midi_files)
midi_notes_df = all_midi_notes_dataframe(midi_files)

midi_stats_summary = midi_stats_df[
    [
        "duration_seconds",
        "num_instruments",
        "num_drum_tracks",
        "num_notes",
        "notes_per_second",
        "avg_pitch",
        "avg_velocity",
        "avg_tempo_bpm",
    ]
].describe().T

midi_stats_csv_path = RESULTS_DIR / "midi_statistics.csv"
midi_load_errors_csv_path = RESULTS_DIR / "midi_load_errors.csv"
midi_stats_df.to_csv(midi_stats_csv_path, index=False)
midi_load_errors_df.to_csv(midi_load_errors_csv_path, index=False)

saved_midi_figure_paths = save_midi_figures(
    midi_stats_df,
    midi_instrument_usage_df,
    midi_notes_df,
)

print(f"Loaded MIDI files: {len(midi_files)}")
print(f"Saved MIDI statistics to: {midi_stats_csv_path}")
print(f"Saved MIDI load errors to: {midi_load_errors_csv_path}")
print("Saved MIDI figures:")
for figure_path in saved_midi_figure_paths:
    print(f"- {figure_path}")

display(midi_stats_summary)
