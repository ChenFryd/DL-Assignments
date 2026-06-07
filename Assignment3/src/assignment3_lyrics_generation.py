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
import matplotlib
try:
    get_ipython  # type: ignore[name-defined]  # noqa: F821
    IN_JUPYTER = True
except NameError:
    matplotlib.use("Agg")  # headless / script mode: no display available
    IN_JUPYTER = False
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
from IPython.display import Audio, display
from tqdm import tqdm


# DL
import torch
from torch.utils.data import Dataset

# WORD2VEC
import gensim.downloader as gensim_downloader
from gensim.models import KeyedVectors

# MIDI
import mido
import pretty_midi

# %% [markdown]
# ## Parameters

# %%
lookback = 7 # Number of previous lyrics to consider for prediction

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
display(train_df.head())
test_df = load_lyrics_dataset(TEST_CSV_PATH)
display(test_df.head())

# %%
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
# The first real call downloads a large model.
music_vec_shape = word_to_word2vec("music").shape
print(f"Word2Vec vector shape for 'music': {music_vec_shape}")  # Expected: (300,)


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
    """Save the active Matplotlib figure into the results directory.

    In Jupyter the figure stays open so the inline backend auto-displays it.
    In script mode it is closed to free memory.
    """
    output_path = RESULTS_DIR / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    if not IN_JUPYTER:
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


# %% [markdown]
# ### Midi To Notes
#
# You will use three variables to represent a note when training the model: **pitch, step and duration**. The pitch is the perceptual quality of the sound as a MIDI note number. The step is the time elapsed from the previous note or start of the track. The duration is how long the note will be playing in seconds and is the difference between the note end and note start times. https://www.tensorflow.org/tutorials/audio/music_generation .

# %%
def one_midi_to_notes_df(midi: pretty_midi.PrettyMIDI) -> pd.DataFrame:
    rows = []

    for instrument in midi.instruments:
        for note in instrument.notes:
            rows.append({
                "instrument_id": instrument.program,
                "instrument_name": instrument.name,
                "is_drum": instrument.is_drum,
                "pitch (note_id)": note.pitch,
                "note_name": pretty_midi.note_number_to_name(note.pitch) if not instrument.is_drum else pretty_midi.note_number_to_drum_name(note.pitch),
                "start": note.start,
                "end": note.end,
                "duration": note.end - note.start,
                "velocity": note.velocity,
            })
    
    notes_df = pd.DataFrame(rows)

    notes_df = notes_df.sort_values(["start", "end"]).reset_index(drop=True)
    notes_df["time_from_last_note_start"] = notes_df["start"].diff().fillna(0)


    return notes_df


# %%
def midi_files_to_notes_df(midi_folder_path: Path) -> pd.DataFrame:
        
    loaded_midis, _ = load_midi_files(midi_folder_path)  # returns (dict, errors_df)

    all_dfs = []

    for midi_path, midi in tqdm(loaded_midis.items()):
        df = one_midi_to_notes_df(midi)

        df["midi_path"] = midi_path
        df["midi_file"] = midi_path.name

        all_dfs.append(df)

    if len(all_dfs) == 0:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


# %%
example_path = MIDI_DIR / "2_Unlimited_-_Get_Ready_for_This.mid"
midi = load_one_midi_file(example_path)

df = one_midi_to_notes_df(midi)
print(f" df shape: {df.shape}")
display(df.head(20))


audio = midi.synthesize(fs=16000)
display(Audio(audio, rate=16000))

# %% [markdown]
# ### Midi to Features

# %%
def extract_midi_features(midi_path: str) -> np.ndarray:

    midi = load_one_midi_file(Path(midi_path))

    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        notes.extend(instrument.notes)

    if len(notes) == 0:
        return np.zeros(5, dtype=np.float32)

    pitches = np.array([note.pitch for note in notes])
    durations = np.array([note.end - note.start for note in notes])
    velocities = np.array([note.velocity for note in notes])

    features = np.array([
        pitches.mean(),
        pitches.std(),
        durations.mean(),
        velocities.mean(),
        len(notes),
    ], dtype=np.float32)

    return features


# %%
def song_to_midi_file_name(artist: str, song_name: str) -> str:
    """Convert artist and song names to the assignment MIDI filename pattern.

    Args:
        artist: Song artist.
        song_name: Song title.

    Returns:
        Expected MIDI filename, such as ``Billy_Joel_-_Piano_Man.mid``.
    """
    artist_part = str(artist).strip().title().replace(" ", "_")
    song_part = str(song_name).strip().title().replace(" ", "_")
    return f"{artist_part}_-_{song_part}.mid"


def normalize_song_key(artist: str, song_name: str) -> tuple[str, str]:
    """Create a case-insensitive matching key for a song."""
    return (
        re.sub(r"[^a-z0-9]+", "", str(artist).lower()),
        re.sub(r"[^a-z0-9]+", "", str(song_name).lower()),
    )


def build_midi_path_lookup(midi_dir: Path) -> dict[tuple[str, str], Path]:
    """Index MIDI paths by normalized artist and song name."""
    midi_lookup = {}

    for midi_path in midi_dir.glob("*.mid"):
        if "_-_" not in midi_path.stem:
            continue

        artist_part, song_part = midi_path.stem.split("_-_", maxsplit=1)
        key = normalize_song_key(
            artist_part.replace("_", " "),
            song_part.replace("_", " "),
        )
        midi_lookup[key] = midi_path

    return midi_lookup


def find_midi_path(
    artist: str,
    song_name: str,
    midi_path_lookup: dict[tuple[str, str], Path],
) -> Path | None:
    """Find the best MIDI path for an artist/song pair."""
    artist_key, song_key = normalize_song_key(artist, song_name)
    exact_path = midi_path_lookup.get((artist_key, song_key))
    if exact_path is not None:
        return exact_path

    artist_matches = [
        (midi_song_key, midi_path)
        for (midi_artist_key, midi_song_key), midi_path in midi_path_lookup.items()
        if midi_artist_key == artist_key
    ]
    prefix_matches = [
        (midi_song_key, midi_path)
        for midi_song_key, midi_path in artist_matches
        if midi_song_key.startswith(song_key) or song_key.startswith(midi_song_key)
    ]

    if not prefix_matches:
        return None

    return sorted(prefix_matches, key=lambda item: len(item[0]))[0][1]


def build_midi_features_dataframe(
    lyrics_df: pd.DataFrame,
    midi_dir: Path = MIDI_DIR,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """Build a dataframe of MIDI feature vectors for songs in a lyrics dataframe.

    Args:
        lyrics_df: DataFrame with ``artist`` and ``song_name`` columns.
        midi_dir: Directory containing the MIDI files.
        raise_on_error: If True, fail when a MIDI file is missing or malformed.
            If False, keep the row and use a NaN feature vector while warning.

    Returns:
        DataFrame indexed by artist and song_name with a midi_features column.
    """
    required_columns = {"artist", "song_name"}
    missing_columns = required_columns - set(lyrics_df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")

    rows = []
    midi_path_lookup = build_midi_path_lookup(midi_dir)

    for row in lyrics_df[["artist", "song_name"]].itertuples(index=False):
        midi_path = find_midi_path(row.artist, row.song_name, midi_path_lookup)

        try:
            if midi_path is None:
                expected_file_name = song_to_midi_file_name(row.artist, row.song_name)
                raise FileNotFoundError(f"Could not find MIDI file like {expected_file_name}")

            midi_features = extract_midi_features(str(midi_path))
        except Exception as error:
            if raise_on_error:
                raise RuntimeError(
                    f"Could not extract MIDI features for {row.artist} - {row.song_name}"
                ) from error

            warnings.warn(
                f"Using NaN MIDI features for {row.artist} - {row.song_name}: {error!r}"
            )
            midi_features = np.full(5, np.nan, dtype=np.float32)

        rows.append(
            {
                "artist": row.artist,
                "song_name": row.song_name,
                "midi_features": midi_features,
            }
        )

    return pd.DataFrame(rows, columns=["artist", "song_name", "midi_features"]).set_index(
        ["artist", "song_name"]
    )


# %%
extract_midi_features(str(example_path))


# %%
midi_features_df = build_midi_features_dataframe(train_df, midi_dir=MIDI_DIR)

print(f"MIDI features dataframe shape: {midi_features_df.shape}")
display(midi_features_df.head(20))


# %% [markdown]
# ## Prepare Lyrics Sequences

# %%
def prepare_lyrics_sequences(
    lyrics_df: pd.DataFrame,
    lookback: int = lookback,
    lyrics_column: str = "lyrics",
    artist_column: str = "artist",
    song_column: str = "song_name",
) -> pd.DataFrame:
    if lookback <= 0:
        raise ValueError("lookback must be a positive integer.")

    required_columns = {artist_column, song_column, lyrics_column}
    missing_columns = required_columns - set(lyrics_df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")

    rows = []

    # For RNN/LSTM: old words first, newest word last
    context_columns = [
        f"lyric_t-{step}"
        for step in range(lookback, 0, -1)
    ]

    for row in lyrics_df[[artist_column, song_column, lyrics_column]].itertuples(index=False):
        tokens = tokenize_lyrics(getattr(row, lyrics_column))

        if len(tokens) <= lookback:
            continue

        for target_position in range(lookback, len(tokens)):
            previous_tokens = tokens[target_position - lookback : target_position]

            rows.append({
                "artist": getattr(row, artist_column),
                "song_name": getattr(row, song_column),
                "target_lyric": tokens[target_position],
                **dict(zip(context_columns, previous_tokens)),
            })

    return pd.DataFrame(
        rows,
        columns=["artist", "song_name", "target_lyric", *context_columns],
    ).set_index(["artist", "song_name"])


# %%
train_lyrics_sequences_df = prepare_lyrics_sequences(train_df, lookback=lookback)
test_lyrics_sequences_df = prepare_lyrics_sequences(test_df, lookback=lookback)

print(f"Train lyrics sequences dataframe shape: {train_lyrics_sequences_df.shape}")
display(train_lyrics_sequences_df.head())

print(f"Test lyrics sequences dataframe shape: {test_lyrics_sequences_df.shape}")
display(test_lyrics_sequences_df.head())

# %% [markdown]
# ## Prepper Dataset and DataLoader
#

# %%
class LyricsMidiPrepperDataset(Dataset):
    """Dataset whose samples combine lyric context and MIDI features."""

    def __init__(
        self,
        lyrics_sequences_df: pd.DataFrame,
        midi_features_df: pd.DataFrame,
        word2vec_model: KeyedVectors | None = None,
        lookback: int = lookback,
        target_word_to_index: dict[str, int] | None = None,
    ) -> None:
        """Create a lyrics-and-MIDI dataset.
        The dataset is built by joining the provided lyrics sequences and MIDI features dataframes on artist and song name, then filtering out rows with missing MIDI features or out-of-vocabulary context words. The target lyric words are mapped to class indices based on the provided mapping or inferred from the data.
        the __getitem__ method returns a tuple of (lyrics_context, song_midi_features, target_lyric) where lyrics_context is a tensor of shape (lookback, 300) containing the Word2Vec embeddings of the context words, midi_features is a tensor of shape (5,) containing the numeric MIDI features, and target is a scalar tensor with the class index of the target lyric word.
        
        
        Args:
            lyrics_sequences_df: DataFrame with indexed by artist and song_name, and columns for target_lyric and lyric context words.
            midi_features_df: DataFrame indexed by artist and song_name, with a midi_features column containing numeric feature vectors.
            word2vec_model: Optional preloaded Gensim keyed vectors object for converting context words to embeddings. If None, the default WORD2VEC_MODEL_NAME will be loaded lazily.
            lookback: Number of previous lyric words in the context (must match the context columns in lyrics_sequences_df).
            target_word_to_index: Optional mapping of target lyric words to class indices. If None, the mapping will be inferred from the unique target_lyric values in lyrics_sequences_df. If provided, rows with target
        
        """

        # Validate input
        if lookback <= 0:
            raise ValueError("lookback must be a positive integer.")
        # context_columns must be computed before the validation set that uses it
        context_columns = [f"lyric_t-{step}" for step in range(lookback, 0, -1)]
        required_lyrics_columns = {"target_lyric", *context_columns}
        if list(lyrics_sequences_df.index.names) != ["artist", "song_name"]:
            raise ValueError("lyrics_sequences_df must be indexed by ['artist', 'song_name'].")
        if list(midi_features_df.index.names) != ["artist", "song_name"]:
            raise ValueError("midi_features_df must be indexed by ['artist', 'song_name'].")
        if missing_columns := required_lyrics_columns - set(lyrics_sequences_df.columns):
            raise KeyError(f"Missing lyric sequence columns: {sorted(missing_columns)}")
        if "midi_features" not in midi_features_df.columns:
            raise KeyError("midi_features_df must contain a 'midi_features' column.")

        # param
        self.lookback = lookback
        self.context_columns = context_columns
        self.word2vec_model = word2vec_model if word2vec_model is not None else load_word2vec_model()

        # Deduplicate MIDI features only (one feature vector per song is enough).
        # Do NOT deduplicate lyrics_sequences_df — it intentionally has many rows
        # per song (one per sliding-window position) and the join is many-to-one.
        midi_features_df = midi_features_df[~midi_features_df.index.duplicated(keep="first")]
        self.df = lyrics_sequences_df.join(midi_features_df[["midi_features"]], how="inner")
        # Remove rows with missing MIDI features or out-of-vocabulary context words.
        self.df = self.df[~self.df["midi_features"].apply(lambda features: np.isnan(np.asarray(features)).any())].copy() # remove rows with NaN MIDI features
        self.df = self.df[self.df[self.context_columns].apply(lambda row: all(word in self.word2vec_model for word in row), axis=1,)].copy() # remove rows with out-of-vocabulary context words

        if target_word_to_index is None:
            print("Warning: Inferring target word classes from data. Consider providing a fixed mapping to ensure consistent class indices across runs.")
            target_words = sorted(self.df["target_lyric"].unique())
            self.word_to_index = {word: index for index, word in enumerate(target_words)}
        else:
            self.word_to_index = dict(target_word_to_index)
            self.df = self.df[self.df["target_lyric"].isin(self.word_to_index)].copy()

        self.index_to_word = {index: word for word, index in self.word_to_index.items()}
        unique_context_words = pd.unique(self.df[self.context_columns].values.ravel())
        self.embedding_cache = {
            word: np.asarray(self.word2vec_model[word], dtype=np.float32)
            for word in unique_context_words
        }
        self.dataset_info = {
            "lookback": self.lookback,
            "num_samples": len(self.df),
            "vocab_size": len(self.word_to_index),
        }

    def __len__(self) -> int:
        """Return the number of valid lyric/MIDI samples."""
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one sample.

        Args:
            index: Sample index.

        Returns:
            Tuple ``(lyrics_context, midi_features, target)`` where
            ``lyrics_context`` has shape ``(lookback, 300)``,
            ``midi_features`` has shape ``(5,)``, and ``target`` is a scalar
            next-word class index.
        """
        row = self.df.iloc[index]
        context_words = row[self.context_columns]
        midi_features = np.asarray(row["midi_features"], dtype=np.float32)
        target_word = row["target_lyric"]
        lyrics_context = np.stack([self.embedding_cache[word] for word in context_words])
        target = self.word_to_index[target_word]

        return (
            torch.tensor(lyrics_context, dtype=torch.float32),
            torch.tensor(midi_features, dtype=torch.float32),
            torch.tensor(target, dtype=torch.long),
        )

    @property
    def vocab_size(self) -> int:
        """Return the number of target-word classes."""
        return len(self.word_to_index)


# %%
# Example usage, after the Word2Vec model is cached locally:
# train_dataset = LyricsMidiPrepperDataset(
#     train_lyrics_sequences_df,
#     midi_features_df,
#     lookback=lookback,
# )
# lyrics_context, midi_features, target = train_dataset[0]
# print(lyrics_context.shape, midi_features.shape, target.shape)

# %%
from torch.utils.data import DataLoader

# %% [markdown]
# ## Vocab + Train/Val Split + DataLoaders

# %%
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from collections import Counter

VOCAB_SIZE = 5000
BATCH_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 15
LEARNING_RATE = 1e-3
WORDS_PER_LINE = 9
GENERATE_NUM_WORDS = 80
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def build_top_k_vocab(lyrics_sequences_df: pd.DataFrame, k: int = VOCAB_SIZE) -> dict[str, int]:
    """Build word-to-index mapping from top-k most frequent target words.

    Limiting vocabulary to top-k words keeps the output layer manageable
    for CPU training (full vocab can exceed 30k unique words).
    """
    word_counts = Counter(lyrics_sequences_df["target_lyric"])
    top_words = [word for word, _ in word_counts.most_common(k)]
    return {word: idx for idx, word in enumerate(top_words)}


def song_level_train_val_split(
    lyrics_df: pd.DataFrame,
    val_fraction: float = VAL_SPLIT,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by unique songs to prevent train/val leakage."""
    songs = lyrics_df[["artist", "song_name"]].drop_duplicates().sample(frac=1, random_state=random_seed)
    n_val = max(1, int(len(songs) * val_fraction))
    val_songs = songs.iloc[:n_val].set_index(["artist", "song_name"])
    train_songs_index = songs.iloc[n_val:].set_index(["artist", "song_name"]).index

    song_index = lyrics_df.set_index(["artist", "song_name"]).index
    return lyrics_df[song_index.isin(train_songs_index)].copy(), lyrics_df[song_index.isin(val_songs.index)].copy()


# %%
train_split_df, val_split_df = song_level_train_val_split(train_df)
print(f"Train songs: {train_split_df['song_name'].nunique()}, Val songs: {val_split_df['song_name'].nunique()}")

train_seqs_df = prepare_lyrics_sequences(train_split_df, lookback=lookback)
val_seqs_df = prepare_lyrics_sequences(val_split_df, lookback=lookback)

target_word_to_index = build_top_k_vocab(train_seqs_df, k=VOCAB_SIZE)
print(f"Vocab size: {len(target_word_to_index)}")

w2v_model = load_word2vec_model()
midi_features_train_df = build_midi_features_dataframe(train_split_df, midi_dir=MIDI_DIR)
midi_features_val_df = build_midi_features_dataframe(val_split_df, midi_dir=MIDI_DIR)

# Normalize MIDI features using train-set statistics.
# The 5 features span very different scales (e.g. avg_pitch ≈ 60–80 vs num_notes ≈ 100–5000).
# Z-score normalization prevents high-magnitude features from dominating.
# dropna() does not work on object-dtype Series of numpy arrays,
# so we filter valid (non-NaN) feature vectors explicitly.
_valid_feats = [v for v in midi_features_train_df["midi_features"].values
                if v is not None and not np.any(np.isnan(v))]
train_midi_matrix = np.stack(_valid_feats)
midi_mean = train_midi_matrix.mean(axis=0)
midi_std  = train_midi_matrix.std(axis=0) + 1e-8  # avoid division by zero

def normalize_midi_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["midi_features"] = df["midi_features"].apply(
        lambda v: ((np.asarray(v, dtype=np.float32) - midi_mean) / midi_std)
        if v is not None and not np.any(np.isnan(v))
        else v
    )
    return df

midi_features_train_df = normalize_midi_features(midi_features_train_df)
midi_features_val_df   = normalize_midi_features(midi_features_val_df)
print(f"MIDI normalization — mean: {midi_mean.round(2)}, std: {midi_std.round(2)}")

train_dataset = LyricsMidiPrepperDataset(
    train_seqs_df, midi_features_train_df,
    word2vec_model=w2v_model, lookback=lookback,
    target_word_to_index=target_word_to_index,
)
val_dataset = LyricsMidiPrepperDataset(
    val_seqs_df, midi_features_val_df,
    word2vec_model=w2v_model, lookback=lookback,
    target_word_to_index=target_word_to_index,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# %% [markdown]
# ## Model Definitions
#
# Three models share the same interface:
#   `forward(lyrics_context, midi_features) -> logits[batch, vocab_size]`
#
# - **LyricsOnlyModel**: baseline, ignores midi_features.
# - **MelodyConcatModel** (Approach A): broadcasts midi_features across the time axis
#   and concatenates to each word embedding before the LSTM.
# - **MelodyHiddenInitModel** (Approach B): projects midi_features into the LSTM's
#   initial (h0, c0) state — melody shapes the hidden dynamics rather than each input.

# %%
class LyricsOnlyModel(nn.Module):
    """LSTM that predicts the next lyric word from a context window.

    Receives no melody information — serves as the control baseline.
    """

    def __init__(self, vocab_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=WORD2VEC_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, lyrics_context: torch.Tensor, midi_features: torch.Tensor) -> torch.Tensor:
        # lyrics_context: (batch, lookback, 300)
        out, _ = self.lstm(lyrics_context)         # (batch, lookback, hidden)
        last = self.dropout(out[:, -1, :])          # (batch, hidden)
        return self.fc(last)                        # (batch, vocab_size)


class MelodyConcatModel(nn.Module):
    """Approach A: MIDI features concatenated to every word-embedding timestep.

    The LSTM input at each step is [word_emb || midi_features] = (300+5) dims,
    allowing the melody to directly modulate every prediction step.
    """

    MIDI_DIM = 5

    def __init__(self, vocab_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=WORD2VEC_DIM + self.MIDI_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, lyrics_context: torch.Tensor, midi_features: torch.Tensor) -> torch.Tensor:
        # lyrics_context: (batch, lookback, 300)
        # midi_features:  (batch, 5)
        batch, seq_len, _ = lyrics_context.shape
        midi_expanded = midi_features.unsqueeze(1).expand(batch, seq_len, self.MIDI_DIM)
        x = torch.cat([lyrics_context, midi_expanded], dim=2)  # (batch, lookback, 305)
        out, _ = self.lstm(x)
        last = self.dropout(out[:, -1, :])
        return self.fc(last)


class MelodyHiddenInitModel(nn.Module):
    """Approach B: MIDI features projected into the LSTM initial hidden state.

    The LSTM input remains 300-dim word embeddings, but the melody is encoded
    into (h0, c0) via a learned projection — a structurally different integration
    that conditions the entire sequence through the hidden dynamics.
    """

    MIDI_DIM = 5

    def __init__(self, vocab_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Projects 5 MIDI scalars to the full (h0, c0) tensor for all layers
        self.midi_proj = nn.Linear(self.MIDI_DIM, num_layers * hidden_size * 2)
        self.lstm = nn.LSTM(
            input_size=WORD2VEC_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, lyrics_context: torch.Tensor, midi_features: torch.Tensor) -> torch.Tensor:
        batch = lyrics_context.shape[0]
        # Project MIDI features to initial hidden and cell state
        proj = self.midi_proj(midi_features)                        # (batch, num_layers*hidden*2)
        proj = proj.view(batch, self.num_layers * 2, self.hidden_size)
        proj = proj.permute(1, 0, 2).contiguous()                   # (num_layers*2, batch, hidden)
        h0 = proj[:self.num_layers]                                 # (num_layers, batch, hidden)
        c0 = proj[self.num_layers:]                                 # (num_layers, batch, hidden)
        out, _ = self.lstm(lyrics_context, (h0, c0))
        last = self.dropout(out[:, -1, :])
        return self.fc(last)


# %% [markdown]
# ## Training Loop

# %%
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    device: torch.device = DEVICE,
) -> list[dict]:
    """Train model and log losses to TensorBoard. Returns per-epoch history.

    Saves the best-val-loss checkpoint to results/<model_name>_best.pt and
    restores it after training so the model is ready for generation.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(str(RESULTS_DIR / "runs" / model_name))

    best_val_loss = float("inf")
    checkpoint_path = RESULTS_DIR / f"{model_name}_best.pt"
    history = []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for lyrics_ctx, midi_feat, targets in train_loader:
            lyrics_ctx = lyrics_ctx.to(device)
            midi_feat = midi_feat.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(lyrics_ctx, midi_feat)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item() * targets.size(0)
            train_n += targets.size(0)

        train_loss = train_loss_sum / train_n

        # --- Val ---
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for lyrics_ctx, midi_feat, targets in val_loader:
                lyrics_ctx = lyrics_ctx.to(device)
                midi_feat = midi_feat.to(device)
                targets = targets.to(device)
                logits = model(lyrics_ctx, midi_feat)
                loss = criterion(logits, targets)
                val_loss_sum += loss.item() * targets.size(0)
                val_n += targets.size(0)

        val_loss = val_loss_sum / val_n

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[{model_name}] Epoch {epoch}/{epochs} — train: {train_loss:.4f}, val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    writer.close()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"[{model_name}] Best val loss: {best_val_loss:.4f} — loaded from {checkpoint_path}")
    return history


# %% [markdown]
# ## Instantiate and Train All Three Models
#
# `load_or_train` checks for an existing checkpoint first.
# If one exists the model is loaded instantly and history is read from the
# TensorBoard event files.  Only models without a saved checkpoint are trained.

# %%
from tensorboard.backend.event_processing import event_accumulator as tb_ea
import math

def read_tb_history(model_name: str) -> list[dict]:
    """Read per-epoch train/val loss from an existing TensorBoard run directory.

    When a run directory contains event files from multiple training sessions,
    EventAccumulator concatenates all of them.  We keep only the *last* entry
    for each step so the chart shows the most recent training run.
    """
    run_dir = RESULTS_DIR / "runs" / model_name
    ea = tb_ea.EventAccumulator(str(run_dir))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if "Loss/train" not in tags or "Loss/val" not in tags:
        return []
    train_events = ea.Scalars("Loss/train")
    val_events   = ea.Scalars("Loss/val")
    # Build dicts keyed by step; later entries overwrite earlier ones (most recent run wins)
    train_by_step = {e.step: e.value for e in train_events}
    val_by_step   = {e.step: e.value for e in val_events}
    steps = sorted(set(train_by_step) & set(val_by_step))
    return [
        {"epoch": s, "train_loss": train_by_step[s], "val_loss": val_by_step[s]}
        for s in steps
    ]


def load_or_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
) -> list[dict]:
    """Load checkpoint + TensorBoard history if available, else train from scratch."""
    checkpoint_path = RESULTS_DIR / f"{model_name}_best.pt"
    if checkpoint_path.exists():
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        )
        model.to(DEVICE)
        history = read_tb_history(model_name)
        epochs_done = len(history)
        print(f"[{model_name}] Loaded checkpoint ({epochs_done} epochs in TensorBoard history).")
        return history
    print(f"[{model_name}] No checkpoint found — training from scratch.")
    return train_model(model, train_loader, val_loader, model_name)


# %%
vocab_size = len(target_word_to_index)

lyrics_only_model     = LyricsOnlyModel(vocab_size)
melody_concat_model   = MelodyConcatModel(vocab_size)
melody_hiddeninit_model = MelodyHiddenInitModel(vocab_size)

print(f"LyricsOnlyModel params:       {sum(p.numel() for p in lyrics_only_model.parameters()):,}")
print(f"MelodyConcatModel params:     {sum(p.numel() for p in melody_concat_model.parameters()):,}")
print(f"MelodyHiddenInitModel params: {sum(p.numel() for p in melody_hiddeninit_model.parameters()):,}")

# %%
history_lyrics_only = load_or_train(lyrics_only_model, train_loader, val_loader, "lyrics_only")

# %%
history_melody_concat = load_or_train(melody_concat_model, train_loader, val_loader, "melody_concat")

# %%
history_melody_hiddeninit = load_or_train(melody_hiddeninit_model, train_loader, val_loader, "melody_hiddeninit")

# %% [markdown]
# ## Training Visualisations
#
# ### Loss curves
# Cross-entropy loss and perplexity (exp(loss)) for each model over training.

# %%
MODEL_DISPLAY_NAMES = {
    "lyrics_only":       "Lyrics Only (Baseline)",
    "melody_concat":     "Melody Concat (Approach A)",
    "melody_hiddeninit": "Melody Hidden-Init (Approach B)",
}
model_histories = [
    ("lyrics_only",       history_lyrics_only),
    ("melody_concat",     history_melody_concat),
    ("melody_hiddeninit", history_melody_hiddeninit),
]

# --- Loss curves ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, (key, history) in zip(axes, model_histories):
    epochs_x = [h["epoch"] for h in history]
    ax.plot(epochs_x, [h["train_loss"] for h in history], label="Train", linewidth=2)
    ax.plot(epochs_x, [h["val_loss"]   for h in history], label="Val",   linewidth=2, linestyle="--")
    ax.set_title(MODEL_DISPLAY_NAMES[key], fontsize=13)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
plt.suptitle("Training & Validation Loss — All Models", fontsize=15, y=1.02)
save_current_figure("training_curves_loss.png")

# %%
# --- Perplexity curves (exp of cross-entropy) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, (key, history) in zip(axes, model_histories):
    epochs_x = [h["epoch"] for h in history]
    ax.plot(epochs_x, [math.exp(h["train_loss"]) for h in history], label="Train", linewidth=2)
    ax.plot(epochs_x, [math.exp(h["val_loss"])   for h in history], label="Val",   linewidth=2, linestyle="--")
    ax.set_title(MODEL_DISPLAY_NAMES[key], fontsize=13)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
plt.suptitle("Training & Validation Perplexity — All Models", fontsize=15, y=1.02)
save_current_figure("training_curves_perplexity.png")

# %%
# --- Model comparison: best validation loss & perplexity bar chart ---
best_val = {key: min(h["val_loss"] for h in hist) for key, hist in model_histories}
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
_labels = [MODEL_DISPLAY_NAMES[k] for k in best_val]
_colors = ["#4f7cac", "#e07b39", "#6aab68"]

bars0 = axes[0].bar(_labels, list(best_val.values()), color=_colors)
axes[0].set_title("Best Validation Loss", fontsize=13)
axes[0].set_ylabel("Cross-Entropy Loss", fontsize=12)
axes[0].tick_params(axis="x", labelsize=10, rotation=15)
for bar, val in zip(bars0, best_val.values()):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=11, color="black", fontweight="bold")

_perp_vals = [math.exp(v) for v in best_val.values()]
bars1 = axes[1].bar(_labels, _perp_vals, color=_colors)
axes[1].set_title("Best Validation Perplexity", fontsize=13)
axes[1].set_ylabel("Perplexity", fontsize=12)
axes[1].tick_params(axis="x", labelsize=10, rotation=15)
for bar, val in zip(bars1, _perp_vals):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=11, color="black", fontweight="bold")

plt.suptitle("Model Comparison — Best Validation Metrics", fontsize=14, y=1.02)
save_current_figure("model_comparison_bar.png")

# %% [markdown]
# ## Text Generation
#
# `generate_lyrics` autoregressively produces one word at a time using a
# sliding window of `lookback` previous words.  Four sampling strategies are
# supported; all are post-processing of the same logit vector so they work
# without retraining:
#
# - **proportional** — multinomial sample from the raw softmax distribution
# - **temperature** — divide logits by T before softmax (T < 1 = sharper,
#   T > 1 = flatter / more random)
# - **top_k** — zero out every logit except the top-k, then sample
# - **nucleus** — keep the smallest set of words whose cumulative probability
#   reaches *p*, zero out the rest, then sample

# %%
import torch.nn.functional as F

FALLBACK_SEED_WORD = "love"  # used when the requested seed word is OOV


def _apply_sampling_strategy(
    logits: torch.Tensor,
    strategy: str,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    """Apply a decoding strategy to a 1-D logit tensor and return a word index."""
    if strategy == "proportional":
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()

    if strategy == "temperature":
        probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
        return torch.multinomial(probs, 1).item()

    if strategy == "top_k":
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        threshold = values[-1]
        filtered = logits.masked_fill(logits < threshold, float("-inf"))
        probs = F.softmax(filtered, dim=-1)
        return torch.multinomial(probs, 1).item()

    if strategy == "nucleus":
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens once cumulative mass exceeds top_p
        remove_mask = cumsum - sorted_probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
        # Scatter back to original ordering
        filtered = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)
        probs = F.softmax(filtered, dim=-1)
        return torch.multinomial(probs, 1).item()

    raise ValueError(f"Unknown strategy '{strategy}'. Choose from: proportional, temperature, top_k, nucleus.")


def generate_lyrics(
    model: nn.Module,
    seed_word: str,
    midi_features: np.ndarray | None,
    word_to_index: dict[str, int],
    index_to_word: dict[int, str],
    word2vec_model: KeyedVectors,
    num_words: int = GENERATE_NUM_WORDS,
    strategy: str = "proportional",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    words_per_line: int = WORDS_PER_LINE,
    device: torch.device = DEVICE,
) -> str:
    """Generate lyrics autoregressively from a seed word and optional MIDI features.

    Args:
        model: Trained LyricsOnlyModel, MelodyConcatModel, or MelodyHiddenInitModel.
        seed_word: First word of the output. Falls back to FALLBACK_SEED_WORD if OOV.
        midi_features: 5-element MIDI feature vector, or None for the lyrics-only model.
        word_to_index: Vocabulary mapping (word → class index).
        index_to_word: Reverse vocabulary (class index → word).
        word2vec_model: Gensim keyed vectors for embedding lookups.
        num_words: Total words to generate.
        strategy: One of 'proportional', 'temperature', 'top_k', 'nucleus'.
        temperature: Temperature for the temperature strategy.
        top_k: k for the top_k strategy.
        top_p: Cumulative probability threshold for the nucleus strategy.
        words_per_line: Insert a newline every this many generated words.
        device: Torch device.

    Returns:
        Generated lyrics as a formatted multi-line string.
    """
    model.eval()

    # Resolve seed word — fall back if not embeddable
    def _can_embed(word: str) -> bool:
        for candidate in (word, word.lower(), word.title()):
            if candidate in word2vec_model:
                return True
        return False

    if not _can_embed(seed_word):
        warnings.warn(f"Seed word '{seed_word}' not in Word2Vec; using '{FALLBACK_SEED_WORD}'.")
        seed_word = FALLBACK_SEED_WORD

    def _embed(word: str) -> np.ndarray:
        for candidate in (word, word.lower(), word.title()):
            if candidate in word2vec_model:
                return np.asarray(word2vec_model[candidate], dtype=np.float32)
        raise KeyError(word)

    # Build initial context: fill the lookback window with the seed word
    context = [seed_word] * lookback

    midi_tensor = torch.zeros(1, 5, device=device)
    if midi_features is not None:
        midi_tensor = torch.tensor(midi_features, dtype=torch.float32, device=device).unsqueeze(0)

    generated_words = []

    with torch.no_grad():
        for step in range(num_words):
            # Embed current context window
            ctx_array = np.stack([_embed(w) for w in context])          # (lookback, 300)
            ctx_tensor = torch.tensor(ctx_array, dtype=torch.float32, device=device).unsqueeze(0)  # (1, lookback, 300)

            logits = model(ctx_tensor, midi_tensor).squeeze(0)           # (vocab_size,)

            # Sample next word
            for _ in range(20):  # retry if sampled word is not embeddable
                word_idx = _apply_sampling_strategy(logits, strategy, temperature, top_k, top_p)
                next_word = index_to_word.get(word_idx, FALLBACK_SEED_WORD)
                if _can_embed(next_word):
                    break
            else:
                next_word = FALLBACK_SEED_WORD

            generated_words.append(next_word)
            context = context[1:] + [next_word]  # slide window

    # Format as song: insert newlines to create lines
    lines, current_line = [], []
    for i, word in enumerate(generated_words):
        current_line.append(word)
        if len(current_line) >= words_per_line:
            lines.append(" ".join(current_line))
            current_line = []
    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


# %% [markdown]
# ## Test Phase
#
# For each of the 3 test songs:
# - Generate lyrics with 3 seed words × 3 models (proportional sampling)
# - Save all outputs to results/generated_lyrics.txt
#
# Then run two analysis experiments:
# 1. **Melody-influence probe** — compare outputs under correct vs. corrupted MIDI
# 2. **Decoding strategy comparison** — proportional / temperature / top_k / nucleus on 2 songs

# %%
SEED_WORDS = ["love", "night", "run"]

# Load test songs and their MIDI features
test_df_loaded = load_lyrics_dataset(TEST_CSV_PATH)
print("Test songs:")
display(test_df_loaded[["artist", "song_name"]])

midi_path_lookup_test = build_midi_path_lookup(MIDI_DIR)
test_midi_features = {}
for row in test_df_loaded.itertuples(index=False):
    midi_path = find_midi_path(row.artist, row.song_name, midi_path_lookup_test)
    if midi_path is not None:
        raw_feat = extract_midi_features(str(midi_path))
        # Apply the same z-score normalization as the training data
        test_midi_features[(row.artist, row.song_name)] = (raw_feat - midi_mean) / midi_std
    else:
        warnings.warn(f"No MIDI found for test song: {row.artist} - {row.song_name}")
        test_midi_features[(row.artist, row.song_name)] = None

index_to_word = {idx: word for word, idx in target_word_to_index.items()}
model_configs = [
    ("lyrics_only",       lyrics_only_model,       None),       # None → model ignores MIDI
    ("melody_concat",     melody_concat_model,     "use_midi"),
    ("melody_hiddeninit", melody_hiddeninit_model, "use_midi"),
]

# %%
# --- 5a: Generate lyrics for all test songs, seed words, and models ---
output_lines = []
all_generated = {}  # key: (song_key, model_name, seed_word) → lyrics string

for row in test_df_loaded.itertuples(index=False):
    song_key = (row.artist, row.song_name)
    midi_feat = test_midi_features[song_key]

    for model_name, model, midi_flag in model_configs:
        effective_midi = midi_feat if midi_flag == "use_midi" else None

        for seed_word in SEED_WORDS:
            lyrics = generate_lyrics(
                model=model,
                seed_word=seed_word,
                midi_features=effective_midi,
                word_to_index=target_word_to_index,
                index_to_word=index_to_word,
                word2vec_model=w2v_model,
                strategy="proportional",
            )
            all_generated[(song_key, model_name, seed_word)] = lyrics

            header = f"\n{'='*60}\nSong: {row.artist} - {row.song_name}\nModel: {model_name} | Seed: '{seed_word}'\n{'='*60}"
            output_lines.append(header)
            output_lines.append(lyrics)

generated_lyrics_path = RESULTS_DIR / "generated_lyrics.txt"
generated_lyrics_path.write_text("\n".join(output_lines), encoding="utf-8")
print(f"Saved {len(all_generated)} generated outputs to {generated_lyrics_path}")

# %%
# --- 5b: Melody-influence probe ---
# Keep the seed word fixed; vary what MIDI is fed to the melody models.
# Strategies: correct MIDI, shuffled MIDI features (same values, shuffled dims),
# and mismatched MIDI (from a different test song).
# Metric: Jaccard similarity of generated word-sets.

def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity between the word-sets of two texts."""
    set_a = set(text_a.split())
    set_b = set(text_b.split())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


probe_seed = SEED_WORDS[0]
test_songs_list = list(test_df_loaded.itertuples(index=False))
probe_rows = []

for row in test_songs_list:
    song_key = (row.artist, row.song_name)
    correct_midi = test_midi_features[song_key]
    if correct_midi is None:
        continue

    # Mismatched MIDI: use another test song's features
    other_midi = next(
        (f for (a, s), f in test_midi_features.items() if (a, s) != song_key and f is not None),
        correct_midi,
    )
    shuffled_midi = correct_midi.copy()
    np.random.shuffle(shuffled_midi)

    for model_name, model, midi_flag in model_configs:
        if midi_flag != "use_midi":
            continue  # probe only applies to melody models

        def _gen(midi_feat, _model=model):  # default-arg captures model at definition time
            return generate_lyrics(
                model=_model, seed_word=probe_seed,
                midi_features=midi_feat,
                word_to_index=target_word_to_index, index_to_word=index_to_word,
                word2vec_model=w2v_model, strategy="proportional",
            )

        gen_correct   = _gen(correct_midi)
        gen_shuffled  = _gen(shuffled_midi)
        gen_mismatched = _gen(other_midi)

        probe_rows.append({
            "song":       f"{row.artist} - {row.song_name}",
            "model":      model_name,
            "jaccard_shuffled":   round(jaccard_similarity(gen_correct, gen_shuffled),   3),
            "jaccard_mismatched": round(jaccard_similarity(gen_correct, gen_mismatched), 3),
        })

probe_df = pd.DataFrame(probe_rows)
print("\nMelody-Influence Probe Results (Jaccard similarity vs. correct MIDI output):")
print("Lower = more different output → melody is being used")
display(probe_df)
probe_df.to_csv(RESULTS_DIR / "melody_probe_results.csv", index=False)

# --- Probe bar chart ---
if not probe_df.empty:
    _model_names = {"melody_concat": "Melody Concat\n(Approach A)", "melody_hiddeninit": "Melody Hidden-Init\n(Approach B)"}
    _models = probe_df["model"].unique()
    _bar_colors = ["#4f7cac", "#e07b39"]
    _dot_colors = ["#111111", "#e74c3c", "#16a085", "#8e44ad", "#d4ac0d"]  # black, red, teal, purple, gold

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (col, title) in zip(axes, [
        ("jaccard_shuffled",    "Shuffled MIDI Features"),
        ("jaccard_mismatched",  "Mismatched Song MIDI"),
    ]):
        means = probe_df.groupby("model")[col].mean().reindex(_models)
        x_pos = np.arange(len(_models))

        bars = ax.bar(x_pos, means.values, color=_bar_colors, alpha=0.75, width=0.5, zorder=2)

        # individual song dots (drawn before value labels so labels sit on top)
        dot_maxes = {xi: means.values[xi] for xi in range(len(_models))}
        for si, song in enumerate(probe_df["song"].unique()):
            row = probe_df[probe_df["song"] == song]
            for xi, m in enumerate(_models):
                v = row.loc[row["model"] == m, col]
                if not v.empty:
                    ax.scatter(xi, v.values[0], color=_dot_colors[si % len(_dot_colors)],
                               s=60, zorder=3, label=song if xi == 0 else "")
                    dot_maxes[xi] = max(dot_maxes[xi], v.values[0])

        # value labels just above the bar, drawn on top of dots via zorder
        for xi, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{means.values[xi]:.3f}", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="black", zorder=4)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([_model_names.get(m, m) for m in _models], fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 0.35)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    axes[0].set_ylabel("Jaccard Similarity (lower → melody matters more)", fontsize=11)
    axes[1].legend(title="Song", fontsize=8, title_fontsize=9, loc="upper right")
    fig.suptitle("Melody-Influence Probe: Output Similarity Under Corrupted MIDI", fontsize=13, y=1.02)
    save_current_figure("melody_probe_bar.png")

# %%
# --- 5c: Decoding strategy comparison ---
# Use the first 2 test songs, MelodyConcatModel (Approach A), seed='love'
strategies = ["proportional", "temperature", "top_k", "nucleus"]
strategy_kwargs = {
    "proportional": {},
    "temperature":  {"temperature": 0.7},
    "top_k":        {"top_k": 50},
    "nucleus":      {"top_p": 0.9},
}

decoding_output = []
for row in test_songs_list[:2]:
    song_key = (row.artist, row.song_name)
    midi_feat = test_midi_features[song_key]
    decoding_output.append(f"\n{'#'*60}\nSong: {row.artist} - {row.song_name} | Seed: 'love'\n{'#'*60}")

    for strat in strategies:
        lyrics = generate_lyrics(
            model=melody_concat_model,
            seed_word="love",
            midi_features=midi_feat,
            word_to_index=target_word_to_index,
            index_to_word=index_to_word,
            word2vec_model=w2v_model,
            strategy=strat,
            **strategy_kwargs[strat],
        )
        decoding_output.append(f"\n--- Strategy: {strat} ---\n{lyrics}")

decoding_comparison_text = "\n".join(decoding_output)
(RESULTS_DIR / "decoding_strategy_comparison.txt").write_text(decoding_comparison_text, encoding="utf-8")
print(decoding_comparison_text)

# %% [markdown]
# ## Decoding Strategy Analysis
#
# Lexical diversity (type-token ratio) and unique-word count for each strategy,
# averaged across the 2 test songs.

# %%
diversity_rows = []
for row in test_songs_list[:2]:
    song_key = (row.artist, row.song_name)
    midi_feat = test_midi_features[song_key]
    for strat in strategies:
        lyrics = generate_lyrics(
            model=melody_concat_model,
            seed_word="love",
            midi_features=midi_feat,
            word_to_index=target_word_to_index,
            index_to_word=index_to_word,
            word2vec_model=w2v_model,
            strategy=strat,
            **strategy_kwargs[strat],
        )
        words = lyrics.split()
        ttr = len(set(words)) / len(words) if words else 0.0
        diversity_rows.append({"strategy": strat, "song": f"{row.artist[:15]}..", "ttr": round(ttr, 3), "unique_words": len(set(words))})

diversity_df = pd.DataFrame(diversity_rows)
diversity_avg = diversity_df.groupby("strategy")[["ttr", "unique_words"]].mean().reindex(strategies)
display(diversity_avg)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
bars0 = axes[0].bar(strategies, diversity_avg["ttr"], color=["#4f7cac", "#e07b39", "#6aab68", "#c45c68"])
axes[0].set_title("Type-Token Ratio by Decoding Strategy", fontsize=13)
axes[0].set_ylabel("TTR (higher = more diverse)", fontsize=11)
axes[0].set_ylim(0, 1.0)
axes[0].grid(axis="y", alpha=0.3)
for bar, val in zip(bars0, diversity_avg["ttr"]):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11, color="black", fontweight="bold")

bars1 = axes[1].bar(strategies, diversity_avg["unique_words"], color=["#4f7cac", "#e07b39", "#6aab68", "#c45c68"])
axes[1].set_title("Unique Words by Decoding Strategy", fontsize=13)
axes[1].set_ylabel("Unique word count", fontsize=11)
axes[1].grid(axis="y", alpha=0.3)
for bar, val in zip(bars1, diversity_avg["unique_words"]):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=11, color="black", fontweight="bold")

plt.suptitle("Lexical Diversity across Sampling Strategies (MelodyConcatModel)", fontsize=13, y=1.02)
save_current_figure("decoding_diversity.png")

