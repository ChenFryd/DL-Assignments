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
        
    loaded_midis = load_midi_files(midi_folder_path) # dict[Path, pretty_midi.PrettyMIDI]

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
        required_lyrics_columns = {"target_lyric", *self.context_columns}
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
        self.context_columns = [f"lyric_t-{step}" for step in range(lookback, 0, -1)]
        self.word2vec_model = word2vec_model if word2vec_model is not None else load_word2vec_model()

        midi_features_df = midi_features_df[~midi_features_df.index.duplicated(keep="first")] # remove duplication
        lyrics_sequences_df = lyrics_sequences_df[~lyrics_sequences_df.index.duplicated(keep="first")] # remove duplication
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

