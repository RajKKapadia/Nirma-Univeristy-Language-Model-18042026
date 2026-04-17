# Nirma University Language Models

This repository contains lecture material for recurrent neural networks in three tracks:

- character-level language modeling with Tiny Shakespeare
- symbolic music generation with a small local melody dataset
- word-level sentiment analysis with a small local review dataset

All tracks include dataset exploration, lecture notes, and a side-by-side comparison of vanilla RNN, GRU, and LSTM behavior.

## Repository Layout

- `src/character_level_model/01_character_dataset_exploration.ipynb`: inspect the dataset, build a vocabulary, and create next-character training pairs.
- `src/character_level_model/02_character_rnn_gru_lstm_comparison.ipynb`: train RNN, GRU, and LSTM models on the same corpus and compare losses plus generated text.
- `src/character_level_model/03_character_sequence_models_lecture_notebook.ipynb`: short theory notebook for lecture delivery.
- `src/music_generation/01_music_dataset_exploration.ipynb`: inspect the symbolic melody dataset, vocabulary, and note-token format.
- `src/music_generation/02_music_rnn_gru_lstm_comparison.ipynb`: train RNN, GRU, and LSTM melody generators, compare losses, and export a sample `.wav`.
- `src/music_generation/03_music_sequence_models_lecture_notebook.ipynb`: short theory notebook for teaching music as sequence modeling.
- `src/sentiment_analysis/01_sentiment_dataset_exploration.ipynb`: inspect the labeled review dataset and build padded token sequences.
- `src/sentiment_analysis/02_sentiment_rnn_gru_lstm_comparison.ipynb`: train sentiment classifiers with RNN, GRU, and LSTM backbones.
- `src/sentiment_analysis/03_sentiment_sequence_models_lecture_notebook.ipynb`: short theory notebook for teaching sentiment classification with sequence models.
- `nirma_university_language_models/character_models.py`: shared helpers for the character-level notebooks.
- `nirma_university_language_models/music_models.py`: shared helpers for symbolic music tokenization, dataset building, generation, and waveform export.
- `nirma_university_language_models/sentiment_models.py`: shared helpers for tokenization, dataset building, sentiment classifiers, training, and inference.
- `src/character_level_model/tinyshakespeare.txt`: checked-in Tiny Shakespeare corpus. The helpers reuse this local file and only download it if it is missing.
- `src/music_generation/simple_melodies.txt`: checked-in symbolic melody dataset used by the music notebooks.
- `src/sentiment_analysis/sentiment_reviews.csv`: checked-in local sentiment dataset used by the sentiment notebooks.

## Setup

This project targets Python `3.12.11` (see `.python-version`).

Use the project virtual environment if it already exists:

```bash
source .venv/bin/activate
```

If you need to create it again with `uv`:

```bash
uv sync
```

If you prefer a standard `venv` + `pip` workflow:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Smoke-check the shared Python module after activating the environment:

```bash
python main.py
```

If you do not want to activate the environment first, run the interpreter directly:

```bash
.venv/bin/python main.py
```

Launch Jupyter and open the notebooks from the repository root. JupyterLab is not declared in `pyproject.toml`, so install it separately if your environment does not already provide it:

```bash
uv run --with jupyterlab jupyter lab
```

## Notes

- The training notebook is intentionally simple and optimized for readability in a lecture, not for training speed.
- The notebooks now import shared helpers instead of redefining the model stack inline, which keeps the teaching flow simpler and makes the code reusable.
- The music track uses symbolic note tokens such as `C4_q` and `REST_e`, which keeps the sequence-model idea visible without requiring a heavy audio-generation stack.
- The sentiment dataset is intentionally small and classroom-friendly. It is suitable for demonstrating the full preprocessing and training pipeline, not for benchmarking production sentiment models.
- If Matplotlib warns about an unwritable config directory in a restricted environment, set `MPLCONFIGDIR` to a writable path such as `/tmp/matplotlib`.
