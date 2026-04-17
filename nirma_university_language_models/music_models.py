from __future__ import annotations

import math
import random
import re
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_MUSIC_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "music_generation" / "simple_melodies.txt"
)
MELODY_END_TOKEN = "<eos>"
DURATION_TO_BEATS = {"w": 4.0, "h": 2.0, "q": 1.0, "e": 0.5, "s": 0.25}
NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}
NOTE_TOKEN_PATTERN = re.compile(r"^(REST|[A-G](?:#|b)?\d)_(w|h|q|e|s)$")


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_music_token_sequences(
    path: str | Path | None = None,
) -> tuple[list[list[str]], Path]:
    dataset_path = Path(path) if path is not None else DEFAULT_MUSIC_DATA_PATH
    sequences = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            sequences.append(line.split())
    return sequences, dataset_path


def melody_lengths(sequences: list[list[str]]) -> list[int]:
    return [len(sequence) for sequence in sequences]


def most_common_music_tokens(
    sequences: list[list[str]],
    limit: int = 20,
) -> list[tuple[str, int]]:
    counter = Counter()
    for sequence in sequences:
        counter.update(sequence)
    return counter.most_common(limit)


def flatten_music_token_sequences(
    sequences: list[list[str]],
    end_token: str = MELODY_END_TOKEN,
) -> list[str]:
    tokens: list[str] = []
    for sequence in sequences:
        tokens.extend(sequence)
        tokens.append(end_token)
    return tokens


def build_music_vocabulary(
    sequences: list[list[str]],
    end_token: str = MELODY_END_TOKEN,
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    vocab = sorted(set(flatten_music_token_sequences(sequences, end_token=end_token)))
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    return vocab, token_to_idx, idx_to_token


def encode_music_tokens(tokens: list[str], token_to_idx: dict[str, int], dtype=np.int64) -> np.ndarray:
    unknown = sorted({token for token in tokens if token not in token_to_idx})
    if unknown:
        raise ValueError(f"tokens contain unknown values: {unknown!r}")
    return np.array([token_to_idx[token] for token in tokens], dtype=dtype)


def decode_music_ids(
    ids,
    idx_to_token: dict[int, str],
    *,
    stop_at_end: bool = False,
    end_token: str = MELODY_END_TOKEN,
) -> list[str]:
    tokens = []
    for idx in ids:
        token = idx_to_token[int(idx)]
        if stop_at_end and token == end_token:
            break
        tokens.append(token)
    return tokens


class MusicSequenceDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int = 16):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def build_music_dataloaders(
    encoded: np.ndarray,
    seq_len: int = 16,
    batch_size: int = 32,
    split_ratio: float = 0.9,
) -> tuple[DataLoader, DataLoader]:
    split_idx = int(split_ratio * len(encoded))
    train_data = encoded[:split_idx]
    val_data = encoded[split_idx:]

    train_ds = MusicSequenceDataset(train_data, seq_len=seq_len)
    val_ds = MusicSequenceDataset(val_data, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


class MusicModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_size: int = 96,
        rnn_type: str = "RNN",
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if rnn_type == "RNN":
            self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        else:
            raise ValueError("rnn_type must be one of: RNN, GRU, LSTM")

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded, hidden)
        logits = self.fc(out)
        return logits, hidden


def evaluate_music_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


def train_music_model(
    rnn_type: str,
    vocab_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    *,
    epochs: int = 8,
    lr: float = 1e-3,
    embed_dim: int = 32,
    hidden_size: int = 96,
) -> tuple[MusicModel, list[float], list[float]]:
    model = MusicModel(
        vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        rnn_type=rnn_type,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1

        train_loss = running_loss / max(count, 1)
        val_loss = evaluate_music_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"{rnn_type} | Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

    return model, train_losses, val_losses


def sample_music_model(
    model: nn.Module,
    token_to_idx: dict[str, int],
    idx_to_token: dict[int, str],
    device: str,
    *,
    start_tokens: list[str] | str | None = None,
    max_new_tokens: int = 24,
    temperature: float = 1.0,
    stop_on_end: bool = True,
    end_token: str = MELODY_END_TOKEN,
) -> list[str]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if start_tokens is None:
        prompt = ["C4_q", "E4_q", "G4_q"]
    elif isinstance(start_tokens, str):
        prompt = start_tokens.split()
    else:
        prompt = list(start_tokens)

    unknown = sorted({token for token in prompt if token not in token_to_idx})
    if unknown:
        raise ValueError(f"start_tokens contain unknown values: {unknown!r}")

    model.eval()
    input_ids = torch.tensor([[token_to_idx[token] for token in prompt]], dtype=torch.long).to(device)
    generated = list(prompt)
    hidden = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, hidden = model(input_ids, hidden)
            last_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            next_token = idx_to_token[next_id.item()]
            if stop_on_end and next_token == end_token:
                break
            generated.append(next_token)
            input_ids = next_id

    return generated


def melody_to_text(tokens: list[str]) -> str:
    return " ".join(tokens)


def parse_note_token(token: str) -> tuple[str, str]:
    match = NOTE_TOKEN_PATTERN.match(token)
    if not match:
        raise ValueError(
            "music tokens must look like 'C4_q', 'F#4_h', or 'REST_e'; "
            f"received {token!r}"
        )
    return match.group(1), match.group(2)


def token_duration_seconds(token: str, tempo_bpm: float = 120.0) -> float:
    _, duration_code = parse_note_token(token)
    beats = DURATION_TO_BEATS[duration_code]
    return 60.0 * beats / tempo_bpm


def note_name_to_frequency(note_name: str) -> float:
    if note_name == "REST":
        return 0.0

    pitch_class = note_name[:-1]
    octave = int(note_name[-1])
    midi_number = 12 * (octave + 1) + NOTE_TO_SEMITONE[pitch_class]
    return 440.0 * math.pow(2.0, (midi_number - 69) / 12.0)


def melody_tokens_to_waveform(
    tokens: list[str],
    *,
    sample_rate: int = 16_000,
    tempo_bpm: float = 120.0,
    amplitude: float = 0.2,
) -> np.ndarray:
    segments = []
    fade_len = max(int(sample_rate * 0.005), 1)

    for token in tokens:
        note_name, _ = parse_note_token(token)
        duration_seconds = token_duration_seconds(token, tempo_bpm=tempo_bpm)
        num_samples = max(int(sample_rate * duration_seconds), 1)
        times = np.linspace(0.0, duration_seconds, num_samples, endpoint=False)

        if note_name == "REST":
            segment = np.zeros(num_samples, dtype=np.float32)
        else:
            frequency = note_name_to_frequency(note_name)
            segment = (amplitude * np.sin(2.0 * np.pi * frequency * times)).astype(np.float32)
            fade = np.linspace(0.0, 1.0, min(fade_len, num_samples), dtype=np.float32)
            segment[: fade.size] *= fade
            segment[-fade.size :] *= fade[::-1]

        segments.append(segment)

    if not segments:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(segments)


def save_waveform_as_wav(
    waveform: np.ndarray,
    path: str | Path,
    *,
    sample_rate: int = 16_000,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clipped = np.clip(waveform, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return output_path
