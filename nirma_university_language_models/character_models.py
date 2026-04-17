from __future__ import annotations

import random
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "character_level_model" / "tinyshakespeare.txt"
)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_tinyshakespeare_dataset(path: str | Path | None = None) -> Path:
    dataset_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if not dataset_path.exists():
        urllib.request.urlretrieve(DATA_URL, dataset_path)
    return dataset_path


def load_tinyshakespeare_text(path: str | Path | None = None) -> tuple[str, Path]:
    dataset_path = ensure_tinyshakespeare_dataset(path)
    return dataset_path.read_text(encoding="utf-8"), dataset_path


def build_vocabulary(text: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return chars, char_to_idx, idx_to_char


def most_common_characters(text: str, limit: int = 20) -> list[tuple[str, int]]:
    return Counter(text).most_common(limit)


def encode_text(text: str, char_to_idx: dict[str, int], dtype=np.int64) -> np.ndarray:
    return np.array([char_to_idx[ch] for ch in text], dtype=dtype)


def decode_ids(ids, idx_to_char: dict[int, str]) -> str:
    return "".join(idx_to_char[int(i)] for i in ids)


def make_input_target_pair(data: np.ndarray, seq_len: int = 80) -> tuple[np.ndarray, np.ndarray]:
    return data[:seq_len], data[1 : seq_len + 1]


def make_sequences(
    data: np.ndarray,
    seq_len: int = 80,
    step: int = 80,
    n_examples: int = 4,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    xs, ys = [], []
    stop = min(len(data) - seq_len - 1, step * n_examples)
    for start in range(0, stop, step):
        xs.append(data[start : start + seq_len])
        ys.append(data[start + 1 : start + seq_len + 1])
    return xs, ys


class CharDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int = 100):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def build_dataloaders(
    encoded: np.ndarray,
    seq_len: int = 100,
    batch_size: int = 64,
    split_ratio: float = 0.9,
) -> tuple[DataLoader, DataLoader]:
    split_idx = int(split_ratio * len(encoded))
    train_data = encoded[:split_idx]
    val_data = encoded[split_idx:]

    train_ds = CharDataset(train_data, seq_len=seq_len)
    val_ds = CharDataset(val_data, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


class CharModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_size: int = 128,
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
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.fc(out)
        return logits, hidden


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


def train_model(
    rnn_type: str,
    vocab_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    embed_dim: int = 64,
    hidden_size: int = 128,
) -> tuple[CharModel, list[float], list[float]]:
    model = CharModel(
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
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1

        train_loss = running_loss / max(count, 1)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"{rnn_type} | Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

    return model, train_losses, val_losses


def sample_from_model(
    model: nn.Module,
    char_to_idx: dict[str, int],
    idx_to_char: dict[int, str],
    device: str,
    start_text: str = "ROMEO:",
    length: int = 300,
    temperature: float = 1.0,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    unknown = {ch for ch in start_text if ch not in char_to_idx}
    if unknown:
        raise ValueError(f"start_text contains unknown characters: {sorted(unknown)!r}")

    model.eval()
    input_ids = torch.tensor([[char_to_idx[ch] for ch in start_text]], dtype=torch.long).to(device)
    generated = start_text
    hidden = None

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_ids, hidden)
            last_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated += idx_to_char[next_id.item()]
            input_ids = next_id

    return generated
