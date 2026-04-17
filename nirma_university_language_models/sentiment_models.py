from __future__ import annotations

import csv
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_SENTIMENT_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "sentiment_analysis" / "sentiment_reviews.csv"
)
LABEL_TO_IDX = {"negative": 0, "positive": 1}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}
TOKEN_PATTERN = re.compile(r"[a-z']+")


def load_sentiment_examples(path: str | Path | None = None) -> tuple[list[dict[str, str]], Path]:
    dataset_path = Path(path) if path is not None else DEFAULT_SENTIMENT_DATA_PATH
    with dataset_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        examples = [
            {"text": row["text"].strip(), "label": row["label"].strip().lower()}
            for row in reader
            if row["text"].strip()
        ]
    return examples, dataset_path


def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def label_distribution(examples: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(example["label"] for example in examples)
    return dict(sorted(counts.items()))


def most_common_tokens(texts: list[str], limit: int = 20) -> list[tuple[str, int]]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize_text(text))
    return counter.most_common(limit)


def build_word_vocabulary(
    texts: list[str],
    min_freq: int = 1,
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize_text(text))

    vocab = ["<pad>", "<unk>"]
    vocab.extend(sorted(token for token, count in counter.items() if count >= min_freq))
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    return vocab, token_to_idx, idx_to_token


def encode_tokens(tokens: list[str], token_to_idx: dict[str, int]) -> list[int]:
    unk_idx = token_to_idx["<unk>"]
    return [token_to_idx.get(token, unk_idx) for token in tokens]


def pad_sequence_to_length(
    token_ids: list[int],
    max_len: int,
    pad_idx: int,
) -> tuple[list[int], int]:
    trimmed = token_ids[:max_len]
    length = max(len(trimmed), 1)
    padded = trimmed + [pad_idx] * max(0, max_len - len(trimmed))
    if not trimmed:
        padded = [pad_idx] * max_len
    return padded, length


class SentimentDataset(Dataset):
    def __init__(
        self,
        sequences: list[list[int]],
        labels: list[int],
        max_len: int,
        pad_idx: int,
    ):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
        self.pad_idx = pad_idx

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded, length = pad_sequence_to_length(self.sequences[idx], self.max_len, self.pad_idx)
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def build_sentiment_dataloaders(
    examples: list[dict[str, str]],
    token_to_idx: dict[str, int],
    max_len: int = 20,
    batch_size: int = 16,
    split_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    encoded_sequences = [
        encode_tokens(tokenize_text(example["text"]), token_to_idx) for example in examples
    ]
    labels = [LABEL_TO_IDX[example["label"]] for example in examples]

    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    split_idx = int(split_ratio * len(indices))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    pad_idx = token_to_idx["<pad>"]
    train_ds = SentimentDataset(
        [encoded_sequences[i] for i in train_idx],
        [labels[i] for i in train_idx],
        max_len=max_len,
        pad_idx=pad_idx,
    )
    val_ds = SentimentDataset(
        [encoded_sequences[i] for i in val_idx],
        [labels[i] for i in val_idx],
        max_len=max_len,
        pad_idx=pad_idx,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


class SentimentRNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_size: int = 64,
        rnn_type: str = "RNN",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if rnn_type == "RNN":
            self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        else:
            raise ValueError("rnn_type must be one of: RNN, GRU, LSTM")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, len(LABEL_TO_IDX))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        outputs, _ = self.rnn(embedded)

        safe_lengths = lengths.clamp(min=1, max=x.size(1))
        gather_index = (safe_lengths - 1).view(-1, 1, 1).expand(-1, 1, outputs.size(-1))
        last_outputs = outputs.gather(1, gather_index).squeeze(1)
        return self.fc(self.dropout(last_outputs))


def evaluate_sentiment_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for x, lengths, labels in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(x, lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    average_loss = total_loss / max(len(loader), 1)
    accuracy = total_correct / max(total_examples, 1)
    return average_loss, accuracy


def train_sentiment_model(
    rnn_type: str,
    vocab_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    *,
    epochs: int = 8,
    lr: float = 1e-3,
    embed_dim: int = 64,
    hidden_size: int = 64,
    dropout: float = 0.2,
) -> tuple[SentimentRNNClassifier, dict[str, list[float]]]:
    model = SentimentRNNClassifier(
        vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        rnn_type=rnn_type,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for x, lengths, labels in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

        train_loss = total_loss / max(len(train_loader), 1)
        train_acc = total_correct / max(total_examples, 1)
        val_loss, val_acc = evaluate_sentiment_model(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"{rnn_type} | Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f}"
        )

    return model, history


def predict_sentiment(
    model: nn.Module,
    text: str,
    token_to_idx: dict[str, int],
    device: str,
    max_len: int = 20,
) -> tuple[str, list[float]]:
    model.eval()
    token_ids = encode_tokens(tokenize_text(text), token_to_idx)
    padded, length = pad_sequence_to_length(token_ids, max_len=max_len, pad_idx=token_to_idx["<pad>"])

    x = torch.tensor([padded], dtype=torch.long).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x, lengths)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).tolist()
        prediction = IDX_TO_LABEL[int(torch.argmax(logits, dim=1).item())]

    return prediction, probabilities
