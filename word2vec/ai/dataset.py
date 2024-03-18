from collections import Counter
from pathlib import Path
from typing import Tuple

import spacy
import torch
from torch.utils.data import Dataset

nlp = spacy.load("en_core_web_sm")


class FriendsDataset(Dataset):
    def __init__(self, filename: Path, window_size: int = 3):
        self.data = self._read_data(filename)
        self.window_size = window_size
        self.vocab, self.word_to_idx, self.idx_to_word = self._build_vocab()
        self.samples = self._generate_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        center_word, context_word = self.samples[idx]
        return torch.tensor(center_word), torch.tensor(context_word)

    def _read_data(self, filename: Path) -> list:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

            # Process each line using SpaCy to remove stop words
            processed_words = [
                word
                for line in lines
                for word in self._remove_stopwords(line)
            ]

            return processed_words

    @staticmethod
    def _remove_stopwords(text: str) -> list:
        # Process text using SpaCy to remove stop words
        doc = nlp(text)
        processed_words = [
            token.text.lower().strip()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.is_alpha
            and len(token.text) > 1
        ]
        return processed_words

    def _build_vocab(self) -> Tuple[list, dict, dict]:
        word_counts = Counter(self.data)
        vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        return vocab, word_to_idx, idx_to_word

    def _generate_samples(self) -> list:
        samples = []
        for i, target_word in enumerate(self.data):
            for j in range(
                max(i - self.window_size, 0),
                min(len(self.data), i + self.window_size + 1),
            ):
                if j != i:
                    samples.append(
                        (self.word_to_idx[target_word], self.word_to_idx[self.data[j]])
                    )
        return samples


if __name__ == "__main__":
    # Suppress UserWarning
    import warnings

    warnings.simplefilter("ignore", UserWarning)

    file = Path("./dataset/S01E01 Monica Gets A Roommate.txt")

    print("-- FriendsDataset --")
    dataset = FriendsDataset(file, window_size=10)
    print(f"Number of samples: {len(dataset)}")
    print(f"Vocab size: {len(dataset.vocab)}")
    print(f"Window size: {dataset.window_size}")
    print(dataset.vocab)
    print(dataset.data)
