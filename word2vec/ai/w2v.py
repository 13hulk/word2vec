from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from word2vec.ai.dataloader import FriendsDataLoader
from word2vec.ai.dataset import FriendsDataset


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target_word: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(target_word)
        predicted = self.linear(embedded)
        return predicted


class W2V:
    def __init__(
        self,
        dataset: FriendsDataset = None,
        dataloader: FriendsDataLoader = None,
        model: nn.Module = None,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
        model_path: Path = None,
    ):
        self.dataset: FriendsDataset | None = dataset
        self.dataloader: FriendsDataLoader | None = dataloader
        self.model: nn.Module | None = model
        self.criterion: nn.Module | None = criterion
        self.optimizer: optim.Optimizer | None = optimizer
        self.epochs: int | None = epochs
        self.model_path: Path | None = model_path

        self.embeddings = None

    def train(self):
        for epoch in range(self.epochs):
            # Track the total loss for each epoch
            total_loss = 0

            for center_word, context_word in self.dataloader:
                # Zero the gradients
                self.optimizer.zero_grad()

                # 1. Forward pass
                output_word = self.model(center_word)
                # 2. Calculate the loss
                loss = self.criterion(output_word, context_word)
                # 3. Backward pass
                loss.backward()
                # 4. Update the weights
                self.optimizer.step()

                total_loss += loss.item()

            # Print loss every 10th epoch
            if (epoch + 1) % 10 == 0:
                total_samples = len(dataset) * (epoch + 1)
                percentage_loss = (total_loss / total_samples) * 100
                print(
                    f"Epoch [{epoch + 1}/{self.epochs}], Loss: {percentage_loss:.4f}%"
                )

        # Save the trained embeddings
        self.embeddings = self.model.embedding.weight.data.numpy()

    def save(self):
        print(f"Saving model at: {self.model_path}")
        torch.save(self.model, self.model_path)

    def load(self):
        print(f"Loading model from: {self.model_path}")
        self.model = torch.load(self.model_path)

    def similarity(self, word1: str, word2: str) -> float:
        # Get the trained embedding weights
        embedding_layer = self.model.embedding
        self.embeddings = embedding_layer.weight.data

        idx1 = self.dataset.word_to_idx[word1]
        idx2 = self.dataset.word_to_idx[word2]
        emb1 = self.embeddings[idx1]
        emb2 = self.embeddings[idx2]

        # Convert NumPy arrays to PyTorch tensors
        emb1_tensor = torch.tensor(emb1)
        emb2_tensor = torch.tensor(emb2)

        # Compute cosine similarity
        cosine_similarity = torch.matmul(emb1_tensor, emb2_tensor.T) / (
            torch.norm(emb1_tensor) * torch.norm(emb2_tensor)
        )

        print(f"Similarity between '{word1}' and '{word2}': {cosine_similarity:.2f}")
        return cosine_similarity


if __name__ == "__main__":
    # Suppress UserWarning
    import warnings

    warnings.simplefilter("ignore", UserWarning)

    # torch.manual_seed(42)
    #
    file = Path("./dataset/S01E01 Monica Gets A Roommate.txt")

    print("-- FriendsDataset --")
    dataset = FriendsDataset(file, window_size=3)
    print(f"Number of samples: {len(dataset)}")
    print(f"Vocab size: {len(dataset.vocab)}")
    print(f"Window size: {dataset.window_size}")
    #
    # print("\n-- FriendsDataLoader --")
    # dataloader = FriendsDataLoader(dataset, batch_size=32, shuffle=True)
    # print(f"Batch size: {dataloader.batch_size}")
    # print(f"Number of batches: {len(dataloader)}")
    #
    # print("\n-- SkipGramModel --")
    # embedding_dim = 100
    # model = SkipGramModel(vocab_size=len(dataset.vocab), embedding_dim=embedding_dim)
    # print(f"Embedding dimension: {embedding_dim}")
    #
    # print("\n-- W2V --")
    # learning_rate = 0.001
    # epochs = 1000
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model_path = Path("trained/word2vec_model.pth")

    w2v = W2V(
        dataset=dataset,
        # dataloader=dataloader,
        # model=model,
        # criterion=criterion,
        # optimizer=optimizer,
        # epochs=epochs,
        model_path=model_path,
    )
    # print(f"Learning rate: {learning_rate}")
    # print(f"Epochs: {epochs}")

    # print("\n-- Training --")
    # w2v.train()
    # print(f"Model trained and saved at: {model_path}")
    # print(f"Model saved at: {model_path}")

    # print("\n-- Saving --")
    # w2v.save(model_path)

    print("\n-- Loading --")
    w2v.load()

    print("\n-- Similarity --")
    w2v.similarity("ross", "rachel")
    w2v.similarity("chandler", "monica")
    w2v.similarity("chandler", "door")
    w2v.similarity("door", "monica")
    w2v.similarity("monica", "chandler")
