from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from word2vec.ai.dataloader import FriendsDataLoader
from word2vec.ai.dataset import FriendsDataset

# Set Seaborn style
sns.set_style("whitegrid")


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
        self.losses = []

    def train(self):
        for epoch in range(self.epochs):
            # Track the total loss for each epoch
            epoch_loss = 0

            size = len(self.dataset)
            for batch, (center_word, context_word) in enumerate(self.dataloader):
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

                epoch_loss += loss.item()

            # Store the average loss for the epoch
            self.losses.append(epoch_loss / size)

            # Print loss every 10th epoch
            if epoch % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.epochs}], Loss: {epoch_loss / size:.6f}"
                )

        # Save the trained embeddings
        self.embeddings = self.model.embedding.weight.data.numpy()

    def save(self):
        print(f"Saving model at: {self.model_path}")
        torch.save(self.model, self.model_path)

    def load(self):
        print(f"Loading model from: {self.model_path}")
        self.model = torch.load(self.model_path)

    def plot_loss(self):
        # Plot the loss graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.epochs + 1), w2v.losses, marker="o", linestyle="-")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.show()

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
        return abs(cosine_similarity)


if __name__ == "__main__":
    # Suppress UserWarning
    import warnings

    warnings.simplefilter("ignore", UserWarning)

    # # torch.manual_seed(42)

    file = Path("./dataset/S01E01 Monica Gets A Roommate.txt")

    print("-- FriendsDataset --")
    friends_dataset = FriendsDataset(file, window_size=3)
    print(f"Number of samples: {len(friends_dataset)}")
    print(f"Vocab size: {len(friends_dataset.vocab)}")
    print(f"Window size: {friends_dataset.window_size}")
    # #
    # print("\n-- FriendsDataLoader --")
    # dataloader = FriendsDataLoader(dataset, batch_size=32, shuffle=True)
    # print(f"Batch size: {dataloader.batch_size}")
    # print(f"Number of batches: {len(dataloader)}")
    #
    # print("\n-- SkipGramModel --")
    # embedding_dim = 100
    # model = SkipGramModel(vocab_size=len(dataset.vocab), embedding_dim=embedding_dim)
    # print(f"Embedding dimension: {embedding_dim}")

    print("\n-- W2V --")
    _learning_rate = 0.0001
    _epochs = 1000
    # _criterion = nn.CrossEntropyLoss()
    # _optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    _model_path = Path("trained/word2vec_model.pth")

    w2v = W2V(
        dataset=friends_dataset,  # dataloader=dataloader,
        # model=model,
        # criterion=_criterion,
        # optimizer=_optimizer,
        # epochs=_epochs,
        model_path=_model_path,
    )
    print(f"Learning rate: {_learning_rate}")
    print(f"Epochs: {_epochs}")

    # print("\n-- Training --")
    # w2v.train()
    # print(f"Model trained and saved at: {model_path}")
    # print(f"Model saved at: {model_path}")
    #
    # print("\n-- Saving --")
    # w2v.save()

    print("\n-- Loading --")
    w2v.load()

    print("\n-- Similarity --")
    w2v.similarity("las", "vegas")
    w2v.similarity("central", "perk")
    w2v.similarity("hump", "hairpiece")
    w2v.similarity("coffee", "hairpiece")
    w2v.similarity("coffee", "central")
