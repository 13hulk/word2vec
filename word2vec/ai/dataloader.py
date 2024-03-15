from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from word2vec.ai.dataset import FriendsDataset


class FriendsDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    file = Path("./dataset/S01E01 Monica Gets A Roommate.txt")

    print("-- FriendsDataset --")
    friends_dataset = FriendsDataset(file, window_size=2)
    print(f"Number of samples: {len(friends_dataset)}")
    print(f"Vocab size: {len(friends_dataset.vocab)}")
    print(f"Window size: {friends_dataset.window_size}")

    print("\n-- FriendsDataLoader --")
    friends_dataloader = FriendsDataLoader(friends_dataset, batch_size=32, shuffle=True)
    print(f"Batch size: {friends_dataloader.batch_size}")
    print(f"Number of batches: {len(friends_dataloader)}")
