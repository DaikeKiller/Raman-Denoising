import torch
from torch.utils.data import Dataset


class RamanNoiseDataset(Dataset):
    def __init__(self, noisy_signals, true_noises):
        """
        Initialize dataset.
        :param noisy_signals: List or tensor of noisy signals.
        :param true_noises: List or tensor of corresponding true noise signals.
        """
        self.noisy_signals = noisy_signals
        self.true_noises = true_noises

    def __len__(self):
        return len(self.noisy_signals)

    def __getitem__(self, idx):
        return self.noisy_signals[idx], self.true_noises[idx]


if __name__ == "__main__":
    num_samples = 100
    spectrum_length = 1000
    noisy_signals = torch.randn(num_samples, spectrum_length)
    true_noises = torch.randn(num_samples, spectrum_length)
    dataset = RamanNoiseDataset(noisy_signals=noisy_signals, true_noises=true_noises)
    noisy_tmp, true_tmp = dataset[0]
    print(f"signal length: {len(dataset)}")
    print(f"shape of a signal: {noisy_tmp.shape}")