import torch
from torch.utils.data import Dataset
import numpy as np
# import math
from scipy.fftpack import dct
# from scipy.signal import find_peaks


class IntegrationDataset(Dataset):
    def __init__(self, input_noises, true_noises):
        """
        Initialize dataset.
        :param noisy_signals: List or tensor of noisy signals.
        :param true_noises: List or tensor of corresponding true noise signals.
        """
        self.input_noises = np.array(input_noises) # make it (num_spectrum, spectrunm_len)
        self.true_noises = np.array(true_noises)
        self.input_noises_dct = None
        self.true_noises_dct = None
    
    def DCT(self):
        self.input_noises_dct = torch.from_numpy(dct(self.input_noises, axis=1, norm='ortho'))
        self.true_noises_dct = torch.from_numpy(dct(self.true_noises, axis=1, norm='ortho'))
        return

    def __len__(self):
        return len(self.input_noises)

    def __getitem__(self, idx):
        return self.input_noises[idx], self.true_noises[idx]


if __name__ == "__main__":
    num_samples = 100
    spectrum_length = 1000
    input_noises = np.random.randn(num_samples, spectrum_length) # the clean data is generated as shape(spectrum_length, num_samples)
    true_noises = np.random.randn(num_samples, spectrum_length)
    dataset = IntegrationDataset(input_noises=input_noises, true_noises=true_noises)
    dataset.DCT()
    noisy_tmp, true_tmp = dataset[0]
    print(f"signal length: {len(dataset)}")
    print(f"shape of a signal: {noisy_tmp.shape}")