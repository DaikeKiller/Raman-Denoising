import torch
from torch.utils.data import Dataset
import numpy as np
# import math
from scipy.fftpack import dct
# from scipy.signal import find_peaks


class SkinGenerationDataset(Dataset):
    def __init__(self, input_spectra, true_spectra):
        """
        Initialize dataset.
        :param noisy_signals: List or tensor of noisy signals.
        :param true_noises: List or tensor of corresponding true noise signals.
        """
        self.input_spectra = np.array(input_spectra) # make it (num_spectrum, spectrunm_len)
        self.true_spectra = np.array(true_spectra)
        self.input_spectra_out = self.input_spectra
        self.true_spectra_out = self.true_spectra
    
    def DCT(self):
        self.input_spectra_dct = torch.from_numpy(dct(self.input_spectra, axis=1, norm='ortho'))
        self.true_spectra_dct = torch.from_numpy(dct(self.true_spectra, axis=1, norm='ortho'))
        self.input_spectra_out = self.input_spectra_dct
        self.true_spectra_out = self.true_spectra_dct
        return
    
    def __len__(self):
        return len(self.input_spectra_out)

    def __getitem__(self, idx):
        return self.input_spectra_out[idx], self.true_spectra_out[idx]


if __name__ == "__main__":
    num_samples = 100
    spectrum_length = 1000
    input_spectra = np.random.randn(num_samples, spectrum_length) # the clean data is generated as shape(spectrum_length, num_samples)
    true_spectra = np.random.randn(num_samples, spectrum_length)
    dataset = SkinGenerationDataset(input_spectra=input_spectra, true_spectra=true_spectra)
    dataset.DCT()
    input_tmp, true_tmp = dataset[0]
    print(f"signal length: {len(dataset)}")
    print(f"shape of a signal: {input_tmp.shape}")