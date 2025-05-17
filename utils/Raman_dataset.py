import torch
from torch.utils.data import Dataset
import numpy as np
import math
from scipy.fftpack import dct
from scipy.signal import find_peaks
import random


class RamanNoiseDataset(Dataset):
    def __init__(self, clean_signals, noise_std_list):
        """
        Initialize dataset.
        :param noisy_signals: List or tensor of noisy signals.
        :param true_noises: List or tensor of corresponding true noise signals.
        """
        self.clean_signals = torch.from_numpy(np.transpose(clean_signals)) # make it (num_spectrum, spectrunm_len)
        self.noise_std_list = noise_std_list
        self.SNR = []
        self.noisy_signals = []
        self.noises = []
        self.gt_signals = []
        self.int_times = []
    
    def generate_noisy_signals(self, SNR_range):
        def get_signal_max(signal):
            # peaks, properties = find_peaks(signal, height=0.1, distance=5, prominence=0.4)
            # peak_amplitudes = properties["peak_heights"]
            # return np.sum(peak_amplitudes)
            return np.max(np.array(signal)), np.argmax(np.array(signal))
        
        min_SNR, max_SNR = SNR_range
        for signal in self.clean_signals:
            signal_max, max_pos = get_signal_max(signal)
            # SNR = np.log10(np.random.uniform(min_SNR, max_SNR))
            SNR = np.random.uniform(min_SNR, max_SNR)
            int_time, noise_std = random.choice(list(self.noise_std_list.items()))
            # target_signal_power = 10**(SNR / 10) * noise_power.item()
            k = (SNR**2 + np.sqrt(SNR**4 + 4*2*noise_std[max_pos]*SNR**2)) / (2 * signal_max)
            signal = signal * k
            noise = torch.normal(0, torch.sqrt(signal + 2 * noise_std))
            noisy_signal = signal + noise
            self.noisy_signals.append(noisy_signal)
            self.noises.append(noise)
            self.SNR.append(SNR)
            self.gt_signals.append(signal)
            self.int_times.append(int_time)
        return
    
    def DCT(self):
        noisy_signals_tmp = torch.stack(self.noisy_signals).numpy()
        noises_tmp = torch.stack(self.noises).numpy()
        gt_signals_tmp = torch.stack(self.gt_signals).numpy()
        self.noisy_signals_dct = torch.from_numpy(dct(noisy_signals_tmp, axis=1, norm='ortho'))
        self.noises_dct = torch.from_numpy(dct(noises_tmp, axis=1, norm='ortho'))
        self.gt_signals_dct = torch.from_numpy(dct(gt_signals_tmp, axis=1, norm='ortho'))
        return

    def __len__(self):
        return len(self.noisy_signals)

    def __getitem__(self, idx):
        return self.noisy_signals[idx], self.noisy_signals_dct[idx], self.noises_dct[idx], self.gt_signals_dct[idx], self.SNR[idx], self.int_times[idx]


if __name__ == "__main__":
    num_samples = 100
    spectrum_length = 1000
    SNR_range = [1.001, 10]
    clean_signals = np.abs(np.random.randn(spectrum_length, num_samples)) # the clean data is generated as shape(spectrum_length, num_samples)
    noise_std_list = {"0.1s": np.abs(np.random.randn(spectrum_length)), "0.5s": np.abs(np.random.randn(spectrum_length))}
    dataset = RamanNoiseDataset(clean_signals=clean_signals, noise_std_list=noise_std_list)
    dataset.generate_noisy_signals(SNR_range=SNR_range)
    dataset.DCT()
    noisy_signal_tmp, noisy_tmp, true_tmp, gt_tmp, SNR, int_times_tmp = dataset[1]
    print(f"signal length: {len(dataset)}")
    print(f"shape of a signal: {noisy_tmp.shape}")