import torch
from torch.utils.data import Dataset
import numpy as np
import math
from scipy.fftpack import dct
from scipy.signal import find_peaks
import random


class RamanNoiseDataset(Dataset):
    def __init__(self, clean_signals, noise_std_list, fluorescence=None):
        """
        Initialize dataset.
        :param noisy_signals: List or tensor of noisy signals.
        :param true_noises: List or tensor of corresponding true noise signals.
        """
        self.clean_signals = torch.from_numpy(np.transpose(clean_signals)) # make it (num_spectrum, spectrunm_len)
        self.fluorescence = torch.from_numpy(np.transpose(fluorescence)) if fluorescence is not None else None
        self.noise_std_list = noise_std_list
        self.SNR = []
        self.noisy_signals = []
        self.noises = []
        self.gt_signals = []
        self.int_times = []
        self.r2f_ranges = []
    
    def generate_noisy_signals(self, SNR_range, r2f_range=None):
        def get_signal_max(signal):
            # peaks, properties = find_peaks(signal, height=0.1, distance=5, prominence=0.4)
            # peak_amplitudes = properties["peak_heights"]
            # return np.sum(peak_amplitudes)
            return np.max(np.array(signal)), np.argmax(np.array(signal))
        
        min_SNR, max_SNR = SNR_range
        if self.fluorescence is not None:
            fluorescence_indices = torch.randperm(self.fluorescence.shape[0])
            shuffled_fluorescence = self.fluorescence[fluorescence_indices]
            for signal, fluorescence_signal in zip(self.clean_signals, shuffled_fluorescence):
                signal_max, max_pos = get_signal_max(signal)
                SNR = np.random.uniform(min_SNR, max_SNR)
                r2f = np.random.uniform(r2f_range[0], r2f_range[1])
                int_time, noise_std = random.choice(list(self.noise_std_list.items()))
                
                f_max = np.max(np.array(fluorescence_signal))
                f_p   = fluorescence_signal[max_pos]
                y     = 2 * noise_std[max_pos]

                # Quadratic is: A*n^2 + B*n + C = 0
                A = (r2f**2) * (f_max**2)
                B = - (SNR**2) * (r2f * f_max + f_p)
                C = - (SNR**2) * y
                disc = B**2 - 4 * A * C
                if disc < 0:
                    raise ValueError("Discriminant is negative: no real solutions for n")
                
                n = (-B + np.sqrt(disc)) / (2 * A)
                m = r2f * n * f_max / signal_max
                
                signal = signal * m
                fluorescence_signal = fluorescence_signal * n
                noise = torch.normal(0, torch.sqrt(signal + fluorescence_signal + 2 * noise_std))
                noisy_signal = signal + noise
                
                self.noisy_signals.append(noisy_signal)
                self.noises.append(noise)
                self.SNR.append(SNR)
                self.gt_signals.append(signal)
                self.int_times.append(int_time)
                self.r2f_ranges.append(r2f)
            return
        else:
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
                self.r2f_ranges.append(0.0)
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
        return self.noisy_signals[idx], self.noisy_signals_dct[idx], self.noises_dct[idx], self.gt_signals_dct[idx], self.SNR[idx], self.int_times[idx], self.r2f_ranges[idx]


if __name__ == "__main__":
    num_samples = 100
    spectrum_length = 1000
    SNR_range = [1.001, 10]
    r2f_range = [0.1, 0.5]
    clean_signals = np.abs(np.random.randn(spectrum_length, num_samples)) # the clean data is generated as shape(spectrum_length, num_samples)
    fluorescence = np.abs(np.random.randn(spectrum_length, num_samples))
    noise_std_list = {"0.1s": np.abs(np.random.randn(spectrum_length)), "0.5s": np.abs(np.random.randn(spectrum_length))}
    dataset = RamanNoiseDataset(clean_signals=clean_signals, noise_std_list=noise_std_list, fluorescence=fluorescence)
    dataset.generate_noisy_signals(SNR_range=SNR_range, r2f_range=r2f_range)
    dataset.DCT()
    params = dataset[1]
    print(f"signal length: {len(dataset)}")
    print(f"shape of a signal: {params[0].shape}")