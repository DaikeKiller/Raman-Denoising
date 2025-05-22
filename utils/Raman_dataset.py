import torch
from torch.utils.data import Dataset
import numpy as np
import math
from scipy.fftpack import dct
from scipy.signal import find_peaks
import random
import matplotlib.pyplot as plt
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
# from AUnet import dct_torch, idct_torch
import torch_dct

def dct_torch(x):
    """
    Compute the DCT Type-II of a batched signal using FFT (differentiable).
    """
    return torch_dct.dct(x, norm='ortho')

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
        self.gt_raman_signals = []
        self.gt_fluorescence_signals = []
        self.int_times = []
        self.r2f_ranges = []
    
    def generate_noisy_signals(self, SNR_range, r2f_range=None):
        def get_signal_max(signal):
            # peaks, properties = find_peaks(signal, height=0.1, distance=5, prominence=0.4)
            # peak_amplitudes = properties["peak_heights"]
            # return np.sum(peak_amplitudes)
            return np.max(np.array(signal)), np.argmax(np.array(signal))
        
        min_SNR, max_SNR = SNR_range
        fluorescence_indices = torch.randperm(self.fluorescence.shape[0])
        shuffled_fluorescence = self.fluorescence[fluorescence_indices]
        for raman_signal, fluorescence_signal in zip(self.clean_signals, shuffled_fluorescence):
            signal_max, max_pos = get_signal_max(raman_signal)
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
            
            raman_signal = raman_signal * m
            fluorescence_signal = fluorescence_signal * n
            signal = raman_signal + fluorescence_signal
            noise = torch.normal(0, torch.sqrt(signal + 2 * noise_std))
            noisy_signal = signal + noise
            
            self.noisy_signals.append(noisy_signal)
            self.noises.append(noise)
            self.SNR.append(SNR)
            self.gt_signals.append(signal)
            self.gt_raman_signals.append(raman_signal)
            self.gt_fluorescence_signals.append(fluorescence_signal)
            self.int_times.append(int_time)
            self.r2f_ranges.append(r2f)
        return
        
    def DCT(self):
        noisy_signals_tmp = torch.stack(self.noisy_signals).numpy()
        noises_tmp = torch.stack(self.noises).numpy()
        gt_signals_tmp = torch.stack(self.gt_signals).numpy()
        gt_raman_signals_tmp = torch.stack(self.gt_raman_signals).numpy()
        gt_fluorescence_signals_tmp = torch.stack(self.gt_fluorescence_signals).numpy()
        self.noisy_signals_dct = torch.from_numpy(dct(noisy_signals_tmp, axis=1, norm='ortho'))
        self.noises_dct = torch.from_numpy(dct(noises_tmp, axis=1, norm='ortho'))
        self.gt_signals_dct = torch.from_numpy(dct(gt_signals_tmp, axis=1, norm='ortho'))
        self.gt_raman_signals_dct = torch.from_numpy(dct(gt_raman_signals_tmp, axis=1, norm='ortho'))
        self.gt_fluorescence_signals_dct = torch.from_numpy(dct(gt_fluorescence_signals_tmp, axis=1, norm='ortho'))
        return

    def __len__(self):
        return len(self.noisy_signals)

    def __getitem__(self, idx):
        return self.noisy_signals[idx], self.noisy_signals_dct[idx], self.noises_dct[idx], self.gt_signals_dct[idx], \
               self.gt_raman_signals_dct[idx], self.gt_fluorescence_signals_dct[idx], self.SNR[idx], self.int_times[idx], self.r2f_ranges[idx]


class DenoiserTestDataset():
    def __init__(self, clean_signals, noise_std_list, fluorescence=None):
        """
        Initialize dataset.
        :param noisy_signals: List or tensor of noisy signals.
        :param true_noises: List or tensor of corresponding true noise signals.
        """
        super(DenoiserTestDataset, self).__init__()
        self.clean_signals = torch.from_numpy(np.transpose(clean_signals)) # make it (num_spectrum, spectrunm_len)
        self.fluorescence = torch.from_numpy(np.transpose(fluorescence)) if fluorescence is not None else None
        self.noise_std_list = noise_std_list
        self.pairs = []
        self.noisy_signals = []
        self.noisy_signals_dct = []
        self.max_pos = []
        self.raman_max = []
    
    def generate_test_signals(self, SNR_range, r2f_range=None, params=None):
        def get_signal_max(signal):
            # peaks, properties = find_peaks(signal, height=0.1, distance=5, prominence=0.4)
            # peak_amplitudes = properties["peak_heights"]
            # return np.sum(peak_amplitudes)
            return np.max(np.array(signal)), np.argmax(np.array(signal))
        
        def get_snr_r2f_pairs(num_pairs):
            min_SNR, max_SNR = SNR_range
            min_r2f, max_r2f = r2f_range
            num_signals = self.clean_signals.shape[0]

            # Generate sorted SNR and r2f values
            SNR_values = np.random.uniform(min_SNR, max_SNR, num_pairs)
            r2f_values = np.random.uniform(min_r2f, max_r2f, num_pairs)

            for SNR_value, r2f_value in zip(SNR_values, r2f_values):
                self.pairs.append((SNR_value, r2f_value))
            
            return
        
        def generate_signals(num_samples, num_noise_per_smaple):
            num_pairs = len(self.pairs)
            signals_overall = []
            max_pos_overall = []
            raman_max_overall = []

            for i in range(num_pairs):
                SNR = self.pairs[i][0]
                r2f = self.pairs[i][1]
                int_time, noise_std = random.choice(list(self.noise_std_list.items()))
                
                # randomly select num_samples indices for clean_signals and fluorescence
                indices = np.random.choice(self.clean_signals.shape[0], num_samples, replace=False)
                clean_selected = self.clean_signals[indices]
                fluorescence_selected = self.fluorescence[indices]
                signals = []
                max_pos_tmp = []
                raman_max = []

                for raman_signal, fluorescence_signal in zip(clean_selected, fluorescence_selected):
                    signal_max, max_pos = get_signal_max(raman_signal)
                    
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
                    
                    raman_signal_scaled = raman_signal * m
                    fluorescence_signal_scaled = fluorescence_signal * n
                    signal = raman_signal_scaled + fluorescence_signal_scaled
                    
                    raman_max_value = raman_signal_scaled[max_pos]
                    raman_max.append(raman_max_value)

                    # Store noisy signals for this sample
                    noisy_signals_per_sample = []
                    for _ in range(num_noise_per_smaple):
                        noise = torch.normal(0, torch.sqrt(signal + 2 * noise_std))
                        noisy_signal = signal + noise
                        noisy_signals_per_sample.append(noisy_signal.unsqueeze(0))  # shape (1, L)
                    # Stack to (num_noise_per_sample, L)
                    noisy_signals_per_sample = torch.cat(noisy_signals_per_sample, dim=0)
                    signals.append(noisy_signals_per_sample.unsqueeze(0))  # shape (1, num_noise_per_sample, L)
                    max_pos_tmp.append(max_pos)

                # Stack
                signals = torch.cat(signals, dim=0)  # shape (1, num_samples)
                signals_overall.append(signals)
                max_pos_overall.append(max_pos_tmp)
                raman_max_overall.append(raman_max)
                
            self.noisy_signals = torch.stack(signals_overall, dim=0)
            self.max_pos = max_pos_overall  # shape (num_pairs, num_samples)
            self.raman_max = raman_max_overall

            return
        
        get_snr_r2f_pairs(num_pairs=params["num_pairs"])
        generate_signals(num_samples=params["num_samples_per_pair"], num_noise_per_smaple=params["num_noise_per_sample"])

        return
    
    def DCT(self):
        noisy_signals_dct_tmp = dct_torch(self.noisy_signals)
        self.noisy_signals_dct = noisy_signals_dct_tmp
        return


if __name__ == "__main__":
    num_samples = 100
    spectrum_length = 1000
    SNR_range = [0.1, 10]
    r2f_range = [0.05, 0.5]
    clean_signals = np.abs(np.random.randn(spectrum_length, num_samples)) # the clean data is generated as shape(spectrum_length, num_samples)
    fluorescence = np.abs(np.random.randn(spectrum_length, num_samples))
    noise_std_list = {"0.1s": np.abs(np.random.randn(spectrum_length)), "0.5s": np.abs(np.random.randn(spectrum_length))}
    dataset = RamanNoiseDataset(clean_signals=clean_signals, noise_std_list=noise_std_list, fluorescence=fluorescence)
    dataset.generate_noisy_signals(SNR_range=SNR_range, r2f_range=r2f_range)
    dataset.DCT()
    params = dataset[1]
    print(f"signal length: {len(dataset)}")
    print(f"shape of a signal: {params[0].shape}")
    
    test_dataset = DenoiserTestDataset(clean_signals=clean_signals, noise_std_list=noise_std_list, fluorescence=fluorescence)
    test_dataset.generate_test_signals(SNR_range=SNR_range, r2f_range=r2f_range, params={"num_pairs": 100, "num_samples_per_pair": 10, "num_noise_per_sample": 5})
    test_dataset.DCT()
    print(f"shape of a test signal: {test_dataset.noisy_signals_dct.shape}")