import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.fftpack import dct
import math


class RamanNoiseDataset(Dataset):
    def __init__(self, clean_signals, true_noises):
        """
        Initialize dataset.
        :param noisy_signals: List or tensor of noisy signals.
        :param true_noises: List or tensor of corresponding true noise signals.
        """
        self.clean_signals = torch.from_numpy(np.transpose(clean_signals)) # make it (num_spectrum, spectrunm_len)
        self.true_noises = torch.from_numpy(true_noises)
        self.noisy_signals = []
        self.noise_out = []
        self.gt_signal = []
        self.SNR = []
    
    def generate_noisy_signals(self, SNR_range):
        min_SNR, max_SNR = SNR_range
        num_of_noise = self.true_noises.shape[0]
        for signal in self.clean_signals:
            idx = np.random.randint(0, num_of_noise)
            noise_tmp = self.true_noises[idx] # the noise power is 1 by default
            noise_power = torch.mean(noise_tmp ** 2)
            noise_tmp = noise_tmp / torch.sqrt(noise_power)
            signal_power = torch.mean(signal ** 2)
            # SNR = np.log10(np.random.uniform(min_SNR, max_SNR))
            SNR = np.random.uniform(min_SNR, max_SNR)
            # target_signal_power = 10**(SNR / 10) * noise_power.item()
            target_signal_power = SNR * noise_power.item()
            target_signal = math.sqrt(target_signal_power) * (signal / math.sqrt(signal_power))
            new_signal = target_signal + noise_tmp
            # give a magnification
            mag = np.random.uniform(0.1, 100)
            new_signal = new_signal * mag
            target_signal = target_signal * mag
            noise_tmp = noise_tmp * mag

            tmp = torch.max(new_signal)
            self.noisy_signals.append(new_signal/tmp)
            self.noise_out.append(noise_tmp/tmp)
            self.gt_signal.append(target_signal/tmp)
            self.SNR.append(SNR)
        return
    
    def DCT(self):
        noisy_signals_tmp = torch.stack(self.noisy_signals).numpy()
        noise_out_tmp = torch.stack(self.noise_out).numpy()
        self.noisy_signals_dct = torch.from_numpy(dct(noisy_signals_tmp, axis=1, norm='ortho'))
        self.noise_out_dct = torch.from_numpy(dct(noise_out_tmp, axis=1, norm='ortho'))
        return

    def __len__(self):
        return len(self.noisy_signals)

    def __getitem__(self, idx):
        return self.noisy_signals_dct[idx], self.noise_out_dct[idx], self.gt_signal[idx], self.SNR[idx]


if __name__ == "__main__":
    num_samples = 100
    spectrum_length = 1000
    SNR_range = [1.001, 10]
    clean_signals = np.random.randn(spectrum_length, num_samples) # the clean data is generated as shape(spectrum_length, num_samples)
    true_noises = np.random.randn(num_samples, spectrum_length)
    dataset = RamanNoiseDataset(clean_signals=clean_signals, true_noises=true_noises)
    dataset.generate_noisy_signals(SNR_range=SNR_range)
    dataset.DCT()
    noisy_tmp, true_tmp, gt_tmp, SNR = dataset[0]
    print(f"signal length: {len(dataset)}")
    print(f"shape of a signal: {noisy_tmp.shape}")