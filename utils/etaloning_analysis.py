import os
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from scipy.signal import detrend
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
import pywt
from PyEMD import EMD
import pickle
from datetime import datetime
import os


# file  = 'data/basics/SRM 2246 certified log normal calculations.xls'
# sheet = 'SRM 2246 Example'

# # we want rows 12–3335 (1-based) in cols D and I ⇒ zero-based skiprows=11, nrows=3335-11=3324
# # usecols='D,I' pulls in exactly those two columns
# df = pd.read_excel(
#     file,
#     sheet_name=sheet,
#     header=None,        # no header row in our slice
#     skiprows=11,        # drop rows 0–10 (i.e. Excel rows 1–11)
#     nrows=3324,         # rows 12 through 3335 inclusive
#     usecols='D,I'       # only columns D and I
# )

# # now df.iloc[:,0] is the D12:D3335 range, df.iloc[:,1] is I12:I3335
# wvn_srm = df.iloc[:, 0].to_numpy()
# i_srm   = df.iloc[:, 1].to_numpy()
# # load wvn
# wvn = loadmat("data/wvn_raw.mat")["wvn"].reshape(-1)
# # interp SRM_interp
# i_srm_interp = np.interp(wvn, wvn_srm, i_srm)
# # load NIST data
# nist = np.loadtxt("data/basics/NIST_0.1s.txt")
# dark = np.loadtxt("data/basics/dark_0.1s.txt")
# nist = nist - dark
# # norm NIST data and SRM_interp
# nist_norm = nist / np.max(nist)
# i_srm_interp = i_srm_interp / np.max(i_srm_interp)
# # get res_curve
# out = gaussian_filter1d(i_srm_interp / nist_norm, sigma=15, mode='nearest')

# # # ============================
# # skin = np.loadtxt("data/basics/ROI1_2s_2.txt")
# # skin = skin - 0.5*np.min(skin)
# # skin = skin / np.max(skin)

# # plt.plot(wvn[331:], skin[331:] / out[331:], label='Skin')
# # plt.plot(wvn[331:], nist[331:] / out[331:], label='NIST')
# # plt.legend()
# # plt.savefig("tmp/zz.jpg")

# # ============================
# nist = nist / out  # Normalize NIST data by the res_curve
# nist = nist[331:]  # Adjust to match the skin data
# wvn = wvn[331:]  # Adjust to match the skin data
# # Step 1: Remove the smooth fluorescence background
# # Option 1: Detrend (for quick use)
# nist_smooth = gaussian_filter1d(nist, sigma=15, mode='nearest')
# nist_detrended = nist / nist_smooth - 1  # or use baseline subtraction if needed
# # nist_detrended = nist - nist_smooth

# signal = nist_detrended
# x = wvn  # Assuming wavenumber is the x-axis

# # Your signal: 'signal' and axis: 'x' should already be defined (e.g., fluorescence signal)
# emd = EMD()
# IMFs = emd(signal)

# # Plot first few IMFs
# plt.figure(figsize=(12, 2 * (len(IMFs) + 1)))
# plt.subplot(len(IMFs)+1, 1, 1)
# plt.plot(x, signal)
# plt.title("Original Signal")

# for i, imf in enumerate(IMFs[:5]):
#     plt.subplot(len(IMFs)+1, 1, i+2)
#     plt.plot(x, imf)
#     plt.title(f"IMF {i+1}")
# plt.tight_layout()
# plt.savefig("tmp/emd_imfs.jpg")

# # Choose some mid-frequency IMFs to reconstruct etalon (typically IMF 2 + 3)
# etaloning_emd = IMFs[1] + IMFs[2] + IMFs[3]

# plt.figure(figsize=(12, 4))
# plt.plot(x, signal, label='Original Signal')
# plt.plot(x, etaloning_emd, label='Etalon Pattern (EMD)', linestyle='--')
# plt.legend()
# plt.title("Etalon Interference Extracted with EMD")
# plt.xlabel("Wavenumber")
# plt.tight_layout()
# plt.savefig("tmp/etaloning_emd.jpg")

class EtaloningGenerator:
    def __init__(self):
        """
        Initialize the EtaloningGenerator with a signal and wavenumber.
        """
        self.etaloning = None
        self.wvn = None
        
    def get_etaloning(self, nist_dir="data/basics/NIST_0.1s.txt", dark_dir="data/basics/dark_0.1s.txt", wvn_dir="data/wvn_raw.mat"):
        """
        Extract etaloning pattern from the signal using EMD.
        """
        file  = 'data/basics/SRM 2246 certified log normal calculations.xls'
        sheet = 'SRM 2246 Example'

        wvn = loadmat(wvn_dir)["wvn"].reshape(-1)
        # we want rows 12–3335 (1-based) in cols D and I ⇒ zero-based skiprows=11, nrows=3335-11=3324
        # usecols='D,I' pulls in exactly those two columns
        df = pd.read_excel(
            file,
            sheet_name=sheet,
            header=None,        # no header row in our slice
            skiprows=11,        # drop rows 0–10 (i.e. Excel rows 1–11)
            nrows=3324,         # rows 12 through 3335 inclusive
            usecols='D,I'       # only columns D and I
        )

        # now df.iloc[:,0] is the D12:D3335 range, df.iloc[:,1] is I12:I3335
        wvn_srm = df.iloc[:, 0].to_numpy()
        i_srm   = df.iloc[:, 1].to_numpy()
        # interp SRM_interp
        i_srm_interp = np.interp(wvn, wvn_srm, i_srm)
        # load NIST data
        nist = np.loadtxt(nist_dir)
        dark = np.loadtxt(dark_dir)
        nist = nist - dark
        # norm NIST data and SRM_interp
        nist_norm = nist / np.max(nist)
        # get res_curve
        out = gaussian_filter1d(i_srm_interp / nist_norm, sigma=15, mode='nearest')
        
        nist = nist / out  # Normalize NIST data by the res_curve
        nist = nist[331:]  # Adjust to match the skin data
        wvn = wvn[331:]  # Adjust to match the skin data
        # Step 1: Remove the smooth fluorescence background
        # Option 1: Detrend (for quick use)
        nist_smooth = gaussian_filter1d(nist, sigma=15, mode='nearest')
        nist_detrended = nist / nist_smooth - 1  # or use baseline subtraction if needed
        # nist_detrended = nist - nist_smooth

        signal = nist_detrended

        # Your signal: 'signal' and axis: 'x' should already be defined (e.g., fluorescence signal)
        emd = EMD()
        IMFs = emd(signal)
        
        etaloning_emd = IMFs[1] + IMFs[2] + IMFs[3]
            
        self.etaloning = etaloning_emd
        self.wvn = wvn
        
        return

    def generate_etaloning(self, random=True, intercept=-1, slope=-1):
        """
        Generate etaloning pattern from the signal using linear curve.
        """
        if random and (intercept == -1 and slope == -1):
            b = np.random.uniform(0.1, 1)
            m = b
            x_len = np.max(self.wvn) - np.min(self.wvn)
            k_min = -m / x_len
            k_max = (1 - m) / x_len
            k = np.random.uniform(k_min, k_max)
        else:
            b = intercept
            k = slope

        if self.wvn is None or self.etaloning is None:
            raise ValueError("Call get_etaloning() first to initialize wavenumber and etaloning.")

        x = self.wvn
        y = k * (x - np.min(x)) + b
        out = y * self.etaloning
        return out
    
    def generate_multiple_etaloning(self, num=10, random=True, intercept=-1, slope=-1):
        """
        Generate multiple etaloning patterns.
        """
        if self.wvn is None or self.etaloning is None:
            raise ValueError("Call get_etaloning() first to initialize wavenumber and etaloning.")
        
        etalonings = []
        for _ in range(num):
            etalonings.append(self.generate_etaloning(random=random, intercept=intercept, slope=slope))
        
        return np.array(etalonings)


if __name__ == "__main__":
    etaloning_gen = EtaloningGenerator()
    etaloning_gen.get_etaloning()
    
    # Generate a single etaloning pattern
    etalonings = etaloning_gen.generate_multiple_etaloning(num=1000, random=True)
    
    os.makedirs("data/generated/etaloning/", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/generated/etaloning/etalonings_{timestamp}_test.pkl"
    with open(filename, "wb") as f:
        pickle.dump(etalonings, f)
    
    # Plot all generated etaloning patterns together, separated along y-axis
    plt.figure(figsize=(10, 5))
    etalonings_plot = etalonings[:5]
    offset = 1.2 * np.max(np.abs(etalonings_plot))  # vertical offset between signals
    for i in range(etalonings_plot.shape[0]):
        plt.plot(etaloning_gen.wvn, etalonings_plot[i] + i * offset, label=f'Etaloning {i+1}')
    plt.title('Generated Etaloning Patterns (Offset)')
    plt.xlabel('Wavenumber')
    plt.yticks([])  # Ignore yticks
    plt.legend()
    plt.savefig("tmp/generated_etaloning.jpg")