from train_nl import *
from scipy.fftpack import idct, dct
import random
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
import cv2
from scipy.io import loadmat
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd


def use_dl_model(model, data, eta):
    """
    Use the trained model to make predictions on the data.
    Args:
        model: The trained model.
        data: The input data for denoising, numpy.array, shape (n, L)
        eta: The etaloning got from NIST, numpy.array, shape (1, L)
    Returns:
        The denoised made by the model.
        The output is a dictionary.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        noisy_signal = torch.from_numpy(data).unsqueeze(1).float().to(device)
        max_values = noisy_signal.max(dim=2, keepdim=True)[0]
        noisy_signal_dct = dct_torch(noisy_signal)
        noisy_signal_dct_norm = noisy_signal_dct / max_values
        
        input = torch.cat([noisy_signal_dct_norm, torch.from_numpy(eta).unsqueeze(0).repeat(data.shape[0], 1, 1).float().to(device)], dim=1)
        out = model(input)
        
        denoised_signal = out["denoised_signal"].squeeze(1).cpu().numpy() * max_values.squeeze(1).cpu().numpy()
        raman_signal = out["raman_signal"].squeeze(1).cpu().numpy() * max_values.squeeze(1).cpu().numpy()
        original_signal = noisy_signal.squeeze(1).cpu().numpy()
    
    output = {
        "denoised_signal": denoised_signal,
        "raman_signal": raman_signal,
        "original_signal": original_signal,
    }
    # Assuming the model has a predict method
    return output

def NIST_processing(NIST_filename, dark_filename, NIST_ref_filename, wvn_filename):
    """
    Process the NIST data file to extract the etaloning and other relevant information.
    Args:
        filename: The path to the NIST data file.
    Returns:
        A dictionary containing the etaloning and other relevant data.
    """
    # file + sheet
    file  = NIST_ref_filename
    sheet = 'SRM 2246 Example'

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
    # load wvn
    wvn = loadmat(wvn_filename)["wvn"].reshape(-1)
    # interp SRM_interp
    i_srm_interp = np.interp(wvn, wvn_srm, i_srm)
    # load NIST data
    nist = np.loadtxt(NIST_filename)
    dark = np.loadtxt(dark_filename)
    nist = nist - dark
    # norm NIST data and SRM_interp
    nist = nist / np.max(nist)
    i_srm_interp = i_srm_interp / np.max(i_srm_interp)
    # get res_curve
    res_curve = gaussian_filter1d(i_srm_interp / nist, sigma=15, mode='nearest')
    
    eta_gen = EtaloningGenerator()
    eta_gen.get_etaloning(nist_dir=NIST_filename, dark_dir=dark_filename, wvn_dir=wvn_filename)
    eta_signal = eta_gen.etaloning
    
    out = {
        "original_nist": nist[wvn_600:],
        "res_curve": res_curve[wvn_600:],
        "wvn": wvn[wvn_600:],
        "i_srm_interp": i_srm_interp[wvn_600:],
        "eta_signal": eta_signal
    }
    
    return out

def read_data_folder(data_dir):
    """
    Read the data from the specified directory.
    Args:
        data_dir: The directory containing the data files.
    Returns:
        A numpy array containing the data.
    """
    # Assuming the data is in a specific format, e.g., CSV or text files
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.txt') or f.endswith('.csv')]
    data = []
    
    for file in data_files:
        file_path = os.path.join(data_dir, file)
        if file.endswith('.txt'):
            data.append(np.loadtxt(file_path)[wvn_600:])
        elif file.endswith('.csv'):
            data.append(pd.read_csv(file_path).values[wvn_600:])
    
    return np.array(data)

def read_data_single(data_file):
    """
    Read a single data file.
    Args:
        data_file: The path to the data file.
    Returns:
        A numpy array containing the data.
    """
    if data_file.endswith('.txt'):
        return np.loadtxt(data_file)[wvn_600:].reshape(1, -1)
    elif data_file.endswith('.csv'):
        return pd.read_csv(data_file).values[wvn_600:].reshape(1, -1)
    else:
        raise ValueError("Unsupported file format. Please use .txt or .csv files.")

def load_model(model_dir):
    """
    Load the trained model from the specified directory.
    Args:
        model_dir: The directory containing the model file.
    Returns:
        The loaded model.
    """
    # model_full_path = "models/pretrained/new_noise_model_05202025_151616_end_to_end.pth"
    model = TwoStageModel()
    model.load_state_dict(torch.load(model_dir))
    return model

def pre_processing(data, dark, NIST_out):
    """
    Pre-process the data using the NIST etaloning and other parameters.
    Args:
        data: The input data to be pre-processed.
        NIST_out: The output from the NIST processing function.
    Returns:
        The pre-processed data.
    """
    return (data - dark) * NIST_out["res_curve"]

def plot(model_out):
    """
    Plot the results from the model output.
    Args:
        model_out: The output from the model.
    """
    plt.figure(figsize=(12, 6))
    idx = np.random.randint(0, model_out["original_signal"].shape[0])
    
    plt.subplot(2, 2, 1)
    plt.plot(model_out["original_signal"][idx], label='Original Signal')
    plt.title('Original Signal')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(model_out["denoised_signal"][idx], label='Denoised Signal', color='green')
    plt.title('Denoised Signal')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(model_out["raman_signal"][idx], label='Raman Signal', color='red')
    plt.title('Raman Signal')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/sani_data.jpg")


def main():
    data_dir = "data/basics/sani_data/"
    dark_dir = "data/basics/dark_2s_quartz.txt"
    model_dir = "models/pretrained/etaloning/end_to_end_06032025_153912.pth"
    NIST_dir = "data/basics/NIST_0.1s.txt"
    NIST_ref_dir = "data/basics/SRM 2246 certified log normal calculations.xls"
    NIST_dark_dir = "data/basics/dark_0.1s.txt"
    wvn_dir = "data/wvn_raw.mat"
    
    global wvn_600
    wvn_600 = 331
    
    data = read_data_folder(data_dir)
    dark = read_data_single(dark_dir)
    model = load_model(model_dir)
    NIST_out = NIST_processing(
        NIST_filename=NIST_dir,
        dark_filename=NIST_dark_dir,
        NIST_ref_filename=NIST_ref_dir,
        wvn_filename=wvn_dir
    )
    
    data_processed = pre_processing(data, dark, NIST_out)
    
    model_out = use_dl_model(model, data_processed, NIST_out["eta_signal"])
    plot(model_out)
    

if __name__ == "__main__":
    main()