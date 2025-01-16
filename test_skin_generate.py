import pickle
import torch
from models.AUnet import AUnet
# from train import read_clean_data
from utils.skin_generation_dataset import SkinGenerationDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.fftpack import idct
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
import pywt
from train_skin_generate import apply_SG


def read_data(filename, type="from_cleaned"):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    if type == "from_cleaned":
        out_signals, gt_signals = data["cleaned_signals"], data["gt_signals"]
    if type == "from_noisy":
        out_signals, gt_signals = data["noisy_signals"], data["gt_signals"]
    if type == "from_SG":
        out_signals, gt_signals = data["noisy_signals"], data["gt_signals"]
        out_signals = apply_SG(out_signals, "results/best_SG_params.pkl")
    return out_signals, gt_signals, data

def norm(signals):
    mean_ = np.mean(signals, axis=1).reshape(-1,1)
    signals_new = signals - mean_
    max_ = np.max(signals_new, axis=1).reshape(-1,1)
    signals_out = signals_new / max_
    return signals_out, mean_, max_

def test_model(model, test_dataloader, device):
    model.to(device)
    model.eval()
    results = []
    with torch.no_grad():  # Disable gradient calculation for testing
        # Progress bar for testing phase
        for input_spectra, _ in tqdm(test_dataloader, desc="Testing", unit="batch"):
            input_spectra = input_spectra.float().unsqueeze(1).to(device)
            predicted_spectra = model(input_spectra).squeeze(1).cpu().numpy()

            predicted_spectra_idct = idct(predicted_spectra, norm="ortho")
            results.append(predicted_spectra_idct)
        results = np.concatenate(results, axis=0)
    return results

def add_other_methods_for_comparison(data):
    # SG filter
    signals = data["noisy_signals"]
    
    def SG_filter(singals):
        window_length = 100  # Should be odd and <= number of data points
        polyorder = 4       # Polynomial order for smoothing

        # Apply SG filter to each spectrum
        cleaned_data = np.apply_along_axis(
            lambda spectrum: savgol_filter(spectrum, window_length=window_length, polyorder=polyorder),
            axis=1,
            arr=signals
        )
        return cleaned_data
    
    def wavelet_denoise(spectrum, wavelet='db4', level=3, thresholding='soft'):
        # Wavelet decomposition
        coeffs = pywt.wavedec(spectrum, wavelet, level=level)
        
        # Thresholding
        sigma = np.median(np.abs(coeffs[-1])) / 0.01  # Estimate noise level
        threshold = sigma * np.sqrt(2 * np.log(len(spectrum)))
        denoised_coeffs = [
            pywt.threshold(c, value=threshold, mode=thresholding) if i > 0 else c
            for i, c in enumerate(coeffs)
        ]
        
        # Reconstruct the denoised signal
        denoised_spectrum = pywt.waverec(denoised_coeffs, wavelet)
        
        # Ensure the reconstructed spectrum has the same length
        return denoised_spectrum[:len(spectrum), :spectrum.shape[1]]
    
    SG_out = SG_filter(signals)
    wavelet_out = wavelet_denoise(signals)
    
    data["SG_denoise"] = SG_out
    data["wavelet_denoise"] = wavelet_out
    
    save_results = os.path.join("results/results_for_skin_test.pkl")
    with open(save_results, 'wb') as f:
        pickle.dump(data, f)
    
    return data

def plot_signals(noisy_signals, cleaned_signals, SG_denoised, skin_gen_from_cleaned, skin_gen_from_noisy, skin_gen_from_SG, gt_signals, SNR_list, num_samples=5, save_path="./results/"):
    # Sort indices based on SNR_list in ascending order
    sorted_indices = np.argsort(SNR_list)
    SNR_list = np.array(SNR_list)[sorted_indices]
    noisy_signals = np.array(noisy_signals)[sorted_indices]
    cleaned_signals = np.array(cleaned_signals)[sorted_indices]
    SG_denoised = np.array(SG_denoised)[sorted_indices]
    skin_gen_from_cleaned = np.array(skin_gen_from_cleaned)[sorted_indices]
    skin_gen_from_noisy = np.array(skin_gen_from_noisy)[sorted_indices]
    skin_gen_from_SG = np.array(skin_gen_from_SG)[sorted_indices]
    gt_signals = np.array(gt_signals)[sorted_indices]
    
    # Get the total number of signals
    total_signals = len(noisy_signals)
    
    # If the requested number of samples exceeds the total signals, limit it to the available number
    num_samples = min(num_samples, total_signals)
    
    selected_indices = []
    for i in range(num_samples):
        selected_indices.append(i*total_signals//num_samples + total_signals//num_samples//10 - 2)
        
    # Select random samples
    # selected_indices = sorted(random.sample(range(len(SNR_list)), num_samples))
    
    # =========== figure for only skin gen from cleaned ================
    fig, axs = plt.subplots(num_samples, 4, figsize=(20, num_samples * 3))
    
    for i, idx in enumerate(selected_indices):
        axs[i, 0].plot(np.linspace(800, 1790, 1981), noisy_signals[idx], label="Original low-SNR spectra")
        axs[i, 0].set_title(f"Original low-SNR spectra, SNR = {SNR_list[idx]:.2f}")
        # axs[i, 0].legend()

        axs[i, 1].plot(np.linspace(800, 1790, 1981), cleaned_signals[idx], label="Denoised spectra with model", color='green')
        axs[i, 1].set_title(f"Denoised spectra with model")
        # axs[i, 3].legend()
        
        axs[i, 2].plot(np.linspace(800, 1790, 1981), skin_gen_from_cleaned[idx], label="Skin generation output", color='green')
        axs[i, 2].set_title(f"Skin generation output")

        axs[i, 3].plot(np.linspace(800, 1790, 1981), gt_signals[idx], label="Pure spectra", color='orange')
        # axs[i, 4].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - cleaned_signals[idx], label="Residual_model", color='black')
        axs[i, 3].set_title(f"Pure spectra")
        # axs[i, 4].legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "signal_skin_gen_from_cleaned_all_test.jpg"))

    fig, axs = plt.subplots(num_samples, 1, figsize=(5, num_samples * 3))
    for i, idx in enumerate(selected_indices):
        # axs[i].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - cleaned_signals[idx], label="Residual_model", color='black')
        axs[i].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - skin_gen_from_cleaned[idx], label="Residual_from_cleaned", color='red')
        # axs[i].set_title(f"Smaple {idx}")
        # axs[i].legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "residual_skin_gen_from_cleaned_all_test.jpg"))
    
    # ==================== figure for only skin gen from noisy ================
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, num_samples * 3))
    
    for i, idx in enumerate(selected_indices):
        axs[i, 0].plot(np.linspace(800, 1790, 1981), noisy_signals[idx], label="Original low-SNR spectra")
        axs[i, 0].set_title(f"Original low-SNR spectra, SNR = {SNR_list[idx]:.2f}")
        # axs[i, 0].legend()
        
        axs[i, 1].plot(np.linspace(800, 1790, 1981), skin_gen_from_noisy[idx], label="Skin generation output", color='green')
        axs[i, 1].set_title(f"Skin generation output")

        axs[i, 2].plot(np.linspace(800, 1790, 1981), gt_signals[idx], label="Pure spectra", color='orange')
        # axs[i, 4].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - cleaned_signals[idx], label="Residual_model", color='black')
        axs[i, 2].set_title(f"Pure spectra")
        # axs[i, 4].legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "signal_skin_gen_from_noisy_all_test.jpg"))

    fig, axs = plt.subplots(num_samples, 1, figsize=(5, num_samples * 3))
    for i, idx in enumerate(selected_indices):
        # axs[i].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - cleaned_signals[idx], label="Residual_model", color='black')
        axs[i].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - skin_gen_from_noisy[idx], label="Residual_from_noisy", color='black')
        # axs[i].set_title(f"Smaple {idx}")
        # axs[i].legend()
    
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "residual_skin_gen_from_noisy_all_test.jpg"))
    
    # ==================== figure for only skin gen from SG ================
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, num_samples * 3))
    
    for i, idx in enumerate(selected_indices):
        axs[i, 0].plot(np.linspace(800, 1790, 1981), noisy_signals[idx], label="Original low-SNR spectra")
        axs[i, 0].set_title(f"Original low-SNR spectra, SNR = {SNR_list[idx]:.2f}")
        # axs[i, 0].legend()
        
        axs[i, 1].plot(np.linspace(800, 1790, 1981), skin_gen_from_SG[idx], label="Skin generation output", color='green')
        axs[i, 1].set_title(f"Skin generation output")

        axs[i, 2].plot(np.linspace(800, 1790, 1981), gt_signals[idx], label="Pure spectra", color='orange')
        # axs[i, 4].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - cleaned_signals[idx], label="Residual_model", color='black')
        axs[i, 2].set_title(f"Pure spectra")
        # axs[i, 4].legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "signal_skin_gen_from_SG_all_test.jpg"))

    fig, axs = plt.subplots(num_samples, 1, figsize=(5, num_samples * 3))
    for i, idx in enumerate(selected_indices):
        # axs[i].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - cleaned_signals[idx], label="Residual_model", color='black')
        axs[i].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - skin_gen_from_SG[idx], label="Residual_from_SG", color='black')
        # axs[i].set_title(f"Smaple {idx}")
        # axs[i].legend()
    
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "residual_skin_gen_from_SG_all_test.jpg"))
    
    # =========== figure for all ===============
    fig, axs = plt.subplots(num_samples, 4, figsize=(25, num_samples * 3))
    
    for i, idx in enumerate(selected_indices):
        axs[i, 0].plot(np.linspace(800, 1790, 1981), noisy_signals[idx], label="Original low-SNR spectra")
        axs[i, 0].set_title(f"Original low-SNR spectra, SNR = {SNR_list[idx]:.2f}")
        # axs[i, 0].legend()
        
        axs[i, 1].plot(np.linspace(800, 1790, 1981), noisy_signals[idx], label="Original low-SNR", color='red', linewidth=1)
        axs[i, 1].plot(np.linspace(800, 1790, 1981), SG_denoised[idx], label="SG filtered", color='blue', linewidth=2)
        axs[i, 1].plot(np.linspace(800, 1790, 1981), cleaned_signals[idx], label="Noise Removal", color='green', linewidth=2)
        axs[i, 1].set_title(f"Denoised output")
        axs[i, 1].legend()
        
        axs[i, 2].plot(np.linspace(800, 1790, 1981), skin_gen_from_noisy[idx], label="Skin generation from low-SNR", color='red', linewidth=1)
        axs[i, 2].plot(np.linspace(800, 1790, 1981), skin_gen_from_SG[idx], label="Skin generation from SG", color='blue', linewidth=2)
        axs[i, 2].plot(np.linspace(800, 1790, 1981), skin_gen_from_cleaned[idx], label="Skin generation from noise_removal", color='green', linewidth=2)
        axs[i, 2].set_title(f"Skin generation output")
        axs[i, 2].legend()

        axs[i, 3].plot(np.linspace(800, 1790, 1981), gt_signals[idx], label="Pure spectra", color='orange', linewidth=2)
        # axs[i, 4].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - cleaned_signals[idx], label="Residual_model", color='black')
        axs[i, 3].set_title(f"Pure spectra")
        # axs[i, 4].legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "signal_skin_gen_from_all_all_test.jpg"))

    fig, axs = plt.subplots(num_samples, 1, figsize=(5, num_samples * 3))
    for i, idx in enumerate(selected_indices):
        # axs[i].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - cleaned_signals[idx], label="Residual_model", color='black')
        axs[i].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - skin_gen_from_cleaned[idx], label="Residual_from_noise_removal", color='green')
        axs[i].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - skin_gen_from_noisy[idx], label="Residual_from_noisy", color='red')
        axs[i].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - skin_gen_from_SG[idx], label="Residual_from_SG", color='black')
        # axs[i].set_title(f"Smaple {idx}")
        axs[i].legend()
        
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "residual_skin_gen_from_all_all_test.jpg"))
        
    
def main(model_path, process_data_type = "from_cleaned", save_flag = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    filename_test = "results/results_for_skin_test.pkl"
    input_spectra_test, true_spectra_test, data = read_data(filename_test, process_data_type)
    input_spectra_test, mean_test, max_test = norm(input_spectra_test)
    true_spectra_test = (true_spectra_test - mean_test) / max_test

    save_data_flag = save_flag
    save_path = "./results/"

    # Load the best trained model
    model_path = model_path
    model = AUnet(1, 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    test_dataset = SkinGenerationDataset(input_spectra=input_spectra_test, true_spectra=true_spectra_test)
    test_dataset.DCT()
    test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Test the model and get predicted spectra and cleaned signals
    # predicted_spectra, cleaned_signals, noisy_signals, moving_window_cleaned, noise_avg_signals, true_spectra, gt_signals, SNR_list = test_on_one_signal(model_HF, model_MF, model_LF, test_dataloader, device)
    output_spectra = test_model(model, test_dataloader, device)
    output_spectra = output_spectra * max_test + mean_test
    
    data["cleaned_signals_" + process_data_type] = output_spectra
    
    if save_data_flag:
        save_results = os.path.join(save_path, "results_for_skin_test.pkl")
        with open(save_results, 'wb') as f:
            pickle.dump(data, f)

    # plot_signals(data["noisy_signals"], data["cleaned_signals"], output_spectra, data["gt_signals"], data["SNR_list"])
    
    print("complete. Results saved.")
    
    return data
    

if __name__ == "__main__":
    model_path_from_cleaned = "models/pretrained/model_12262024_140651_skin_generation.pth"
    model_path_from_noisy = "models/pretrained/model_12302024_095023_skin_generation_from_noisy.pth"
    model_path_from_SG = "models/pretrained/model_01132025_110850_skin_generation_from_SG.pth"
    data = main(model_path=model_path_from_cleaned, process_data_type = "from_cleaned", save_flag = True)
    data = main(model_path=model_path_from_noisy, process_data_type = "from_noisy", save_flag = True)
    data = main(model_path=model_path_from_noisy, process_data_type = "from_SG", save_flag = True)
    # data = add_other_methods_for_comparison(data)
    
    plot_signals(data["noisy_signals"], data["cleaned_signals"], data["SG_denoise"], data["cleaned_signals_from_cleaned"], \
                 data["cleaned_signals_from_noisy"], data["cleaned_signals_from_SG"], data["gt_signals"], data["SNR_list"])