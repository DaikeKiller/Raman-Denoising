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


def read_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    cleaned_signals, gt_signals = data["cleaned_signals"], data["gt_signals"]
    return cleaned_signals, gt_signals, data

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

def plot_signals(noisy_signals, cleaned_signals, skin_gen, gt_signals, SNR_list, num_samples=5, save_path="./results/"):
    # Sort indices based on SNR_list in ascending order
    sorted_indices = np.argsort(SNR_list)
    SNR_list = np.array(SNR_list)[sorted_indices]
    noisy_signals = np.array(noisy_signals)[sorted_indices]
    cleaned_signals = np.array(cleaned_signals)[sorted_indices]
    skin_gen = np.array(skin_gen)[sorted_indices]
    gt_signals = np.array(gt_signals)[sorted_indices]
    
    # Get the total number of signals
    total_signals = len(noisy_signals)
    
    # If the requested number of samples exceeds the total signals, limit it to the available number
    num_samples = min(num_samples, total_signals)
    
    selected_indices = []
    for i in range(num_samples):
        selected_indices.append(i*total_signals//num_samples + total_signals//num_samples//10)
        
    # Select random samples
    # selected_indices = sorted(random.sample(range(len(SNR_list)), num_samples))
    
    fig, axs = plt.subplots(num_samples, 4, figsize=(20, num_samples * 3))
    
    for i, idx in enumerate(selected_indices):
        axs[i, 0].plot(np.linspace(800, 1790, 1981), noisy_signals[idx], label="Original low-SNR spectra")
        axs[i, 0].set_title(f"Original low-SNR spectra, SNR = {SNR_list[idx]:.2f}")
        # axs[i, 0].legend()

        axs[i, 1].plot(np.linspace(800, 1790, 1981), cleaned_signals[idx], label="Denoised spectra with model", color='green')
        axs[i, 1].set_title(f"Denoised spectra with model")
        # axs[i, 3].legend()
        
        axs[i, 2].plot(np.linspace(800, 1790, 1981), skin_gen[idx], label="Skin generation output", color='green')
        axs[i, 2].set_title(f"Skin generation output")

        axs[i, 3].plot(np.linspace(800, 1790, 1981), gt_signals[idx], label="Pure spectra", color='orange')
        # axs[i, 4].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - cleaned_signals[idx], label="Residual_model", color='black')
        axs[i, 3].set_title(f"Pure spectra")
        # axs[i, 4].legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "signal_skin_gen_all_test.jpg"))

    fig, axs = plt.subplots(num_samples, 1, figsize=(5, num_samples * 3))
    for i, idx in enumerate(selected_indices):
        axs[i].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - cleaned_signals[idx], label="Residual_model", color='black')
        axs[i].plot(np.linspace(800, 1790, 1981), gt_signals[idx] - skin_gen[idx], label="Residual_moving_window", color='red')
        # axs[i].set_title(f"Smaple {idx}")
        axs[i].legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "residual_skin_gen_all_test.jpg"))

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    filename_test = "results/results_for_skin_test.pkl"
    input_spectra_test, true_spectra_test, data = read_data(filename_test)
    input_spectra_test, mean_test, max_test = norm(input_spectra_test)
    true_spectra_test = (true_spectra_test - mean_test) / max_test

    save_data_flag = False
    save_path = "./results/"

    # Load the best trained model
    model_path = "models/pretrained/model_12262024_140651_skin_generation.pth"
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

    plot_signals(data["noisy_signals"], data["cleaned_signals"], output_spectra, data["gt_signals"], data["SNR_list"])
    
    print("complete. Results saved.")