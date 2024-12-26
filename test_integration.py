import pickle
import torch
from models.AUnet import AUnet
# from train import read_clean_data
from utils.Integrate_dataset import IntegrationDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.fftpack import idct
import matplotlib.pyplot as plt


def read_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    noisy_signals = data["noisy_signals"]
    cleaned_signals, gt_signals = data["cleaned_signals"], data["gt_signals"]
    return noisy_signals - cleaned_signals, noisy_signals - gt_signals, data

def test_model(model, test_dataloader, device):
    model.to(device)
    model.eval()
    results = []
    with torch.no_grad():  # Disable gradient calculation for testing
        # Progress bar for testing phase
        for input_noises, _ in tqdm(test_dataloader, desc="Testing", unit="batch"):
            input_noises = input_noises.float().unsqueeze(1).to(device)
            center_factor = torch.mean(input_noises, dim=2, keepdim=True)
            input_noises = input_noises - center_factor
            predicted_noises = model(input_noises).squeeze(1).cpu().numpy()
            predicted_noises = predicted_noises + center_factor.squeeze(1).cpu().numpy()

            # predicted_noises_idct = idct(predicted_noises, norm="ortho")
            results.append(predicted_noises)
        results = np.concatenate(results, axis=0)
    return results

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    filename_test = "results/results_all_pV_test.pkl"
    input_noises_test, true_noises_test, data = read_data(filename_test)

    save_data_flag = False
    save_path = "./results/"

    # Load the best trained model
    model_path = "models/pretrained/model_12172024_163524_integration.pth"
    model = AUnet(1, 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    test_dataset = IntegrationDataset(input_noises=input_noises_test, true_noises=true_noises_test)
    # test_dataset.DCT()
    test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Test the model and get predicted noises and cleaned signals
    # predicted_noises, cleaned_signals, noisy_signals, moving_window_cleaned, noise_avg_signals, true_noises, gt_signals, SNR_list = test_on_one_signal(model_HF, model_MF, model_LF, test_dataloader, device)
    output_noises = test_model(model, test_dataloader, device)

    # plt.figure(figsize=())
    
    print("complete. Results saved.")