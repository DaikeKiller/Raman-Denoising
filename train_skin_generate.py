import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from models.Network import RamanNoiseNet, RamanNoiseNet_HF, RamanNoiseNet_LF
from models.AUnet import AUnet
from utils.skin_generation_dataset import SkinGenerationDataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from scipy.signal import resample
from scipy.fftpack import idct
from scipy.signal import savgol_filter


def read_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    cleaned_signals, gt_signals = data["noisy_signals"], data["gt_signals"]
    return cleaned_signals, gt_signals

def apply_SG(signals, sg_params_dir):
    with open(sg_params_dir, 'rb') as file:
        sg_params = pickle.load(file)
    if sg_params is not None:
        window_length, polyorder = sg_params
        sg_filtered_signals = np.apply_along_axis(
            lambda spectrum: savgol_filter(spectrum, window_length=window_length, polyorder=polyorder),
            axis=1,
            arr=signals
        )
    return sg_filtered_signals

def norm(signals):
    mean_ = np.mean(signals, axis=1).reshape(-1,1)
    signals_new = signals - mean_
    max_ = np.max(signals_new, axis=1).reshape(-1,1)
    signals_out = signals_new / max_
    return signals_out, mean_, max_

def reload_train_dataloader():
    train_dataset = SkinGenerationDataset(input_spectra=input_spectra_train, true_spectra=true_spectra_train)
    train_dataset.DCT()
    print("-------- Reloaded Dataset ---------")
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training Function
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device, save_path):
    model.to(device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        if (epoch) % 1 == 0 and epoch != 0:
            train_dataloader = reload_train_dataloader()
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} Training', unit="batch")

        for inputs, gt in progress_bar:
            # Move data to the appropriate device
            inputs = inputs.unsqueeze(1).float().to(device)  # Shape: (batch_size, 1, length)
            gt = gt.unsqueeze(1).float().to(device)      # Shape: (batch_size, 1, length)
            
            # # center the signal
            # centered = torch.mean(inputs, dim=2, keepdim=True)
            # inputs = inputs - centered
            # gt = gt - centered

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(inputs)
            
            loss = 100 * criterion(outputs, gt)

            # # Calculate IDCT of output and ground truth for time-domain loss
            # idct_output = torch.tensor(
            #     idct(outputs.detach().cpu().numpy(), type=2, norm='ortho', axis=-1),
            #     dtype=outputs.dtype, device=device
            # )
            # idct_target = torch.tensor(
            #     idct(gt.detach().cpu().numpy(), type=2, norm='ortho', axis=-1),
            #     dtype=gt.dtype, device=device
            # )
            # idct_loss = criterion(idct_output, idct_target)

            # Regularize the mean difference between output and ground truth
            # mean_reg_loss = (outputs.mean() - gt.mean()) ** 2

            # loss = 1000 * dct_loss + 1000 * idct_loss + 10 * mean_reg_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        # Validation Phase
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation for validation
            for inputs, gt in val_dataloader:
                inputs = inputs.unsqueeze(1).float().to(device)
                gt = gt.unsqueeze(1).float().to(device)
                
                # # center the signal
                # centered = torch.mean(inputs, dim=2, keepdim=True)
                # inputs = inputs - centered
                # gt = gt - centered
                       
                # true_noise = normalization(true_noise)

                outputs = model(inputs)

                loss = 100 * criterion(outputs, gt)

                # idct_output = torch.tensor(
                #     idct(outputs.detach().cpu().numpy(), type=2, norm='ortho', axis=-1),
                #     dtype=outputs.dtype, device=device
                # )
                # idct_target = torch.tensor(
                #     idct(gt.detach().cpu().numpy(), type=2, norm='ortho', axis=-1),
                #     dtype=gt.dtype, device=device
                # )
                # idct_loss = criterion(idct_output, idct_target)

                # Regularize the mean difference between output and ground truth
                # mean_reg_loss = (outputs.mean() - gt.mean()) ** 2

                # loss = 1000 * dct_loss + 1000 * idct_loss + 10 * mean_reg_loss

                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        # Check if this is the best validation loss and save the model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Epoch [{epoch+1}/{num_epochs}] - New best model saved with val loss: {best_val_loss:.4f}")

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    print('Training complete.')
    return train_losses, val_losses
    
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    filename_train = "results/results_for_skin_train.pkl"
    input_spectra_train, true_spectra_train = read_data(filename_train)
    input_spectra_train = apply_SG(input_spectra_train, "results/best_SG_params.pkl")
    input_spectra_train, mean_train, max_train = norm(input_spectra_train)
    true_spectra_train = (true_spectra_train - mean_train) / max_train
    filename_val = "results/results_for_skin_val.pkl"
    input_spectra_val, true_spectra_val = read_data(filename_val)
    input_spectra_val = apply_SG(input_spectra_val, "results/best_SG_params.pkl")
    input_spectra_val, mean_val, max_val = norm(input_spectra_val)
    true_spectra_val = (true_spectra_val - mean_val) / max_val
    
    num_epochs = 400
    batch_size = 32
    learning_rate = 2e-6
    save_dir = "models/pretrained/"
    timestamp = time.strftime("%m%d%Y_%H%M%S")

    save_name = f"model_{timestamp}_skin_generation_from_SG.pth"
    save_path = os.path.join(save_dir, save_name)
    
    model = AUnet(1, 1)
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.99)
    
    train_dataset = SkinGenerationDataset(input_spectra=input_spectra_train, true_spectra=true_spectra_train)
    train_dataset.DCT()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = SkinGenerationDataset(input_spectra=input_spectra_val, true_spectra=true_spectra_val)
    val_dataset.DCT()
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    train_loss, val_loss = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device, save_path)
    