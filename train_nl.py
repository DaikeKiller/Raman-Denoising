import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.Network import RamanNoiseNet, RamanNoiseNet_HF, RamanNoiseNet_LF
from models.AUnet import AUnet
from utils.Raman_dataset import RamanNoiseDataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.signal import resample
from scipy.fftpack import idct
import math


def read_clean_data(clean_dir, customized_noise=False, pV=False):
    if pV:
        with open(clean_dir, 'rb') as file:
            clean_data = pickle.load(file)
    else:
        with open(clean_dir, 'rb') as file:
            concentrations, clean_data = pickle.load(file)
        # clean_data = resample(clean_data, 603, axis=0)
    if customized_noise is True:
        noise_data = np.random.randn(1000, clean_data.shape[0])
    else:
        noise_data = None
    if pV:
        return clean_data, noise_data
    else:
        return clean_data, noise_data, concentrations

def read_noise_data(file_name):
    std = np.loadtxt(file_name)
    return std

def normalization_for_loss(signal):
    # Generate normalization factor tensor
    # factor = [np.log(50*a) / (8*np.log(50)) for a in range(1, signal.shape[2]+1)]
    factor = [0.5*(a+10) for a in range(1, signal.shape[2]+1)]
    factor = np.array(factor)
    factor = torch.from_numpy(np.reshape(factor, [1, 1, -1])).float()

    # Move the factor to the same device as the signal
    factor = factor.to(signal.device)
    return signal * factor

def normalization_for_input(signal):
    # shape (batch_size, 1, signal_length)
    return signal / torch.max(signal, dim=2, keepdim=True)[0]

def reload_train_dataloader():
    train_dataset = RamanNoiseDataset(clean_signals=train_signal, noise_std=noise_std)
    train_dataset.generate_noisy_signals(SNR_range=SNR_range)
    train_dataset.DCT()
    print("-------- Reloaded Dataset ---------")
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

import math

def idct_torch(x_dct):
    """
    Compute the IDCT type-II of a batched signal using the inverse FFT.
    """
    n, c, length = x_dct.size()
    
    # Scale the input coefficients
    x_dct = x_dct.clone()  # Avoid modifying the original tensor
    x_dct[:, :, 0] *= (2 ** 0.5)  # Scale the first coefficient
    x_dct *= (length / 2) ** 0.5  # Scale all coefficients
    
    # Create the extended symmetric signal
    x_dct_ext = torch.cat((x_dct, x_dct.flip(dims=[-1])), dim=-1)
    
    # Perform inverse FFT
    x_ifft = torch.fft.irfft(x_dct_ext, n=2 * length, dim=-1)
    
    # Extract the original length
    x_reconstructed = x_ifft[..., :length]
    
    return x_reconstructed

# Training Function
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device, save_path, clip="full"):
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

        for noisy_signal, true_noise, _, _ in progress_bar:
            # Move data to the appropriate device
            noisy_signal = noisy_signal.unsqueeze(1).float().to(device)  # Shape: (batch_size, 1, length)
            true_noise = true_noise.unsqueeze(1).float().to(device)      # Shape: (batch_size, 1, length)
            # print(device)
            if clip == "high":
                noisy_signal = noisy_signal[:,:,81+950-50:]
                true_noise = true_noise[:,:,81+950-50:]
                gt = noisy_signal - true_noise
            elif clip == "mid":
                noisy_signal = noisy_signal[:,:,81-50:81+950]
                true_noise = true_noise[:,:,81-50:81+950]
                gt = noisy_signal - true_noise
            elif clip == "low":
                noisy_signal = noisy_signal[:,:,1:81]
                true_noise = true_noise[:,:,1:81]
                gt = noisy_signal - true_noise
                # noise_residual = true_noise
                # true_noise = normalization_for_loss(true_noise)
            elif clip != "full":
                Warning("please input a valid string to the param *clip")
            
            # center the signal
            centered = torch.mean(noisy_signal, dim=2, keepdim=True)
            noisy_signal = noisy_signal - centered
            true_noise = true_noise - centered
            gt = gt - centered

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(noisy_signal)
            pred = noisy_signal - outputs
            pred_idct = idct_torch(pred)
            gt_idct = idct_torch(gt)
            # if clip == "low":
                # outputs = normalization_for_loss(outputs)
            # if clip == "low":
            #     dct_loss = criterion(normalization_for_loss(outputs), normalization_for_loss(noise_residual))
            # else:
            #     dct_loss = criterion(outputs, noise_residual)
            loss_ = criterion(pred_idct, gt_idct)

            # Regularize the mean difference between output and ground truth
            mean_reg_loss = (pred_idct.mean() - gt_idct.mean()) ** 2

            loss = 1000 * loss_ + 10 * mean_reg_loss

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
            for noisy_signal, true_noise, _, _ in val_dataloader:
                noisy_signal = noisy_signal.unsqueeze(1).float().to(device)
                true_noise = true_noise.unsqueeze(1).float().to(device)

                if clip == "high":
                    noisy_signal = noisy_signal[:,:,81+950-50:]
                    true_noise = true_noise[:,:,81+950-50:]
                    gt = noisy_signal - true_noise
                elif clip == "mid":
                    noisy_signal = noisy_signal[:,:,81-50:81+950]
                    true_noise = true_noise[:,:,81-50:81+950]
                    gt = noisy_signal - true_noise
                elif clip == "low":
                    noisy_signal = noisy_signal[:,:,1:81]
                    true_noise = true_noise[:,:,1:81]
                    gt = noisy_signal - true_noise
                    # noise_residual = true_noise
                    # true_noise = normalization_for_loss(true_noise)
                elif clip != "full":
                    Warning("please input a valid string to the param *clip")
            
                # center the signal
                centered = torch.mean(noisy_signal, dim=2, keepdim=True)
                noisy_signal = noisy_signal - centered
                true_noise = true_noise - centered
                gt = gt - centered

                optimizer.zero_grad()  # Zero the gradients

                # Forward pass
                outputs = model(noisy_signal)
                pred = noisy_signal - outputs
                pred_idct = idct_torch(pred)
                gt_idct = idct_torch(gt)
                # if clip == "low":
                    # outputs = normalization_for_loss(outputs)
                # if clip == "low":
                #     dct_loss = criterion(normalization_for_loss(outputs), normalization_for_loss(noise_residual))
                # else:
                #     dct_loss = criterion(outputs, noise_residual)
                loss_ = criterion(pred_idct, gt_idct)

                # Regularize the mean difference between output and ground truth
                mean_reg_loss = (pred_idct.mean() - gt_idct.mean()) ** 2

                loss = 1000 * loss_ + 10 * mean_reg_loss

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

    train_dir = "data/generated/pV_new_noise_model_training_05132025_181058.pkl"
    val_dir = "data/generated/pV_new_noise_model_val_05142025_135815.pkl"
    noise_dir = "data/noise/std/std_0.1s.txt"
    SNR_range = [0.01, 10]

    # Hyperparameters
    num_epochs = 400
    batch_size = 32
    learning_rate_HF = 2e-5
    learning_rate_MF = 2e-4
    learning_rate_LF = 3e-5
    save_dir = "models/pretrained/"
    timestamp = time.strftime("%m%d%Y_%H%M%S")

    save_name_HF = f"model_{timestamp}_HF_NL.pth"
    save_path_HF = os.path.join(save_dir, save_name_HF)
    save_name_MF = f"model_{timestamp}_MF_NL.pth"
    save_path_MF = os.path.join(save_dir, save_name_MF)
    save_name_LF = f"model_{timestamp}_LF_NL.pth"
    save_path_LF = os.path.join(save_dir, save_name_LF)

    # Initialize model, loss function, and optimizer
    # model = RamanNoiseNet()
    model_HF = AUnet(1, 1)
    criterion_HF = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    optimizer_HF = optim.Adam(model_HF.parameters(), lr=learning_rate_HF, weight_decay=0.01)
    model_MF = AUnet(1, 1)
    criterion_MF = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    optimizer_MF = optim.Adam(model_MF.parameters(), lr=learning_rate_MF)
    model_LF = AUnet(1, 1)
    criterion_LF = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    optimizer_LF = optim.Adam(model_LF.parameters(), lr=learning_rate_LF)
    
    # train_signal_skin, _, train_concentrations = read_clean_data(clean_dir=train_dir, customized_noise=False)
    # val_signal_skin, _, val_concentrations = read_clean_data(clean_dir=val_dir, customized_noise=False)
    train_signal, _ = read_clean_data(clean_dir=train_dir, customized_noise=False, pV=True)
    val_signal, _ = read_clean_data(clean_dir=val_dir, customized_noise=False, pV=True)
    noise_std = read_noise_data(noise_dir)

    # Create Dataset and DataLoader
    train_dataset = RamanNoiseDataset(clean_signals=train_signal, noise_std=noise_std)
    train_dataset.generate_noisy_signals(SNR_range=SNR_range)
    train_dataset.DCT()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = RamanNoiseDataset(clean_signals=val_signal, noise_std=noise_std)
    val_dataset.generate_noisy_signals(SNR_range=SNR_range)
    val_dataset.DCT()
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    train_loss_HF, val_loss_HF = train_model(model_HF, train_dataloader, val_dataloader, criterion_HF, optimizer_HF, num_epochs, device, save_path_HF, clip="high")
    train_loss_MF, val_loss_MF = train_model(model_MF, train_dataloader, val_dataloader, criterion_MF, optimizer_MF, num_epochs, device, save_path_MF, clip="mid")
    train_loss_LF, val_loss_LF = train_model(model_LF, train_dataloader, val_dataloader, criterion_LF, optimizer_LF, num_epochs, device, save_path_LF, clip="low")

    # plt.figure
    # plt.subplot(3,1,1)
    # plt.plot(range(num_epochs), train_loss_HF)
    # plt.plot(range(num_epochs), val_loss_HF)
    # plt.legend(["train loss", "validation loss"])
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.title("High Frequency")
    # plt.subplot(3,1,2)
    # plt.plot(range(num_epochs), train_loss_MF)
    # plt.plot(range(num_epochs), val_loss_MF)
    # plt.legend(["train loss", "validation loss"])
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.title("Mid Frequency")
    # plt.subplot(3,1,3)
    # plt.plot(range(400), train_loss_LF)
    # plt.plot(range(400), val_loss_LF)
    # plt.legend(["train loss", "validation loss"])
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.title("Low Frequency")
    # plt.show()
    # plt.savefig("results/training_loss.jpg")