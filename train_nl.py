import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from models.Network import RamanNoiseNet, RamanNoiseNet_HF, RamanNoiseNet_LF
from models.AUnet import AUnet, Double_AUnet
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

# def normalization_for_loss(signal):
#     # Generate normalization factor tensor
#     # factor = [np.log(50*a) / (8*np.log(50)) for a in range(1, signal.shape[2]+1)]
#     factor = [0.5*(a+10) for a in range(1, signal.shape[2]+1)]
#     factor = np.array(factor)
#     factor = torch.from_numpy(np.reshape(factor, [1, 1, -1])).float()

#     # Move the factor to the same device as the signal
#     factor = factor.to(signal.device)
#     return signal * factor

# def normalization_for_input(signal):
#     # shape (batch_size, 1, signal_length)
#     return signal / torch.max(signal, dim=2, keepdim=True)[0]

def reload_train_dataloader():
    train_dataset = RamanNoiseDataset(clean_signals=train_signal, noise_std_list=noise_std_dict, fluorescence=fluo_train_signal)
    train_dataset.generate_noisy_signals(SNR_range=SNR_range, r2f_range=r2f_range)
    train_dataset.DCT()
    print("-------- Reloaded Dataset ---------")
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

import math
import glob

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
            print(f"Epoch {epoch}: Dataset size = {len(train_dataloader.dataset)}")
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} Training', unit="batch")

        for noisy_signal, noisy_signal_dct, _, gt_signal_dct, gt_raman_signal_dct, _, _, _, _ in progress_bar:
            # Move data to the appropriate device
            noisy_signal = noisy_signal.unsqueeze(1).float().to(device)  # Shape: (batch_size, 1, length)
            noisy_signal_dct = noisy_signal_dct.unsqueeze(1).float().to(device)  # Shape: (batch_size, 1, length)
            gt_signal_dct = gt_signal_dct.unsqueeze(1).float().to(device)
            gt_raman_signal_dct = gt_raman_signal_dct.unsqueeze(1).float().to(device)
            # get the max of the noisy_signal
            max_values = noisy_signal.max(dim=2, keepdim=True)[0]
            noisy_signal_dct_norm = noisy_signal_dct / max_values
            gt_signal_dct_norm = gt_signal_dct / max_values
            gt_raman_signal_dct_norm = gt_raman_signal_dct / max_values
            # print(device)
            # if clip == "high":
            #     noisy_signal_dct_norm = noisy_signal_dct_norm[:,:,51+321:]
            #     gt_signal_dct_norm = gt_signal_dct_norm[:,:,51+321:]
            # elif clip == "mid":
            #     noisy_signal_dct_norm = noisy_signal_dct_norm[:,:,51:51+321]
            #     gt_signal_dct_norm = gt_signal_dct_norm[:,:,51:51+321]
            # elif clip == "low":
            #     noisy_signal_dct_norm = noisy_signal_dct_norm[:,:,:51]
            #     gt_signal_dct_norm = gt_signal_dct_norm[:,:,:51]
            if clip != "full":
                Warning("please input a valid string to the param *clip")
                
            
            # # center the signal
            # centered = torch.mean(noisy_signal, dim=2, keepdim=True)
            # noisy_signal = noisy_signal - centered
            # true_noise = true_noise - centered
            # gt = gt - centered

            optimizer.zero_grad()  # Zero the gradients

            # # Forward pass
            # denoise_outputs_dct, raman_outputs_dct = model(noisy_signal_dct_norm)
            # denoise_outputs = idct_torch(denoise_outputs_dct)
            # raman_outputs = idct_torch(raman_outputs_dct)
            
            # gt_idct = idct_torch(gt_signal_dct_norm)
            # loss_denoise = criterion(denoise_outputs, gt_idct)
            
            # gt_raman_idct = idct_torch(gt_raman_signal_dct_norm)
            # loss_raman = criterion(raman_outputs, gt_raman_idct)
            
            # loss = 1000 * (0.6 * loss_denoise + 0.4 * loss_raman)
            
            # Forward pass for Raman only
            outputs = model(noisy_signal_dct_norm)
            pred = noisy_signal_dct_norm - outputs
            pred_idct = idct_torch(pred)
            gt_idct = idct_torch(gt_signal_dct_norm)
            # if clip == "low":
                # outputs = normalization_for_loss(outputs)
            # if clip == "low":
            #     dct_loss = criterion(normalization_for_loss(outputs), normalization_for_loss(noise_residual))
            # else:
            #     dct_loss = criterion(outputs, noise_residual)
            loss = 1000 * criterion(pred_idct, gt_idct)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        # plot the process
        # Detach tensors and move to cpu for plotting
        # denoise_outputs_plot = denoise_outputs[0, 0, :].detach().cpu().numpy()
        # gt_idct_plot = gt_idct[0, 0, :].detach().cpu().numpy()
        # raman_outputs_plot = raman_outputs[0, 0, :].detach().cpu().numpy()
        # gt_raman_idct_plot = gt_raman_idct[0, 0, :].detach().cpu().numpy()

        # if epoch % 10 == 0:
        #     plt.figure()
        #     plt.subplot(1,2,1)
        #     plt.plot(denoise_outputs_plot, label="denoised")
        #     plt.plot(gt_idct_plot, label="gt")
        #     plt.legend()
        #     plt.subplot(1,2,2)
        #     plt.plot(raman_outputs_plot, label="predicted raman")
        #     plt.plot(gt_raman_idct_plot, label="gt raman")
        #     plt.legend()
        #     plt.title(f"Epoch {epoch+1}/{num_epochs}")
        #     plt.savefig(f"tmp/training_epoch_{epoch+1}.jpg")
        #     plt.close()
            
        # Validation Phase
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation for validation
            for noisy_signal, noisy_signal_dct, _, gt_signal_dct, gt_raman_signal_dct, _, _, _, _ in progress_bar:
                # Move data to the appropriate device
                noisy_signal = noisy_signal.unsqueeze(1).float().to(device)  # Shape: (batch_size, 1, length)
                noisy_signal_dct = noisy_signal_dct.unsqueeze(1).float().to(device)  # Shape: (batch_size, 1, length)
                gt_signal_dct = gt_signal_dct.unsqueeze(1).float().to(device)
                gt_raman_signal_dct = gt_raman_signal_dct.unsqueeze(1).float().to(device)
                # get the max of the noisy_signal
                max_values = noisy_signal.max(dim=2, keepdim=True)[0]
                noisy_signal_dct_norm = noisy_signal_dct / max_values
                gt_signal_dct_norm = gt_signal_dct / max_values
                gt_raman_signal_dct_norm = gt_raman_signal_dct / max_values
                # print(device)
                # if clip == "high":
                #     noisy_signal_dct_norm = noisy_signal_dct_norm[:,:,51+321:]
                #     gt_signal_dct_norm = gt_signal_dct_norm[:,:,51+321:]
                # elif clip == "mid":
                #     noisy_signal_dct_norm = noisy_signal_dct_norm[:,:,51:51+321]
                #     gt_signal_dct_norm = gt_signal_dct_norm[:,:,51:51+321]
                # elif clip == "low":
                #     noisy_signal_dct_norm = noisy_signal_dct_norm[:,:,:51]
                #     gt_signal_dct_norm = gt_signal_dct_norm[:,:,:51]
                if clip != "full":
                    Warning("please input a valid string to the param *clip")
                    
                
                # # center the signal
                # centered = torch.mean(noisy_signal, dim=2, keepdim=True)
                # noisy_signal = noisy_signal - centered
                # true_noise = true_noise - centered
                # gt = gt - centered

                optimizer.zero_grad()  # Zero the gradients

                # # Forward pass
                # denoise_outputs_dct, raman_outputs_dct = model(noisy_signal_dct_norm)
                # denoise_outputs = idct_torch(denoise_outputs_dct)
                # raman_outputs = idct_torch(raman_outputs_dct)
                
                # gt_idct = idct_torch(gt_signal_dct_norm)
                # loss_denoise = criterion(denoise_outputs, gt_idct)
                
                # gt_raman_idct = idct_torch(gt_raman_signal_dct_norm)
                # loss_raman = criterion(raman_outputs, gt_raman_idct)
                
                # loss = 1000 * (0.6 * loss_denoise + 0.4 * loss_raman)
                
                # Forward pass for Raman only
                outputs = model(noisy_signal_dct_norm)
                pred = noisy_signal_dct_norm - outputs
                pred_idct = idct_torch(pred)
                gt_idct = idct_torch(gt_signal_dct_norm)
                # if clip == "low":
                    # outputs = normalization_for_loss(outputs)
                # if clip == "low":
                #     dct_loss = criterion(normalization_for_loss(outputs), normalization_for_loss(noise_residual))
                # else:
                #     dct_loss = criterion(outputs, noise_residual)
                loss = 1000 * criterion(pred_idct, gt_idct)

                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        # plot the process
        # Detach tensors and move to cpu for plotting
        # denoise_outputs_plot = denoise_outputs[0, 0, :].detach().cpu().numpy()
        # gt_idct_plot = gt_idct[0, 0, :].detach().cpu().numpy()
        # raman_outputs_plot = raman_outputs[0, 0, :].detach().cpu().numpy()
        # gt_raman_idct_plot = gt_raman_idct[0, 0, :].detach().cpu().numpy()

        # if epoch % 10 == 0:
        #     plt.figure()
        #     plt.subplot(1,2,1)
        #     plt.plot(denoise_outputs_plot, label="denoised")
        #     plt.plot(gt_idct_plot, label="gt")
        #     plt.legend()
        #     plt.subplot(1,2,2)
        #     plt.plot(raman_outputs_plot, label="predicted raman")
        #     plt.plot(gt_raman_idct_plot, label="gt raman")
        #     plt.legend()
        #     plt.title(f"Epoch {epoch+1}/{num_epochs}")
        #     plt.savefig(f"tmp/val_epoch_{epoch+1}.jpg")
        #     plt.close()

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
    fluo_train_dir = "data/generated/poly_new_noise_model_train_fluorescence_05182025_185627.pkl"
    val_dir = "data/generated/pV_new_noise_model_val_05142025_135815.pkl"
    fluo_val_dir = "data/generated/poly_new_noise_model_val_fluorescence_05182025_185801.pkl"
    noise_dir = "data/noise/std"
    SNR_range = [0.01, 10]
    r2f_range = [0.01, 0.5]

    # Hyperparameters
    num_epochs = 200
    batch_size = 32
    learning_rate_full = 5e-5
    save_dir = "models/pretrained/"
    timestamp = time.strftime("%m%d%Y_%H%M%S")

    save_name_full = f"new_noise_model_{timestamp}_full_with_fluo_in_signal.pth"
    save_path_full = os.path.join(save_dir, save_name_full)

    # Initialize model, loss function, and optimizer
    model_full = AUnet(1, 1)
    criterion_full = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    optimizer_full = optim.Adam(model_full.parameters(), lr=learning_rate_full)
    
    # train_signal_skin, _, train_concentrations = read_clean_data(clean_dir=train_dir, customized_noise=False)
    # val_signal_skin, _, val_concentrations = read_clean_data(clean_dir=val_dir, customized_noise=False)
    train_signal, _ = read_clean_data(clean_dir=train_dir, customized_noise=False, pV=True)
    val_signal, _ = read_clean_data(clean_dir=val_dir, customized_noise=False, pV=True)
    fluo_train_signal, _ = read_clean_data(clean_dir=fluo_train_dir, customized_noise=False, pV=True)
    fluo_val_signal, _ = read_clean_data(clean_dir=fluo_val_dir, customized_noise=False, pV=True)
    noise_std_dict = {}
    txt_files = glob.glob(os.path.join(noise_dir, "*.txt"))
    for txt_file in txt_files:
        std = read_noise_data(txt_file)
        key = os.path.splitext(os.path.basename(txt_file))[0]
        noise_std_dict[key] = std

    # Create Dataset and DataLoader
    train_dataset = RamanNoiseDataset(clean_signals=train_signal, noise_std_list=noise_std_dict, fluorescence=fluo_train_signal)
    train_dataset.generate_noisy_signals(SNR_range=SNR_range, r2f_range=r2f_range)
    train_dataset.DCT()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = RamanNoiseDataset(clean_signals=val_signal, noise_std_list=noise_std_dict, fluorescence=fluo_val_signal)
    val_dataset.generate_noisy_signals(SNR_range=SNR_range, r2f_range=r2f_range)
    val_dataset.DCT()
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    # train_loss_HF, val_loss_HF = train_model(model_HF, train_dataloader, val_dataloader, criterion_HF, optimizer_HF, 50, device, save_path_HF, clip="high")
    # train_loss_MF, val_loss_MF = train_model(model_MF, train_dataloader, val_dataloader, criterion_MF, optimizer_MF, 20, device, save_path_MF, clip="mid")
    # train_loss_LF, val_loss_LF = train_model(model_LF, train_dataloader, val_dataloader, criterion_LF, optimizer_LF, num_epochs, device, save_path_LF, clip="low")
    train_loss_full, val_loss_full = train_model(model_full, train_dataloader, val_dataloader, criterion_full, optimizer_full, num_epochs, device, save_path_full, clip=None)

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
    
    plt.figure
    plt.plot(range(num_epochs), train_loss_full)
    plt.plot(range(num_epochs), val_loss_full)
    plt.legend(["train loss", "validation loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Full Frequency")
    plt.savefig("results/training_loss_full.jpg")