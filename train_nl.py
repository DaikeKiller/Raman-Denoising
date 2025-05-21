import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from models.Network import RamanNoiseNet, RamanNoiseNet_HF, RamanNoiseNet_LF
from models.AUnet import AUnet, TwoStageModel, idct_torch, dct_torch
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

def poly_fit(data: torch.Tensor, poly_order_range: tuple[int, int]):
    """
    Fit each 1×L fluorescence sample in `data` to the best polynomial within the given order range.

    Args:
        data: Tensor of shape (B, 1, L), the fluorescence baselines on GPU.
        poly_order_range: Tuple (min_order, max_order), inclusive.

    Returns:
        best_coeffs: Tensor of shape (B, 1, P_max), where P_max = max_order + 1.
                     For each sample, these are the coefficients (highest power first) 
                     of the best-fit polynomial of order in [min_order, max_order].
        best_orders: Tensor of shape (B,), the chosen order for each sample.
    """
    device = data.device
    B, C, L = data.shape
    assert C == 1, "Expected data of shape (B,1,L)"
    min_order, max_order = poly_order_range
    orders = list(range(min_order, max_order + 1))
    P_max = max_order + 1

    # Precompute the x-axis (wavenumbers) on GPU
    wvn = torch.linspace(600., 1790., L, device=device)

    # Prepare target Y of shape (B, L, 1)
    Y = data.view(B, L, 1)

    coefs_list = []
    rss_list   = []

    for p in orders:
        P = p + 1
        # Build Vandermonde matrix V of shape (L, P)
        V = torch.stack([wvn**i for i in reversed(range(P))], dim=1)  # (L, P)

        # Broadcast to batch: A of shape (B, L, P)
        A = V.unsqueeze(0).expand(B, L, P)

        # Solve least-squares A @ c = Y  → c has shape (B, P, 1)
        sol = torch.linalg.lstsq(A, Y).solution  # (B, P, 1)

        # Compute residuals: pred = A @ sol  → (B, L, 1)
        pred = torch.bmm(A, sol)  # (B, L, 1)
        rss  = ((Y - pred)**2).sum(dim=(1,2))  # (B,)

        # Pad sol to length P_max if needed: from (B,P,1) → (B,P_max,1)
        if P < P_max:
            pad = torch.zeros(B, P_max - P, 1, device=device, dtype=sol.dtype)
            sol = torch.cat([sol, pad], dim=1)  # (B, P_max, 1)

        coefs_list.append(sol)  # each is (B, P_max, 1)
        rss_list.append(rss)    # each is (B,)

    # Stack over order dimension: Coefs (O, B, P_max, 1), RSS (O, B)
    Coefs = torch.stack(coefs_list, dim=0)
    RSS   = torch.stack(rss_list,   dim=0)

    # Find best order index for each sample
    best_idx = RSS.argmin(dim=0)  # (B,)
    b_idx    = torch.arange(B, device=device)

    # Gather the corresponding coefficients: shape (B, P_max, 1)
    best_coefs = Coefs[best_idx, b_idx]  # advanced indexing

    # Reshape to (B, 1, P_max)
    best_coeffs = best_coefs.squeeze(-1).unsqueeze(1)  # (B, 1, P_max)
    # Evaluate the fit for the first sample in the batch
    # with torch.no_grad():
    #     # Get the coefficients and order for the first sample
    #     coeff = best_coeffs[0, 0]  # shape: (P_max,)
    #     order = best_idx[0].item() + min_order  # actual polynomial order

    #     # Generate x-axis (wavenumbers)
    #     x = torch.linspace(600., 1790., L, device=device).cpu().numpy()
    #     # Get the original fluorescence signal (first sample)
    #     y = data[0, 0].cpu().numpy()

    #     # Only use the fitted coefficients up to the selected order
    #     coeff_np = coeff[:order+1].cpu().numpy()
    #     # Evaluate the polynomial fit
    #     y_fit = np.polyval(coeff_np, x)

    #     # Plot and save
    #     plt.figure()
    #     plt.plot(x, y, label="Original")
    #     plt.plot(x, y_fit, label=f"Fitted (order={order})")
    #     plt.legend()
    #     plt.title("Polynomial Fit to Fluorescence Baseline")
    #     plt.xlabel("Wavenumber")
    #     plt.ylabel("Intensity")
    #     plt.savefig("tmp/haha.jpg")
    #     plt.close()
    return best_coeffs, best_idx

def get_loss_coeffs(pred_coeffs: torch.Tensor,
             gt_coeffs:   torch.Tensor,
             criterion:   nn.Module) -> torch.Tensor:
    """
    Align predicted and ground-truth coefficient tensors along the last dimension
    by padding with zeros or truncating, then computes a loss via the given criterion.

    Args:
        pred_coeffs: Tensor of shape (B, 1, P_pred)
        gt_coeffs:   Tensor of shape (B, 1, P_gt)
        criterion:   A loss module (e.g. nn.MSELoss())

    Returns:
        loss: Scalar tensor
    """
    B, Cp, Pp = pred_coeffs.shape
    _, Cg, Pg = gt_coeffs.shape
    assert Cp == 1 and Cg == 1, "Channel dimension must be 1"

    # Target length
    P = max(Pp, Pg)
    device = pred_coeffs.device

    # Pad or truncate pred_coeffs
    if Pp < P:
        pad = torch.zeros(B, 1, P - Pp, device=device, dtype=pred_coeffs.dtype)
        aligned_pred = torch.cat([pred_coeffs, pad], dim=2)
    else:
        aligned_pred = pred_coeffs[..., :P]

    # Pad or truncate gt_coeffs
    if Pg < P:
        pad = torch.zeros(B, 1, P - Pg, device=device, dtype=gt_coeffs.dtype)
        aligned_gt = torch.cat([gt_coeffs, pad], dim=2)
    else:
        aligned_gt = gt_coeffs[..., :P]

    # Compute loss
    loss = criterion(aligned_pred, aligned_gt)
    return loss

# Training Function
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device, save_path, clip="full"):
    model.to(device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Load pretrained weights for model.denoiser and freeze its parameters
    pretrained_path = "models/pretrained/new_noise_model_05202025_083413_full_with_fluo_in_signal.pth"
    if hasattr(model, "denoiser"):
        model.denoiser.load_state_dict(torch.load(pretrained_path, map_location=device))
        for param in model.denoiser.parameters():
            param.requires_grad = False
        print("Pretrained weights loaded and parameters frozen for model.denoiser.")

    for epoch in range(num_epochs):
        if (epoch) % 1 == 0 and epoch != 0:
            train_dataloader = reload_train_dataloader()
            print(f"Epoch {epoch}: Dataset size = {len(train_dataloader.dataset)}")

        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} Training', unit="batch")

        for noisy_signal, noisy_signal_dct, _, gt_signal_dct, gt_raman_signal_dct, gt_fluo_signal_dct, _, _, _ in progress_bar:
            # Move data to the appropriate device
            noisy_signal = noisy_signal.unsqueeze(1).float().to(device)  # Shape: (batch_size, 1, length)
            noisy_signal_dct = noisy_signal_dct.unsqueeze(1).float().to(device)  # Shape: (batch_size, 1, length)
            gt_signal_dct = gt_signal_dct.unsqueeze(1).float().to(device)
            gt_raman_signal_dct = gt_raman_signal_dct.unsqueeze(1).float().to(device)
            gt_fluo_signal_dct = gt_fluo_signal_dct.unsqueeze(1).float().to(device)
            # get the max of the noisy_signal
            max_values = noisy_signal.max(dim=2, keepdim=True)[0]
            noisy_signal_dct_norm = noisy_signal_dct / max_values
            gt_signal_dct_norm = gt_signal_dct / max_values
            gt_raman_signal_dct_norm = gt_raman_signal_dct / max_values
            gt_fluo_signal_dct_norm = gt_fluo_signal_dct / max_values
            gt_fluo_signal_idct_norm = idct_torch(gt_fluo_signal_dct_norm)
            
            # gt_coeffs, _ = poly_fit(gt_fluo_signal_idct_norm, [2, 6])

            if clip != "full":
                Warning("please input a valid string to the param *clip")

            optimizer.zero_grad()  # Zero the gradients

            # -------- Forward pass ------------
            out = model(noisy_signal_dct_norm)
            denoise_outputs = out["denoised_signal"]
            fluo_signal = out["poly_raw"]
            fluo_signal_dct = out["poly_dct"]
            raman_outputs = out["raman_signal"]
            raman_outputs_dct = out["raman_dct"]
            gt_idct = idct_torch(gt_signal_dct_norm)
            
            gt_raman_idct = idct_torch(gt_raman_signal_dct_norm)
            loss_raman = criterion(raman_outputs, gt_raman_idct)
            
            loss_raman_dct = criterion(gt_raman_signal_dct_norm, raman_outputs_dct)
            
            # # L2 regularization for coeffs
            # l2_coeffs = torch.mean(coeffs ** 2)
            # l1_coeffs = torch.mean(torch.abs(coeffs))
            # loss = 1000 * (0.3 * loss_raman + 0.7 * loss_raman_dct + 0.5 * loss_coeffs) + 1000 * l2_coeffs
            loss = 1000 * loss_raman + 500 * loss_raman_dct

            # -----------  Forward pass for Raman only ------------
            # outputs = model(noisy_signal_dct_norm)
            # pred = noisy_signal_dct_norm - outputs
            # pred_idct = idct_torch(pred)
            # gt_idct = idct_torch(gt_signal_dct_norm)
            # loss = 1000 * criterion(pred_idct, gt_idct)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        # plot the process
        # Detach tensors and move to cpu for plotting
        denoise_outputs_plot = denoise_outputs[0, 0, :].detach().cpu().numpy()
        gt_idct_plot = gt_idct[0, 0, :].detach().cpu().numpy()
        raman_outputs_plot = raman_outputs[0, 0, :].detach().cpu().numpy()
        gt_raman_idct_plot = gt_raman_idct[0, 0, :].detach().cpu().numpy()

        if epoch % 5 == 0:
            plt.figure()
            plt.subplot(1,2,1)
            plt.plot(denoise_outputs_plot, label="denoised")
            plt.plot(gt_idct_plot, label="gt")
            plt.legend()
            plt.subplot(1,2,2)
            plt.plot(raman_outputs_plot, label="predicted raman")
            plt.plot(gt_raman_idct_plot, label="gt raman")
            plt.legend()
            plt.title(f"Epoch {epoch+1}/{num_epochs}")
            plt.savefig(f"tmp/training_epoch_{epoch+1}.jpg")
            plt.close()
        
        # pred_idct_plot = pred_idct[0, 0, :].detach().cpu().numpy()
        # gt_idct_plot = gt_idct[0, 0, :].detach().cpu().numpy()
        
        # if epoch % 1 == 0:
        #     plt.figure()
        #     plt.plot(pred_idct_plot, label="denoised")
        #     plt.plot(gt_idct_plot, label="gt")
        #     plt.legend()
        #     plt.title(f"Epoch {epoch+1}/{num_epochs}")
        #     plt.savefig(f"tmp/training_epoch_{epoch+1}.jpg")
        #     plt.close()
            
        # Validation Phase
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation for validation
            for noisy_signal, noisy_signal_dct, _, gt_signal_dct, gt_raman_signal_dct, gt_fluo_signal_dct, _, _, _ in progress_bar:
                # Move data to the appropriate device
                noisy_signal = noisy_signal.unsqueeze(1).float().to(device)  # Shape: (batch_size, 1, length)
                noisy_signal_dct = noisy_signal_dct.unsqueeze(1).float().to(device)  # Shape: (batch_size, 1, length)
                gt_signal_dct = gt_signal_dct.unsqueeze(1).float().to(device)
                gt_raman_signal_dct = gt_raman_signal_dct.unsqueeze(1).float().to(device)
                gt_fluo_signal_dct = gt_fluo_signal_dct.unsqueeze(1).float().to(device)
                # get the max of the noisy_signal
                max_values = noisy_signal.max(dim=2, keepdim=True)[0]
                noisy_signal_dct_norm = noisy_signal_dct / max_values
                gt_signal_dct_norm = gt_signal_dct / max_values
                gt_raman_signal_dct_norm = gt_raman_signal_dct / max_values
                gt_fluo_signal_dct_norm = gt_fluo_signal_dct / max_values
                gt_fluo_signal_idct_norm = idct_torch(gt_fluo_signal_dct_norm)
                
                # gt_coeffs, _ = poly_fit(gt_fluo_signal_idct_norm, [2, 6])

                if clip != "full":
                    Warning("please input a valid string to the param *clip")

                optimizer.zero_grad()  # Zero the gradients

                # -------- Forward pass ------------
                out = model(noisy_signal_dct_norm)
                denoise_outputs = out["denoised_signal"]
                fluo_signal = out["poly_raw"]
                fluo_signal_dct = out["poly_dct"]
                raman_outputs = out["raman_signal"]
                raman_outputs_dct = out["raman_dct"]
                gt_idct = idct_torch(gt_signal_dct_norm)
                
                gt_raman_idct = idct_torch(gt_raman_signal_dct_norm)
                loss_raman = criterion(raman_outputs, gt_raman_idct)
                
                loss_raman_dct = criterion(gt_raman_signal_dct_norm, raman_outputs_dct)
                
                # # L2 regularization for coeffs
                # l2_coeffs = torch.mean(coeffs ** 2)
                # l1_coeffs = torch.mean(torch.abs(coeffs))
                # loss = 1000 * (0.3 * loss_raman + 0.7 * loss_raman_dct + 0.5 * loss_coeffs) + 1000 * l2_coeffs
                loss = 1000 * loss_raman + 500 * loss_raman_dct
                
                # ---------- Forward pass for denoising only ---------------
                # outputs = model(noisy_signal_dct_norm)
                # pred = noisy_signal_dct_norm - outputs
                # pred_idct = idct_torch(pred)
                # gt_idct = idct_torch(gt_signal_dct_norm)
                # loss = 1000 * criterion(pred_idct, gt_idct)

                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        # plot the process
        # Detach tensors and move to cpu for plotting
        denoise_outputs_plot = denoise_outputs[0, 0, :].detach().cpu().numpy()
        gt_idct_plot = gt_idct[0, 0, :].detach().cpu().numpy()
        raman_outputs_plot = raman_outputs[0, 0, :].detach().cpu().numpy()
        gt_raman_idct_plot = gt_raman_idct[0, 0, :].detach().cpu().numpy()

        if epoch % 5 == 0:
            plt.figure()
            plt.subplot(1,2,1)
            plt.plot(denoise_outputs_plot, label="denoised")
            plt.plot(gt_idct_plot, label="gt")
            plt.legend()
            plt.subplot(1,2,2)
            plt.plot(raman_outputs_plot, label="predicted raman")
            plt.plot(gt_raman_idct_plot, label="gt raman")
            plt.legend()
            plt.title(f"Epoch {epoch+1}/{num_epochs}")
            plt.savefig(f"tmp/val_epoch_{epoch+1}.jpg")
            plt.close()
        
        # pred_idct_plot = pred_idct[0, 0, :].detach().cpu().numpy()
        # gt_idct_plot = gt_idct[0, 0, :].detach().cpu().numpy()
        
        # if epoch % 1 == 0:
        #     plt.figure()
        #     plt.plot(pred_idct_plot, label="denoised")
        #     plt.plot(gt_idct_plot, label="gt")
        #     plt.legend()
        #     plt.title(f"Epoch {epoch+1}/{num_epochs}")
        #     plt.savefig(f"tmp/training_epoch_{epoch+1}.jpg")
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
    r2f_range = [0.05, 0.5]
    signal_length = 693

    # Hyperparameters
    num_epochs = 200
    batch_size = 32
    learning_rate_full = 2e-5
    save_dir = "models/pretrained/"
    timestamp = time.strftime("%m%d%Y_%H%M%S")

    save_name_full = f"new_noise_model_{timestamp}_end_to_end_wvn_domain.pth"
    save_path_full = os.path.join(save_dir, save_name_full)

    # Initialize model, loss function, and optimizer
    model_full = TwoStageModel()
    # model_full = AUnet(1, 1)
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