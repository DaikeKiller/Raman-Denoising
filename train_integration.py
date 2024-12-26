import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from models.Network import RamanNoiseNet, RamanNoiseNet_HF, RamanNoiseNet_LF
from models.AUnet import AUnet
from utils.Integrate_dataset import IntegrationDataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from scipy.signal import resample
from scipy.fftpack import idct


def read_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    noisy_signals = data["noisy_signals"]
    cleaned_signals, gt_signals = data["cleaned_signals"], data["gt_signals"]
    return noisy_signals - cleaned_signals, noisy_signals - gt_signals

def reload_train_dataloader():
    train_dataset = IntegrationDataset(input_noises=input_noises_train, true_noises=true_noises_train)
    # train_dataset.DCT()
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
            
            loss = criterion(outputs, gt)

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

                loss = criterion(outputs, gt)

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
    
    filename_train = "results/results_all_pV_train.pkl"
    input_noises_train, true_noises_train = read_data(filename_train)
    filename_val = "results/results_all_pV_val.pkl"
    input_noises_val, true_noises_val = read_data(filename_val)
    
    num_epochs = 400
    batch_size = 32
    learning_rate = 2e-6
    save_dir = "models/pretrained/"
    timestamp = time.strftime("%m%d%Y_%H%M%S")

    save_name = f"model_{timestamp}_integration.pth"
    save_path = os.path.join(save_dir, save_name)
    
    model = AUnet(1, 1)
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.99)
    
    train_dataset = IntegrationDataset(input_noises=input_noises_train, true_noises=true_noises_train)
    # train_dataset.DCT()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = IntegrationDataset(input_noises=input_noises_val, true_noises=true_noises_val)
    # val_dataset.DCT()
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    train_loss, val_loss = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device, save_path)
    