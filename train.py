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

def read_noise_data(root_folder):
    # Step 1: Load .txt files data
    txt_data = []
    for file in os.listdir(root_folder):
        if file.endswith('.txt'):
            file_path = os.path.join(root_folder, file)
            # Read the data from the file
            with open(file_path, 'r') as f:
                data = f.read().strip().split()  # Adjust based on file format
                data = [float(x) for x in data]  # Convert to float (or int, based on data)
                # data = data[421:]
            power = np.std(data)
            bias = np.mean(data)
            if power > 0:  # Avoid division by zero
                data = (data - bias) / power
            txt_data.append(data)  # Add the data to the list
    # Convert list of lists to a NumPy array
    txt_array = np.array(txt_data, dtype=np.float64)
    # Step 2: Split data into train, val, and test sets
    train_data, temp_data = train_test_split(txt_array, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    # Step 3: Save the test data to a pickle file
    test_pickle_file = os.path.join(root_folder, 'test_data.pkl')
    with open(test_pickle_file, 'wb') as f:
        pickle.dump(test_data, f)
    train_pickle_file = os.path.join(root_folder, 'train_data.pkl')
    with open(train_pickle_file, 'wb') as f:
        pickle.dump(train_data, f)
    val_pickle_file = os.path.join(root_folder, 'val_data.pkl')
    with open(val_pickle_file, 'wb') as f:
        pickle.dump(val_data, f)
    # Return the train, val, and test sets
    return train_data, val_data

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
    train_dataset = RamanNoiseDataset(clean_signals=train_signal, true_noises=train_noise)
    train_dataset.generate_noisy_signals(SNR_range=SNR_range)
    train_dataset.DCT()
    print("-------- Reloaded Dataset ---------")
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

            if clip == "high":
                noisy_signal = noisy_signal[:,:,81+950-50:]
                true_noise = true_noise[:,:,81+950-50:]
                # noise_residual = noisy_signal - true_noise
            elif clip == "mid":
                noisy_signal = noisy_signal[:,:,81-50:81+950]
                true_noise = true_noise[:,:,81-50:81+950]
                # noise_residual = noisy_signal - true_noise
            elif clip == "low":
                noisy_signal = noisy_signal[:,:,1:81]
                true_noise = true_noise[:,:,1:81]
                # noise_residual = true_noise
                # true_noise = normalization_for_loss(true_noise)
            elif clip != "full":
                Warning("please input a valid string to the param *clip")
            
            # center the signal
            centered = torch.mean(noisy_signal, dim=2, keepdim=True)
            noisy_signal = noisy_signal - centered
            true_noise = true_noise - centered

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(noisy_signal)
            # if clip == "low":
                # outputs = normalization_for_loss(outputs)
            # if clip == "low":
            #     dct_loss = criterion(normalization_for_loss(outputs), normalization_for_loss(noise_residual))
            # else:
            #     dct_loss = criterion(outputs, noise_residual)
            dct_loss = criterion(outputs, true_noise)

            # Calculate IDCT of output and ground truth for time-domain loss
            idct_output = torch.tensor(
                idct(outputs.detach().cpu().numpy(), type=2, norm='ortho', axis=-1),
                dtype=outputs.dtype, device=device
            )
            idct_target = torch.tensor(
                idct(true_noise.detach().cpu().numpy(), type=2, norm='ortho', axis=-1),
                dtype=true_noise.dtype, device=device
            )
            idct_loss = criterion(idct_output, idct_target)

            # Regularize the mean difference between output and ground truth
            mean_reg_loss = (outputs.mean() - true_noise.mean()) ** 2

            if clip != "low":
                loss = 1000 * dct_loss + 100 * idct_loss + 10 * mean_reg_loss
            else:
                loss = 100 * dct_loss + 1 * idct_loss

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
                    # noise_residual = noisy_signal - true_noise
                elif clip == "mid":
                    noisy_signal = noisy_signal[:,:,81-50:81+950]
                    true_noise = true_noise[:,:,81-50:81+950]
                    # noise_residual = noisy_signal - true_noise
                elif clip == "low":
                    noisy_signal = noisy_signal[:,:,1:81]
                    true_noise = true_noise[:,:,1:81]
                    # noise_residual = true_noise
                    # true_noise = normalization_for_loss(true_noise)
                elif clip != "full":
                    Warning("please input a valid string to the param *clip")
                
                # center the signal
                centered = torch.mean(noisy_signal, dim=2, keepdim=True)
                noisy_signal = noisy_signal - centered
                true_noise = true_noise - centered
                       
                # true_noise = normalization(true_noise)

                outputs = model(noisy_signal)

                # if clip == "low":
                #     dct_loss = criterion(normalization_for_loss(outputs), normalization_for_loss(noise_residual))
                # else:
                #     dct_loss = criterion(outputs, noise_residual)
                dct_loss = criterion(outputs, true_noise)

                idct_output = torch.tensor(
                    idct(outputs.detach().cpu().numpy(), type=2, norm='ortho', axis=-1),
                    dtype=outputs.dtype, device=device
                )
                idct_target = torch.tensor(
                    idct(true_noise.detach().cpu().numpy(), type=2, norm='ortho', axis=-1),
                    dtype=true_noise.dtype, device=device
                )
                idct_loss = criterion(idct_output, idct_target)

                # Regularize the mean difference between output and ground truth
                mean_reg_loss = (outputs.mean() - true_noise.mean()) ** 2

                if clip != "low":
                    # loss = 1000 * dct_loss + 100 * idct_loss + 2000 * mean_reg_loss
                    loss = 1000 * dct_loss + 100 * idct_loss + 10 * mean_reg_loss
                else:
                    loss = 100 * dct_loss + 1 * idct_loss

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

    train_dir = "data/generated/generated_skin_spectrum_11012024_143219.pkl"
    val_dir = "data/generated/generated_skin_spectrum_11012024_143226.pkl"
    train_dir_pV = "data/generated/raman_pesudo_Vioget_train_11182024_110514.pkl"
    val_dir_pV = "data/generated/raman_pesudo_Vioget_val_11182024_110718.pkl"
    noise_dir = "data/noise/processed_new"
    SNR_range = [0.01, 10]

    # Hyperparameters
    num_epochs = 400
    batch_size = 32
    learning_rate_HF = 2e-5
    learning_rate_MF = 2e-4
    learning_rate_LF = 3e-5
    save_dir = "models/pretrained/"
    timestamp = time.strftime("%m%d%Y_%H%M%S")

    save_name_HF = f"model_{timestamp}_HF.pth"
    save_path_HF = os.path.join(save_dir, save_name_HF)
    save_name_MF = f"model_{timestamp}_MF.pth"
    save_path_MF = os.path.join(save_dir, save_name_MF)
    save_name_LF = f"model_{timestamp}_LF.pth"
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
    
    train_signal_skin, _, train_concentrations = read_clean_data(clean_dir=train_dir, customized_noise=False)
    val_signal_skin, _, val_concentrations = read_clean_data(clean_dir=val_dir, customized_noise=False)
    train_signal_pV, _ = read_clean_data(clean_dir=train_dir_pV, customized_noise=False, pV=True)
    val_signal_pV, _ = read_clean_data(clean_dir=val_dir_pV, customized_noise=False, pV=True)
    train_noise, val_noise = read_noise_data(noise_dir)

    # train_signal = np.concatenate((train_signal_skin, train_signal_pV), axis=1)
    # val_signal = np.concatenate((val_signal_skin, val_signal_pV), axis=1)
    train_signal = train_signal_pV
    val_signal = val_signal_pV

    # Create Dataset and DataLoader
    train_dataset = RamanNoiseDataset(clean_signals=train_signal, true_noises=train_noise)
    train_dataset.generate_noisy_signals(SNR_range=SNR_range)
    train_dataset.DCT()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = RamanNoiseDataset(clean_signals=val_signal, true_noises=val_noise)
    val_dataset.generate_noisy_signals(SNR_range=SNR_range)
    val_dataset.DCT()
    val_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    train_loss_HF, val_loss_HF = train_model(model_HF, train_dataloader, val_dataloader, criterion_HF, optimizer_HF, num_epochs, device, save_path_HF, clip="high")
    train_loss_MF, val_loss_MF = train_model(model_MF, train_dataloader, val_dataloader, criterion_MF, optimizer_MF, num_epochs, device, save_path_MF, clip="mid")
    train_loss_LF, val_loss_LF = train_model(model_LF, train_dataloader, val_dataloader, criterion_LF, optimizer_LF, 400, device, save_path_LF, clip="low")

    plt.figure
    plt.subplot(3,1,1)
    plt.plot(range(num_epochs), train_loss_HF)
    plt.plot(range(num_epochs), val_loss_HF)
    plt.legend(["train loss", "validation loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("High Frequency")
    plt.subplot(3,1,2)
    plt.plot(range(num_epochs), train_loss_MF)
    plt.plot(range(num_epochs), val_loss_MF)
    plt.legend(["train loss", "validation loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Mid Frequency")
    plt.subplot(3,1,3)
    plt.plot(range(400), train_loss_LF)
    plt.plot(range(400), val_loss_LF)
    plt.legend(["train loss", "validation loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Low Frequency")
    plt.show()
    plt.savefig("results/training_loss.jpg")

