import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.Network import RamanNoiseNet, RamanNoiseNet_Clip, RamanNoiseNet_HF, RamanNoiseNet_LF
from utils.Raman_dataset import RamanNoiseDataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.signal import resample


def read_clean_data(clean_dir, customized_noise=False):
    with open(clean_dir, 'rb') as file:
        concentrations, clean_data = pickle.load(file)
        # clean_data = resample(clean_data, 603, axis=0)
    if customized_noise is True:
        noise_data = np.random.randn(1000, clean_data.shape[0])
    else:
        noise_data = None
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
            power = np.mean([a ** 2 for a in data])
            if power > 0:  # Avoid division by zero
                data = data / np.sqrt(power)
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
    # Return the train, val, and test sets
    return train_data, val_data

def normalization(signal):
    # Generate normalization factor tensor
    # factor = [np.log(50*a) / (8*np.log(50)) for a in range(1, signal.shape[2]+1)]
    factor = [0.01*a + 0.1 for a in range(1, signal.shape[2]+1)]
    factor = np.array(factor)
    factor = torch.from_numpy(np.reshape(factor, [1, 1, -1])).float()

    # Move the factor to the same device as the signal
    factor = factor.to(signal.device)
    return signal * factor

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
        if (epoch) % 5 == 0 and epoch != 0:
            train_dataloader = reload_train_dataloader()
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} Training', unit="batch")

        for noisy_signal, true_noise in progress_bar:
            # Move data to the appropriate device
            noisy_signal = noisy_signal.unsqueeze(1).float().to(device)  # Shape: (batch_size, 1, length)
            true_noise = true_noise.unsqueeze(1).float().to(device)      # Shape: (batch_size, 1, length)

            if clip == "high":
                noisy_signal = noisy_signal[:,:,81:]
                true_noise = true_noise[:,:,81:]
                true_noise = noisy_signal - true_noise # for high frequency, we use residual learning
            elif clip == "low":
                noisy_signal = noisy_signal[:,:,:81]
                true_noise = true_noise[:,:,:81]
            elif clip != "full":
                Warning("please input a valid string to the param *clip")

            # true_noise = normalization(true_noise)

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(noisy_signal)
            # outputs = normalization(outputs)
            loss = criterion(outputs, true_noise)

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
            for noisy_signal, true_noise in val_dataloader:
                noisy_signal = noisy_signal.unsqueeze(1).float().to(device)
                true_noise = true_noise.unsqueeze(1).float().to(device)
                if clip == "high":
                    noisy_signal = noisy_signal[:,:,81:]
                    true_noise = true_noise[:,:,81:]
                    true_noise = noisy_signal - true_noise # for high frequency, we use residual learning
                elif clip == "low":
                    noisy_signal = noisy_signal[:,:,:81]
                    true_noise = true_noise[:,:,:81]
                elif clip != "full":
                    Warning("please input a valid string to the param *clip")
                
                # true_noise = normalization(true_noise)

                outputs = model(noisy_signal)
                # outputs = normalization(outputs)
                loss = criterion(outputs, true_noise)

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

    train_dir = "data/generated/generated_skin_spectrum_10072024_173907.pkl"
    val_dir = "data/generated/generated_skin_spectrum_10022024_104349.pkl"
    noise_dir = "data/noise/processed"
    SNR_range = [-2, 2]

    # Hyperparameters
    num_epochs = 100
    batch_size = 32
    learning_rate_HF = 0.0002
    learning_rate_LF = 0.0002
    save_dir = "models/pretrained/"
    timestamp = time.strftime("%m%d%Y_%H%M%S")
    save_name_HF = f"model_{timestamp}_HF.pth"
    save_path_HF = os.path.join(save_dir, save_name_HF)
    save_name_LF = f"model_{timestamp}_LF.pth"
    save_path_LF = os.path.join(save_dir, save_name_LF)

    # Initialize model, loss function, and optimizer
    # model = RamanNoiseNet()
    model_HF = RamanNoiseNet_HF()
    criterion_HF = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    optimizer_HF = optim.Adam(model_HF.parameters(), lr=learning_rate_HF, weight_decay=0.01)
    model_LF = RamanNoiseNet_LF()
    criterion_LF = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    optimizer_LF = optim.Adam(model_LF.parameters(), lr=learning_rate_LF)
    
    train_signal, _, train_concentrations = read_clean_data(clean_dir=train_dir, customized_noise=False)
    val_signal, _, val_concentrations = read_clean_data(clean_dir=val_dir, customized_noise=False)
    train_noise, val_noise = read_noise_data(noise_dir)

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
    # train_loss_LF, val_loss_LF = train_model(model_LF, train_dataloader, val_dataloader, criterion_LF, optimizer_LF, num_epochs, device, save_path_LF, clip="low")

    plt.figure
    plt.subplot(2,1,1)
    plt.plot(range(num_epochs), train_loss_HF)
    plt.plot(range(num_epochs), val_loss_HF)
    plt.legend(["train loss", "validation loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("High Frequency")
    # plt.subplot(2,1,2)
    # plt.plot(range(num_epochs), train_loss_LF)
    # plt.plot(range(num_epochs), val_loss_LF)
    # plt.legend(["train loss", "validation loss"])
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.title("Low Frequency")
    # plt.show()

