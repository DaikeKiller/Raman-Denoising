import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.Network import RamanNoiseNet
from utils.Raman_dataset import RamanNoiseDataset
import pickle
import numpy as np


def read_data(clean_dir, noise=None):
    with open(clean_dir, 'rb') as file:
        concentrations, clean_data = pickle.load(file)
    if noise is None:
        noise = np.random.randn(1000, clean_data.shape[0])
    return clean_data, noise, concentrations

# Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (noisy_signal, true_noise) in enumerate(dataloader):
            # Move data to the appropriate device
            noisy_signal = noisy_signal.unsqueeze(1).float().to(device)  # Shape: (batch_size, 1, length)
            true_noise = true_noise.unsqueeze(1).float().to(device)      # Shape: (batch_size, 1, length)

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(noisy_signal)
            loss = criterion(outputs, true_noise)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print('Training complete.')


if __name__ == "__main__":
    # Setup: device, hyperparameters, etc.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dir = "data/generated/generated_skin_spectrum_10072024_173907.pkl"
    SNR_range = [0, 5]

    # Hyperparameters
    num_epochs = 100
    batch_size = 16
    learning_rate = 0.005

    # Initialize model, loss function, and optimizer
    model = RamanNoiseNet()
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    clean_signal, noise, concentrations = read_data(clean_dir=dir)

    # Create Dataset and DataLoader
    dataset = RamanNoiseDataset(clean_signals=clean_signal, true_noises=noise)
    dataset.generate_noisy_signals(SNR_range=SNR_range)
    dataset.DCT()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs, device)


