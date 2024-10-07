import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.Network import RamanNoiseNet
from utils.Raman_dataset import RamanNoiseDataset


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

    # Hyperparameters
    num_epochs = 100
    batch_size = 16
    learning_rate = 0.005

    # Initialize model, loss function, and optimizer
    model = RamanNoiseNet()
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Example data (replace with your real data)
    num_samples = 100  # Number of samples
    spectrum_length = 1000  # Length of each spectrum
    noisy_signals = torch.randn(num_samples, spectrum_length)  # Random noisy signals
    true_noises = torch.randn(num_samples, spectrum_length)    # Corresponding true noise signals

    # Create Dataset and DataLoader
    dataset = RamanNoiseDataset(noisy_signals, true_noises)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs, device)


