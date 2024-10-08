from train import *
from scipy.fftpack import idct
import random


# Function to test the model and clean the signals
def test_model(model, test_dataloader, device):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    predicted_noises = []
    cleaned_signals = []
    noisy_signals = []
    true_noises = []
    
    with torch.no_grad():  # Disable gradient calculation for testing
        # Progress bar for testing phase
        for noisy_signal, true_noise in tqdm(test_dataloader, desc="Testing", unit="batch"):
            # Move data to the appropriate device
            noisy_signal = noisy_signal.unsqueeze(1).float().to(device)  # Shape: (batch_size, 1, length)
            
            # Predict noise using the trained model
            predicted_noise = model(noisy_signal)
            
            # Perform inverse DCT on predicted noise
            predicted_noise = predicted_noise.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension
            idct_predicted_noise = idct(predicted_noise, type=2, norm='ortho', axis=1)
            
            # Subtract predicted noise from noisy signal to clean the signal
            noisy_signal_np = noisy_signal.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension
            noisy_signal_np = idct(noisy_signal_np, type=2, norm='ortho', axis=1)
            cleaned_signal = noisy_signal_np - idct_predicted_noise

            true_noise_np = true_noise.squeeze(1).cpu().numpy()
            true_noise_np = idct(true_noise_np, type=2, norm='ortho', axis=1)
            
            # Store the results
            predicted_noises.append(idct_predicted_noise)
            cleaned_signals.append(cleaned_signal)
            noisy_signals.append(noisy_signal_np)
            true_noises.append(true_noise_np)
    
    # Concatenate results into numpy arrays
    predicted_noises = np.concatenate(predicted_noises, axis=0)
    cleaned_signals = np.concatenate(cleaned_signals, axis=0)
    noisy_signals = np.concatenate(noisy_signals, axis=0)
    true_noises = np.concatenate(true_noises, axis=0)
    
    return predicted_noises, cleaned_signals, noisy_signals, true_noises

def plot_signals(noisy_signals, cleaned_signals, test_signals, num_samples=5):
    # Get the total number of signals
    total_signals = len(noisy_signals)
    
    # If the requested number of samples exceeds the total signals, limit it to the available number
    num_samples = min(num_samples, total_signals)
    
    # Select random samples
    indices = random.sample(range(total_signals), num_samples)
    
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, num_samples * 3))
    
    for i, idx in enumerate(indices):
        axs[i, 0].plot(noisy_signals[idx], label="Noisy Signal")
        axs[i, 0].set_title(f"Noisy Signal {idx}")
        axs[i, 0].legend()

        axs[i, 1].plot(cleaned_signals[idx], label="Cleaned Signal", color='orange')
        axs[i, 1].set_title(f"Cleaned Signal {idx}")
        axs[i, 1].legend()

        axs[i, 2].plot(test_signals[:,idx], label="Test Signal", color='green')
        axs[i, 2].set_title(f"Test Signal {idx}")
        axs[i, 2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dir = "data/generated/generated_skin_spectrum_10022024_104256.pkl"  # Test data
    SNR_range = [0, 5]

    # Load the best trained model
    model_path = "models/pretrained/model_10082024_130046.pth"
    model = RamanNoiseNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Load test data
    test_signal, test_noise, test_concentrations = read_data(clean_dir=test_dir)

    # Create Dataset and DataLoader
    test_dataset = RamanNoiseDataset(clean_signals=test_signal, true_noises=test_noise)
    test_dataset.generate_noisy_signals(SNR_range=SNR_range)
    test_dataset.DCT()  # Apply DCT on the test data
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Test the model and get predicted noises and cleaned signals
    predicted_noises, cleaned_signals, noisy_signals, true_noise = test_model(model, test_dataloader, device)
    print(noisy_signals.shape)

    plot_signals(noisy_signals, cleaned_signals, test_signal, num_samples=5)

    print("Testing complete. Results saved.")

