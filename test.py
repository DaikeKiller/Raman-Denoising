from train import *
from scipy.fftpack import idct
import random


# Function to test the model and clean the signals
def test_model(model_HF, model_LF, test_dataloader, device):
    model_HF.to(device)
    model_HF.eval()  # Set the model to evaluation mode
    model_LF.to(device)
    model_LF.eval()  # Set the model to evaluation mode
    
    predicted_noises = []
    cleaned_signals = []
    noisy_signals = []
    true_noises = []
    SNR_list = []
    
    with torch.no_grad():  # Disable gradient calculation for testing
        # Progress bar for testing phase
        for noisy_signal, true_noise, SNR in tqdm(test_dataloader, desc="Testing", unit="batch"):
            # Move data to the appropriate device
            noisy_signal = noisy_signal.unsqueeze(1).float().to(device)  # Shape: (batch_size, 1, length)
            true_noise_np = true_noise.squeeze(1).cpu().numpy()

            noisy_signal_HF = noisy_signal[:, :, 81:]
            noisy_signal_LF = noisy_signal[:, :, :81]

            predicted_noise_HF_residual = model_HF(noisy_signal_HF) 
            predicted_noise_LF = model_LF(noisy_signal_LF)
            predicted_noise_HF_residual = predicted_noise_HF_residual.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension
            predicted_noise_LF = predicted_noise_LF.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension

            noisy_signal_np = noisy_signal.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension
            predicted_noise_HF = noisy_signal_np[:, 81:] - predicted_noise_HF_residual
            predicted_noise = np.concatenate((predicted_noise_LF, predicted_noise_HF), axis=1)

            idct_predicted_noise = idct(predicted_noise, type=2, norm='ortho', axis=1)
            # idct_predicted_noise_power = np.mean(idct_predicted_noise ** 2, axis=1)
            # idct_predicted_noise = idct_predicted_noise / np.sqrt(idct_predicted_noise_power[:, np.newaxis])
            
            # Subtract predicted noise from noisy signal to clean the signal
            idct_noisy_signal_np = idct(noisy_signal_np, type=2, norm='ortho', axis=1)
            cleaned_signal = idct_noisy_signal_np - idct_predicted_noise

            true_noise_np = true_noise.squeeze(1).cpu().numpy()
            idct_true_noise_np = idct(true_noise_np, type=2, norm='ortho', axis=1)
            
            # Store the results
            predicted_noises.append(idct_predicted_noise)
            cleaned_signals.append(cleaned_signal)
            noisy_signals.append(idct_noisy_signal_np)
            true_noises.append(idct_true_noise_np)
            SNR_list.append(SNR)
    
    # Concatenate results into numpy arrays
    predicted_noises = np.concatenate(predicted_noises, axis=0)
    cleaned_signals = np.concatenate(cleaned_signals, axis=0)
    noisy_signals = np.concatenate(noisy_signals, axis=0)
    true_noises = np.concatenate(true_noises, axis=0)
    SNR_list = np.concatenate(SNR_list, axis=0)
    
    return predicted_noises, cleaned_signals, noisy_signals, true_noises, SNR_list

def plot_signals(noisy_signals, cleaned_signals, test_signals, SNR_List, num_samples=5):
    # Get the total number of signals
    total_signals = len(noisy_signals)
    
    # If the requested number of samples exceeds the total signals, limit it to the available number
    num_samples = min(num_samples, total_signals)
    
    # Select random samples
    indices = random.sample(range(total_signals), num_samples)
    
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, num_samples * 3))
    
    for i, idx in enumerate(indices):
        axs[i, 0].plot(noisy_signals[idx], label="Noisy Signal")
        axs[i, 0].set_title(f"Noisy Signal, SNR = {SNR_list[idx]:.2f}")
        axs[i, 0].legend()

        axs[i, 1].plot(cleaned_signals[idx], label="Cleaned Signal", color='green')
        axs[i, 1].set_title(f"Cleaned Signal")
        axs[i, 1].legend()

        axs[i, 2].plot(test_signals[:,idx], label="Test True Signal", color='orange')
        axs[i, 2].plot(test_signals[:,idx] - cleaned_signals[idx], label="Residual", color='black')
        axs[i, 2].set_title(f"Test True Signal")
        axs[i, 2].legend()

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(num_samples, 1, figsize=(15, num_samples * 3))
    for i, idx in enumerate(indices):
        axs[i].plot(noisy_signals[idx] - test_signals[:,idx], label="Real Noise")
        axs[i].plot(noisy_signals[idx] - cleaned_signals[idx], label="Predicted Noise", color='green')
        axs[i].plot(cleaned_signals[idx] - test_signals[:,idx], label="Difference", color='black')
        # axs[i].set_title(f"Smaple {idx}")
        axs[i].legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dir = "data/generated/generated_skin_spectrum_10022024_104256.pkl"  # Test data
    test_noise_dir = "data/noise/processed/test_data.pkl"
    SNR_range = [-8, 0]

    # Load the best trained model
    model_HF_path = "models/pretrained/model_10292024_090528_HF.pth"
    model_HF = RamanNoiseNet_HF()
    model_HF.load_state_dict(torch.load(model_HF_path))
    model_HF.eval()  # Set the model to evaluation mode
    model_LF_path = "models/pretrained/model_10282024_230802_LF.pth"
    model_LF = RamanNoiseNet_LF()
    model_LF.load_state_dict(torch.load(model_LF_path))
    model_LF.eval()  # Set the model to evaluation mode

    # Load test data
    test_signal, _, test_concentrations = read_clean_data(clean_dir=test_dir, customized_noise=False)
    with open(test_noise_dir, 'rb') as file:
        test_noise = pickle.load(file)

    # Create Dataset and DataLoader
    test_dataset = RamanNoiseDataset(clean_signals=test_signal, true_noises=test_noise)
    test_dataset.generate_noisy_signals(SNR_range=SNR_range)
    test_dataset.DCT()  # Apply DCT on the test data
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Test the model and get predicted noises and cleaned signals
    predicted_noises, cleaned_signals, noisy_signals, true_noise, SNR_list = test_model(model_HF, model_LF, test_dataloader, device)
    print(noisy_signals.shape)

    plot_signals(noisy_signals, cleaned_signals, test_signal, SNR_list, num_samples=5)

    print("Testing complete. Results saved.")

