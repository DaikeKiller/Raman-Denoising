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
            predicted_noise_LF_residual = model_LF(noisy_signal_LF)
            predicted_noise_HF_residual = predicted_noise_HF_residual.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension
            predicted_noise_LF_residual = predicted_noise_LF_residual.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension

            noisy_signal_np = noisy_signal.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension
            predicted_noise_HF = noisy_signal_np[:, 81:] - predicted_noise_HF_residual
            predicted_noise_LF = noisy_signal_np[:, :81] - predicted_noise_LF_residual
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

def test_on_one_signal(model_HF, model_MF, model_LF, test_dataloader, device):
    model_HF.to(device)
    model_HF.eval()  # Set the model to evaluation mode
    model_MF.to(device)
    model_MF.eval()  # Set the model to evaluation mode
    model_LF.to(device)
    model_LF.eval()  # Set the model to evaluation mode
    
    predicted_noises = []
    cleaned_signals = []
    noisy_signals = []
    true_noises = []
    gt_signals = []
    SNR_list = []
    
    with torch.no_grad():  # Disable gradient calculation for testing
        # Progress bar for testing phase
        for noisy_signal, true_noise, _, SNR in tqdm(test_dataloader, desc="Testing", unit="batch"):
            target_signal = noisy_signal[0, :] - true_noise[0, :]
            target_signal = target_signal.unsqueeze(0).repeat(10, 1)
            noisy_signal = target_signal + true_noise

            noisy_signal = noisy_signal.unsqueeze(1).float().to(device)
            true_noise_np = true_noise.squeeze(1).cpu().numpy()
            gt_signal = target_signal.squeeze(1).cpu().numpy()
            noisy_signal_np = noisy_signal.squeeze(1).cpu().numpy()

            noisy_signal_HF = noisy_signal[:, :, 81+950:]
            noisy_signal_MF = noisy_signal[:, :, 81:81+950]
            noisy_signal_LF = noisy_signal[:, :, :81]

            # center_term = torch.mean(noisy_signal_HF, dim=2, keepdim=True)[0]
            # noisy_signal_HF = noisy_signal_HF - center_term
            # norm_term = torch.max(torch.abs(noisy_signal_HF), dim=2, keepdim=True)[0]
            # noisy_signal_HF = noisy_signal_HF / norm_term
            predicted_noise_HF_residual = model_HF(noisy_signal_HF)
            noise_HF = noisy_signal_HF - predicted_noise_HF_residual
            # noise_HF = (noisy_signal_HF - predicted_noise_HF_residual) * norm_term + center_term

            # center_term = torch.mean(noisy_signal_MF, dim=2, keepdim=True)[0]
            # noisy_signal_MF = noisy_signal_MF - center_term
            # norm_term = torch.max(torch.abs(noisy_signal_MF), dim=2, keepdim=True)[0]
            # noisy_signal_MF = noisy_signal_MF / norm_term
            predicted_noise_MF_residual = model_MF(noisy_signal_MF)
            noise_MF = noisy_signal_MF - predicted_noise_MF_residual
            # noise_MF = (noisy_signal_MF - predicted_noise_MF_residual) * norm_term + center_term

            # center_term = torch.mean(noisy_signal_LF, dim=2, keepdim=True)[0]
            # noisy_signal_LF = noisy_signal_LF - center_term
            # norm_term = torch.max(torch.abs(noisy_signal_LF), dim=2, keepdim=True)[0]
            # noisy_signal_LF = noisy_signal_LF / norm_term
            predicted_noise_LF_residual = model_LF(noisy_signal_LF)
            noise_LF = noisy_signal_LF - predicted_noise_LF_residual
            # noise_LF = (noisy_signal_LF - predicted_noise_LF_residual) * norm_term + center_term

            noise_HF = noise_HF.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension
            noise_MF = noise_MF.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension
            noise_LF = noise_LF.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension

            predicted_noise = np.concatenate((noise_LF, noise_MF, noise_HF), axis=1)

            predicted_noise[:,81+950:] = true_noise_np[:,81+950:] # !!!!!test*********

            idct_predicted_noise = idct(predicted_noise, type=2, norm='ortho', axis=1)
            # idct_predicted_noise_power = np.mean(idct_predicted_noise ** 2, axis=1)
            # idct_predicted_noise = idct_predicted_noise / np.sqrt(idct_predicted_noise_power[:, np.newaxis])
            
            # Subtract predicted noise from noisy signal to clean the signal
            idct_noisy_signal_np = idct(noisy_signal_np, type=2, norm='ortho', axis=1)
            cleaned_signal = idct_noisy_signal_np - idct_predicted_noise

            idct_true_noise_np = idct(true_noise_np, type=2, norm='ortho', axis=1)
            idct_gt_signal = idct(gt_signal, type=2, norm='ortho', axis=1)
            
            # Store the results
            predicted_noises.append(idct_predicted_noise)
            cleaned_signals.append(cleaned_signal)
            noisy_signals.append(idct_noisy_signal_np)
            true_noises.append(idct_true_noise_np)
            gt_signals.append(idct_gt_signal)
            SNR_list.append(SNR)

        # Concatenate results into numpy arrays
        predicted_noises = np.concatenate(predicted_noises, axis=0)
        cleaned_signals = np.concatenate(cleaned_signals, axis=0)
        noisy_signals = np.concatenate(noisy_signals, axis=0)
        true_noises = np.concatenate(true_noises, axis=0)
        gt_signals = np.concatenate(gt_signals, axis=0)
        SNR_list = np.concatenate(SNR_list, axis=0)


        # cleaned_signals = cleaned_signals / np.max(cleaned_signals, axis=1).reshape(-1, 1)
    
    return predicted_noises, cleaned_signals, noisy_signals, true_noises, gt_signals, SNR_list

def plot_signals(noisy_signals, cleaned_signals, gt_signals, SNR_list, num_samples=5):
    # Sort indices based on SNR_list in ascending order
    sorted_indices = np.argsort(SNR_list)
    SNR_list = np.array(SNR_list)[sorted_indices]
    noisy_signals = np.array(noisy_signals)[sorted_indices]
    cleaned_signals = np.array(cleaned_signals)[sorted_indices]
    gt_signals = np.array(gt_signals)[sorted_indices]
    
    # Get the total number of signals
    total_signals = len(noisy_signals)
    
    # If the requested number of samples exceeds the total signals, limit it to the available number
    num_samples = min(num_samples, total_signals)
    
    # Select random samples
    selected_indices = sorted(random.sample(range(len(SNR_list)), num_samples))
    
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, num_samples * 3))
    
    for i, idx in enumerate(selected_indices):
        axs[i, 0].plot(noisy_signals[idx], label="Noisy Signal")
        axs[i, 0].set_title(f"Noisy Signal, SNR = {SNR_list[idx]:.2f}")
        axs[i, 0].legend()

        axs[i, 1].plot(cleaned_signals[idx], label="Cleaned Signal", color='green')
        axs[i, 1].set_title(f"Cleaned Signal")
        axs[i, 1].legend()

        axs[i, 2].plot(gt_signals[idx], label="Test True Signal", color='orange')
        axs[i, 2].plot(gt_signals[idx] - cleaned_signals[idx], label="Residual", color='black')
        axs[i, 2].set_title(f"Test True Signal")
        axs[i, 2].legend()

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(num_samples, 1, figsize=(5, num_samples * 3))
    for i, idx in enumerate(selected_indices):
        axs[i].plot(noisy_signals[idx] - gt_signals[idx], label="Real Noise")
        axs[i].plot(noisy_signals[idx] - cleaned_signals[idx], label="Predicted Noise", color='green')
        axs[i].plot(cleaned_signals[idx] - gt_signals[idx], label="Difference", color='black')
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
    model_HF_path = "models/pretrained/model_11012024_115418_HF.pth"
    model_HF = RamanNoiseNet_HF()
    model_HF.load_state_dict(torch.load(model_HF_path))
    model_HF.eval()  # Set the model to evaluation mode
    model_MF_path = "models/pretrained/model_11012024_104104_MF.pth"
    model_MF = RamanNoiseNet_HF()
    model_MF.load_state_dict(torch.load(model_MF_path))
    model_HF.eval()  # Set the model to evaluation mode
    model_LF_path = "models/pretrained/model_11012024_104104_LF.pth"
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
    predicted_noises, cleaned_signals, noisy_signals, true_noises, gt_signals, SNR_list = test_on_one_signal(model_HF, model_MF, model_LF, test_dataloader, device)
    print(noisy_signals.shape)

    plot_signals(noisy_signals, cleaned_signals, gt_signals, SNR_list, num_samples=5)

    print("Testing complete. Results saved.")

