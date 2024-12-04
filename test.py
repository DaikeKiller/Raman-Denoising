from train import *
from scipy.fftpack import idct
import random
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
import cv2


def moving_average(signal, window_size=5):
    """
    Apply a simple moving average filter to the input signal.
    
    Parameters:
        signal (np.ndarray): Input signal of shape (signal_length,).
        window_size (int): Size of the moving average window.
    
    Returns:
        np.ndarray: Smoothed signal of the same shape as input.
    """
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

def extend_signal_left(signals, num_points=5, method='linear'):
    """
    Extend each signal in the batch by adding `num_points` to the left using interpolation.
    
    Parameters:
        signals (np.ndarray): Input signals of shape (batch_size, signal_length).
        num_points (int): Number of points to add to the left of each signal.
        method (str): Interpolation method, either 'linear' or 'spline'.
    
    Returns:
        np.ndarray: Extended signals of shape (batch_size, signal_length + num_points).
    """
    batch_size, signal_length = signals.shape
    extended_length = signal_length + num_points
    extended_signals = np.zeros((batch_size, extended_length))

    # Indices for the original and new points
    x_original = np.arange(signal_length)
    x_new = np.arange(-num_points, 0)  # New indices to add to the left

    for i in range(batch_size):
        # Choose interpolation method
        if method == 'linear':
            # Linear extrapolation
            linear_interpolator = np.polyfit(x_original[:num_points], signals[i, :num_points], 1)
            new_points = np.polyval(linear_interpolator, x_new)

        elif method == 'spline':
            # Spline extrapolation with cubic spline
            spline_interpolator = interp1d(x_original, signals[i], kind='cubic', fill_value="extrapolate")
            new_points = spline_interpolator(x_new)

        else:
            raise ValueError("Invalid method. Choose either 'linear' or 'spline'.")

        # Concatenate the new points with the original signal
        extended_signals[i] = np.concatenate((new_points, signals[i]))

    return extended_signals

def soft_low_pass_filter_dct(dct_signal, cutoff=1031, transition_width=50):
    """
    Apply a soft low-pass filter to the DCT signal by tapering coefficients
    above the cutoff frequency using a Hann window.

    Parameters:
        dct_signal (np.ndarray): Input DCT signals with shape (batch_size, dct_length).
        cutoff (int): Cutoff frequency for the low-pass filter.
        transition_width (int): Width of the transition band for tapering. Frequencies
                                in this range around the cutoff will be gradually reduced.

    Returns:
        np.ndarray: Soft-filtered DCT signals with high frequencies smoothly attenuated.
    """
    batch_size, dct_length = dct_signal.shape

    # Ensure the cutoff frequency and transition width are within bounds
    cutoff = min(cutoff, dct_length)
    end_transition = min(cutoff + transition_width, dct_length)

    # Initialize a Hann window for the transition band
    window = np.hanning(2 * transition_width)
    transition_window = window[transition_width:]  # Use only the second half of the window

    # Create a filter mask with ones up to the cutoff frequency
    filter_mask = np.ones(dct_length)
    filter_mask[cutoff:end_transition] = transition_window  # Apply tapering window
    filter_mask[end_transition:] = 0  # Zero out frequencies above the transition band

    # Apply the filter mask to each DCT signal in the batch
    filtered_dct_signal = dct_signal * filter_mask

    return filtered_dct_signal

# Function to test the model and clean the signals
def test_model(model_HF, model_MF, model_LF, test_dataloader, device):
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
    moving_window_cleaned = []
    noise_avg_signals = []
    
    with torch.no_grad():  # Disable gradient calculation for testing
        # Progress bar for testing phase
        for noisy_signal, true_noise, _, SNR in tqdm(test_dataloader, desc="Testing", unit="batch"):
            # target_signal = noisy_signal[9, :] - true_noise[9, :]
            # target_signal = target_signal.unsqueeze(0).repeat(true_noise.shape[0], 1)
            # noisy_signal = target_signal + true_noise

            target_signal = noisy_signal - true_noise
            noisy_signal = noisy_signal.unsqueeze(1).float().to(device)
            true_noise_np = true_noise.squeeze(1).cpu().numpy()
            gt_signal = target_signal.squeeze(1).cpu().numpy()
            noisy_signal_np = noisy_signal.squeeze(1).cpu().numpy()

            noisy_signal_HF = noisy_signal[:, :, 81+950-50:]
            noisy_signal_MF = noisy_signal[:, :, 81-50:81+950]
            noisy_signal_LF = noisy_signal[:, :, :81]

            HF_center_factor = torch.mean(noisy_signal_HF, dim=2, keepdim=True)
            MF_center_factor = torch.mean(noisy_signal_MF, dim=2, keepdim=True)
            LF_center_factor = torch.mean(noisy_signal_LF, dim=2, keepdim=True)
            noisy_signal_HF = noisy_signal_HF - HF_center_factor
            noisy_signal_MF = noisy_signal_MF - MF_center_factor
            noisy_signal_LF = noisy_signal_LF - LF_center_factor

            # center_term = torch.mean(noisy_signal_HF, dim=2, keepdim=True)[0]
            # noisy_signal_HF = noisy_signal_HF - center_term
            # norm_term = torch.max(torch.abs(noisy_signal_HF), dim=2, keepdim=True)[0]
            # noisy_signal_HF = noisy_signal_HF / norm_term
            predicted_noise_HF_residual = model_HF(noisy_signal_HF).squeeze(1).cpu().numpy()
            predicted_noise_HF_residual = predicted_noise_HF_residual + HF_center_factor.squeeze(1).cpu().numpy()
            # noise_HF = noisy_signal_HF - predicted_noise_HF_residual
            # noise_HF = (noisy_signal_HF - predicted_noise_HF_residual) * norm_term + center_term

            # center_term = torch.mean(noisy_signal_MF, dim=2, keepdim=True)[0]
            # noisy_signal_MF = noisy_signal_MF - center_term
            # norm_term = torch.max(torch.abs(noisy_signal_MF), dim=2, keepdim=True)[0]
            # noisy_signal_MF = noisy_signal_MF / norm_term
            predicted_noise_MF_residual = model_MF(noisy_signal_MF).squeeze(1).cpu().numpy()
            predicted_noise_MF_residual = predicted_noise_MF_residual + MF_center_factor.squeeze(1).cpu().numpy()
            # noise_MF = noisy_signal_MF - predicted_noise_MF_residual
            # noise_MF = (noisy_signal_MF - predicted_noise_MF_residual) * norm_term + center_term

            # center_term = torch.mean(noisy_signal_LF, dim=2, keepdim=True)[0]
            # noisy_signal_LF = noisy_signal_LF - center_term
            # norm_term = torch.max(torch.abs(noisy_signal_LF), dim=2, keepdim=True)[0]
            # noisy_signal_LF = noisy_signal_LF / norm_term
            predicted_noise_LF = model_LF(noisy_signal_LF).squeeze(1).cpu().numpy()
            # predicted_noise_LF = predicted_noise_LF + LF_center_factor.squeeze(1).cpu().numpy()
            predicted_noise_LF_residual = noisy_signal_LF.squeeze(1).cpu().numpy() - predicted_noise_LF
            # predicted_noise_LF_residual = predicted_noise_LF

            predicted_noise_residual = np.concatenate((predicted_noise_LF_residual[:,:81-5], predicted_noise_MF_residual[:,45:-25], predicted_noise_HF_residual[:,25:]), axis=1)
            predicted_noise_residual = soft_low_pass_filter_dct(predicted_noise_residual, 1031, 50)

            predicted_noise = noisy_signal_np - predicted_noise_residual

            # noise_HF = noise_HF.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension
            # noise_MF = noise_MF.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension
            # noise_LF = noise_LF.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension

            # predicted_noise = np.concatenate((noise_LF, noise_MF, noise_HF), axis=1)

            # predicted_noise[:,:81] = true_noise_np[:,:81] # !!!!!test******

            idct_predicted_noise = idct(predicted_noise, type=2, norm='ortho', axis=1)
            
            # Subtract predicted noise from noisy signal to clean the signal
            idct_noisy_signal_np = idct(noisy_signal_np, type=2, norm='ortho', axis=1)
            cleaned_signal = idct_noisy_signal_np - idct_predicted_noise
            remove_num = 3 # to deal with zero-point spike
            cleaned_signal[:,:remove_num] = cleaned_signal[:,remove_num:remove_num*2]

            # cleaned_signal = als_baseline_correction(cleaned_signal, lam=1e8, p=0.001, n_iter=10)
            # cleaned_signal = rolling_ball_baseline(cleaned_signal, window_size=200)
            # cleaned_signal = bilateral_cdf(cleaned_signal)
            # cleaned_signal[:,0] = cleaned_signal[:,1]

            # cleaned_signal[cleaned_signal < 0.05] = 0

            idct_true_noise_np = idct(true_noise_np, type=2, norm='ortho', axis=1)
            idct_gt_signal = idct(gt_signal, type=2, norm='ortho', axis=1)

            # moving avg
            idct_noisy_signal_np_filtered = np.array([moving_average(signal, 13) for signal in idct_noisy_signal_np])

            # noise avg
            idct_noisy_signal_np_avg = [np.mean(idct_noisy_signal_np[:10,:], axis=0) for i in range(idct_noisy_signal_np.shape[0])]
            idct_noisy_signal_np_avg = np.array(idct_noisy_signal_np_avg)
            
            # Store the results
            predicted_noises.append(idct_predicted_noise)
            cleaned_signals.append(cleaned_signal)
            noisy_signals.append(idct_noisy_signal_np)
            moving_window_cleaned.append(idct_noisy_signal_np_filtered)
            noise_avg_signals.append(idct_noisy_signal_np_avg)
            true_noises.append(idct_true_noise_np)
            gt_signals.append(idct_gt_signal)
            SNR_list.append(SNR)

        # Concatenate results into numpy arrays
        predicted_noises = np.concatenate(predicted_noises, axis=0)
        cleaned_signals = np.concatenate(cleaned_signals, axis=0)
        noisy_signals = np.concatenate(noisy_signals, axis=0)
        moving_window_cleaned = np.concatenate(moving_window_cleaned, axis=0)
        noise_avg_signals = np.concatenate(noise_avg_signals, axis=0)
        true_noises = np.concatenate(true_noises, axis=0)
        gt_signals = np.concatenate(gt_signals, axis=0)
        SNR_list = np.concatenate(SNR_list, axis=0)


        # cleaned_signals = cleaned_signals / np.max(cleaned_signals, axis=1).reshape(-1, 1)
    
    return predicted_noises, cleaned_signals, noisy_signals, moving_window_cleaned, noise_avg_signals, true_noises, gt_signals, SNR_list

def als_baseline_correction(spectra, lam=1e6, p=0.001, n_iter=10):
    """
    Perform Asymmetric Least Squares (ALS) baseline correction on a set of spectra.

    Parameters:
    spectra (numpy.ndarray): Input array of shape (n, 1981), where n is the number of spectra.
    lam (float): Smoothing parameter (lambda), controls the smoothness of the baseline.
    p (float): Asymmetry parameter, controls how much positive residuals are penalized.
    n_iter (int): Number of iterations for fitting the baseline.

    Returns:
    numpy.ndarray: Baseline corrected spectra of shape (n, 1981).
    """
    n, m = spectra.shape
    baselines = np.zeros((n, m))
    corrected_spectra = np.zeros((n, m))
    
    for i in range(n):
        y = spectra[i, :]
        D = np.diff(np.eye(m), 2, axis=0)
        D = lam * D.T @ D
        w = np.ones(m)
        for _ in range(n_iter):
            W = np.diag(w)
            Z = W + D
            baseline = np.linalg.solve(Z, w * y)
            w = p * (y > baseline) + (1 - p) * (y < baseline)
        baselines[i, :] = baseline
        corrected_spectra[i, :] = y - baseline
    
    return corrected_spectra

def rolling_ball_baseline(spectra, window_size=100):
    """
    Perform Rolling Ball baseline correction on a set of Raman spectra.

    Parameters:
    spectra (numpy.ndarray): Input array of shape (n, m), where n is the number of spectra and m is the spectrum length.
    window_size (int): Size of the rolling ball window used to estimate the baseline.

    Returns:
    numpy.ndarray: Baseline corrected spectra of shape (n, m).
    """
    n, m = spectra.shape
    baselines = np.zeros((n, m))
    corrected_spectra = np.zeros((n, m))

    for i in range(n):
        y = spectra[i, :]
        # Apply uniform filter as a rolling ball (minimum filter)
        smoothed = uniform_filter1d(y, size=window_size, mode='nearest')
        baselines[i, :] = smoothed
        corrected_spectra[i, :] = y - smoothed
    
    return corrected_spectra

def bilateral_cdf(spectra, d=150, sigma_color=10, sigma_space=300):
    n, m = spectra.shape
    cdf = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cdf[i, j] = spectra[i, j] + cdf[i, j-1] if j != 0 else spectra[i, j]
    
    filtered_cdfs = np.zeros((n, m))
    for i in range(n):
        # OpenCV's bilateral filter works on 2D images, so we need to reshape the signal to (m, 1)
        cdf_reshaped = cdf[i, :].astype(np.float32).reshape(-1, 1)
        filtered_cdf = cv2.bilateralFilter(cdf_reshaped, d, sigma_color, sigma_space)
        filtered_cdfs[i, :] = filtered_cdf.flatten()

    cleaned = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cleaned[i, j] = filtered_cdfs[i, j] - filtered_cdfs[i, j-1] if j != 0 else filtered_cdfs[i, j]

    return cleaned

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
    moving_window_cleaned = []
    noise_avg_signals = []
    
    with torch.no_grad():  # Disable gradient calculation for testing
        # Progress bar for testing phase
        for noisy_signal, true_noise, _, SNR in tqdm(test_dataloader, desc="Testing", unit="batch"):
            target_signal = noisy_signal[9, :] - true_noise[9, :]
            target_signal = target_signal.unsqueeze(0).repeat(true_noise.shape[0], 1)
            target_signal = torch.from_numpy(np.array([a * b for a, b, in zip(target_signal, np.sqrt(SNR))]))
            noisy_signal = target_signal + true_noise

            target_signal = noisy_signal - true_noise
            noisy_signal = noisy_signal.unsqueeze(1).float().to(device)
            true_noise_np = true_noise.squeeze(1).cpu().numpy()
            gt_signal = target_signal.squeeze(1).cpu().numpy()
            noisy_signal_np = noisy_signal.squeeze(1).cpu().numpy()

            noisy_signal_HF = noisy_signal[:, :, 81+950-50:]
            noisy_signal_MF = noisy_signal[:, :, 81-50:81+950]
            noisy_signal_LF = noisy_signal[:, :, :81]

            HF_center_factor = torch.mean(noisy_signal_HF, dim=2, keepdim=True)
            MF_center_factor = torch.mean(noisy_signal_MF, dim=2, keepdim=True)
            LF_center_factor = torch.mean(noisy_signal_LF, dim=2, keepdim=True)
            noisy_signal_HF = noisy_signal_HF - HF_center_factor
            noisy_signal_MF = noisy_signal_MF - MF_center_factor
            noisy_signal_LF = noisy_signal_LF - LF_center_factor

            # center_term = torch.mean(noisy_signal_HF, dim=2, keepdim=True)[0]
            # noisy_signal_HF = noisy_signal_HF - center_term
            # norm_term = torch.max(torch.abs(noisy_signal_HF), dim=2, keepdim=True)[0]
            # noisy_signal_HF = noisy_signal_HF / norm_term
            predicted_noise_HF_residual = model_HF(noisy_signal_HF).squeeze(1).cpu().numpy()
            predicted_noise_HF_residual = predicted_noise_HF_residual + HF_center_factor.squeeze(1).cpu().numpy()
            # noise_HF = noisy_signal_HF - predicted_noise_HF_residual
            # noise_HF = (noisy_signal_HF - predicted_noise_HF_residual) * norm_term + center_term

            # center_term = torch.mean(noisy_signal_MF, dim=2, keepdim=True)[0]
            # noisy_signal_MF = noisy_signal_MF - center_term
            # norm_term = torch.max(torch.abs(noisy_signal_MF), dim=2, keepdim=True)[0]
            # noisy_signal_MF = noisy_signal_MF / norm_term
            predicted_noise_MF_residual = model_MF(noisy_signal_MF).squeeze(1).cpu().numpy()
            predicted_noise_MF_residual = predicted_noise_MF_residual + MF_center_factor.squeeze(1).cpu().numpy()
            # noise_MF = noisy_signal_MF - predicted_noise_MF_residual
            # noise_MF = (noisy_signal_MF - predicted_noise_MF_residual) * norm_term + center_term

            # center_term = torch.mean(noisy_signal_LF, dim=2, keepdim=True)[0]
            # noisy_signal_LF = noisy_signal_LF - center_term
            # norm_term = torch.max(torch.abs(noisy_signal_LF), dim=2, keepdim=True)[0]
            # noisy_signal_LF = noisy_signal_LF / norm_term
            predicted_noise_LF = model_LF(noisy_signal_LF).squeeze(1).cpu().numpy()
            # predicted_noise_LF = predicted_noise_LF + LF_center_factor.squeeze(1).cpu().numpy()
            predicted_noise_LF_residual = noisy_signal_LF.squeeze(1).cpu().numpy() - predicted_noise_LF
            # predicted_noise_LF_residual = predicted_noise_LF

            predicted_noise_residual = np.concatenate((predicted_noise_LF_residual[:,:81-5], predicted_noise_MF_residual[:,45:-25], predicted_noise_HF_residual[:,25:]), axis=1)
            predicted_noise_residual = soft_low_pass_filter_dct(predicted_noise_residual, 1031, 50)

            predicted_noise = noisy_signal_np - predicted_noise_residual

            # noise_HF = noise_HF.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension
            # noise_MF = noise_MF.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension
            # noise_LF = noise_LF.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension

            # predicted_noise = np.concatenate((noise_LF, noise_MF, noise_HF), axis=1)

            # predicted_noise[:,:81] = true_noise_np[:,:81] # !!!!!test******

            idct_predicted_noise = idct(predicted_noise, type=2, norm='ortho', axis=1)
            
            # Subtract predicted noise from noisy signal to clean the signal
            idct_noisy_signal_np = idct(noisy_signal_np, type=2, norm='ortho', axis=1)
            cleaned_signal = idct_noisy_signal_np - idct_predicted_noise
            remove_num = 3 # to deal with zero-point spike
            cleaned_signal[:,:remove_num] = cleaned_signal[:,remove_num:remove_num*2]

            # cleaned_signal = als_baseline_correction(cleaned_signal, lam=1e8, p=0.001, n_iter=10)
            # cleaned_signal = rolling_ball_baseline(cleaned_signal, window_size=200)
            # cleaned_signal = bilateral_cdf(cleaned_signal)
            # cleaned_signal[:,0] = cleaned_signal[:,1]

            # cleaned_signal[cleaned_signal < 0.05] = 0

            idct_true_noise_np = idct(true_noise_np, type=2, norm='ortho', axis=1)
            idct_gt_signal = idct(gt_signal, type=2, norm='ortho', axis=1)

            # moving avg
            idct_noisy_signal_np_filtered = np.array([moving_average(signal, 13) for signal in idct_noisy_signal_np])

            # noise avg
            idct_noisy_signal_np_avg = [np.mean(idct_noisy_signal_np[:10,:], axis=0) for i in range(idct_noisy_signal_np.shape[0])]
            idct_noisy_signal_np_avg = np.array(idct_noisy_signal_np_avg)
            
            # Store the results
            predicted_noises.append(idct_predicted_noise)
            cleaned_signals.append(cleaned_signal)
            noisy_signals.append(idct_noisy_signal_np)
            moving_window_cleaned.append(idct_noisy_signal_np_filtered)
            noise_avg_signals.append(idct_noisy_signal_np_avg)
            true_noises.append(idct_true_noise_np)
            gt_signals.append(idct_gt_signal)
            SNR_list.append(SNR)

        # Concatenate results into numpy arrays
        predicted_noises = np.concatenate(predicted_noises, axis=0)
        cleaned_signals = np.concatenate(cleaned_signals, axis=0)
        noisy_signals = np.concatenate(noisy_signals, axis=0)
        moving_window_cleaned = np.concatenate(moving_window_cleaned, axis=0)
        noise_avg_signals = np.concatenate(noise_avg_signals, axis=0)
        true_noises = np.concatenate(true_noises, axis=0)
        gt_signals = np.concatenate(gt_signals, axis=0)
        SNR_list = np.concatenate(SNR_list, axis=0)


        # cleaned_signals = cleaned_signals / np.max(cleaned_signals, axis=1).reshape(-1, 1)
    
    return predicted_noises, cleaned_signals, noisy_signals, moving_window_cleaned, noise_avg_signals, true_noises, gt_signals, SNR_list

def plot_signals(noisy_signals, cleaned_signals, moving_window_cleaned, noise_avg_signals, gt_signals, SNR_list, num_samples=5, save_path="./results/"):
    # Sort indices based on SNR_list in ascending order
    sorted_indices = np.argsort(SNR_list)
    SNR_list = np.array(SNR_list)[sorted_indices]
    noisy_signals = np.array(noisy_signals)[sorted_indices]
    cleaned_signals = np.array(cleaned_signals)[sorted_indices]
    moving_window_cleaned = np.array(moving_window_cleaned)[sorted_indices]
    gt_signals = np.array(gt_signals)[sorted_indices]
    
    # Get the total number of signals
    total_signals = len(noisy_signals)
    
    # If the requested number of samples exceeds the total signals, limit it to the available number
    num_samples = min(num_samples, total_signals)
    
    selected_indices = []
    for i in range(num_samples):
        selected_indices.append(i*total_signals//num_samples + total_signals//num_samples//10)
        
    # Select random samples
    # selected_indices = sorted(random.sample(range(len(SNR_list)), num_samples))
    
    fig, axs = plt.subplots(num_samples, 5, figsize=(25, num_samples * 3))
    
    for i, idx in enumerate(selected_indices):
        axs[i, 0].plot(np.linspace(800, 1800, 1981), noisy_signals[idx], label="Noisy Signal")
        axs[i, 0].set_title(f"Noisy Signal, SNR = {SNR_list[idx]:.2f}")
        # axs[i, 0].legend()

        axs[i, 1].plot(np.linspace(800, 1800, 1981), moving_window_cleaned[idx], label="Cleaned Signal with moving avg (window_size=11)", color='green')
        axs[i, 1].set_title(f"Cleaned Signal with moving avg (window_size=11)")
        # axs[i, 1].legend()

        axs[i, 2].plot(np.linspace(800, 1800, 1981), noise_avg_signals[idx], label="Cleaned Signal with noise avg (avg num=10)", color='green')
        axs[i, 2].set_title(f"Cleaned Signal with noise avg (avg num=10)")

        axs[i, 3].plot(np.linspace(800, 1800, 1981), cleaned_signals[idx], label="Cleaned Signal with model", color='green')
        axs[i, 3].set_title(f"Cleaned Signal with model")
        # axs[i, 3].legend()

        axs[i, 4].plot(np.linspace(800, 1800, 1981), gt_signals[idx], label="Test True Signal", color='orange')
        axs[i, 4].plot(np.linspace(800, 1800, 1981), gt_signals[idx] - cleaned_signals[idx], label="Residual_model", color='black')
        axs[i, 4].set_title(f"Test True Signal")
        axs[i, 4].legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "signal.png"))

    fig, axs = plt.subplots(num_samples, 1, figsize=(5, num_samples * 3))
    for i, idx in enumerate(selected_indices):
        axs[i].plot(np.linspace(800, 1800, 1981), gt_signals[idx] - cleaned_signals[idx], label="Residual_model", color='black')
        axs[i].plot(np.linspace(800, 1800, 1981), gt_signals[idx] - moving_window_cleaned[idx], label="Residual_moving_window", color='gray')
        axs[i].plot(np.linspace(800, 1800, 1981), gt_signals[idx] - noise_avg_signals[idx], label="Residual_noise_avg", color='brown')
        # axs[i].set_title(f"Smaple {idx}")
        axs[i].legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "residual.png"))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dir = "data/generated/generated_skin_spectrum_11012024_143232.pkl"  # Test data
    test_dir_pV = "data/generated/raman_pesudo_Vioget_test_11182024_110755.pkl"  # Test data
    test_noise_dir = "data/noise/processed_new/test_data.pkl"
    SNR_range = [0.01, 1]
    # SNR_range = [np.log10(a) for a in SNR_range]

    save_data_flag = False
    save_path = "./results/"

    # Load the best trained model
    model_HF_path = "models/pretrained/model_11252024_231410_HF.pth"
    model_HF = RamanNoiseNet_HF()
    model_HF.load_state_dict(torch.load(model_HF_path))
    model_HF.eval()  # Set the model to evaluation mode
    model_MF_path = "models/pretrained/model_11252024_231410_MF.pth"
    model_MF = RamanNoiseNet_HF()
    model_MF.load_state_dict(torch.load(model_MF_path))
    model_HF.eval()  # Set the model to evaluation mode
    model_LF_path = "models/pretrained/model_11252024_231410_LF.pth"
    model_LF = AUnet(1, 1)
    # model_LF = RamanNoiseNet_LF()
    model_LF.load_state_dict(torch.load(model_LF_path))
    model_LF.eval()  # Set the model to evaluation mode

    # Load test data
    test_signal_skin, _, test_concentrations = read_clean_data(clean_dir=test_dir, customized_noise=False)
    # Load test data
    test_signal_pV, _ = read_clean_data(clean_dir=test_dir_pV, customized_noise=False, pV=True)
    with open(test_noise_dir, 'rb') as file:
        test_noise = pickle.load(file)

    # test_signal = np.concatenate((test_signal_skin, test_signal_pV), axis=1)
    test_signal = test_signal_skin

    # Create Dataset and DataLoader
    test_dataset = RamanNoiseDataset(clean_signals=test_signal, true_noises=test_noise)
    test_dataset.generate_noisy_signals(SNR_range=SNR_range)
    test_dataset.DCT()  # Apply DCT on the test data
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Test the model and get predicted noises and cleaned signals
    predicted_noises, cleaned_signals, noisy_signals, moving_window_cleaned, noise_avg_signals, true_noises, gt_signals, SNR_list = test_on_one_signal(model_HF, model_MF, model_LF, test_dataloader, device)
    # predicted_noises, cleaned_signals, noisy_signals, moving_window_cleaned, noise_avg_signals, true_noises, gt_signals, SNR_list = test_model(model_HF, model_MF, model_LF, test_dataloader, device)

    # save
    if save_data_flag:
        save_data = {"noisy_signals": noisy_signals, "cleaned_signals": cleaned_signals, "moving_window_cleaned": moving_window_cleaned, \
                    "noise_avg_signals": noise_avg_signals, "gt_signals": gt_signals, "SNR_list": SNR_list}
        save_results = os.path.join(save_path, 'results_0.01to1.pkl')
        with open(save_results, 'wb') as f:
            pickle.dump(save_data, f)

    plot_signals(noisy_signals, cleaned_signals, moving_window_cleaned, noise_avg_signals, gt_signals, SNR_list, num_samples=5)

    print("Testing complete. Results saved.")

