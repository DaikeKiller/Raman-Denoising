from train_nl import *
from scipy.fftpack import idct, dct
import random
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
import cv2
from scipy.io import loadmat


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
    int_time_list = []
    r2f_list = []
    
    with torch.no_grad():  # Disable gradient calculation for testing
        # Progress bar for testing phase
        for noisy_signal, noisy_signal_dct, _, gt_signal_dct, SNR, int_time, r2f in tqdm(test_dataloader, desc="Testing", unit="batch"):

            noisy_signal = noisy_signal.unsqueeze(1).float().to(device)
            noisy_signal_dct = noisy_signal_dct.unsqueeze(1).float().to(device)
            gt_signal_dct = gt_signal_dct.unsqueeze(1).float().to(device)
            # get the max of the noisy_signal
            max_values = noisy_signal.max(dim=2, keepdim=True)[0]
            noisy_signal_dct_norm = noisy_signal_dct / max_values
            # true_noise_np = true_noise.squeeze(1).cpu().numpy()
            # gt_signal = target_signal.squeeze(1).cpu().numpy()
            # noisy_signal_np = noisy_signal.squeeze(1).cpu().numpy()

            noisy_signal_dct_norm_HF = noisy_signal_dct_norm[:, :, 51+321:]
            noisy_signal_dct_norm_MF = noisy_signal_dct_norm[:, :, 51:51+321]
            noisy_signal_dct_norm_LF = noisy_signal_dct_norm[:, :, :51]

            # HF_center_factor = torch.mean(noisy_signal_HF, dim=2, keepdim=True)
            # MF_center_factor = torch.mean(noisy_signal_MF, dim=2, keepdim=True)
            # LF_center_factor = torch.mean(noisy_signal_LF, dim=2, keepdim=True)
            # noisy_signal_HF = noisy_signal_HF - HF_center_factor
            # noisy_signal_MF = noisy_signal_MF - MF_center_factor
            # noisy_signal_LF = noisy_signal_LF - LF_center_factor

            predicted_noise_dct_HF = model_HF(noisy_signal_dct_norm_HF).squeeze(1).cpu().numpy()
            predicted_noise_dct_MF = model_MF(noisy_signal_dct_norm_MF).squeeze(1).cpu().numpy()
            predicted_noise_dct_LF = model_LF(noisy_signal_dct_norm_LF).squeeze(1).cpu().numpy()

            predicted_noise_dct = np.concatenate((predicted_noise_dct_LF, predicted_noise_dct_MF, predicted_noise_dct_HF), axis=1) * max_values.squeeze(1).cpu().numpy()
            # predicted_noise[:,:50] = true_noise_np[:,:50] # !!!!!test******

            predicted_noise = idct(predicted_noise_dct, type=2, norm='ortho', axis=1)
            
            # Subtract predicted noise from noisy signal to clean the signal
            cleaned_signal_dct = noisy_signal_dct.squeeze(1).cpu().numpy() - predicted_noise_dct
            cleaned_signal = noisy_signal.squeeze(1).cpu().numpy() - predicted_noise
            # remove_num = 5 # to deal with zero-point spike
            # cleaned_signal[:,:remove_num] = cleaned_signal[:,remove_num:remove_num*2]

            # cleaned_signal = als_baseline_correction(cleaned_signal, lam=1e8, p=0.001, n_iter=10)
            # cleaned_signal = rolling_ball_baseline(cleaned_signal, window_size=200)
            # cleaned_signal = bilateral_cdf(cleaned_signal)
            # cleaned_signal[:,0] = cleaned_signal[:,1]

            # cleaned_signal[cleaned_signal < 0.05] = 0

            true_noise_dct = noisy_signal_dct.squeeze(1).cpu().numpy() - gt_signal_dct.squeeze(1).cpu().numpy()
            true_noise = idct(true_noise_dct, type=2, norm='ortho', axis=1)
            gt_signal = idct(gt_signal_dct.squeeze(1).cpu().numpy(), type=2, norm='ortho', axis=1)
            
            # Store the results
            predicted_noises.append(predicted_noise)
            cleaned_signals.append(cleaned_signal)
            noisy_signals.append(noisy_signal.squeeze(1).cpu().numpy())
            true_noises.append(true_noise)
            gt_signals.append(gt_signal)
            SNR_list.append(SNR)
            int_time_list.append(int_time)
            r2f_list.append(r2f)

        # Concatenate results into numpy arrays
        predicted_noises = np.concatenate(predicted_noises, axis=0)
        cleaned_signals = np.concatenate(cleaned_signals, axis=0)
        noisy_signals = np.concatenate(noisy_signals, axis=0)
        true_noises = np.concatenate(true_noises, axis=0)
        gt_signals = np.concatenate(gt_signals, axis=0)
        SNR_list = np.concatenate(SNR_list, axis=0)
        int_times = np.concatenate(int_time_list, axis=0)
        r2fs = np.concatenate(r2f_list, axis=0)


        # cleaned_signals = cleaned_signals / np.max(cleaned_signals, axis=1).reshape(-1, 1)
    
    return predicted_noises, cleaned_signals, noisy_signals, true_noises, gt_signals, SNR_list, int_times, r2fs

# Function to test the model and clean the signals
def test_model_full(model_full, test_dataloader, device):
    model_full.to(device)
    model_full.eval()  # Set the model to evaluation mode
    
    predicted_noises = []
    cleaned_signals = []
    noisy_signals = []
    true_noises = []
    gt_signals = []
    SNR_list = []
    int_time_list = []
    r2f_list = []
    
    with torch.no_grad():  # Disable gradient calculation for testing
        # Progress bar for testing phase
        for noisy_signal, noisy_signal_dct, _, gt_signal_dct, gt_raman_dct, gt_flu_dct, SNR, int_time, r2f in tqdm(test_dataloader, desc="Testing", unit="batch"):

            noisy_signal = noisy_signal.unsqueeze(1).float().to(device)
            noisy_signal_dct = noisy_signal_dct.unsqueeze(1).float().to(device)
            gt_signal_dct = gt_signal_dct.unsqueeze(1).float().to(device)
            # get the max of the noisy_signal
            max_values = noisy_signal.max(dim=2, keepdim=True)[0]
            noisy_signal_dct_norm = noisy_signal_dct / max_values
            # true_noise_np = true_noise.squeeze(1).cpu().numpy()
            # gt_signal = target_signal.squeeze(1).cpu().numpy()
            # noisy_signal_np = noisy_signal.squeeze(1).cpu().numpy()

            # HF_center_factor = torch.mean(noisy_signal_HF, dim=2, keepdim=True)
            # MF_center_factor = torch.mean(noisy_signal_MF, dim=2, keepdim=True)
            # LF_center_factor = torch.mean(noisy_signal_LF, dim=2, keepdim=True)
            # noisy_signal_HF = noisy_signal_HF - HF_center_factor
            # noisy_signal_MF = noisy_signal_MF - MF_center_factor
            # noisy_signal_LF = noisy_signal_LF - LF_center_factor

            predicted_noise_dct = model_full(noisy_signal_dct_norm).squeeze(1).cpu().numpy() * max_values.squeeze(1).cpu().numpy()

            # predicted_noise[:,:50] = true_noise_np[:,:50] # !!!!!test******

            predicted_noise = idct(predicted_noise_dct, type=2, norm='ortho', axis=1)
            
            # Subtract predicted noise from noisy signal to clean the signal
            cleaned_signal_dct = noisy_signal_dct.squeeze(1).cpu().numpy() - predicted_noise_dct
            cleaned_signal = noisy_signal.squeeze(1).cpu().numpy() - predicted_noise
            # remove_num = 5 # to deal with zero-point spike
            # cleaned_signal[:,:remove_num] = cleaned_signal[:,remove_num:remove_num*2]

            # cleaned_signal = als_baseline_correction(cleaned_signal, lam=1e8, p=0.001, n_iter=10)
            # cleaned_signal = rolling_ball_baseline(cleaned_signal, window_size=200)
            # cleaned_signal = bilateral_cdf(cleaned_signal)
            # cleaned_signal[:,0] = cleaned_signal[:,1]

            # cleaned_signal[cleaned_signal < 0.05] = 0

            true_noise_dct = noisy_signal_dct.squeeze(1).cpu().numpy() - gt_signal_dct.squeeze(1).cpu().numpy()
            true_noise = idct(true_noise_dct, type=2, norm='ortho', axis=1)
            gt_signal = idct(gt_signal_dct.squeeze(1).cpu().numpy(), type=2, norm='ortho', axis=1)
            
            # Store the results
            predicted_noises.append(predicted_noise)
            cleaned_signals.append(cleaned_signal)
            noisy_signals.append(noisy_signal.squeeze(1).cpu().numpy())
            true_noises.append(true_noise)
            gt_signals.append(gt_signal)
            SNR_list.append(SNR)
            int_time_list.append(int_time)
            r2f_list.append(r2f)

        # Concatenate results into numpy arrays
        predicted_noises = np.concatenate(predicted_noises, axis=0)
        cleaned_signals = np.concatenate(cleaned_signals, axis=0)
        noisy_signals = np.concatenate(noisy_signals, axis=0)
        true_noises = np.concatenate(true_noises, axis=0)
        gt_signals = np.concatenate(gt_signals, axis=0)
        SNR_list = np.concatenate(SNR_list, axis=0)
        int_times = np.concatenate(int_time_list, axis=0)
        r2fs = np.concatenate(r2f_list, axis=0)


        # cleaned_signals = cleaned_signals / np.max(cleaned_signals, axis=1).reshape(-1, 1)
    
    return predicted_noises, cleaned_signals, noisy_signals, true_noises, gt_signals, SNR_list, int_times, r2fs

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
    def get_signal_power(signal):
            # peaks, properties = find_peaks(signal, height=0.1, distance=5, prominence=0.4)
            # peak_amplitudes = properties["peak_heights"]
            # return np.sum(peak_amplitudes)
            return np.max(np.array(signal))
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
            target_signal = noisy_signal[101, :] - true_noise[101, :]
            target_signal_idct = idct(np.array(target_signal), norm="ortho")
            target_signal_power = get_signal_power(target_signal_idct)
            target_signal = target_signal.unsqueeze(0).repeat(true_noise.shape[0], 1)
            target_signal = torch.from_numpy(np.array([a * b for a, b, in zip(target_signal, SNR)])) / target_signal_power
            noisy_signal = target_signal + true_noise
            
            noise_avg_signal = np.array(noisy_signal.clone())
            for i in range(target_signal.shape[0]):
                selected_noise_idx = np.random.randint(0, target_signal.shape[0], size=(9,1))
                selected_noise_idx = np.concatenate([np.array([[i]]), selected_noise_idx], axis=0)
                noise_added = np.mean(idct(np.array(true_noise[selected_noise_idx]), norm="ortho"), axis=0)
                noise_avg_signal[i] = noise_added + idct(np.array(target_signal[i]), norm="ortho")

            noisy_signal = noisy_signal.unsqueeze(1).float().to(device)
            true_noise_np = true_noise.squeeze(1).cpu().numpy()
            gt_signal = target_signal.squeeze(1).cpu().numpy()
            noisy_signal_np = noisy_signal.squeeze(1).cpu().numpy()

            noisy_signal_HF = noisy_signal[:, :, 81+950-50:]
            noisy_signal_MF = noisy_signal[:, :, 81-50:81+950]
            noisy_signal_LF = noisy_signal[:, :, 1:81]
            noisy_signal_LF_include_dc = noisy_signal[:, :, :81]

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
            predicted_noise_HF = model_HF(noisy_signal_HF).squeeze(1).cpu().numpy()
            predicted_noise_HF = predicted_noise_HF + HF_center_factor.squeeze(1).cpu().numpy()
            # noise_HF = noisy_signal_HF - predicted_noise_HF_residual
            # noise_HF = (noisy_signal_HF - predicted_noise_HF_residual) * norm_term + center_term

            # center_term = torch.mean(noisy_signal_MF, dim=2, keepdim=True)[0]
            # noisy_signal_MF = noisy_signal_MF - center_term
            # norm_term = torch.max(torch.abs(noisy_signal_MF), dim=2, keepdim=True)[0]
            # noisy_signal_MF = noisy_signal_MF / norm_term
            predicted_noise_MF = model_MF(noisy_signal_MF).squeeze(1).cpu().numpy()
            predicted_noise_MF = predicted_noise_MF + MF_center_factor.squeeze(1).cpu().numpy()
            # noise_MF = noisy_signal_MF - predicted_noise_MF_residual
            # noise_MF = (noisy_signal_MF - predicted_noise_MF_residual) * norm_term + center_term

            # center_term = torch.mean(noisy_signal_LF, dim=2, keepdim=True)[0]
            # noisy_signal_LF = noisy_signal_LF - center_term
            # norm_term = torch.max(torch.abs(noisy_signal_LF), dim=2, keepdim=True)[0]
            # noisy_signal_LF = noisy_signal_LF / norm_term
            predicted_noise_LF = np.zeros((noisy_signal.shape[0], 81))
            predicted_noise_LF[:, 1:] = model_LF(noisy_signal_LF).squeeze(1).cpu().numpy()
            # predicted_noise_LF_residual = noisy_signal_LF_include_dc.squeeze(1).cpu().numpy() - predicted_noise_LF
            # predicted_noise_LF_residual = predicted_noise_LF

            predicted_noise = np.concatenate((predicted_noise_LF[:,:81-5], predicted_noise_MF[:,45:-25], predicted_noise_HF[:,25:]), axis=1)
            # predicted_noise_residual = soft_low_pass_filter_dct(predicted_noise_residual, 800, 50)

            # predicted_noise = noisy_signal_np - predicted_noise_residual

            # noise_HF = noise_HF.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension
            # noise_MF = noise_MF.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension
            # noise_LF = noise_LF.squeeze(1).cpu().numpy()  # Convert to numpy and remove channel dimension

            # predicted_noise = np.concatenate((noise_LF, noise_MF, noise_HF), axis=1)

            # predicted_noise[:,:50] = true_noise_np[:,:50] # !!!!!test******

            idct_predicted_noise = idct(predicted_noise, type=2, norm='ortho', axis=1)
            
            # Subtract predicted noise from noisy signal to clean the signal
            idct_noisy_signal_np = idct(noisy_signal_np, type=2, norm='ortho', axis=1)
            cleaned_signal = idct_noisy_signal_np - idct_predicted_noise
            remove_num = 5 # to deal with zero-point spike
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
            # idct_noisy_signal_np_avg = [np.mean(idct_noisy_signal_np[:10,:], axis=0) for i in range(idct_noisy_signal_np.shape[0])]
            idct_noisy_signal_np_avg = np.array(noise_avg_signal)
            
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
        
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    freq = np.linspace(0,1981/990,1981)
    plt.plot(freq, dct(true_noises[10], norm="ortho"), label="Real noise", alpha=0.4)
    plt.plot(freq, dct(predicted_noises[10], norm="ortho"), label="Predicted noise", alpha=0.4)
    plt.plot(freq, dct(true_noises[10] - predicted_noises[10], norm="ortho"), label="Difference (real minus predicted)")
    plt.xlabel("frequency (sample/wavenumber)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("DCT of noise")
    plt.legend()
    plt.subplot(1,2,2)
    wvn = np.linspace(800,1790,1981)
    plt.plot(wvn, true_noises[10], label="Real noise", alpha=0.4)
    plt.plot(wvn, predicted_noises[10], label="Predicted noise", alpha=0.4)
    plt.plot(wvn, true_noises[10] - predicted_noises[10], label="Difference (real minus predicted)")
    plt.xlabel(r'wavenumber ($\mathrm{cm}^{-1}$)')
    plt.ylabel("Intensity (a.u.)")
    plt.title("Noise in wavenumbers")
    plt.legend()
    plt.savefig("results/model_result_example.jpg")


        # cleaned_signals = cleaned_signals / np.max(cleaned_signals, axis=1).reshape(-1, 1)
    
    return predicted_noises, cleaned_signals, noisy_signals, moving_window_cleaned, noise_avg_signals, true_noises, gt_signals, SNR_list

def plot_signals(noisy_signals, cleaned_signals, gt_signals, SNR_list, int_time_list, r2f_list, num_samples=5, save_path="./tmp/"):
    # Sort indices based on SNR_list in ascending order
    sorted_indices = np.argsort(SNR_list)
    SNR_list = np.array(SNR_list)[sorted_indices]
    noisy_signals = np.array(noisy_signals)[sorted_indices]
    cleaned_signals = np.array(cleaned_signals)[sorted_indices]
    gt_signals = np.array(gt_signals)[sorted_indices]
    int_time_list = np.array(int_time_list)[sorted_indices]
    
    # Get the total number of signals
    total_signals = len(noisy_signals)
    
    # If the requested number of samples exceeds the total signals, limit it to the available number
    num_samples = min(num_samples, total_signals)
    
    selected_indices = []
    for i in range(num_samples):
        selected_indices.append(i*total_signals//num_samples + total_signals//num_samples//10)
        
    # Select random samples
    # selected_indices = sorted(random.sample(range(len(SNR_list)), num_samples))
    
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, num_samples * 3))
    
    wvn = loadmat("data/wvn_raw.mat")["wvn"].reshape(-1)
    wvn = wvn[331:]
    
    for i, idx in enumerate(selected_indices):
        axs[i, 0].plot(wvn, noisy_signals[idx], label="Noisy Signal")
        int_time_str = str(int_time_list[idx])
        if int_time_str.startswith("std_"):
            int_time_str = int_time_str[4:]
        axs[i, 0].set_title(f"Noisy Signal, SNR = {SNR_list[idx]:.2f}, int$\_$time = {int_time_str}")
        # axs[i, 0].legend()

        axs[i, 1].plot(wvn, cleaned_signals[idx], label="Cleaned Signal with model", color='green')
        axs[i, 1].set_title(f"Cleaned Signal with model")
        # axs[i, 3].legend()

        axs[i, 2].plot(wvn, gt_signals[idx], label="Test True Signal", color='orange')
        axs[i, 2].set_title(f"Test True Signal")

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "result_new_noise_model.jpg"))

    fig, axs = plt.subplots(num_samples, 1, figsize=(5, num_samples * 3))
    for i, idx in enumerate(selected_indices):
        axs[i].plot(wvn, gt_signals[idx] - cleaned_signals[idx], label="Residual_model", color='black')
        # axs[i].set_title(f"Smaple {idx}")
        axs[i].legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "residual_new_noise_model.jpg"))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dir = "data/generated/pV_new_noise_model_test_05132025_181123.pkl"
    test_noise_dir = "data/noise/std"
    fluo_test_dir = "data/generated/poly_new_noise_model_test_fluorescence_05182025_185808.pkl"
    # test_dir_skin = "data/generated/generated_skin_spectrum_12302024_163012.pkl"
    # test_noise_dir = "data/noise/processed_new/val_data.pkl"
    
    SNR_range = [0.01, 10]
    r2f_range = [0.01, 0.5]
    # SNR_range = [np.log10(a) for a in SNR_range]

    save_data_flag = False
    save_path = "./tmp/"

    # Load the best trained model
    model_full_path = "models/pretrained/new_noise_model_05182025_195707_full_with_fluo_in_signal.pth"
    model_full = AUnet(1, 1)
    model_full.load_state_dict(torch.load(model_full_path))
    model_full.eval()  # Set the model to evaluation mode

    # Load test data
    test_signal, _ = read_clean_data(clean_dir=test_dir, customized_noise=False, pV=True)
    fluo_test_signal, _ = read_clean_data(clean_dir=fluo_test_dir, customized_noise=False, pV=True)
    
    noise_std_dict = {}
    txt_files = glob.glob(os.path.join(test_noise_dir, "*.txt"))
    for txt_file in txt_files:
        std = read_noise_data(txt_file)
        key = os.path.splitext(os.path.basename(txt_file))[0]
        noise_std_dict[key] = std

    # Create Dataset and DataLoader
    test_dataset = RamanNoiseDataset(clean_signals=test_signal, noise_std_list=noise_std_dict, fluorescence=fluo_test_signal)
    test_dataset.generate_noisy_signals(SNR_range=SNR_range, r2f_range=r2f_range)
    test_dataset.DCT()  # Apply DCT on the test data
    test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Test the model and get predicted noises and cleaned signals
    # predicted_noises, cleaned_signals, noisy_signals, moving_window_cleaned, noise_avg_signals, true_noises, gt_signals, SNR_list = test_on_one_signal(model_HF, model_MF, model_LF, test_dataloader, device)
    # predicted_noises, cleaned_signals, noisy_signals, true_noises, gt_signals, SNR_list, int_time_list = test_model(model_HF, model_MF, model_LF, test_dataloader, device)
    predicted_noises, cleaned_signals, noisy_signals, true_noises, gt_signals, SNR_list, int_time_list, r2f_list = test_model_full(model_full, test_dataloader, device)

    # save
    if save_data_flag:
        save_data = {"noisy_signals": noisy_signals, "cleaned_signals": cleaned_signals, "gt_signals": gt_signals, "SNR_list": SNR_list}
        save_results = os.path.join(save_path, 'results_for_skin_train.pkl')
        with open(save_results, 'wb') as f:
            pickle.dump(save_data, f)

    plot_signals(noisy_signals, cleaned_signals, gt_signals, SNR_list, int_time_list, r2f_list, num_samples=5)

    print("Testing complete. Results saved.")

