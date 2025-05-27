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
    raman_signals = []
    gt_raman = []
    
    with torch.no_grad():  # Disable gradient calculation for testing
        # Progress bar for testing phase
        for noisy_signal, noisy_signal_dct, _, gt_signal_dct, gt_raman_dct, gt_flu_dct, SNR, int_time, r2f in tqdm(test_dataloader, desc="Testing", unit="batch"):

            noisy_signal = noisy_signal.unsqueeze(1).float().to(device)
            noisy_signal_dct = noisy_signal_dct.unsqueeze(1).float().to(device)
            gt_signal_dct = gt_signal_dct.unsqueeze(1).float().to(device)
            # get the max of the noisy_signal
            max_values = noisy_signal.max(dim=2, keepdim=True)[0]
            noisy_signal_dct_norm = noisy_signal_dct / max_values
            gt_signal_dct_norm = gt_signal_dct / max_values

            out = model_full(noisy_signal_dct_norm)
            
            denoised_signal = out["denoised_signal"].squeeze(1).cpu().numpy() * max_values.squeeze(1).cpu().numpy()
            raman_signal = out["raman_signal"].squeeze(1).cpu().numpy() * max_values.squeeze(1).cpu().numpy()
            gt_signal = idct(gt_signal_dct.squeeze(1).cpu().numpy(), type=2, norm='ortho', axis=1)
            predicted_noise = noisy_signal.squeeze(1).cpu().numpy() - denoised_signal
            true_noise = noisy_signal.squeeze(1).cpu().numpy() - gt_signal
            
            # predicted_noise_dct = out.squeeze(1).cpu().numpy() * max_values.squeeze(1).cpu().numpy()
            # predicted_noise = idct(predicted_noise_dct, type=2, norm='ortho', axis=1)
            # denoised_signal = noisy_signal.squeeze(1).cpu().numpy() - predicted_noise
            # gt_signal = idct(gt_signal_dct.squeeze(1).cpu().numpy(), type=2, norm='ortho', axis=1)
            # true_noise = noisy_signal.squeeze(1).cpu().numpy() - gt_signal
            
            
            # Store the results
            predicted_noises.append(predicted_noise)
            true_noises.append(true_noise)
            noisy_signals.append(noisy_signal.squeeze(1).cpu().numpy())
            cleaned_signals.append(denoised_signal)
            gt_signals.append(gt_signal)
            raman_signals.append(raman_signal)
            gt_raman.append(idct(gt_raman_dct.squeeze(1).cpu().numpy(), type=2, norm='ortho', axis=1))
            
            SNR_list.append(SNR)
            int_time_list.append(int_time)
            r2f_list.append(r2f)

        # Concatenate results into numpy arrays
        predicted_noises = np.concatenate(predicted_noises, axis=0)
        cleaned_signals = np.concatenate(cleaned_signals, axis=0)
        noisy_signals = np.concatenate(noisy_signals, axis=0)
        true_noises = np.concatenate(true_noises, axis=0)
        gt_signals = np.concatenate(gt_signals, axis=0)
        raman_signals = np.concatenate(raman_signals, axis=0)
        gt_raman = np.concatenate(gt_raman, axis=0)
        
        SNR_list = np.concatenate(SNR_list, axis=0)
        int_times = np.concatenate(int_time_list, axis=0)
        r2fs = np.concatenate(r2f_list, axis=0)
        
        out = {
            "predicted_noises": predicted_noises,
            "cleaned_signals": cleaned_signals,
            "noisy_signals": noisy_signals,
            "true_noises": true_noises,
            "gt_signals": gt_signals,
            "raman_signals": raman_signals,
            "gt_raman": gt_raman,
            "SNR_list": SNR_list,
            "int_times": int_times,
            "r2fs": r2fs
        }
    
    return out


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

def plot_signals(output, num_samples=5, save_path="./tmp/"):
    
    SNR_list = output["SNR_list"]
    noisy_signals = output["noisy_signals"]
    cleaned_signals = output["cleaned_signals"]
    gt_signals = output["gt_signals"]
    int_time_list = output["int_times"]
    r2f_list = output["r2fs"]
    raman_signals = output["raman_signals"]
    gt_raman = output["gt_raman"]
    
    # Sort indices based on SNR_list in ascending order
    sorted_indices = np.argsort(SNR_list)
    SNR_list = np.array(SNR_list)[sorted_indices]
    noisy_signals = np.array(noisy_signals)[sorted_indices]
    cleaned_signals = np.array(cleaned_signals)[sorted_indices]
    gt_signals = np.array(gt_signals)[sorted_indices]
    int_time_list = np.array(int_time_list)[sorted_indices]
    raman_signals = np.array(raman_signals)[sorted_indices]
    gt_raman = np.array(gt_raman)[sorted_indices]
    r2f_list = np.array(r2f_list)[sorted_indices]
    
    # Get the total number of signals
    total_signals = len(noisy_signals)
    
    # If the requested number of samples exceeds the total signals, limit it to the available number
    num_samples = min(num_samples, total_signals)
    
    selected_indices = []
    for i in range(num_samples):
        selected_indices.append(i*total_signals//num_samples + total_signals//num_samples//10)
        
    # Select random samples
    # selected_indices = sorted(random.sample(range(len(SNR_list)), num_samples))
    
    fig, axs = plt.subplots(num_samples, 3, figsize=(30, num_samples * 3))
    
    wvn = loadmat("data/wvn_raw.mat")["wvn"].reshape(-1)
    wvn = wvn[331:]
    
    for i, idx in enumerate(selected_indices):
        axs[i, 0].plot(wvn, noisy_signals[idx], label="Noisy Signal")
        int_time_str = str(int_time_list[idx])
        if int_time_str.startswith("std_"):
            int_time_str = int_time_str[4:]
        axs[i, 0].set_title(f"Noisy Signal, SNR = {SNR_list[idx]:.2f}, r2f = {r2f_list[idx]:.2f}, int_time = {int_time_str}")
        # axs[i, 0].legend()

        axs[i, 1].plot(wvn, gt_signals[idx], label="True Signal", color='orange', linewidth=3)
        axs[i, 1].plot(wvn, cleaned_signals[idx], label="Cleaned Signal with model", color='green', linewidth=3)
        axs[i, 1].set_title(f"Denoised")
        axs[i, 1].legend(fontsize=18)
        
        axs[i, 2].plot(wvn, gt_raman[idx], label="True Raman", color='orange', linewidth=3)
        axs[i, 2].plot(wvn, raman_signals[idx], label="Pred Raman", color='green', linewidth=3)
        axs[i, 2].set_title(f"Raman")
        axs[i, 2].legend(fontsize=18)
    
    for ax_row in axs:
        for ax in (ax_row if isinstance(ax_row, (list, np.ndarray)) else [ax_row]):
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.set_xlabel(ax.get_xlabel(), fontsize=20)
            ax.set_ylabel(ax.get_ylabel(), fontsize=20)
            ax.title.set_fontsize(22)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "result_new_noise_model.jpg"))

    # fig, axs = plt.subplots(num_samples, 1, figsize=(5, num_samples * 3))
    # for i, idx in enumerate(selected_indices):
    #     axs[i].plot(wvn, gt_signals[idx] - cleaned_signals[idx], label="Residual_model", color='black')
    #     # axs[i].set_title(f"Smaple {idx}")
    #     axs[i].legend()

    # plt.tight_layout()
    # plt.show()
    # plt.savefig(os.path.join(save_path, "residual_new_noise_model.jpg"))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test_dir = "data/generated/pV_new_noise_model_test_05132025_181123.pkl"
    test_dir = "data/generated/generated_skin_spectrum_05262025_131135.pkl"
    test_noise_dir = "data/noise/std"
    fluo_test_dir = "data/generated/poly_new_noise_model_test_fluorescence_05182025_185808.pkl"
    
    SNR_range = [0.01, 10]
    r2f_range = [0.05, 0.5]
    # SNR_range = [np.log10(a) for a in SNR_range]

    save_data_flag = False
    save_path = "./tmp/"

    # Load the best trained model
    model_full_path = "models/pretrained/new_noise_model_05202025_211748_end_to_end_wvn_domain.pth"
    # model_full_path = "models/pretrained/new_noise_model_05202025_151616_end_to_end.pth"
    model_full = TwoStageModel()
    model_full.load_state_dict(torch.load(model_full_path))
    model_full.eval()  # Set the model to evaluation mode

    # # Optionally load another weight into model_full.denoiser
    # denoiser_weight_path = "models/pretrained/denoiser_only_weights.pth"
    # if os.path.exists(denoiser_weight_path):
    #     model_full.denoiser.load_state_dict(torch.load(denoiser_weight_path))

    # Load test data
    # test_signal, _ = read_clean_data(clean_dir=test_dir, customized_noise=False, pV=True)
    test_signal, _, _ = read_clean_data(clean_dir=test_dir, customized_noise=False, pV=False)
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
    output = test_model_full(model_full, test_dataloader, device)

    # # save
    # if save_data_flag:
    #     save_data = {"noisy_signals": noisy_signals, "cleaned_signals": cleaned_signals, "gt_signals": gt_signals, "SNR_list": SNR_list}
    #     save_results = os.path.join(save_path, 'results_for_skin_train.pkl')
    #     with open(save_results, 'wb') as f:
    #         pickle.dump(save_data, f)

    plot_signals(output, num_samples=5)

    print("Testing complete. Results saved.")

