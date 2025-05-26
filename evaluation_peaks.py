import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix
from train_nl import *
from test import test_model_full
from models.AUnet import fit_modpoly_range
import torch
import os
import matplotlib.pyplot as plt

def get_peaks(singal, prominence):
    """
    Get the peaks of signals using scipy's find_peaks function.

    Parameters:
    signal (np.ndarray): The input signals as a 2D numpy array of shape (n, L).
    threshold (float): The threshold above which to consider a peak (as minimum height).

    Returns:
    list: A list of tuples for each sample, each containing (peak_values, peak_indices).
    """

    peaks_info = []
    for sig in singal:
        peak_indices, properties = find_peaks(
                                        sig, 
                                        height=0.01,
                                        width=(10, 200),        # e.g. only peaks wider than 5 and narrower than 50 samples
                                        prominence=prominence,       # only peaks that stand out by at least 0.1 intensity units
                                        distance=5,          # at least 10 samples between peaks
                                        )
        peak_values = sig[peak_indices]
        peaks_info.append((peak_values, peak_indices))
    return peaks_info

def compare_peaks(peaks_info_all, peaks_info_ref):
    """
    Compare the peaks of multiple models with the reference peaks.

    Parameters:
    peaks_info_all (list of list): Each element is peaks_info from a model (list of (peak_values, peak_indices)).
    peaks_info_ref (list): Reference peaks_info (list of (peak_values, peak_indices)).

    Returns:
    list: For each model, a list of dicts for each sample with keys:
        - 'missing_peaks': count of reference peaks not found in model
        - 'artifact_peaks': count of model peaks not in reference
        - 'peak_value_bias': mean absolute difference of matched peak values
        - 'peak_shifts': mean absolute difference of matched peak indices
    """

    results = []
    for peaks_info_model in peaks_info_all:
        model_results = []
        for (model_vals, model_idx), (ref_vals, ref_idx) in zip(peaks_info_model, peaks_info_ref):
            # Match peaks by closest index (within a tolerance)
            tolerance = 5  # You can adjust this value
            matched_model = []
            matched_ref = []
            unmatched_model = set(range(len(model_idx)))
            unmatched_ref = set(range(len(ref_idx)))

            for i, r_idx in enumerate(ref_idx):
                diffs = np.abs(model_idx - r_idx)
                if len(diffs) == 0 or np.min(diffs) > tolerance:
                    continue
                m = np.argmin(diffs)
                if m in unmatched_model:
                    matched_model.append(m)
                    matched_ref.append(i)
                    unmatched_model.remove(m)
                    unmatched_ref.remove(i)

            missing_peaks = len(unmatched_ref)
            artifact_peaks = len(unmatched_model)
            if matched_model:
                peak_value_bias = np.mean(np.abs(model_vals[matched_model] - ref_vals[matched_ref]))
                peak_shifts = np.mean(np.abs(model_idx[matched_model] - ref_idx[matched_ref]))
            else:
                peak_value_bias = np.nan
                peak_shifts = np.nan

            model_results.append({
                'missing_peaks': missing_peaks,
                'artifact_peaks': artifact_peaks,
                'peak_value_bias': peak_value_bias,
                'peak_shifts': peak_shifts,
                'ref_peaks_count': len(ref_idx)
            })
        results.append(model_results)
    return results

def evaluate_peaks(model_results):
    """
    Summarize the peak comparison results.

    Parameters:
    model_results (list): Output from compare_peaks (list of list of dicts).

    Returns:
    list: For each model, a dict with:
        - 'missing_peak_ratio': ratio of missing peaks to reference peaks
        - 'artifact_peak_ratio': ratio of artifact peaks to reference peaks
        - 'peak_value_bias_mean': mean of peak value bias
        - 'peak_value_bias_std': std of peak value bias
        - 'peak_shifts_mean': mean of peak shifts
        - 'peak_shifts_std': std of peak shifts
    """
    summary = []
    for results in model_results:
        missing = [r['missing_peaks'] for r in results]
        artifact = [r['artifact_peaks'] for r in results]
        ref_peaks_count = [r['ref_peaks_count'] for r in results]

        total_peaks_sum = np.sum(ref_peaks_count)
        missing_sum = np.sum(missing)
        artifact_sum = np.sum(artifact)

        bias = [r['peak_value_bias'] for r in results if not np.isnan(r['peak_value_bias'])]
        shifts = [r['peak_shifts'] for r in results if not np.isnan(r['peak_shifts'])]

        summary.append({
            'missing_peak_ratio': missing_sum / total_peaks_sum if total_peaks_sum > 0 else np.nan,
            'artifact_peak_ratio': artifact_sum / total_peaks_sum if total_peaks_sum > 0 else np.nan,
            'peak_value_bias_mean': np.mean(bias) if bias else np.nan,
            'peak_value_bias_std': np.std(bias) if bias else np.nan,
            'peak_shifts_mean': np.mean(shifts) if shifts else np.nan,
            'peak_shifts_std': np.std(shifts) if shifts else np.nan
        })
    return summary

class PolyFit():
    # Placeholder for the PolyFit model
    def apply(self, signal):
        signal_torch = torch.from_numpy(signal).float().unsqueeze(1)
        best_poly, _ = fit_modpoly_range(signal_torch, x = None, poly_range=(3, 6), threshold=0.05, max_iter=25)
        return signal - best_poly.squeeze(1).cpu().numpy()

def use_compare_models(denoised_signals, models):
    """
    Compare the denoised signals with the models.

    Parameters:
    denoised_signals (np.ndarray): The denoised signals.
    models (list): List of model instances.

    Returns:
    list: List of denoised signals from each model.
    """
    compare_models_output = []
    for model in models:
        output = model.apply(denoised_signals)
        output = output / np.max(output, axis=1, keepdims=True)
        compare_models_output.append(output)
    return compare_models_output

def plot(peaks_all, peaks_ref, signals_all, signals_ref, num_samples=5, save_dir="tmp/peaks_vis"):
    """
    Plot signals and their detected peaks for each sample (row) and model (column).

    Parameters:
    peaks_all (list): List of peaks_info for each model (including DL model as first).
    peaks_ref (list): Reference peaks_info.
    signals_all (list): List of np.ndarray signals for each model (same order as peaks_all).
    signals_ref (np.ndarray): Reference signals (ground truth).
    num_samples (int): Number of samples to plot.
    save_dir (str): Directory to save plots.
    """

    os.makedirs(save_dir, exist_ok=True)
    num_models = len(peaks_all)
    col_num = num_models + 1  # +1 for reference
    row_num = min(num_samples, len(signals_ref))
    model_names = ["DL Model"] + [f"Model {i+1}" for i in range(1, num_models)]

    fig, axes = plt.subplots(row_num, col_num, figsize=(4 * col_num, 3 * row_num), squeeze=False)
    for n in range(row_num):
        # Reference column (m=0)
        ax = axes[n, 0]
        ax.plot(signals_ref[n], color="black", linewidth=2, label="Reference Signal")
        ref_peaks = peaks_ref[n][1]
        ax.plot(ref_peaks, signals_ref[n][ref_peaks], "ko", label="Reference Peaks", markersize=7, fillstyle='none')
        ax.set_title(f"Sample {n} - Reference")
        ax.set_xlabel("Index")
        ax.set_ylabel("Intensity")
        ax.legend()
        # Model columns (m=1,...)
        for m in range(num_models):
            ax = axes[n, m + 1]
            model_signal = signals_all[m][n]
            peaks_idx = peaks_all[m][n][1]
            ax.plot(model_signal, label=f"{model_names[m]} Signal", alpha=0.7)
            ax.plot(peaks_idx, model_signal[peaks_idx], "x", label=f"{model_names[m]} Peaks")
            ax.set_title(f"Sample {n} - {model_names[m]}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Intensity")
            ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"samples_subplot_peaks.png"))
    plt.close()


def main():
    # ---------------- the models ---------------
    dl_model = TwoStageModel()
    # Load the model
    dl_model_path = "models/pretrained/new_noise_model_05202025_211748_end_to_end_wvn_domain.pth"
    dl_model.load_state_dict(torch.load(dl_model_path))
    
    regular_model = PolyFit()
    
    models = [regular_model]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    # ---------------- the data ---------------
    # Load the input data
    test_dir = "data/generated/pV_new_noise_model_test_05132025_181123.pkl"
    test_noise_dir = "data/noise/std"
    fluo_test_dir = "data/generated/poly_new_noise_model_test_fluorescence_05182025_185808.pkl"
    
    SNR_range = [0.01, 10]
    r2f_range = [0.05, 0.5]
    
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
    
    output = test_model_full(dl_model, test_dataloader, device)
    
    denoised_singals = output['cleaned_signals']
    raman_dl = output['raman_signals']
    gt_raman = output['gt_raman']
    
    raman_dl = raman_dl / np.max(raman_dl, axis=1, keepdims=True)
    gt_raman = gt_raman / np.max(gt_raman, axis=1, keepdims=True)

    compare_models_output = use_compare_models(denoised_singals, models)
    
    summaries = []
    prominences = np.linspace(0.01, 0.5, 10)
    for prominence in prominences:
        peaks_raman = get_peaks(raman_dl, prominence)
        peaks_compare = []
        for compare in compare_models_output:
            peaks_compare_tmp = get_peaks(compare, prominence)
            peaks_compare.append(peaks_compare_tmp)
        peaks_all = [peaks_raman] + peaks_compare
        peaks_ref = get_peaks(gt_raman, prominence)
        results = compare_peaks(peaks_all, peaks_ref)
        summary = evaluate_peaks(results)
        summaries.append(summary)

    # Prepare data for plotting (one color per model)
    num_models = len(summaries[0])
    model_names = ["DL Model"] + ["PolyFit"]

    # Collect metrics for each model across all prominences
    missing_peak_ratios = [[] for _ in range(num_models)]
    artifact_peak_ratios = [[] for _ in range(num_models)]
    peak_value_bias_means = [[] for _ in range(num_models)]
    peak_value_bias_stds = [[] for _ in range(num_models)]
    peak_shift_means = [[] for _ in range(num_models)]
    peak_shift_stds = [[] for _ in range(num_models)]

    for summary_sub in summaries:
        for i in range(num_models):
            missing_peak_ratios[i].append(summary_sub[i]['missing_peak_ratio'])
            artifact_peak_ratios[i].append(summary_sub[i]['artifact_peak_ratio'])
            peak_value_bias_means[i].append(summary_sub[i]['peak_value_bias_mean'])
            peak_value_bias_stds[i].append(summary_sub[i]['peak_value_bias_std'])
            peak_shift_means[i].append(summary_sub[i]['peak_shifts_mean'])
            peak_shift_stds[i].append(summary_sub[i]['peak_shifts_std'])

    colors = plt.cm.tab10.colors

    # Create a single figure with 2 subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    label_fontsize = 18
    tick_fontsize = 15
    title_fontsize = 20
    legend_fontsize = 14

    # Plot 1: Missing and Artifact Peak Ratios
    ax = axes[0]
    for i in range(num_models):
        ax.plot(prominences, missing_peak_ratios[i], marker='o', color=colors[i], label=f"{model_names[i]} - Missing Peak Ratio")
        ax.plot(prominences, artifact_peak_ratios[i], marker='s', color=colors[i], linestyle='--', label=f"{model_names[i]} - Artifact Peak Ratio")
    ax.set_xlabel('Prominence', fontsize=label_fontsize)
    ax.set_ylabel('Ratio', fontsize=label_fontsize)
    ax.set_title('Missing & Artifact Peak Ratios vs Prominence', fontsize=title_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Plot 2: Peak Value Bias and Peak Shift (mean ± std) with dual y-axis
    ax1 = axes[1]
    ax2 = ax1.twinx()

    # Convert shift counts to wavenumber (1 count = 1.717 wavenumber)
    peak_shift_means_wn = [[v * 1.717 if v is not np.nan else np.nan for v in vals] for vals in peak_shift_means]
    peak_shift_stds_wn = [[v * 1.717 if v is not np.nan else np.nan for v in vals] for vals in peak_shift_stds]

    for i in range(num_models-1, -1, -1):
        ax1.errorbar(prominences, peak_value_bias_means[i], yerr=peak_value_bias_stds[i], marker='^', color=colors[i], linestyle=':', label=f"{model_names[i]} - Peak Value Bias")
        ax2.errorbar(prominences, peak_shift_means_wn[i], yerr=peak_shift_stds_wn[i], marker='v', color=colors[i], linestyle='-', label=f"{model_names[i]} - Peak Shift")

    ax1.set_xlabel('Prominence', fontsize=label_fontsize)
    ax1.set_ylabel('Peak Value Bias (unitless)', fontsize=label_fontsize)
    ax2.set_ylabel('Peak Shift (wavenumber [$cm^{-1}$])', fontsize=label_fontsize)
    ax1.set_title('Peak Value Bias & Shift vs Prominence', fontsize=title_fontsize)

    # Adjust y-axis limits to not start from 0, but add some margin
    y1min, y1max = ax1.get_ylim()
    y2min, y2max = ax2.get_ylim()
    ax1.set_ylim(y1min, y1max + 1 * (y1max - y1min))
    ax2.set_ylim(y2min - 8 * (y1max - y1min), y2max + 0.1 * (y2max - y2min))

    # Set tick label sizes
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=legend_fontsize)

    plt.tight_layout()
    plt.savefig("tmp/peaks_vis/peak_metrics.png")
    plt.close()
    

    # Prepare signals_all for plotting: [raman_dl] + compare_models_output
    signals_all = [raman_dl] + compare_models_output
    plot(peaks_all, peaks_ref, signals_all, gt_raman, num_samples=5)
    
    return summary
    

if __name__ == "__main__":
    summary = main()
    print(summary[0])
    print(summary[1])