import numpy as np
import torch
from scipy.fftpack import dct, idct
from test import soft_low_pass_filter_dct
from models.Network import RamanNoiseNet, RamanNoiseNet_HF, RamanNoiseNet_LF
from models.AUnet import AUnet
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pywt
from sklearn.metrics import mean_squared_error
import os
import pickle


def read_noise_data(noise_file):
    with open(noise_file, 'rb') as f:
        data_list = pickle.load(f)
    return np.array(data_list)

def read_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    noisy_signals, gt_signals = data["noisy_signals"], data["gt_signals"]
    return noisy_signals, gt_signals

def use_model_dc_input(model_HF, model_MF, model_LF, signal, device, dct_flag=False):
    model_HF.to(device)
    model_HF.eval()  # Set the model to evaluation mode
    model_MF.to(device)
    model_MF.eval()  # Set the model to evaluation mode
    model_LF.to(device)
    model_LF.eval()
    
    noisy_signal_wvn_domain = np.expand_dims(signal, axis=1)
    noisy_signal_np = dct(noisy_signal_wvn_domain, norm="ortho")
    noisy_signal = torch.from_numpy(noisy_signal_np)
    noisy_signal = noisy_signal.float().to(device)
    
    # pre_processing the signals
    noisy_signal_HF = noisy_signal[:, :, 81+950-50:]
    noisy_signal_MF = noisy_signal[:, :, 81-50:81+950]
    noisy_signal_LF = noisy_signal[:, :, 1:81]
    noisy_signal_LF_includ_dc = noisy_signal[:, :, :81]

    HF_center_factor = torch.mean(noisy_signal_HF, dim=2, keepdim=True)
    MF_center_factor = torch.mean(noisy_signal_MF, dim=2, keepdim=True)
    LF_center_factor = torch.mean(noisy_signal_LF, dim=2, keepdim=True)
    noisy_signal_HF = noisy_signal_HF - HF_center_factor
    noisy_signal_MF = noisy_signal_MF - MF_center_factor
    noisy_signal_LF = noisy_signal_LF - LF_center_factor
    
    with torch.no_grad():
        # run the model and pipeline
        predicted_noise_HF = model_HF(noisy_signal_HF).squeeze(1).cpu().numpy()
        predicted_noise_HF = predicted_noise_HF + HF_center_factor.squeeze(1).cpu().numpy()
        predicted_noise_MF = model_MF(noisy_signal_MF).squeeze(1).cpu().numpy()
        predicted_noise_MF = predicted_noise_MF + MF_center_factor.squeeze(1).cpu().numpy()
        predicted_noise_LF = np.zeros((noisy_signal_LF.shape[0], noisy_signal_LF.shape[-1] + 1))
        predicted_noise_LF[:, 1:] = model_LF(noisy_signal_LF).squeeze(1).cpu().numpy()
        # predicted_noise_LF_residual = noisy_signal_LF_includ_dc.squeeze(1).cpu().numpy() - predicted_noise_LF
    
    predicted_noise = np.concatenate((predicted_noise_LF[:,:81-5], predicted_noise_MF[:,45:-25], predicted_noise_HF[:,25:]), axis=1)
    # predicted_noise_residual = soft_low_pass_filter_dct(predicted_noise_residual, 1031, 50)

    noisy_signal_np = noisy_signal_np.squeeze(1)
    
    # predicted_noise = noisy_signal_np - predicted_noise_residual
    
    # for test ================
    # noise_dct = dct(noise.reshape(1,-1), norm="ortho")
    # predicted_noise[:,:1981] = noise_dct[:,:1981]
    
    idct_predicted_noise = idct(predicted_noise, type=2, norm='ortho', axis=1)
            
    # Subtract predicted noise from noisy signal to clean the signal
    idct_noisy_signal_np = idct(noisy_signal_np, type=2, norm='ortho', axis=1)
    cleaned_signal = idct_noisy_signal_np - idct_predicted_noise
    # remove_num = 3 # to deal with zero-point spike
    # cleaned_signal[:,:remove_num] = cleaned_signal[:,remove_num:remove_num*2]
    
    if dct_flag:
        cleaned_signal = dct(cleaned_signal, norm="ortho", axis=1)
    
    return cleaned_signal

def apply_filters_with_tuning(train_signals, train_ground_truth, signals, dct_flag=False):
    """
    Apply Savitzky-Golay (SG) and Wavelet denoising filters on signals after automatic tuning.
    
    Parameters:
        signals (np.ndarray): Noisy signals, shape (n, 1981), where n is the number of signals.
        ground_truth (np.ndarray, optional): Clean signals for evaluation, shape (n, 1981).
                                            If None, no objective tuning will be performed.

    Returns:
        dict: A dictionary with denoised signals and best parameters for SG and wavelet filters.
    """
    results = {}

    # Helper function for SG filter tuning
    def tune_sg_filter(signals, ground_truth):
        best_params = None
        best_score = float('inf')
        
        for window_length in range(11, 201, 2):  # Must be odd
            for polyorder in range(1, min(5, window_length)):
                try:
                    denoised_signals = np.apply_along_axis(
                        lambda spectrum: savgol_filter(spectrum, window_length=window_length, polyorder=polyorder),
                        axis=1,
                        arr=signals
                    )
                    if ground_truth is not None:
                        score = mean_squared_error(ground_truth, denoised_signals)
                        if score < best_score:
                            best_score = score
                            best_params = (window_length, polyorder)
                except ValueError:
                    continue
        return best_params

    # Helper function for Wavelet filter tuning
    def tune_wavelet_filter(signals, ground_truth):
        best_params = None
        best_score = float('inf')
        
        for wavelet in ['db4', 'sym5', 'coif1']:
            for level in range(1, 5):
                for thresholding in ['soft', 'hard']:
                    denoised_signals = []
                    for spectrum in signals:
                        try:
                            coeffs = pywt.wavedec(spectrum, wavelet, level=level)
                            sigma = np.median(np.abs(coeffs[-1])) / 0.01
                            threshold = sigma * np.sqrt(2 * np.log(len(spectrum)))
                            denoised_coeffs = [
                                pywt.threshold(c, value=threshold, mode=thresholding) if i > 0 else c
                                for i, c in enumerate(coeffs)
                            ]
                            denoised_spectrum = pywt.waverec(denoised_coeffs, wavelet)
                            denoised_signals.append(denoised_spectrum[:len(spectrum)])
                        except Exception:
                            denoised_signals.append(spectrum)
                    
                    denoised_signals = np.array(denoised_signals)
                    if ground_truth is not None:
                        score = mean_squared_error(ground_truth, denoised_signals)
                        if score < best_score:
                            best_score = score
                            best_params = (wavelet, level, thresholding)
        return best_params

    # Perform SG filter tuning
    sg_params = tune_sg_filter(train_signals, train_ground_truth)
    if sg_params is not None:
        window_length, polyorder = sg_params
        sg_filtered_signals = np.apply_along_axis(
            lambda spectrum: savgol_filter(spectrum, window_length=window_length, polyorder=polyorder),
            axis=1,
            arr=signals
        )
        
        if dct_flag:
            sg_filtered_signals = dct(sg_filtered_signals, norm="ortho", axis=1)
            
        results["SG_filtered_signals"] = sg_filtered_signals
        results["SG_best_params"] = sg_params
        with open("results/best_SG_params.pkl", "wb") as f:
            pickle.dump(sg_params, f)

    # Perform Wavelet filter tuning
    wavelet_params = tune_wavelet_filter(train_signals, train_ground_truth)
    if wavelet_params is not None:
        wavelet, level, thresholding = wavelet_params
        wavelet_filtered_signals = []
        for spectrum in signals:
            coeffs = pywt.wavedec(spectrum, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.01
            threshold = sigma * np.sqrt(2 * np.log(len(spectrum)))
            denoised_coeffs = [
                pywt.threshold(c, value=threshold, mode=thresholding) if i > 0 else c
                for i, c in enumerate(coeffs)
            ]
            denoised_spectrum = pywt.waverec(denoised_coeffs, wavelet)
            wavelet_filtered_signals.append(denoised_spectrum[:len(spectrum)])
        wavelet_filtered_signals = np.array(wavelet_filtered_signals)
        
        if dct_flag:
            wavelet_filtered_signals = dct(wavelet_filtered_signals, norm="ortho", axis=1)
            
        results["Wavelet_filtered_signals"] = wavelet_filtered_signals
        results["Wavelet_best_params"] = wavelet_params
        with open("results/best_wv_params.pkl", "wb") as f:
            pickle.dump(wavelet_params, f)

    return results


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_HF_path = "models/pretrained/model_12172024_155938_HF.pth"
    model_HF = AUnet(1, 1)
    model_HF.load_state_dict(torch.load(model_HF_path))
    model_HF.eval()  # Set the model to evaluation mode
    model_MF_path = "models/pretrained/model_12172024_102817_MF.pth"
    model_MF = AUnet(1, 1)
    model_MF.load_state_dict(torch.load(model_MF_path))
    model_HF.eval()  # Set the model to evaluation mode
    model_LF_path = "models/pretrained/model_12172024_082602_LF.pth"
    model_LF = AUnet(1, 1)
    # model_LF = RamanNoiseNet_LF()
    model_LF.load_state_dict(torch.load(model_LF_path))
    model_LF.eval()  # Set the model to evaluation mode
    
    noise_dir = "data/noise/processed_new/test_data.pkl"
    
    noise = read_noise_data(noise_dir)
    signal = np.zeros(noise.shape)
    noisy_signals = signal + noise
    
    # load noisy signals for SG and wavelet training
    train_signals, trian_ground_truth = read_data("results/results_all_pV_test.pkl")
    
    dc_val = 0
    cleaned_signal_form_model = use_model_dc_input(model_HF, model_MF, model_LF, noisy_signals, device, dct_flag=True)
    cleaned_signal_form_others = apply_filters_with_tuning(train_signals, trian_ground_truth, noisy_signals, dct_flag=True)
    
    # plot
    plt.figure(figsize=(28, 10))
    wvn = np.linspace(0, 2, 1981)
    
    plt.subplot(1, 3, 1)
    plt.plot(wvn, cleaned_signal_form_others["Wavelet_filtered_signals"][0], alpha=0.9)
    plt.plot(wvn, cleaned_signal_form_others["SG_filtered_signals"][0], alpha=0.9)
    plt.plot(wvn, cleaned_signal_form_model[0], alpha=0.9)
    plt.legend(["wavelet", "SG", "DL"], fontsize=14)
    plt.xlabel('Frequency (cycles/cm$^{-1}$)', fontsize=16)
    plt.ylabel('Intensity', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Comparison of denoised signals", fontsize=16)
    
    plt.subplot(1, 3, 2)
    range_ = range(0, 200)
    plt.plot(wvn[range_], cleaned_signal_form_others["Wavelet_filtered_signals"][0, range_], alpha=0.6)
    plt.plot(wvn[range_], cleaned_signal_form_others["SG_filtered_signals"][0, range_], alpha=0.6)
    plt.plot(wvn[range_], cleaned_signal_form_model[0, range_], alpha=0.6)
    plt.legend(["wavelet", "SG", "DL"], fontsize=14)
    plt.xlabel('Frequency (cycles/cm$^{-1}$)', fontsize=16)
    plt.ylabel('Intensity', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Zoomed in for low frequency", fontsize=16)
    
    plt.subplot(1, 3, 3)
    range_ = range(1100, 1400)
    plt.plot(wvn[range_], cleaned_signal_form_others["Wavelet_filtered_signals"][0, range_], alpha=0.4)
    plt.plot(wvn[range_], cleaned_signal_form_others["SG_filtered_signals"][0, range_], alpha=0.9)
    plt.plot(wvn[range_], cleaned_signal_form_model[0, range_], alpha=0.9)
    plt.legend(["wavelet", "SG", "DL"], fontsize=14)
    plt.xlabel('Frequency (cycles/cm$^{-1}$)', fontsize=16)
    plt.ylabel('Intensity', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(-0.02, 0.02)
    plt.title("Zoomed in for high frequency", fontsize=16)
    
    plt.savefig("tmp/only_noise_input_pV.jpg")
    
    # now I will calculate the difference for the all the signals bwtween SG and DL
    diff = np.abs(cleaned_signal_form_others["SG_filtered_signals"]) - np.abs(cleaned_signal_form_model)
    
    # Then I will plot all the difference spectra together as gray and calculate the mean as black
    plt.figure(figsize=(28, 10))
    wvn = np.linspace(0, 2, 1981)
    
    plt.subplot(1,2,1)
    for i in range(len(diff)):
        if i == len(diff) - 1:
            plt.plot(wvn, diff[i], alpha=0.6, color="gray", label="Individual Differences")
        else:
            plt.plot(wvn, diff[i], alpha=0.6, color="gray")
    plt.plot(wvn, np.mean(diff, axis=0), color="black", label="Mean Difference")
    plt.xlabel('Frequency (cycles/cm$^{-1}$)', fontsize=16)
    plt.ylabel('Intensity', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Difference between SG and DL Denoised Signals", fontsize=16)
    plt.legend(fontsize=16)
    
    plt.subplot(1,2,2)
    range_ = range(1200, 1979)
    for i in range(len(diff)):
        if i == len(diff) - 1:
            plt.plot(wvn[range_], diff[i, range_], alpha=0.6, color="gray", label="Individual Differences")
        else:
            plt.plot(wvn[range_], diff[i, range_], alpha=0.6, color="gray")
    plt.plot(wvn[range_], np.mean(diff[:, range_], axis=0), color="black", label="Mean Difference")
    plt.xlabel('Frequency (cycles/cm$^{-1}$)', fontsize=16)
    plt.ylabel('Intensity', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Zoomed In", fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig("tmp/diff_SG_DL.jpg")
    
    # do the same thing for wavelet and DL
    diff = np.abs(cleaned_signal_form_others["Wavelet_filtered_signals"]) - np.abs(cleaned_signal_form_model)
    # plot
    plt.figure(figsize=(28, 10))
    wvn = np.linspace(0, 2, 1981)
    
    plt.subplot(1,2,1)
    for i in range(len(diff)):
        if i == len(diff) - 1:
            plt.plot(wvn, diff[i], alpha=0.6, color="gray", label="Individual Differences")
        else:
            plt.plot(wvn, diff[i], alpha=0.6, color="gray")
    plt.plot(wvn, np.mean(diff, axis=0), color="black", label="Mean Difference")
    plt.xlabel('Frequency (cycles/cm$^{-1}$)', fontsize=16)
    plt.ylabel('Intensity', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Difference between Wavelet and DL Denoised Signals", fontsize=16)
    plt.legend(fontsize=16)
    
    plt.subplot(1,2,2)
    range_ = range(1200, 1979)
    for i in range(len(diff)):
        if i == len(diff) - 1:
            plt.plot(wvn[range_], diff[i, range_], alpha=0.6, color="gray", label="Individual Differences")
        else:
            plt.plot(wvn[range_], diff[i, range_], alpha=0.6, color="gray")
    plt.plot(wvn[range_], np.mean(diff[:, range_], axis=0), color="black", label="Mean Difference")
    plt.xlabel('Frequency (cycles/cm$^{-1}$)', fontsize=16)
    plt.ylabel('Intensity', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Zoomed In", fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig("tmp/diff_wv_DL.jpg")