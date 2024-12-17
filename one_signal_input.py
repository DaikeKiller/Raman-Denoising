import numpy as np
import torch
from scipy.fftpack import dct, idct
from test import soft_low_pass_filter_dct
from models.Network import RamanNoiseNet, RamanNoiseNet_HF, RamanNoiseNet_LF
from models.AUnet import AUnet
import matplotlib.pyplot as plt


def use_model_dc_input(model_HF, model_MF, model_LF, noise_dir, device, dc_val):
    model_HF.to(device)
    model_HF.eval()  # Set the model to evaluation mode
    model_MF.to(device)
    model_MF.eval()  # Set the model to evaluation mode
    model_LF.to(device)
    model_LF.eval()
    
    signal = np.zeros((1, 1, 1981)) + dc_val
    
    with open(noise_dir, 'r') as f:
        data = f.read().strip().split()  # Adjust based on file format
        data = [float(x) for x in data]  # Convert to float (or int, based on data)
        # data = data[421:]
    power = np.std(data)
    bias = np.mean(data)
    if power > 0:  # Avoid division by zero
        data = (data - bias) / power
    
    noise = np.array(data).reshape(1, 1, -1)
    
    noisy_signal_wvn_domain = signal + noise
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
        predicted_noise_HF_residual = model_HF(noisy_signal_HF).squeeze(1).cpu().numpy()
        predicted_noise_HF_residual = predicted_noise_HF_residual + HF_center_factor.squeeze(1).cpu().numpy()
        predicted_noise_MF_residual = model_MF(noisy_signal_MF).squeeze(1).cpu().numpy()
        predicted_noise_MF_residual = predicted_noise_MF_residual + MF_center_factor.squeeze(1).cpu().numpy()
        predicted_noise_LF = np.zeros((1, 81))
        predicted_noise_LF[:, 1:] = model_LF(noisy_signal_LF).squeeze(1).cpu().numpy()
        predicted_noise_LF_residual = noisy_signal_LF_includ_dc.squeeze(1).cpu().numpy() - predicted_noise_LF
    
    predicted_noise_residual = np.concatenate((predicted_noise_LF_residual[:,:81-5], predicted_noise_MF_residual[:,45:-25], predicted_noise_HF_residual[:,25:]), axis=1)
    predicted_noise_residual = soft_low_pass_filter_dct(predicted_noise_residual, 1031, 50)

    noisy_signal_np = noisy_signal_np.squeeze(1)
    
    predicted_noise = noisy_signal_np - predicted_noise_residual
    
    # for test ================
    # noise_dct = dct(noise.reshape(1,-1), norm="ortho")
    # predicted_noise[:,:1981] = noise_dct[:,:1981]
    
    idct_predicted_noise = idct(predicted_noise, type=2, norm='ortho', axis=1)
            
    # Subtract predicted noise from noisy signal to clean the signal
    idct_noisy_signal_np = idct(noisy_signal_np, type=2, norm='ortho', axis=1)
    cleaned_signal = idct_noisy_signal_np - idct_predicted_noise
    # remove_num = 3 # to deal with zero-point spike
    # cleaned_signal[:,:remove_num] = cleaned_signal[:,remove_num:remove_num*2]
    
    return cleaned_signal, idct_noisy_signal_np, signal.squeeze(1)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_HF_path = "models/pretrained/model_12162024_163819_HF.pth"
    model_HF = RamanNoiseNet_HF()
    model_HF.load_state_dict(torch.load(model_HF_path))
    model_HF.eval()  # Set the model to evaluation mode
    model_MF_path = "models/pretrained/model_12162024_163819_MF.pth"
    model_MF = RamanNoiseNet_HF()
    model_MF.load_state_dict(torch.load(model_MF_path))
    model_HF.eval()  # Set the model to evaluation mode
    model_LF_path = "models/pretrained/model_12162024_163819_LF.pth"
    model_LF = AUnet(1, 1)
    # model_LF = RamanNoiseNet_LF()
    model_LF.load_state_dict(torch.load(model_LF_path))
    model_LF.eval()  # Set the model to evaluation mode
    
    noise_dir = "data/noise/processed_new/dark_1s_101.txt"
    
    dc_val = 0
    cleaned_signal, noisy_signal_np, signal = use_model_dc_input(model_HF, model_MF, model_LF, noise_dir, device, dc_val=dc_val)
    
    # plot
    plt.figure(figsize=(15, 5))
    wvn = np.linspace(800, 1790, 1981)
    plt.subplot(1,2,1)
    plt.plot(wvn, noisy_signal_np[0])
    plt.plot(wvn, cleaned_signal[0], linewidth=3)
    plt.title("noisy_signal")
    plt.legend(["noisy_signal", "cleaned_signal"])
    # plt.subplot(1,2,2)
    # plt.plot(wvn, cleaned_signal[0])
    # plt.title("cleaned_signal")
    plt.subplot(1,2,2)
    plt.plot(wvn, signal[0])
    plt.title("pure signal")
    plt.savefig("results/only_noise_input.jpg")