from utils.Raman_dataset import DenoiserTestDataset
from train_nl import *
from scipy.fftpack import idct, dct
import random
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
import cv2
from scipy.io import loadmat
from scipy.signal import savgol_filter
import torch
import numpy as np
import pywt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.patches import Patch
import matplotlib.pyplot as plt


def deep_learning(input_data, model):
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_data = input_data.float().to(device)
        out = model(input_data)
        denoised_signal = out["denoised_signal"].cpu()
        raman_signal = out["raman_signal"].cpu()
    return denoised_signal, raman_signal

class SG_model():
    def filter(self, input_data, target_data=None, window_length_range=(5, 21, 2), polyorder_range=(2, 4)):
        # input_data: torch tensor of shape (n, 1, L)
        # target_data: torch tensor of shape (n, 1, L), required for optimization

        input_np = input_data.cpu().numpy()
        n, c, L = input_np.shape
        best_output = None
        best_score = float('inf')
        best_params = (None, None)

        if target_data is None:
            # If no target, just use default params
            window_length = 11
            polyorder = 3
            output_np = []
            for i in range(n):
                filtered = savgol_filter(input_np[i, 0], window_length=window_length, polyorder=polyorder)
                output_np.append(filtered)
            output_np = np.stack(output_np, axis=0)  # shape (n, L)
            output_np = output_np[:, np.newaxis, :]  # shape (n, 1, L)
            return torch.from_numpy(output_np).type_as(input_data)

        target_np = target_data.cpu().numpy()
        for window_length in range(*window_length_range):
            if window_length % 2 == 0:
                continue  # window_length must be odd
            for polyorder in range(*polyorder_range):
                if polyorder >= window_length:
                    continue
                output_np = []
                for i in range(n):
                    filtered = savgol_filter(input_np[i, 0], window_length=window_length, polyorder=polyorder)
                    output_np.append(filtered)
                output_np = np.stack(output_np, axis=0)
                score = np.mean((output_np - target_np[:, 0, :]) ** 2)
                if score < best_score:
                    best_score = score
                    best_output = output_np
                    best_params = (window_length, polyorder)
        best_output = best_output[:, np.newaxis, :]
        return torch.from_numpy(best_output).type_as(input_data)

class WaveletModel():
    def wavelet_denoise(self, signal, wavelet, level, mode='soft'):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
        denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode=mode) for c in coeffs[1:]]
        return pywt.waverec(denoised_coeffs, wavelet)[:len(signal)]

    def filter(self, input_data, target_data=None, wavelet_list=['db4', 'sym5'], level_range=(1, 5)):
        input_np = input_data.cpu().numpy()
        n, c, L = input_np.shape
        best_output = None
        best_score = float('inf')
        best_params = (None, None)

        if target_data is None:
            # Use default params
            wavelet = 'db4'
            level = 2
            output_np = []
            for i in range(n):
                filtered = self.wavelet_denoise(input_np[i, 0], wavelet, level)
                output_np.append(filtered)
            output_np = np.stack(output_np, axis=0)
            output_np = output_np[:, np.newaxis, :]
            return torch.from_numpy(output_np).type_as(input_data)

        target_np = target_data.cpu().numpy()
        for wavelet in wavelet_list:
            for level in range(level_range[0], level_range[1]+1):
                output_np = []
                for i in range(n):
                    filtered = self.wavelet_denoise(input_np[i, 0], wavelet, level)
                    output_np.append(filtered)
                output_np = np.stack(output_np, axis=0)
                score = np.mean((output_np - target_np[:, 0, :]) ** 2)
                if score < best_score:
                    best_score = score
                    best_output = output_np
                    best_params = (wavelet, level)
        best_output = best_output[:, np.newaxis, :]
        return torch.from_numpy(best_output).type_as(input_data)

def get_outputs(input_data, input_data_dct, gt, models, plot=False):
    deep_learning_model = models[0]
    compare_models = models[1:]
    out_overall = []
    
    # reshape the signal first without copy
    original_shape = input_data.shape
    input_data = input_data.reshape(-1, 1, original_shape[-1])
    input_data_dct = input_data_dct.reshape(-1, 1, original_shape[-1])
    
    max_values = input_data.max(dim=2, keepdims=True)[0]
    input_data_dct_norm = input_data_dct / max_values
    input_data_norm = input_data / max_values
    
    dl_denoised, dl_raman = deep_learning(input_data_dct_norm, deep_learning_model)
    dl_denoised = dl_denoised * max_values
    dl_denoised = dl_denoised.reshape(original_shape)
    out_overall.append(dl_denoised)
    
    for model in compare_models:
        out = model.filter(input_data_norm)
        out = out * max_values
        out = out.reshape(original_shape)
        out_overall.append(out)
    
    input_data = input_data.reshape(original_shape)
    input_data_dct = input_data_dct.reshape(original_shape)
      
    if plot:
        wvn = loadmat("data/wvn_raw.mat")["wvn"].reshape(-1)
        wvn = wvn[331:]
        model_names = ["DL", "SG", "Wavelet"]
        # Randomly select 3 indices from dim 0, 3 from dim 1, 1 from dim 2
        idx0 = random.sample(range(out_overall[0].shape[0]), 3)
        idx1 = random.sample(range(out_overall[0].shape[1]), 3)
        idx2 = random.sample(range(out_overall[0].shape[2]), 1)
        num_models = len(out_overall)
        offsets = [2, 0.5, 1, 1.5, 0]  # Noisy, SG, Wavelet, DL, GT (top to bottom)
        colors = ['gray', 'g', 'b', 'k', "#FFA500"]
        labels = ['Noisy', 'DL', 'SG', 'Wavelet', 'GT']
        fig, axes = plt.subplots(3, 3, figsize=(20, 30))
        for i, i0 in enumerate(idx0):
            for j, j1 in enumerate(idx1):
                ax = axes[i, j]
                # Get original noisy signal and its max for normalization
                noisy_signal = input_data[i0, j1, idx2[0], :].cpu().numpy()
                max_val = np.max(noisy_signal)
                # Plot noisy input (top)
                ax.plot(wvn, noisy_signal / max_val + offsets[0], label=labels[0], color=colors[0], alpha=0.6)
                # Plot SG, Wavelet, DL (order: SG, Wavelet, DL)
                for m, out in enumerate(out_overall):
                    denoised_signal = out[i0, j1, idx2[0], :].cpu().numpy()
                    ax.plot(wvn, denoised_signal / max_val + offsets[m+1], label=labels[m+1], color=colors[m+1])
                # Plot GT (bottom)
                gt_signal = gt[i0][j1].cpu().numpy() if hasattr(gt[i0][j1], 'cpu') else gt[i0][j1]
                ax.plot(wvn, gt_signal / max_val + offsets[4], label=labels[4], color=colors[-1])
                # Remove y axis label and ticks
                ax.set_yticks([])
                ax.set_ylabel('')
                if i == 0 and j == 0:
                    ax.legend()
                ax.set_xlabel('Wavenumber (cm$^{-1}$)')
                ax.tick_params(axis='x', labelsize=18)
                ax.tick_params(axis='both', labelsize=16)
                ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=20)
        plt.tight_layout()
        plt.savefig('tmp/denoised_signals_comparison.png')
        
        
    return out_overall

def get_SNR_improvement(peak_pos, out_signals, raman_max, pairs):
    SNR_improvement = []
    for out in out_signals:
        # out: (a, b, c, 1000), peak_pos: (a, b, 1)
        out = out.cpu().numpy()
        a, b, c, L = out.shape
        SNR_tmp = np.zeros((a, b))
        SNR = np.zeros((a))
        for i in range(a):
            for j in range(b):
                k = int(peak_pos[i][j])
                raman_value = raman_max[i][j]
                # std along c at position k
                SNR_tmp[i, j] = 10 * np.log10((raman_value / np.std(out[i, j, :, k])) / pairs[i][0])
            SNR[i] = np.mean(SNR_tmp[i, :])
        SNR_improvement.append(SNR)
    return SNR_improvement

def plot(SNR_improvement, pairs):
    model_names = ["DL", "SG", "Wavelet"]
    surface_cmaps = ['Greens', 'Reds', 'Blues']
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    colors = ['r', 'g', 'b']
    legend_patches = []
    for i, snr_impr in enumerate(SNR_improvement):
        zs = np.array(snr_impr)
        # Create grid for surface plot
        grid_x, grid_y = np.mgrid[xs.min():xs.max():100j, ys.min():ys.max():100j]
        grid_z = griddata((xs, ys), zs, (grid_x, grid_y), method='cubic')
        surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=surface_cmaps[i], edgecolor='none', alpha=0.5)
        # For legend
        legend_patches.append(Patch(color=plt.get_cmap(surface_cmaps[i])(0.7), label=model_names[i]))
    ax.set_xlabel('SNR')
    ax.set_ylabel('r2f')
    ax.set_zlabel('SNR Improvement')
    ax.set_title('SNR Improvement Comparison')
    ax.legend(handles=legend_patches, loc='best')
    plt.tight_layout()
    plt.savefig('tmp/snr_improvement_all_models.png')
    plt.close(fig)
    # Save as interactive HTML for rotation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import plotly.graph_objs as go
    import plotly.offline as py

    model_names = ["DL", "SG", "Wavelet"]
    colors = ['green', 'blue', 'black']

    fig_plotly = go.Figure()
    for i, snr_impr in enumerate(SNR_improvement):
        xs = np.array([p[0] for p in pairs])
        ys = np.array([p[1] for p in pairs])
        zs = np.array(snr_impr)
        fig_plotly.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers',
            marker=dict(size=3, color=colors[i]),
            name=model_names[i]
        ))

    fig_plotly.update_layout(
        scene=dict(
            xaxis_title='SNR',
            yaxis_title='r2f',
            zaxis_title='SNR Improvement'
        ),
        title='SNR Improvement Comparison (Interactive)',
        legend=dict(x=0, y=1)
    )
    py.plot(fig_plotly, filename='tmp/snr_improvement_all_models_interactive.html', auto_open=False)


if __name__ == "__main__":
    # ---------------- the models ---------------
    dl_model = TwoStageModel()
    # Load the model
    dl_model_path = "models/pretrained/new_noise_model_05202025_211748_end_to_end_wvn_domain.pth"
    dl_model.load_state_dict(torch.load(dl_model_path))
    
    sg_model = SG_model()
    wavelet_model = WaveletModel()
    
    models = [dl_model, sg_model, wavelet_model]
    
    # Load the input data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ---------------- the data ---------------
    test_dir = "data/generated/pV_new_noise_model_test_05132025_181123.pkl"
    test_noise_dir = "data/noise/std"
    fluo_test_dir = "data/generated/poly_new_noise_model_test_fluorescence_05182025_185808.pkl"
    
    SNR_range = [0.01, 20]
    r2f_range = [0.05, 0.5]
    
    test_signal, _ = read_clean_data(clean_dir=test_dir, customized_noise=False, pV=True)
    fluo_test_signal, _ = read_clean_data(clean_dir=fluo_test_dir, customized_noise=False, pV=True)
    
    noise_std_dict = {}
    txt_files = glob.glob(os.path.join(test_noise_dir, "*.txt"))
    for txt_file in txt_files:
        std = read_noise_data(txt_file)
        key = os.path.splitext(os.path.basename(txt_file))[0]
        noise_std_dict[key] = std
        
    # ---------------- conduction ---------------
    pairs_all = []
    SNR_improvement_all = [np.array([]) for _ in range(len(models))]
    for _ in range(5):
        
        # Create Dataset and DataLoader
        test_dataset = DenoiserTestDataset(clean_signals=test_signal, noise_std_list=noise_std_dict, fluorescence=fluo_test_signal)
        test_dataset.generate_test_signals(SNR_range=SNR_range, r2f_range=r2f_range, params={"num_pairs": 100, "num_samples_per_pair": 5, "num_noise_per_sample": 10})
        test_dataset.DCT()
        
        input_data = test_dataset.noisy_signals
        input_data_dct = test_dataset.noisy_signals_dct
        pairs = test_dataset.pairs
        peak_pos = test_dataset.max_pos
        raman_max = test_dataset.raman_max
        gt = test_dataset.gt

        # Call the function
        out_signals = get_outputs(input_data, input_data_dct, gt, models, plot=False)
        SNR_improvement = get_SNR_improvement(peak_pos, out_signals, raman_max, pairs)
        
        pairs_all += pairs
        SNR_improvement_all = [np.concatenate((SNR_improvement_all[i], SNR_improvement[i])) for i in range(len(SNR_improvement_all))]
        
    plot(SNR_improvement_all, pairs_all)