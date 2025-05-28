from test import test_model_full
from train_nl import *
from scipy.ndimage import gaussian_filter1d
from evaluation_peaks import PolyFit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error
from scipy.io import loadmat
from scipy.interpolate import interp1d


def deep_learning_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test_dir = "data/generated/pV_new_noise_model_test_05132025_181123.pkl"
    test_dir = "data/generated/generated_skin_spectrum_05262025_131135.pkl"
    test_noise_dir = "data/noise/std"
    fluo_test_dir = "data/generated/poly_new_noise_model_test_fluorescence_05182025_185808.pkl"
    
    SNR_range = [0.01, 20]
    r2f_range = [0.05, 0.5]
    # SNR_range = [np.log10(a) for a in SNR_range]

    # Load the best trained model
    model_full_path = "models/pretrained/new_noise_model_05202025_211748_end_to_end_wvn_domain.pth"
    # model_full_path = "models/pretrained/new_noise_model_05202025_151616_end_to_end.pth"
    model_full = TwoStageModel()
    model_full.load_state_dict(torch.load(model_full_path))
    model_full.eval()  # Set the model to evaluation mode

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
    output = test_model_full(model_full, test_dataloader, device)
    
    return output

def traditional_model(data):
    sigma = 2  # You can adjust sigma for smoothing strength
    output = gaussian_filter1d(data, sigma=sigma, axis=1)
    
    regular_model = PolyFit()
    output = regular_model.apply(output)
        
    return output

def plot(signal1, singal2, original, gt, SNR_list):

    idx = np.random.randint(0, signal1.shape[0])
    offset = 0.7  # vertical offset between signals
    plt.figure(figsize=(6, 10))
    wvn = np.linspace(600, 1790, signal1.shape[1])
    plt.plot(wvn, original[idx] / np.max(original[idx]) + 3*offset, label='Original', linestyle='-', color='gray')
    plt.plot(wvn, signal1[idx] / np.max(original[idx]) + 2*offset, label='Traditional Model', linestyle='-', color='blue')
    plt.plot(wvn, singal2[idx] / np.max(original[idx]) + offset, label='Deep Learning Model', linestyle='-', color='green')
    plt.plot(wvn, gt[idx] / np.max(original[idx]), label='Ground Truth', linestyle='-', color="#FFA500")
    plt.legend(loc='upper left')
    plt.title(f"Original SNR = {SNR_list[idx]: .2f}")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig("results/evaluation_skin.png", dpi=300, bbox_inches='tight')
    
    return

def get_concentrations(basis, signals):
    # Interpolate basis from (1981, 7) to (693, 7)
    old_x = np.linspace(600, 1790, basis.shape[0])
    new_x = np.linspace(600, 1790, 693)
    interp_func = interp1d(old_x, basis, axis=0, kind='linear')
    basis = interp_func(new_x)
    coeffs = []
    for signal in signals:
        signal = signal / np.max(signal)
        coefficients, residual = nnls(basis, signal)
        # coefficients = coefficients / np.sqrt(np.sum(coefficients ** 2))
        coeffs.append(coefficients)
    return coeffs

def compare_concentrations(concentrations_dl, concentrations_traditional, concentrations_gt, SNR_list):
    concentrations_dl = np.array(concentrations_dl)
    concentrations_traditional = np.array(concentrations_traditional)
    concentrations_gt = np.array(concentrations_gt)
    SNR_list = np.array(SNR_list)

    for snr_thr, label in zip([7, 7], ['SNR ≤ 7', 'SNR 7~20']):
        if label == 'SNR ≤ 7':
            idx = SNR_list <= snr_thr
        else:
            idx = SNR_list > snr_thr

        gt = concentrations_gt[idx]
        pred_dl = concentrations_dl[idx]
        pred_trad = concentrations_traditional[idx]

        mse_dl = mean_squared_error(gt, pred_dl)
        mse_trad = mean_squared_error(gt, pred_trad)

        plt.figure(figsize=(12, 5))
        components = ["Collagen", "Elastin", "Triolein", "Nucleus", "Keratin", "Ceramide", "Water"]
        for i in range(7):
            plt.subplot(2, 4, i+1)
            plt.scatter(gt[:, i], pred_dl[:, i], alpha=0.5, label='DL', color='tab:blue', s=10)
            plt.scatter(gt[:, i], pred_trad[:, i], alpha=0.5, label='Traditional', color='tab:orange', s=10)
            # Fit and plot linear regression lines
            if len(gt[:, i]) > 1:
                # DL fit
                coef_dl = np.polyfit(gt[:, i], pred_dl[:, i], 1)
                fit_dl = np.polyval(coef_dl, gt[:, i])
                plt.plot(gt[:, i], fit_dl, color="#002466", linewidth=2)  # darker blue
                # Traditional fit
                coef_trad = np.polyfit(gt[:, i], pred_trad[:, i], 1)
                fit_trad = np.polyval(coef_trad, gt[:, i])
                plt.plot(gt[:, i], fit_trad, color='#994c00', linewidth=2)  # darker orange
            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlabel('True Concentration')
            plt.ylabel('Predicted')
            plt.title(components[i])
            if i == 0:
                plt.legend()
        plt.suptitle(f'Predicted vs GT Concentrations ({label})\nMSE DL: {mse_dl:.4f}, MSE Traditional: {mse_trad:.4f}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'results/concentration_scatter_{label.replace(" ", "_")}.png', dpi=300)
        plt.close()
    
    return

def main():
    output = deep_learning_model()
    
    noisy_signals = output["noisy_signals"]
    cleaned_signals = output["cleaned_signals"]
    raman_signals = output["raman_signals"]
    gt_raman = output["gt_raman"]
    SNR_list = output["SNR_list"]
    
    # Apply traditional model
    traditional_output = traditional_model(noisy_signals)
    
    # Plot the results
    plot(traditional_output, raman_signals, noisy_signals, gt_raman, SNR_list)
    
    basis = loadmat("data/basics/basis_calibrated.mat")
    basis = basis["basis"]
    concentrations_dl = get_concentrations(basis, raman_signals)
    concentrations_traditional = get_concentrations(basis, traditional_output)
    concentrations_gt = get_concentrations(basis, gt_raman)
    
    compare_concentrations(concentrations_dl, concentrations_traditional, concentrations_gt, SNR_list)

if __name__ == "__main__":
    main()