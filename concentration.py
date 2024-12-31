from train import *
from scipy.optimize import nnls
from scipy.io import loadmat
from scipy.stats import ttest_rel
import pandas as pd


def get_concentrations(basis, signals):
    coeffs = []
    for signal in signals:
        signal = signal / np.max(signal)
        coefficients, residual = nnls(basis, signal)
        # coefficients = coefficients / np.sqrt(np.sum(coefficients ** 2))
        coeffs.append(coefficients)
    return coeffs

def plot_concentration(coeffs, data, save_names, SNR_ranges, filename_suffix):
    selected_components = list(range(7))
    plt.figure(figsize=(20, 8))  # Optional: Adjust figure size for better layout
    for i, selected_component in enumerate(selected_components):
        plt.subplot(2, 4, i + 1)
        coeffs_origin = coeffs["origin"]
        SNR_list = data["SNR_list"]  # Assuming SNR_list is in data
        for name, coeff in coeffs.items():
            if name != "origin":
                residual = np.abs(coeff - coeffs_origin) / coeffs_origin
                for SNR_range in SNR_ranges:
                    mask = (SNR_list >= SNR_range[0]) & (SNR_list < SNR_range[1])
                    plt.scatter(
                        coeffs_origin[mask, selected_component], 
                        residual[mask, selected_component], 
                        label=f"{name} SNR {SNR_range[0]}-{SNR_range[1]}", 
                        s=SNR_list[mask] * 10,  # Scale SNR_list to adjust point size
                        alpha=0.3 if SNR_range != (0, 1) else 0.5  # Add transparency (0 = fully transparent, 1 = fully opaque)
                    )
        # plt.axhline(y=0, color='r', linestyle='--')  # Add horizontal line at y=0
        # plt.yscale('symlog', linthresh=0.01)  # Set y-axis to symmetric log scale
        plt.yscale('log')
        plt.xlabel("Original Concentration", fontsize=13)
        plt.ylabel("Normalized predicted concentration difference", fontsize=10)
        plt.title(save_names[i], fontsize=14)  # Optional: Larger title font
        plt.legend(fontsize=10)  # Optional: Add legend with adjusted font size
    plt.subplot(2, 4, 8)
    name_list = ["Collagen", "Elastin", "Triolein", "Nucleus", "Keratin", "Ceramide", "Water"]
    for i in range(7):
        plt.plot(np.linspace(800, 1790, 1981), basis[:,i] + i*1.5, color="black")
    plt.xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=13)
    plt.title("Biophysical Model Basis", fontsize=16)
    plt.yticks([i*1.5 for i in range(7)], labels=name_list, fontsize=15)
    plt.tight_layout()  # Optional: Adjust subplot layout
    plt.savefig(f"results/concentrations_{filename_suffix}.jpg")
    
def get_SNR(cleaned, gt):
    return np.max(gt, axis=1) / np.std(cleaned[:, 1440-25:1440+25] - gt[:, 1440-25:1440+25], axis=1)

# signals, _, concentrations = read_clean_data("data/generated/generated_skin_spectrum_11012024_143232.pkl")
with open("./results/results_for_skin_test.pkl", 'rb') as file:
    data = pickle.load(file)

signals_origin = data["gt_signals"]
signals_from_cleaned = data["cleaned_signals_from_cleaned"]
signals_from_noisy = data["cleaned_signals_from_noisy"]
signals_SG = data["SG_denoise"]
signals_wavelet = data["wavelet_denoise"]
basis = loadmat("data/basics/basis_calibrated.mat")
basis = basis["basis"]

signal_datasets = {
    "origin": signals_origin,
    "from_cleaned": signals_from_cleaned,
    "from_noisy": signals_from_noisy,
    # "SG": signals_SG,
    # "wavelet": signals_wavelet
}

# Dictionary to store the results
coeffs = {}
SNR_improve = {}

# Apply `get_concentrations` to each dataset
for name, signals in signal_datasets.items():
    print(f"Processing {name} signals...")
    coeffs_tmp = get_concentrations(basis, signals)
    coeffs[name] = np.array(coeffs_tmp)
    if name != "origin":
        SNR_improve[name] = get_SNR(signals, signal_datasets["origin"]) / data["SNR_list"]

with open("./results/coeffs.pkl", 'wb') as f:
    pickle.dump(coeffs, f)

save_names = ["Collagen", "Elastin", "Triolein", "Nucleus", "Keratin", "Ceramide", "Water"]
SNR_ranges = [(0, 10), (0, 1), (1, 3), (3, 6), (6, 10)]

for i, SNR_range in enumerate(SNR_ranges):
    plot_concentration(coeffs, data, save_names, [SNR_range], f"SNR_{SNR_range[0]}_{SNR_range[1]}")

# draw SNR improvement
plt.figure()
origin_SNR = data["SNR_list"]
for name, SNR in SNR_improve.items():
    plt.scatter(origin_SNR, 10*np.log10(SNR), label=name, alpha=0.5, s=10)
plt.legend()
plt.xlabel("Original SNR (ratio)")
plt.ylabel("SNR improvement (dB)")
plt.savefig("results/SNR_improvement.jpg")

# p_values = []
# for i in range(basis.shape[1]):
#     origin_basis = coeffs_origin_np[:, i]
#     cleaned_basis = coeffs_cleaned_np[:, i]
#     t_statistic, p_value = ttest_rel(origin_basis, cleaned_basis)
#     p_values.append(p_value)

# # Create a DataFrame
# df = pd.DataFrame(p_values)

# # Save to CSV
# df.to_csv("./results/p_values_from_cleaned.csv", index=False)