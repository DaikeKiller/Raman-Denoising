from train import *
from scipy.optimize import nnls
from scipy.io import loadmat
from scipy.stats import ttest_rel
import pandas as pd
from scipy.optimize import curve_fit


def get_concentrations(basis, signals):
    coeffs = []
    for signal in signals:
        signal = signal / np.max(signal)
        coefficients, residual = nnls(basis, signal)
        # coefficients = coefficients / np.sqrt(np.sum(coefficients ** 2))
        coeffs.append(coefficients)
    return coeffs

def plot_concentration(coeffs, data, save_names, SNR_ranges, filename_suffix, norm=False):
    if norm:
        coeffs_norm = {}
        for name, coeff in coeffs.items():
            coeffs_norm[name] = coeff / np.sum(coeff, axis=1).reshape(-1, 1)
        coeffs = coeffs_norm
    selected_components = list(range(7))
    plt.figure(figsize=(20, 8))  # Optional: Adjust figure size for better layout
    for i, selected_component in enumerate(selected_components):
        plt.subplot(2, 4, i + 1)
        coeffs_origin = coeffs["origin"]
        SNR_list = data["SNR_list"]  # Assuming SNR_list is in data
        for j , (name, coeff) in enumerate(coeffs.items()):
            if name != "origin":
                coeff = coeff / coeffs_origin if norm else coeff
                for SNR_range in SNR_ranges:
                    mask = (SNR_list >= SNR_range[0]) & (SNR_list < SNR_range[1])
                    # Fit a linear line
                    if not norm:
                        scatter = plt.scatter(
                            coeffs_origin[mask, selected_component], 
                            coeff[mask, selected_component], 
                            label=f"{name} SNR {SNR_range[0]}-{SNR_range[1]}", 
                            s=SNR_list[mask] * 10,  # Scale SNR_list to adjust point size
                            alpha=0.3 if SNR_range != (0, 1) else 0.5,  # Add transparency (0 = fully transparent, 1 = fully opaque),
                            color=(
                                '#1f77b4' if name == "from noise removal" else 
                                '#ff7f0e' if name == "from noisy" else 
                                '#2ca02c' if name == "from SG" else 
                                '#d62728' if name == "raw" else 
                                'red'  # Optional: Set color based on dataset
                            )
                        )
                        
                        slope, intercept = np.polyfit(coeffs_origin[mask, selected_component], coeff[mask, selected_component], 1)
                        line_x = np.linspace(np.min(coeffs_origin[mask, selected_component]), np.max(coeffs_origin[mask, selected_component]), 1000)
                        line_y = slope * line_x + intercept
                        color = scatter.get_facecolor()[0]
                        plot = plt.plot(
                            line_x, 
                            line_y, 
                            linestyle='-', 
                            linewidth=3,
                            color=color * 0.8,  # Darken the color
                            alpha=1,  # Add transparency (0 = fully transparent, 1 = fully opaque)
                            label=None  # Prevent this plot from appearing in the legend
                        )
                        # Calculate R-squared value
                        residuals = coeff[mask, selected_component] - (slope * coeffs_origin[mask, selected_component] + intercept)
                        ss_res = np.sum(residuals**2)
                        ss_tot = np.sum((coeff[mask, selected_component] - np.mean(coeff[mask, selected_component]))**2)
                        r_squared = 1 - (ss_res / ss_tot)
                        # Add R-squared value to the plot
                        plt.text(
                            0.95, 
                            0.05 + j * 0.05,  # Adjust vertical position to avoid overlap
                            f"$R^2$={r_squared:.2f}", 
                            fontsize=12, 
                            color=color,
                            alpha=1,  # Add transparency (0 = fully transparent, 1 = fully opaque)
                            transform=plt.gca().transAxes,
                            horizontalalignment='right',
                            verticalalignment='bottom'
                        )
                    # if norm
                    else:
                        scatter = plt.scatter(
                            coeffs_origin[mask, selected_component], 
                            coeff[mask, selected_component], 
                            label=f"{name} SNR {SNR_range[0]}-{SNR_range[1]}", 
                            s=SNR_list[mask] * 10,  # Scale SNR_list to adjust point size
                            alpha=0.3 if SNR_range != (0, 1) else 0.5,  # Add transparency (0 = fully transparent, 1 = fully opaque),
                            color=(
                                '#1f77b4' if name == "from noise removal" else 
                                '#ff7f0e' if name == "from noisy" else 
                                '#2ca02c' if name == "from SG" else 
                                '#d62728' if name == "raw" else 
                                'red'  # Optional: Set color based on dataset
                            )
                        )
                        color = scatter.get_facecolor()[0]
                        coeff_fit = coeff * coeffs_origin
                        slope, intercept = np.polyfit(coeffs_origin[mask, selected_component], coeff_fit[mask, selected_component], 1)
                        # Calculate R-squared value
                        residuals = (coeff_fit[mask, selected_component] - (slope * coeffs_origin[mask, selected_component] + intercept)) / coeffs_origin[mask, selected_component]
                        ss_res = np.sum(residuals**2)
                        ss_tot = np.sum((coeff_fit[mask, selected_component] / coeffs_origin[mask, selected_component] - np.mean(coeff_fit[mask, selected_component] / coeffs_origin[mask, selected_component]))**2)
                        r_squared = 1 - (ss_res / ss_tot)
                        
                        line_x = np.linspace(np.min(coeffs_origin[mask, selected_component]), np.max(coeffs_origin[mask, selected_component]), 1000)
                        line_y = slope * line_x + intercept
                        plt.plot(
                            line_x, 
                            line_y / line_x, 
                            linestyle='-', 
                            linewidth=3,
                            color=color * 0.8,  # Darken the color
                            alpha=1,  # Add transparency (0 = fully transparent, 1 = fully opaque)
                            label=None  # Prevent this plot from appearing in the legend
                        )
                        
                        # Add R-squared value to the plot
                        plt.text(
                            0.95, 
                            0.05 + j * 0.05,  # Adjust vertical position to avoid overlap
                            f"$R^2$={r_squared:.2f}", 
                            fontsize=12, 
                            color=color,
                            alpha=1,  # Add transparency (0 = fully transparent, 1 = fully opaque)
                            transform=plt.gca().transAxes,
                            horizontalalignment='right',
                            verticalalignment='bottom'
                        )
        
        if norm:
            plt.axhline(y=1, color='black', linestyle='--', label=None)
        else:
            plt.plot([0, 1], [0, 1], color='black', linestyle='--', label=None)
        plt.xlabel("Normalized Original Concentration", fontsize=12) if norm else plt.xlabel("Original Concentration", fontsize=13)
        plt.ylabel("Normalized predicted concentration", fontsize=12) if norm else plt.ylabel("Predicted concentration", fontsize=13)
        plt.title(save_names[i], fontsize=14)  # Optional: Larger title font
        plt.legend(fontsize=10)  # Optional: Add legend with adjusted font size
        if not norm:
            plt.xlim(min(coeffs_origin[:, selected_component])+0.1, max(coeffs_origin[:, selected_component])+0.1)  # Set x-axis range
            plt.ylim(min(coeffs_origin[:, selected_component])+0.1, max(coeffs_origin[:, selected_component])+0.1)  # Set y-axis range
        # plt.xlim(0, 1)  # Set x-axis range
        # plt.ylim(0, 1)  # Set y-axis range
        if norm:
            plt.yscale("log")  # Optional: Set y-axis scale
        plt.tight_layout()  # Optional: Adjust subplot layout
    plt.subplot(2, 4, 8)
    name_list = ["Collagen", "Elastin", "Triolein", "Nucleus", "Keratin", "Ceramide", "Water"]
    for i in range(7):
        plt.plot(np.linspace(800, 1790, 1981), basis[:,i] + i*1.5, color="black")
    plt.xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=13)
    plt.title("Biophysical Model Basis", fontsize=16)
    plt.yticks([i*1.5 for i in range(7)], labels=name_list, fontsize=15)
    plt.tight_layout()  # Optional: Adjust subplot layout
    plt.savefig(f"results/norm_concentrations_{filename_suffix}.jpg") if norm else plt.savefig(f"results/concentrations_{filename_suffix}.jpg")
    
def get_SNR(cleaned, gt):
    return np.max(gt, axis=1) / np.std(cleaned[:, 1440-25:1440+25] - gt[:, 1440-25:1440+25], axis=1)

# signals, _, concentrations = read_clean_data("data/generated/generated_skin_spectrum_11012024_143232.pkl")
with open("./results/results_for_skin_test.pkl", 'rb') as file:
    data = pickle.load(file)

signals_origin = data["gt_signals"]
signals_from_cleaned = data["cleaned_signals_from_cleaned"]
signals_from_noisy = data["cleaned_signals_from_noisy"]
signals_SG = data["cleaned_signals_from_SG"]
signals_raw = data["noisy_signals"]
# signals_wavelet = data["wavelet_denoise"]
basis = loadmat("data/basics/basis_calibrated.mat")
basis = basis["basis"]

signal_datasets = {
    "origin": signals_origin,
    "from noise removal": signals_from_cleaned,
    "from noisy": signals_from_noisy,
    "from SG": signals_SG,
    "raw": signals_raw,
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
    plot_concentration(coeffs, data, save_names, [SNR_range], f"SNR_{SNR_range[0]}_{SNR_range[1]}", norm=True)

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