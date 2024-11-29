from train import *
from scipy.optimize import nnls
from scipy.io import loadmat
from scipy.stats import ttest_rel
import pandas as pd


# signals, _, concentrations = read_clean_data("data/generated/generated_skin_spectrum_11012024_143232.pkl")
with open("./results/results_0.01to1.pkl", 'rb') as file:
    data = pickle.load(file)

signals_origin = data["gt_signals"]
signals_cleaned = data["cleaned_signals"]
basis = loadmat("data/basics/basis_calibrated.mat")
basis = basis["basis"]

def get_concentrations(basis, signals):
    coeffs = []
    for signal in signals:
        signal = signal / np.max(signal)
        coefficients, residual = nnls(basis, signal)
        # coefficients = coefficients / np.sqrt(np.sum(coefficients ** 2))
        coeffs.append(coefficients)
    return coeffs

coeffs_origin = get_concentrations(basis, signals_origin)
coeffs_cleaned = get_concentrations(basis, signals_cleaned)

save_coeffs = {"origin": coeffs_origin, "cleaned": coeffs_cleaned}

with open("./results/coeffs_0.01to1.pkl", 'wb') as f:
    pickle.dump(save_coeffs, f)

coeffs_origin_np = np.array(coeffs_origin)
coeffs_cleaned_np = np.array(coeffs_cleaned)

p_values = []
for i in range(basis.shape[1]):
    origin_basis = coeffs_origin_np[:, i]
    cleaned_basis = coeffs_cleaned_np[:, i]
    t_statistic, p_value = ttest_rel(origin_basis, cleaned_basis)
    p_values.append(p_value)

# Create a DataFrame
df = pd.DataFrame(p_values)

# Save to CSV
df.to_csv("./results/p_values_0.01to1.csv", index=False)