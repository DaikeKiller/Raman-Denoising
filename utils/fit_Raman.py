import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils.pseudo_Voiget import RamanGenerator
import warnings


def read_in_data(path=None):
    data = {}
    if path is None:
        warnings.warn("read_in_data: ====== No input data, generating simulated data. =======")
        # Create simulated noisy data using the RamanGenerator
        params = {
            'peak_positions': [300, 800, 1200, 607],
            'amplitude': [0.1, 0.9, 0.05, 0.45],
            'FWHM': [25, 30, 50, 200],
            'baseline_flag': True,
            'peak_position_baseline': 1000,
            'amplitude_baseline': 0.01,
            'FWHM_baseline': 2000,
        }

        # Assuming wvn is already provided
        wvn = np.linspace(100, 2000, 1000)
        generator = RamanGenerator(wvn=wvn)
        generator.generate(params)
        smooth_data = generator.getData()  # Smooth, less noisy spectrum
        data["wvn"] = wvn
        data["original_spectrum"] = smooth_data
        data["smoothed_spectrum"] = smooth_data
    else:
        mat_data = loadmat(path)
        data["wvn"] = mat_data["data"]["wvn"][0,0]
        data["original_spectrum"] = mat_data["data"]["original"][0,0]
        data["smoothed_spectrum"] = mat_data["data"]["baseline"][0,0]
    return data

# Define a fitting function using your RamanGenerator to model the spectrum
def raman_model(wvn, *params):
    N = len(params) // 3
    peak_positions = params[:N]
    FWHM = params[N:2*N]
    amplitude = params[2*N:3*N]

    # Prepare input params for the RamanGenerator
    input_params = {
        'peak_positions': peak_positions,
        'FWHM': FWHM,
        'amplitude': amplitude,
        # 'spectra_boundary': [min(wvn), max(wvn)],
        # 'spectra_resolution': len(wvn),
        'baseline_flag': False,  # Assuming we don't want to fit the baseline right now
        # 'peak_position_baseline': 1000,
        # 'amplitude_baseline': 0.01,
        # 'FWHM_baseline': 2000,
    }

    # Generate the spectrum using RamanGenerator
    generator = RamanGenerator(wvn=wvn)
    generator.generate(input_params)
    
    # Return the generated data
    return generator.getData()

def fit_raman_spectrum(x, y, initial_guess):
    popt, _ = curve_fit(raman_model, x, y, p0=initial_guess, maxfev=1000, ftol=1e-2, xtol=1e-2)
    fitted_spectrum = raman_model(x, *popt)
    residuals = y - fitted_spectrum
    return popt, fitted_spectrum, residuals

# Iteratively fit the Raman spectrum by detecting peaks in the residuals
def iterative_fit_raman_spectrum(x, y, initial_guess, min_peak_height=0.05, max_iterations=10):
    current_guess = initial_guess
    iteration = 0
    residuals = y.copy()

    while iteration < max_iterations:
        # Step 1: Fit the spectrum with the current guess
        fitted_params, fitted_spectrum, residuals = fit_raman_spectrum(x, y, current_guess)

        # Step 2: Detect new peaks in the residuals
        peaks, _ = find_peaks(residuals, height=min_peak_height)  # Adjust height threshold as needed
        new_peak_positions = x[peaks]
        new_amplitudes = residuals[peaks]
        new_FWHM = np.full(len(new_peak_positions), 30)  # Assume a default FWHM for new peaks

        if len(new_peak_positions) == 0:
            print(f"Converged after {iteration} iterations")
            break  # No new peaks found, stop the iteration

        # Step 3: Update the current guess by adding the new peaks
        current_guess = np.concatenate([
            fitted_params[:len(fitted_params)//3], new_peak_positions,  # Peak positions
            fitted_params[len(fitted_params)//3:2*len(fitted_params)//3], new_FWHM,  # FWHMs
            fitted_params[2*len(fitted_params)//3:], new_amplitudes  # Amplitudes
        ])

        iteration += 1

    return fitted_params, fitted_spectrum, residuals

# Define a function to ensure the fitted spectrum is not larger than the target
def residuals_with_constraints(params, wvn, target_spectrum):
    fitted_spectrum = raman_model(wvn, *params)
    residuals = fitted_spectrum - target_spectrum
    # Clip residuals to ensure that fitted_spectrum <= target_spectrum
    return np.clip(residuals, 0, None)

# Fit the Raman spectrum with constraints using least_squares
def constrained_fit_raman_spectrum(x, y, initial_guess, bounds):
    result = least_squares(residuals_with_constraints, initial_guess, args=(x, y), bounds=bounds)
    fitted_params = result.x
    fitted_spectrum = raman_model(x, *fitted_params)
    residuals = y - fitted_spectrum
    return fitted_params, fitted_spectrum, residuals

def blend_spectrum(model_spectrum, real_spectrum, alpha=0.5):
    """
    Blends the model spectrum with the real spectrum based on the blending factor alpha.
    alpha = 0.0 means 100% real data, alpha = 1.0 means 100% model spectrum.
    """
    return (1 - alpha) * real_spectrum + alpha * model_spectrum


if __name__ == "__main__":
    file_path = "data/p22_BCC_stage2_clinic_section_epidermis_2s_1.mat"
    data = read_in_data(path=file_path)
    wvn = data["wvn"].reshape(-1)
    smooth_data = data["smoothed_spectrum"].reshape(-1)
    original_data = data["original_spectrum"].reshape(-1)
    normalization_factor = np.max(smooth_data)
    smooth_data = smooth_data / normalization_factor
    # original_data = original_data / np.max(original_data)

    # Start with an initial guess (e.g., one peak to begin)
    max_idx = np.argmax(smooth_data)  # Index of the maximum intensity
    max_position = wvn[max_idx]  # Wavenumber corresponding to the max intensity
    max_amplitude = smooth_data[max_idx]  # Amplitude of the peak
    intial_FWHM = 30
    initial_guess = [max_position, intial_FWHM, max_amplitude]  # Initial guess for one peak: position, FWHM, amplitude

    # Perform iterative fitting
    fitted_params, fitted_spectrum, residuals = iterative_fit_raman_spectrum(wvn, smooth_data, initial_guess, min_peak_height=0.05, max_iterations=10)

    # Blend the fitted spectrum with the original smooth data
    blended_spectrum = blend_spectrum(fitted_spectrum, smooth_data, alpha=0.4)

    # recover to spectrum before normalization
    smooth_data = smooth_data * normalization_factor
    blended_spectrum = blended_spectrum * normalization_factor
    # residuals = residuals * normalization_factor

    # Plot the results
    plt.subplot(2,1,1)
    plt.plot(wvn, original_data, label='Original Spectrum', color="blue")
    plt.plot(wvn, smooth_data, label='Smooth Spectrum', color="orange")
    plt.plot(wvn, blended_spectrum, label='Fitted Spectrum', linestyle='--', color="green")
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(wvn, original_data-smooth_data, label='Residuals of orginal-smoothed Spectrum', color="orange")
    plt.plot(wvn, original_data-blended_spectrum, label='Residuals of orginal-fitted Spectrum', linestyle='--', color="green")
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()

    plt.plot(wvn, original_data, label='Original Spectrum', color="blue")
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Intensity')
    plt.title("Original")
    plt.show()

    plt.plot(wvn, smooth_data, label='Smoothed Spectrum', color="orange")
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Intensity')
    plt.title("Smoothed")
    plt.show()

    plt.plot(wvn, blended_spectrum, label='Fitted Spectrum', color="green")
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Intensity')
    plt.title("Fitted")
    plt.show()