import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pseudo_Voiget import RamanGenerator

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
    popt, _ = curve_fit(raman_model, x, y, p0=initial_guess, maxfev=10000, ftol=1e-9, xtol=1e-9)
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


if __name__ == "__main__":
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

    # Start with an initial guess (e.g., one peak to begin)
    max_idx = np.argmax(smooth_data)  # Index of the maximum intensity
    max_position = wvn[max_idx]  # Wavenumber corresponding to the max intensity
    max_amplitude = smooth_data[max_idx]  # Amplitude of the peak
    intial_FWHM = 30
    initial_guess = [max_position, intial_FWHM, max_amplitude]  # Initial guess for one peak: position, FWHM, amplitude

    # Perform iterative fitting
    fitted_params, fitted_spectrum, residuals = iterative_fit_raman_spectrum(wvn, smooth_data, initial_guess, min_peak_height=0.01, max_iterations=10)

    # Plot the results
    plt.plot(wvn, smooth_data, label='Smooth Spectrum')
    plt.plot(wvn, fitted_spectrum, label='Fitted Spectrum', linestyle='--')
    plt.plot(wvn, residuals, label='Residuals', linestyle=':')
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()