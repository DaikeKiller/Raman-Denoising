import numpy as np
from scipy.optimize import curve_fit
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

# Example usage to fit a distorted or noisy spectrum
def fit_raman_spectrum(x, y, initial_guess):
    # Use curve_fit to fit the Raman model to the noisy/distorted spectrum
    popt, _ = curve_fit(raman_model, x, y, p0=initial_guess)
    return popt

if __name__ == "__main__":
    target_data = None
    wvn = None

    # Initial guess for fitting: (peak positions, FWHMs, amplitudes)
    initial_guess = [300, 800, 1200, 25, 30, 50, 0.1, 0.9, 0.05]

    # Fit the Raman spectrum
    fitted_params = fit_raman_spectrum(wvn, target_data, initial_guess)

    # Generate the fitted spectrum using the optimized parameters
    fitted_spectrum = raman_model(wvn, *fitted_params)

    # Plot the results
    plt.plot(wvn, target_data, label='Noisy Spectrum')
    plt.plot(wvn, fitted_spectrum, label='Fitted Spectrum', linestyle='--')
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()