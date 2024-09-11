import numpy as np
import matplotlib.pyplot as plt
import warnings


class RamanGenerator:
    def __init__(self, noise=None, wvn=None):
        self.data = None
        self.noise = noise
        self.wvn = wvn

    def generate(self, params):
        """
        Generate Raman spectra based on the input parameters.
        
        params: dict
            A dictionary containing parameters for the Raman spectra generation.
            Example params can include:
            - 'peak_positions': list of floats, positions of the peaks
            - 'amplitude': list of floats, amplitudes of the peaks
            - 'FWHM': float, width of the Gaussian peaks
        """
        # Gaussian function based on the provided equation
        def gaussian(x, mu, w):
            sigma = w / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        # Lorentzian function based on the provided equation
        def lorentzian(x, mu, w):
            return (1 / np.pi) * (w / 2) / ((x - mu) ** 2 + (w / 2) ** 2)

        # Pseudo-Voigt function combining Gaussian and Lorentzian parts
        def pseudo_voigt(x, mu, w, rho=0.6785):
            return rho * gaussian(x, mu, w) + (1 - rho) * lorentzian(x, mu, w)
        
        # Function to generate multi-peak Raman spectrum
        def generate_raman_spectrum(x, A, U, W):
            spectrum = np.zeros_like(x)
            N = len(U)  # Number of peaks
            for i in range(N):
                peak = pseudo_voigt(x, U[i], W[i])
                peak_norm = peak / np.max(peak)
                spectrum += A[i] * peak_norm  # Add each peak to the spectrum
            return spectrum
        
        def generate_raman_baseline(x, A_baseline, U_baseline, W_baseline):
            peak = pseudo_voigt(x, U_baseline, W_baseline)
            peak_norm = peak / np.max(peak)
            baseline = A_baseline * peak_norm
            return baseline

        # Extract parameters
        peak_positions = params.get('peak_positions')
        amplitude = params.get('amplitude')
        FWHM = params.get('FWHM')
        peak_position_baseline = params.get('peak_position_baseline')
        amplitude_baseline = params.get('amplitude_baseline')
        FWHM_baseline = params.get('FWHM_baseline')
        baseline_flag = params.get('baseline_flag')

        if self.wvn is None:
            warnings.warn("No wavenumber data was input. Generating spectrum based on customized wavenumber.")
            spec_bound = params.get('spectra_boundary')
            spec_resolution = params.get('spectra_resolution')
            self.wvn = np.linspace(spec_bound[0], spec_bound[1], (spec_bound[1]-spec_bound[0])*spec_resolution + 1)

        self.data = generate_raman_spectrum(self.wvn, amplitude, peak_positions, FWHM)

        if baseline_flag is True:
            baseline = generate_raman_baseline(self.wvn, amplitude_baseline, peak_position_baseline, FWHM_baseline)
            self.data = self.data + baseline

        return

    def addNoise(self):
        """
        Add nosie to the generated Raman spectra data.
        """
        if self.noise is None:
            warnings.warn("No noise has been input to the class. Input *args(your noise data) when call this class.")
            return
        self.data = self.data + self.noise
        return

    def getData(self):
        """
        Return the generated Raman spectra data.
        """
        if self.data is None:
            raise ValueError("No data has been generated yet. Call generate() first.")
        return self.data
    
    def getWVN(self):
        """
        Return the Raman spectra wavenumber.
        """
        return self.wvn


if __name__ == "__main__":
    # Example usage:
    params = {
        'peak_positions': [300, 800, 1200],
        'amplitude': [0.1, 0.9, 0.05],
        'FWHM': [25, 30, 50],
        # === Below is for customized wavenumber, if you already
        # have a wavenumber as input, you can ignore those params ===
        'spectra_boundary': [100, 2000],
        'spectra_resolution': 1000,
        # === Below is for adding a baseline, if you do not want to add
        # a base line, you can set 'baseline_flag': False and ignore other params ===
        'baseline_flag': True,
        'peak_position_baseline': 1000,
        'amplitude_baseline': 0.01,
        'FWHM_baseline': 2000,
    }

    Generator = RamanGenerator()
    Generator.generate(params)
    data = Generator.getData()
    wvn = Generator.getWVN()

    # Plot the resulting Raman spectrum
    plt.plot(wvn, data, label='Simulated Raman Spectrum')
    plt.xlabel('Raman Shift (cm$^{-1}$)')
    plt.ylabel('Intensity')
    plt.title('Simulated Raman Spectrum with Pseudo-Voigt Function')
    plt.legend()
    plt.show()