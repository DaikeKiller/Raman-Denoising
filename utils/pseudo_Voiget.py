import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
import os
import time


class RamanGenerator:
    def __init__(self, noise=None, wvn=None):
        self.data = None
        self.noise = noise
        self.wvn = wvn
        self.output = None

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
            self.wvn = np.linspace(spec_bound[0], spec_bound[1], int(np.ceil((spec_bound[1]-spec_bound[0]) / spec_resolution + 1)))

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
    
    def generate_multiple_spectra(self, num_spectra=10000, max_peak_num=16, spectra_range=(800, 1800), amplitude_range=(0.05, 1.0), \
                                  num_datapt=1981, fwhm_range=(10, 100), noise_level=0, save_flag=True, save_path='data/generated/raman_spectra.pkl'):
        """
        Generate multiple Raman spectra and save them to a pickle file.
        
        Parameters:
        - num_spectra: int
            Number of spectra to generate.
        - fwhm_range: tuple of int
            Range for FWHM values for each peak (default is 10 to 100 cm^-1).
        - noise_level: float
            Standard deviation for noise to be added to each spectrum.
        - output_file: str
            Path to the output pickle file.
        """
        spectra = np.zeros((num_datapt, num_spectra))

        i = 0
        while i < num_spectra:
            # Randomize parameters for each spectrum
            num_peaks = np.random.randint(1, max_peak_num)
            peak_positions = np.random.uniform(spectra_range[0], spectra_range[1], num_peaks)
            amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], num_peaks)
            fwhms = np.random.uniform(fwhm_range[0], fwhm_range[1], num_peaks)

            for j in range(num_peaks):
                if (peak_positions[j] + 1.5*fwhms[j] < spectra_range[1]) and (peak_positions[j] - 1.5*fwhms[j] > spectra_range[0]):
                    valid_spectrum = True
                else:
                    valid_spectrum = False
                    break
            
            if not valid_spectrum:
                continue

            params = {
                'peak_positions': peak_positions,
                'amplitude': amplitudes,
                'FWHM': fwhms,
                'spectra_boundary': [spectra_range[0], spectra_range[1]],
                'spectra_resolution': (spectra_range[1] - spectra_range[0]) / (num_datapt - 1),
                'baseline_flag': False,
                'peak_position_baseline': 1000,
                'amplitude_baseline': 0.01,
                'FWHM_baseline': 2000,
            }
            
            self.generate(params)  # Generate a single spectrum
            if noise_level > 0:
                self.addNoise()  # Optionally add noise
            
            spectra[:, i] = self.getData()  # Store generated spectrum
            i += 1
        
        if save_flag:
            # Save spectra to a pickle file
            with open(save_path, 'wb') as file:
                pickle.dump(spectra, file)
        
        print(f"Generated {num_spectra} spectra and saved to {save_path}")
        self.output = spectra
    
    def plot(self):
        num_spectra = self.output.shape[1]
        if num_spectra > 10:
            plot_spectra = self.output[:, :10]
        else:
            plot_spectra = self.output

        plt.figure()
        for i in range(plot_spectra.shape[1]):
            plt.plot(range(1, self.output.shape[0]+1), self.output[:,i])
        plt.xlabel("pseudo_wavenumber", fontsize=15)
        plt.ylabel("Intensity", fontsize=15)


if __name__ == "__main__":
    save_path = "data/generated"
    save_name = "raman_pesudo_Vioget_test"
    timestamp = time.strftime("%m%d%Y_%H%M%S")
    file_name = save_name + "_" + timestamp + ".pkl"
    save_name = os.path.join(save_path, file_name)
    num_spectra = 1000
    max_peak_num = 30
    spectra_range = (800, 1800)
    amplitude_range = (0.05, 1.0)
    num_datapt = 1981
    fwhm_range = (10, 200) 
    noise_level = 0 
    save_flag = True

    generator = RamanGenerator()
    generator.generate_multiple_spectra(num_spectra=num_spectra, max_peak_num=max_peak_num, spectra_range=spectra_range, amplitude_range=amplitude_range, \
                                        num_datapt=num_datapt, fwhm_range=fwhm_range, noise_level=noise_level, save_flag=save_flag, save_path=save_name)
    generator.plot()
    plt.show()