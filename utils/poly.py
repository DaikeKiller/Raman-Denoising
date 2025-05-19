import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time


class FluorescenceGenerator:
    def __init__(self, wvn=None):
        """
        Generator for polynomial fluorescence backgrounds.
        wvn: optional array of wavenumbers; if None, defined per call.
        """
        self.wvn = wvn
        self.output = None

    def generate_baseline(self, num_datapt, spectra_range, poly_order_range, poly_coeff_range):
        """
        Generate a single fluorescence baseline as a random polynomial,
        ensuring all values are positive within the x range.

        - num_datapt: int, number of data points
        - spectra_range: tuple (min_wvn, max_wvn)
        - poly_order_range: tuple (min_order, max_order)
        - poly_coeff_range: tuple (min_coeff, max_coeff)
        """
        low, high = spectra_range
        x = np.linspace(low, high, num_datapt)
        for _ in range(100):  # Try up to 100 times to get a positive baseline
            order = np.random.randint(poly_order_range[0], poly_order_range[1] + 1)
            coeffs = np.random.uniform(poly_coeff_range[0], poly_coeff_range[1], order + 1)
            baseline = np.polyval(coeffs, x)
            min_val = np.min(baseline)
            if min_val > 0:
                break
            # If not all positive, shift baseline up
            baseline = baseline - min_val + 1e-6
            if np.min(baseline) > 0:
                break
        else:
            raise RuntimeError("Failed to generate a positive baseline after 100 attempts.")
        self.wvn = x
        self.output = baseline / np.max(baseline)  # Normalize to max value of 1
        return self.output

    def generate_multiple_baselines(self,
                                     num_spectra,
                                     num_datapt,
                                     spectra_range,
                                     poly_order_range,
                                     poly_coeff_range,
                                     save_flag,
                                     save_path):
        """
        Generate multiple fluorescence baselines and optionally save to pickle.

        Returns a numpy array of shape (num_datapt, num_spectra).
        """
        baselines = np.zeros((num_datapt, num_spectra))
        for i in range(num_spectra):
            baselines[:, i] = self.generate_baseline(
                num_datapt,
                spectra_range,
                poly_order_range,
                poly_coeff_range
            )
        self.output = baselines / np.max(baselines, axis=0)  # Normalize each spectrum to max value of 1
        # Save if requested
        if save_flag:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(baselines, f)
            print(f"Saved {num_spectra} fluorescence baselines to {save_path}")
        return self.output

    def plot(self, max_plot=10):
        """
        Plot up to max_plot baseline curves.
        """
        if self.output is None:
            raise ValueError("No baselines generated. Call generate_multiple_baselines() first.")
        num_spectra = self.output.shape[1]
        to_plot = self.output[:, :min(max_plot, num_spectra)]
        plt.figure()
        for i in range(to_plot.shape[1]):
            plt.plot(self.wvn, to_plot[:, i])
        plt.xlabel("Wavenumber")
        plt.ylabel("Fluorescence Intensity")
        plt.title("Polynomial Fluorescence Baselines")
        plt.show()


if __name__ == "__main__":
    # Example usage
    save_path = "data/generated"
    save_name = "poly_new_noise_model_test_fluorescence"
    timestamp = time.strftime("%m%d%Y_%H%M%S")
    file_name = save_name + "_" + timestamp + ".pkl"
    save_final = os.path.join(save_path, file_name)

    gen = FluorescenceGenerator()
    baselines = gen.generate_multiple_baselines(
        num_spectra=1000,
        num_datapt=693,
        spectra_range=(600, 1790),
        poly_order_range=(3, 6),
        poly_coeff_range=(-1, 1),
        save_flag=True,
        save_path=save_final
    )
    gen.plot()
    plt.savefig(os.path.join("results", save_name + ".png"))