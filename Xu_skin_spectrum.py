import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import pickle
import time
from scipy.interpolate import interp1d

class SkinSimulator:
    def __init__(self, basis) -> None:
        self.basis = basis
        self.concentrations = None
        self.spectra = None
        self.spectraLength = self.basis.shape[0]
        self.componentNum = self.basis.shape[1]
    
    def generate(self, concentrations=None, nums=None) -> None:
        if concentrations is None:
            if nums is None:
                nums = 1
            self.concentrations = np.random.uniform(0, 1, (self.componentNum, nums))
        else:
            self.concentrations = np.transpose(np.array(concentrations))
            if self.concentrations.shape[0] != self.componentNum:
                Warning("Input concentration needs to have f-by-n shape. f is the \
                              numder of basis spectra, n is the number of spectra to generate. \
                              Got f does not match.")
            if nums is not None:
                if self.concentrations.shape[1] != nums:
                    Warning("The number of spectra (nums) provided does not match the second \
                                dimension of the input concentrations.")
            for i in range(self.concentrations.shape[1]):
                con_tmp = self.concentrations[:, i]
                factor = np.sqrt(np.sum((con_tmp**2)))
                self.concentrations[:, i] = self.concentrations[:, i] / factor
        
        self.spectra = self.basis @ self.concentrations
        original_length = self.spectra.shape[0]
        target_length = 693
        x_original = np.linspace(0, 1, original_length)
        x_target = np.linspace(0, 1, target_length)
        interp_func = interp1d(x_original, self.spectra, axis=0, kind='linear')
        self.spectra = interp_func(x_target)
        self.spectraLength = target_length
        return
    
    def plot(self) -> None:
        if self.spectra.shape[1] > 10:
            plot_spectra = self.spectra[:, :10]
        else:
            plot_spectra = self.spectra
        plt.figure()
        plt.plot(range(1,self.spectraLength+1), plot_spectra)
        plt.xlabel("pseudo_wavenumber", fontsize=15)
        plt.ylabel("Intensity", fontsize=15)
        plt.title("Generated skin spectrum", fontsize=15, fontweight='bold')
        # plt.show()
    
    def plot_basis(self) -> None:
        name_list = ["Collagen", "Elastin", "Triolein", "Nucleus", "Keratin", "Ceramide", "Water"]
        plt.figure(figsize=(10, 7))
        for i in range(self.componentNum):
            # plt.subplot(self.componentNum, 1, i+1)
            # plt.title(name_list[i])
            plt.plot(np.linspace(600, 1790, self.spectraLength), self.basis[:,i] + i*1.5, color="black")
        plt.xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=13)
        plt.title("Biophysical Model Basis", fontsize=16)
        plt.yticks([i*1.5 for i in range(self.componentNum)], labels=name_list, fontsize=15)
            # plt.ylabel("Intensity", fontsize=15)
            # plt.title(f"Basis_{i}", fontsize=15, fontweight='bold')
        # plt.show()
    
    def getData(self, saveFlag=None, saveDir=None) -> list:
        out = [self.concentrations, self.spectra]
        if saveFlag is None:
            return out
            
        if saveDir is None:
            saveDir = 'data/generated/'
    
        # Check if the directory exists, if not, create it
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        timestamp = time.strftime("%m%d%Y_%H%M%S")
        filename = f"generated_skin_spectrum_{timestamp}.pkl"
        file_path = os.path.join(saveDir, filename)
        
        # Save the data as a pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(out, f)
        
        return out


if __name__ == "__main__":
    basis = loadmat("data/basics/basis_calibrated.mat")
    basis = basis["basis"]
    given_concentrations = None
    num_to_generate = 1000 # if given_concentrations is provided, you can set this to None
    save_flag = True
    save_path = None # if this is None, the file will be saved to a default path "data/generated/"

    Generator = SkinSimulator(basis=basis)
    Generator.generate(concentrations=given_concentrations, nums=num_to_generate)
    # test_con = [[0.1, 0.8, 0.6, 0.7, 0.3, 0.1, 0.9], [1, 1, 1, 1, 1, 1, 100]]
    # Generator.generate(concentrations=test_con)
    # Generator.plot()
    # Generator.plot_basis()
    # plt.show()
    # plt.savefig("./results/basis.png")

    [concentrations, spectrum] = Generator.getData(saveFlag=save_flag, saveDir=save_path)
    print(spectrum.shape)
