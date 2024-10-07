import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import pickle
import time

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
            self.concentrations = np.random.rand(self.componentNum, nums)
        else:
            self.concentrations = np.array(concentrations)
            if self.concentrations.shape[0] != self.componentNum:
                Warning("Input concentration needs to have f-by-n shape. f is the \
                              numder of basis spectra, n is the number of spectra to generate. \
                              Got f does not match.")
            if nums is not None:
                if self.concentrations.shape[1] != nums:
                    Warning("The number of spectra (nums) provided does not match the second \
                                dimension of the input concentrations.")
        
        self.spectra = self.basis @ self.concentrations
        return
    
    def plot(self) -> None:
        plt.plot(range(1,self.spectraLength+1), self.spectra)
        plt.xlabel("pseudo_wavenumber", fontsize=15)
        plt.ylabel("Intensity", fontsize=15)
        plt.title("Generated skin spectrum", fontsize=15, fontweight='bold')
        plt.show()
    
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
    num_to_generate = 10000 # if given_concentrations is provided, you can set this to None
    save_flag = True
    save_path = None # if this is None, the file will be saved to a default path "data/generated/"

    Generator = SkinSimulator(basis=basis)
    Generator.generate(concentrations=given_concentrations, nums=num_to_generate)
    # Generator.plot()

    [concentrations, spectrum] = Generator.getData(saveFlag=save_flag, saveDir=save_path)
    print(spectrum.shape)
