import os
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
import pandas as pd


def get_res_curve():
    # file + sheet
    file  = 'data/basics/SRM 2246 certified log normal calculations.xls'
    sheet = 'SRM 2246 Example'

    # we want rows 12–3335 (1-based) in cols D and I ⇒ zero-based skiprows=11, nrows=3335-11=3324
    # usecols='D,I' pulls in exactly those two columns
    df = pd.read_excel(
        file,
        sheet_name=sheet,
        header=None,        # no header row in our slice
        skiprows=11,        # drop rows 0–10 (i.e. Excel rows 1–11)
        nrows=3324,         # rows 12 through 3335 inclusive
        usecols='D,I'       # only columns D and I
    )

    # now df.iloc[:,0] is the D12:D3335 range, df.iloc[:,1] is I12:I3335
    wvn_srm = df.iloc[:, 0].to_numpy()
    i_srm   = df.iloc[:, 1].to_numpy()
    # load wvn
    wvn = loadmat("data/wvn_raw.mat")["wvn"].reshape(-1)
    # interp SRM_interp
    i_srm_interp = np.interp(wvn, wvn_srm, i_srm)
    # load NIST data
    nist = np.loadtxt("data/basics/NIST_0.1s.txt")
    dark = np.loadtxt("data/basics/dark_0.1s.txt")
    nist = nist - dark
    # norm NIST data and SRM_interp
    nist = nist / np.max(nist)
    i_srm_interp = i_srm_interp / np.max(i_srm_interp)
    # get res_curve
    out = gaussian_filter1d(i_srm_interp / nist, sigma=15, mode='nearest')
    return out

def resp_cali(data, res_curve):
    data = data * res_curve
    return data[331:]

def pre_processing(base_folder):
    # Define the paths
    folders = ["0.1s", "0.2s", "0.5s", "1s"]
    # output_folder = os.path.join(base_folder, "processed_new_noise_model")

    # # Create the output folder if it doesn't exist
    # os.makedirs(output_folder, exist_ok=True)
    
    # get calibration curve
    res_curve = get_res_curve()

    # Process each folder
    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        # output_subfolder = output_folder
        # os.makedirs(output_subfolder, exist_ok=True)

        # Get all .txt files in the folder and sort them
        txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
        
        data_all = []
        # pipeline
        for i in range(len(txt_files) - 1):
            file_path = os.path.join(folder_path, txt_files[i])

            # Load the data from the files
            data = np.loadtxt(file_path)

            # system resposnse calibration
            data_res = resp_cali(data, res_curve)
            data_all.append(data_res)
            
            # # Save the result to a new file
            # output_file = os.path.join(output_subfolder, txt_files[i])
            # np.savetxt(output_file, result, fmt="%.6f")
        
        # get std
        data_all = np.array(data_all)
        std = np.std(data_all, axis=0)
        folder_path = "data/noise/std"
        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, f"std_{folder}.txt")
        np.savetxt(save_path, std, fmt="%.6f")


# Function to validate if the each wavenumber is a Gaussian
def plot_and_average(output_folder, integration_time, num_samples=100):
    # Get all files for the specified integration time
    files = [f for f in os.listdir(output_folder) if f"dark_{integration_time}_" in f]
    if len(files) < num_samples:
        print(f"Not enough files for {integration_time}. Found {len(files)}, needed {num_samples}.")
        return

    # Randomly select the specified number of files
    selected_files = random.sample(files, num_samples)

    # Load the data and calculate the average
    plt.figure()
    data_list = []
    for i, file in enumerate(selected_files):
        file_path = os.path.join(output_folder, file)
        data = np.loadtxt(file_path)
        data_list.append(data)
        plt.plot(data, label="individuals", color="gray") if i == 0 else plt.plot(data, color="gray") # Plot each selected file with gray color

    # Calculate the average
    avg_data = np.mean(data_list, axis=0)

    # Plot the average
    plt.plot(avg_data, label=f"Average {integration_time}", linewidth=2, color="black")
    plt.title(f"Random {num_samples} files and Average for {integration_time}")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join("tmp", f"average_{integration_time}.png"))
    
    return avg_data

# validate if each wavenumber is a Gaussian
def validate_gaussian(output_folder, integration_time):
    # Get all files for the specified integration time
    files = [f for f in os.listdir(output_folder) if f"dark_{integration_time}_" in f]
    
    data_all = []
    for i, file in enumerate(files):
        file_path = os.path.join(output_folder, file)
        data = np.loadtxt(file_path)
        data_all.append(data)
    data_all = np.array(data_all)
    
    # # randomly select 5 wavenumbers and plot the ditribution
    # if i == 0:  # Only select random indices once
    #     random_indices = random.sample(range(len(data)), 5)  # Randomly select 5 indices
    # for idx in random_indices:
    #     if i == 0:  # Initialize the distribution list for each index
    #         distributions = {index: [] for index in random_indices}
    #     distributions[idx].append(data_all[:, idx])
    # # Plot the distributions for the selected indices
    # plt.figure(figsize=(15, 10))
    # i = 0
    # for idx, values in distributions.items():
    #     plt.subplot(2, 3, i + 1)
    #     plt.hist(values, bins=30, alpha=0.7, label=f"Wavenumber {idx}")
    #     plt.title(f"Wavenumber {idx / 1024 * 1200 + 600: .2f} cm-1")
    #     i += 1
    # plt.savefig(os.path.join("tmp", f"distribution_{integration_time}.png"))
    
    # plot the shpiro test
    W_test = []
    K_test = []
    for i in range(data_all.shape[1]):
        W_test.append(st.shapiro(data_all[:, i])[1])
        K_test.append(st.normaltest(data_all[:, i])[1])
    plt.figure()
    plt.plot(np.linspace(600, 1800, 1024), W_test)
    plt.title(f"Shapiro test p-values for {integration_time}")
    plt.xlabel("Wavenumber")
    plt.ylabel("p-value")
    plt.axhline(y=0.05, color='r', linestyle='--')
    plt.savefig(os.path.join("tmp", f"Shapiro_{integration_time}.png"))
    plt.close()
    plt.figure()
    plt.plot(np.linspace(600, 1800, 1024), K_test)
    plt.title(f"D’Agostino‑Pearson test p-values for {integration_time}")
    plt.xlabel("Wavenumber")
    plt.ylabel("p-value")
    plt.axhline(y=0.05, color='r', linestyle='--')
    plt.savefig(os.path.join("tmp", f"normtest_{integration_time}.png"))
    plt.close()


if __name__ == "__main__":
    # Define the base folder containing the data
    base_folder = "data/noise"
    
    # Pre-process the data
    pre_processing(base_folder)
    
    # Plot and calculate average for each integration time
    # res = []
    # times = ["0.1s", "0.2s", "0.5s", "1s"]
    # for integration_time in times:
    #     avg = plot_and_average(output_folder, integration_time)
    #     res.append(avg)
    # plt.figure()
    # for i, spec in enumerate(res):
    #     plt.plot(spec + 5 * i, label=times[i])  # Plot the average with a slight offset
    # plt.legend()
    # plt.savefig(os.path.join("tmp", "all_averages.png"))
    # # Plot and calculate average for each integration time
    # res = []
    # times = ["0.1s", "0.2s", "0.5s", "1s"]
    # for integration_time in times:
    #     avg = plot_and_average(output_folder, integration_time)
    #     res.append(avg)
    # plt.figure()
    # for i, spec in enumerate(res):
    #     plt.plot(spec + 5 * i, label=times[i])  # Plot the average with a slight offset
    # plt.legend()
    # plt.savefig(os.path.join("tmp", "all_averages.png"))

    # plt.figure()
    # # Validate Gaussian distribution for each integration time
    # for integration_time in times:
    #     validate_gaussian(output_folder, integration_time)
    # plt.savefig(os.path.join("tmp", "distributions.png"))