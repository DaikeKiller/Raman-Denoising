import os
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as st

# Define the paths
base_folder = "data/noise"
folders = ["0.1s", "0.2s", "0.5s", "1s"]
output_folder = os.path.join(base_folder, "processed_new_noise_model")

# # Create the output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Process each folder
# for folder in folders:
#     folder_path = os.path.join(base_folder, folder)
#     output_subfolder = output_folder
#     os.makedirs(output_subfolder, exist_ok=True)

#     # Get all .txt files in the folder and sort them
#     txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

#     # Subtract consecutive files
#     for i in range(len(txt_files) - 1):
#         file1_path = os.path.join(folder_path, txt_files[i])
#         file2_path = os.path.join(folder_path, txt_files[i + 1]) if i + 1 < len(txt_files) else os.path.join(folder_path, txt_files[0])

#         # Load the data from the files
#         data1 = np.loadtxt(file1_path)
#         data2 = np.loadtxt(file2_path)

#         # Subtract the data
#         result = data1 - data2

#         # Save the result to a new file
#         output_file = os.path.join(output_subfolder, txt_files[i])
#         np.savetxt(output_file, result, fmt="%.6f")

# print("Processing complete. Files saved in 'processed_new_noise_model'.")

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

# Plot and calculate average for each integration time
res = []
times = ["0.1s", "0.2s", "0.5s", "1s"]
for integration_time in times:
    avg = plot_and_average(output_folder, integration_time)
    res.append(avg)
plt.figure()
for i, spec in enumerate(res):
    plt.plot(spec + 5 * i, label=times[i])  # Plot the average with a slight offset
plt.legend()
plt.savefig(os.path.join("tmp", "all_averages.png"))

plt.figure()
# Validate Gaussian distribution for each integration time
for integration_time in times:
    validate_gaussian(output_folder, integration_time)
plt.savefig(os.path.join("tmp", "distributions.png"))