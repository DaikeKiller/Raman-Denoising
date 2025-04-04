import os
import numpy as np
import random
import matplotlib.pyplot as plt

# Define the paths
base_folder = "data/noise"
folders = ["0.1s", "0.2s", "0.5s", "1s"]
output_folder = os.path.join(base_folder, "processed_new_noise_model")

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each folder
for folder in folders:
    folder_path = os.path.join(base_folder, folder)
    output_subfolder = output_folder
    os.makedirs(output_subfolder, exist_ok=True)

    # Get all .txt files in the folder and sort them
    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

    # Subtract consecutive files
    for i in range(len(txt_files) - 1):
        file1_path = os.path.join(folder_path, txt_files[i])
        file2_path = os.path.join(folder_path, txt_files[i + 1]) if i + 1 < len(txt_files) else os.path.join(folder_path, txt_files[0])

        # Load the data from the files
        data1 = np.loadtxt(file1_path)
        data2 = np.loadtxt(file2_path)

        # Subtract the data
        result = data1 - data2

        # Save the result to a new file
        output_file = os.path.join(output_subfolder, txt_files[i])
        np.savetxt(output_file, result, fmt="%.6f")

print("Processing complete. Files saved in 'processed_new_noise_model'.")

# Function to plot and calculate average
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