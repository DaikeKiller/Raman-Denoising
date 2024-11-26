import numpy as np
import os
from scipy.fftpack import dct
import matplotlib.pyplot as plt


def read_noise_data_for_visualize(root_folder):
    # Step 1: Load .txt files data
    txt_data = []
    for file in os.listdir(root_folder):
        if file.endswith('.txt'):
            file_path = os.path.join(root_folder, file)
            # Read the data from the file
            with open(file_path, 'r') as f:
                data = f.read().strip().split()  # Adjust based on file format
                data = [float(x) for x in data]  # Convert to float (or int, based on data)
                # data = data[421:]
            # power = np.mean([a ** 2 for a in data])
            # if power > 0:  # Avoid division by zero
            #     data = data / np.sqrt(power)
            txt_data.append(data)  # Add the data to the list
    # Convert list of lists to a NumPy array
    txt_array = np.array(txt_data, dtype=np.float64)
    # Return the train, val, and test sets
    return txt_array

def visualize(noises, num=10, normalize=False, DCT=True):
    noise_100ms = np.random.randint(0, 1000, num)
    noise_200ms = np.random.randint(1000, 2000, num)
    noise_500ms = np.random.randint(2000, 3000, num)
    noise_1000ms = np.random.randint(3000, 4000, num)

    noises_new = noises[np.concatenate([noise_100ms, noise_200ms, noise_500ms, noise_1000ms]), :]

    if normalize:
        for i in range(noises_new.shape[0]):
            noise = noises_new[i, :]
            power = np.mean([a ** 2 for a in noise])
            if power > 0:  # Avoid division by zero
                noises_new[i, :] = noise / np.sqrt(power)
    
    if DCT:
        for i in range(noises_new.shape[0]):
            noises_new[i, :] = dct(noises_new[i, :], norm='ortho')
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(np.transpose(noises_new[:num, :]))
    plt.title("100ms")
    plt.subplot(2,2,2)
    plt.plot(np.transpose(noises_new[num:2*num, :]))
    plt.title("200ms")
    plt.subplot(2,2,3)
    plt.plot(np.transpose(noises_new[2*num:3*num, :]))
    plt.title("500ms")
    plt.subplot(2,2,4)
    plt.plot(np.transpose(noises_new[3*num:, :]))
    plt.title("1000ms")

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(np.mean(noises_new[:num, :], axis=0))
    plt.title("100ms")
    plt.subplot(2,2,2)
    plt.plot(np.mean(noises_new[num:2*num, :], axis=0))
    plt.title("200ms")
    plt.subplot(2,2,3)
    plt.plot(np.mean(noises_new[2*num:3*num, :], axis=0))
    plt.title("500ms")
    plt.subplot(2,2,4)
    plt.plot(np.mean(noises_new[3*num:, :], axis=0))
    plt.title("1000ms")



if __name__ == "__main__":
    noise_dir = "data/noise/processed_new"
    noises = read_noise_data_for_visualize(noise_dir)
    visualize(noises=noises, num=10, normalize=True, DCT=False)
    visualize(noises=noises, num=10, normalize=True, DCT=True)
    plt.show()
    