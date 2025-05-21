import numpy as np
import torch
from scipy.fftpack import dct as sp_dct, idct as sp_idct
import matplotlib.pyplot as plt
import torch_dct


if __name__ == "__main__":
    # 1) Generate a random signal
    np.random.seed(0)
    L = 693
    x_np = np.random.randn(32, 1, L).astype(np.float32)
    x_torch = torch.from_numpy(x_np)
    
    # 2) Scipy DCT & IDCT
    x_dct_sp = sp_dct(x_np, type=2, norm='ortho')
    x_rec_sp = sp_idct(x_dct_sp, type=2, norm='ortho')

    # 3) Torch DCT & IDCT
    x_dct_torch = torch_dct.dct(x_torch, norm='ortho')
    x_rec_torch = torch_dct.idct(x_dct_torch, norm='ortho')

    # 4) Plot
    plt.figure()
    plt.plot(x_np[0, 0, :],      label='Original')
    plt.plot(x_rec_torch[0, 0, :], label='Torch IDCT(DCT(x))')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('Original vs Reconstructions')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tmp/zz.jpg')