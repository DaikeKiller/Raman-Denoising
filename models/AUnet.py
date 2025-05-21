import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct

def idct_torch(x_dct):
    """
    Compute the IDCT type-II of a batched signal using the inverse FFT.
    """
    return torch_dct.idct(x_dct, norm='ortho')

def dct_torch(x):
    """
    Compute the DCT Type-II of a batched signal using FFT (differentiable).
    """
    return torch_dct.dct(x, norm='ortho')

def generate_poly(coeffs, signal_length):
    """
    Generate a polynomial background curve for each sample using predicted coefficients.
    
    Args:
        coeffs: Tensor of shape (B, 1, N), polynomial coefficients per sample.

    Returns:
        poly_curve: Tensor of shape (B, 1, L)
    """
    B, _, N = coeffs.shape
    L = signal_length  # fixed signal length
    wvn = torch.linspace(600, 1790, L, device=coeffs.device)

    # Create Vandermonde matrix: (L, N) where powers go from N-1 to 0
    vander = torch.stack([wvn**p for p in reversed(range(N))], dim=1)  # (L, N)

    # Expand for batch: (B, L, N)
    vander = vander.unsqueeze(0).repeat(B, 1, 1)

    # Reshape coeffs: (B, N, 1)
    coeffs = coeffs.squeeze(1).unsqueeze(-1)

    # Multiply: (B, L, N) @ (B, N, 1) → (B, L, 1)
    poly_curve = torch.bmm(vander, coeffs)  # (B, L, 1)
    poly_curve = poly_curve.transpose(1, 2)  # (B, 1, L)

    return poly_curve

def fit_modpoly(y_time: torch.Tensor,
                x: torch.Tensor = None,
                poly_order: int = 5,
                threshold: float = 0.05,
                max_iter: int = 25) -> torch.Tensor:
    """
    Iterative robust polynomial baseline removal (ModPoly) for a single polynomial order.
    Returns the baseline curve for each sample.
    """
    B, C, L = y_time.shape
    assert C == 1, "Expected input shape (B,1,L)"
    device = y_time.device

    # build or validate x axis
    if x is None:
        x = torch.linspace(600., 1790., L, device=device)
    else:
        x = x.to(device)
    # Vandermonde basis: powers from poly_order down to 0, shape (L, P)
    P = poly_order + 1
    powers = torch.arange(poly_order, -1, -1, device=device).unsqueeze(0)  # (1, P)
    V0 = x.unsqueeze(1) ** powers                                        # (L, P)
    V = V0.unsqueeze(0).expand(B, L, P)                                   # (B, L, P)
    
    original = y_time.squeeze(1).clone()   # (B, L)
    y_fit    = original.clone()
    prev_norm = None

    for _ in range(max_iter):
        Y = y_fit.unsqueeze(2)                   # (B, L, 1)
        sol = torch.linalg.lstsq(V, Y).solution   # (B, P, 1)
        fit = torch.bmm(V, sol).squeeze(2)        # (B, L)
        resid = y_fit - fit                      # (B, L)
        norm  = resid.norm(dim=1)                # (B,)
        if prev_norm is not None:
            pct_diff = (norm - prev_norm).abs() / norm
            if torch.all(pct_diff < threshold):
                break
        prev_norm = norm
        # mask positive residuals (peaks)
        mask = resid > 0                         # (B, L)
        original[mask] = fit[mask]
        y_fit = original

    return fit.unsqueeze(1)  # (B, 1, L)


def fit_modpoly_range(y_time: torch.Tensor,
                      x: torch.Tensor = None,
                      poly_range: tuple[int, int] = (3, 6),
                      threshold: float = 0.05,
                      max_iter: int = 25):
    """
    Try all polynomial orders in [min_order, max_order] and select the best baseline fit
    by minimal residual sum of squares for each sample independently.

    Args:
        y_time:      Tensor (B,1,L) time-domain signal
        x:           Tensor (L,) wavenumbers or None
        poly_range:  (min_order, max_order) inclusive
        threshold:   convergence criterion for each order
        max_iter:    max iterations per order

    Returns:
        best_poly:   Tensor (B,1,L) best baseline per sample
        best_orders: Tensor (B,) chosen order for each sample
    """
    B, C, L = y_time.shape
    orders = range(poly_range[0], poly_range[1] + 1)
    device = y_time.device

    poly_list = []
    rss_list  = []

    for order in orders:
        baseline = fit_modpoly(
            y_time, x,
            poly_order=order,
            threshold=threshold,
            max_iter=max_iter
        )                                # (B,1,L)
        resid = y_time - baseline        # (B,1,L)
        rss   = resid.pow(2).sum(dim=2)  # (B,1)
        rss = rss.squeeze(1)             # (B,)
        poly_list.append(baseline)
        rss_list.append(rss)

    # Stack: shapes (O, B, 1, L) and (O, B)
    poly_stack = torch.stack(poly_list, dim=0)
    rss_stack  = torch.stack(rss_list,  dim=0)

    # Select best order per sample
    best_idx  = rss_stack.argmin(dim=0)         # (B,)
    batch_idx = torch.arange(B, device=device)
    best_poly = poly_stack[best_idx, batch_idx]  # (B,1,L)

    return best_poly, best_idx

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return self.sigmoid(attention) * x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class AUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AUnet, self).__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.attn1 = nn.Sequential(SpatialAttention(), ChannelAttention(64))
        self.enc2 = DoubleConv(64, 128)
        self.attn2 = nn.Sequential(SpatialAttention(), ChannelAttention(128))
        self.enc3 = DoubleConv(128, 256)
        self.attn3 = nn.Sequential(SpatialAttention(), ChannelAttention(256))
        self.enc4 = DoubleConv(256, 512)
        self.attn4 = nn.Sequential(SpatialAttention(), ChannelAttention(512))
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.attn_dec4 = nn.Sequential(SpatialAttention(), ChannelAttention(512))
        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.attn_dec3 = nn.Sequential(SpatialAttention(), ChannelAttention(256))
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.attn_dec2 = nn.Sequential(SpatialAttention(), ChannelAttention(128))
        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.attn_dec1 = nn.Sequential(SpatialAttention(), ChannelAttention(64))
        
        # Output layer
        self.out_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc1 = self.attn1(enc1)
        enc2 = self.enc2(F.max_pool1d(enc1, kernel_size=2, ceil_mode=True))
        enc2 = self.attn2(enc2)
        enc3 = self.enc3(F.max_pool1d(enc2, kernel_size=2, ceil_mode=True))
        enc3 = self.attn3(enc3)
        enc4 = self.enc4(F.max_pool1d(enc3, kernel_size=2, ceil_mode=True))
        enc4 = self.attn4(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool1d(enc4, kernel_size=2, ceil_mode=True))
        
        # Decoder path
        dec4 = self.upconv4(bottleneck)
        dec4 = F.interpolate(dec4, size=enc4.size(2))  # Ensure matching size for concatenation
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)
        dec4 = self.attn_dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = F.interpolate(dec3, size=enc3.size(2))  # Ensure matching size for concatenation
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)
        dec3 = self.attn_dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = F.interpolate(dec2, size=enc2.size(2))  # Ensure matching size for concatenation
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)
        dec2 = self.attn_dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = F.interpolate(dec1, size=enc1.size(2))  # Ensure matching size for concatenation
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)
        dec1 = self.attn_dec1(dec1)
        
        # Output layer
        out = self.out_conv(dec1)
        return out


class PolyRegressor(nn.Module):
    def __init__(self, signal_length, poly_order=6, feat_ch=64):
        super().__init__()
        self.poly_order = poly_order
        # a few 1×1 convs to turn the (B,1,L) denoised signal into a feature map
        self.backbone = nn.Sequential(
            nn.Conv1d(1, feat_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feat_ch, feat_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # global pooling + FC to output coeffs
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),     # (B, feat_ch, 1)
            nn.Flatten(start_dim=1),     # (B, feat_ch)
            nn.Linear(feat_ch, poly_order+1),  # (B, C)
            nn.Unflatten(1, (1, poly_order+1)) # (B,1,C)
        )
        self.signal_length = signal_length

    def forward(self, x):
        # Optionally you could work in time-domain (idct) here before regressing coeffs
        # but we'll stick in DCT domain as before.
        f = self.backbone(x)          # (B, feat_ch, L)
        coeffs = self.head(f)                    # (B, 1, C)

        poly_raw = generate_poly(coeffs, self.signal_length)   # (B,1,L)
        poly_dct = dct_torch(poly_raw)               # (B,1,L)
        return coeffs, poly_raw, poly_dct


class TwoStageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser = AUnet(1, 1)
        # self.regressor = PolyRegressor(signal_length=signal_length, poly_order=poly_order)
        self.raman_learner = AUnet(2, 1)

    def forward(self, x_dct):
        # Stage 1: denoise
        noise_dct = self.denoiser(x_dct)
        denoised_dct = x_dct - noise_dct # (B,1,L)
        denoised = idct_torch(denoised_dct)            # (B,1,L)

        # # Stage 2: estimate polynomial
        # coeffs, poly_raw, poly_dct = self.regressor(denoised)
        # # (optional) final clean: subtract in DCT or time domain
        # raman_signal = denoised - poly_raw
        # raman_dct  = denoised_dct - poly_dct
        
        fitted_fluo, _ = fit_modpoly_range(denoised, poly_range=(3, 6), threshold=0.05, max_iter=25)
        fitted_fluo_dct = dct_torch(fitted_fluo)  # (B,1,L)
        # next_input = torch.cat((denoised_dct, fitted_fluo_dct), dim=1)  # (B,2,L)
        next_input = torch.cat((denoised, fitted_fluo), dim=1)  # (B,2,L)
        
        # raman_dct = self.raman_learner(next_input)
        # raman_signal = idct_torch(raman_dct)
        raman_signal = self.raman_learner(next_input)
        raman_dct = dct_torch(raman_signal)

        return {
            "denoised_dct": denoised_dct,
            "denoised_signal":     denoised,
            # "coeffs":       coeffs,
            "poly_raw":     fitted_fluo,
            "poly_dct":     fitted_fluo_dct,
            "raman_dct":    raman_dct,
            "raman_signal":   raman_signal
        }


# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(8, 1, 693).to(device)  # Move input to GPU
    signal_length = x.shape[2]
    poly_order = 6
    model = TwoStageModel().to(device)  # Move model to GPU
    # Load pretrained weights for denoiser
    state_dict = torch.load("models/pretrained/new_noise_model_05202025_083413_full_with_fluo_in_signal.pth", map_location=device)
    model.denoiser.load_state_dict(state_dict)
    # Freeze denoiser parameters
    for param in model.denoiser.parameters():
        param.requires_grad = False
    out = model(x)
    print(out["denoised_signal"].shape)  # Should be (8, 1, 693)
    print(out["raman_signal"].shape)  # Should be (8, 1, 693)
