import torch
import torch.nn.functional as F


def gaussian_kernel(sigma, device, dtype):
    radius = max(1, int(3 * sigma + .5))
    a = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-a.square() / (2 * sigma**2))
    k /= k.sum()
    return k[:, None] * k[None, :]


def blur(x, sigma=1.2):
    k = gaussian_kernel(sigma, x.device, x.dtype)
    weights = k.expand(x.shape[1], 1, *k.shape)
    # Circular boundaries make this symmetric convolution its exact adjoint.
    pad = k.shape[0] // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode='circular'), weights, groups=x.shape[1])


class Operator:
    def __init__(self, task, sigma=1.2, factor=4):
        self.task, self.sigma, self.factor = task, sigma, factor

    def forward(self, x):
        if self.task == 'gaussian_blur':
            return blur(x, self.sigma)
        return F.avg_pool2d(x, self.factor)

    def adjoint(self, y):
        if self.task == 'gaussian_blur':
            return blur(y, self.sigma)
        return y.repeat_interleave(self.factor, -2).repeat_interleave(self.factor, -1) / self.factor**2

    def backprojection(self, y):
        if self.task == 'gaussian_blur':
            return y
        return F.interpolate(y, scale_factor=self.factor, mode='nearest')

    def observe(self, x, noise_sigma, seed):
        y = self.forward(x)
        g = torch.Generator(device=x.device).manual_seed(seed)
        # No clipping: keep the additive Gaussian observation model exact.
        return y + noise_sigma * torch.randn(y.shape, generator=g, device=y.device)

    def correction(self, x, y, step=.2):
        return x - step * self.adjoint(self.forward(x) - y)
