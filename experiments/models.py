import math

import torch
from torch import nn
import torch.nn.functional as F


class PriorCNN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, 3, 3, padding=1))

    def forward(self, y_image):
        return (y_image + self.net(y_image)).clamp(0, 1)


class Block(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(incoming, outgoing, 3, padding=1),
                                 nn.GroupNorm(4, outgoing), nn.SiLU(),
                                 nn.Conv2d(outgoing, outgoing, 3, padding=1),
                                 nn.GroupNorm(4, outgoing), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class ConditionalDenoiser(nn.Module):
    """Small pilot U-Net, explicitly trained with null and spatially mixed conditions."""
    def __init__(self, width=32):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.width = width
        self.first = Block(10, width)  # noisy image, y, mixed CNN condition, reliability weight
        self.down = Block(width, width * 2)
        self.up = Block(width * 3, width)
        self.output = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, x, t, y_image, condition, weight):
        f = torch.exp(-math.log(10000) * torch.arange(self.width//2, device=x.device) / (self.width//2-1))
        emb = t.float()[:, None] * f[None]
        emb = self.time(torch.cat((emb.sin(), emb.cos()), dim=1))
        first = self.first(torch.cat((x, y_image * 2-1, condition * 2-1, weight), 1))
        first = first + emb[..., None, None]
        lower = self.down(F.avg_pool2d(first, 2))
        up = F.interpolate(lower, size=first.shape[-2:], mode='nearest')
        return self.output(self.up(torch.cat((up, first), 1)))


def alpha_schedule(steps, device):
    s = torch.linspace(0, 1, steps+1, device=device)
    abar = torch.cos((s+.008)/1.008 * math.pi/2).square()
    abar /= abar[0].clone()
    beta = (1 - abar[1:] / abar[:-1]).clamp(.0001, .999)
    return beta, torch.cumprod(1-beta, dim=0)
