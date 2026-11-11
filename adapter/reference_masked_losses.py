"""FP32, single-process losses for reference code/50 targets.

Native call signatures: regression(zeros, input); denoiser(latent, condition, sigma).
No augmentation or additional random patching is performed.
"""
from dataclasses import dataclass
import math
import torch


@dataclass
class LossResult:
    numerator: torch.Tensor
    valid_pixels: int
    empty_samples: int

    @property
    def loss(self):
        return None if self.valid_pixels == 0 else self.numerator / self.valid_pixels


def finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f'Non-finite {name}')


def validate_batch(batch):
    x, y, m = (batch[k] for k in ('input', 'target', 'mask'))
    if x.ndim != 4 or y.ndim != 4 or y.shape[1] != 1 or m.shape != y.shape:
        raise ValueError('Expected input BCHW and target/mask B1HW')
    if x.shape[0] != y.shape[0] or x.shape[2:] != y.shape[2:] or y.numel() == 0:
        raise ValueError('Incompatible or empty batch shapes')
    if x.dtype != torch.float32 or y.dtype != torch.float32 or m.dtype != torch.bool:
        raise ValueError('Expected float32 inputs/targets and boolean mask')
    if x.device != y.device or m.device != y.device:
        raise ValueError('Batch tensors must share a device')
    finite(x, 'conditioning'); finite(y[m], 'valid targets')
    # Do not multiply by mask: NaN * 0 remains NaN.
    return x, torch.where(m, y, 0.0), m


def masked_squared_error(prediction, target, mask, weight=None):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('Prediction, target and mask must have identical shapes')
    if mask.dtype != torch.bool or prediction.dtype != torch.float32 or target.dtype != torch.float32:
        raise ValueError('Expected float32 prediction/target and boolean mask')
    if prediction.device != target.device or mask.device != target.device:
        raise ValueError('Loss tensors must share a device')
    finite(prediction, 'prediction'); finite(target[mask], 'valid targets')
    error = (prediction[mask] - target[mask]).square()
    if weight is not None:
        if weight.shape != (target.shape[0], 1, 1, 1) or weight.device != target.device:
            raise ValueError('Weight must have shape B111 on target device')
        finite(weight, 'EDM weights')
        if (weight <= 0).any():
            raise ValueError('Weights must be positive')
        error = error * weight.expand_as(target)[mask]
    numerator = error.sum()
    finite(numerator, 'loss numerator')
    empty = int((~mask.flatten(1).any(1)).sum().item())
    return LossResult(numerator, int(mask.sum().item()), empty)


def empty_result(y, m):
    return LossResult(y.new_zeros(()), 0, m.shape[0])


def regression_loss(net, batch):
    x, y, m = validate_batch(batch)
    if not m.any():
        return empty_result(y, m)
    prediction = net(torch.zeros_like(y), x)
    return masked_squared_error(prediction, y, m)


class MaskedResidualEDMLoss:
    """Masked clean residual + full-domain independent Gaussian corruption.

    Regression must already be frozen and in eval mode. Invalid targets never
    enter residual, mean conditioning, or the denoiser. sigma/noise may be supplied
    for reproducible validation; noise denotes unit normal epsilon, not sigma*epsilon.
    """
    def __init__(self, *, p_mean=-1.2, p_std=1.2, sigma_data=0.5, hr_mean_conditioning=False):
        if not all(math.isfinite(v) for v in (p_mean, p_std, sigma_data)) or p_std < 0 or sigma_data <= 0:
            raise ValueError('Invalid EDM parameters')
        self.p_mean, self.p_std, self.sigma_data = p_mean, p_std, sigma_data
        self.hr_mean_conditioning = hr_mean_conditioning

    def __call__(self, net, regression_net, batch, *, generator=None, sigma=None, noise=None):
        x, y, m = validate_batch(batch)
        if regression_net.training or any(p.requires_grad for p in regression_net.parameters()):
            raise ValueError('Regression must be eval() and requires_grad_(False)')
        if not m.any():
            return empty_result(y, m)
        with torch.no_grad():
            mean = regression_net(torch.zeros_like(y), x)
        if mean.shape != y.shape or mean.dtype != y.dtype or mean.device != y.device:
            raise ValueError('Invalid regression output schema')
        finite(mean, 'regression mean')
        residual = torch.where(m, y - mean, 0.0)
        shape = (y.shape[0], 1, 1, 1)
        if sigma is None:
            sigma = (torch.randn(shape, device=y.device, generator=generator) * self.p_std + self.p_mean).exp()
        if sigma.shape != shape or sigma.device != y.device or sigma.dtype != torch.float32:
            raise ValueError('sigma must be float32 B111 on target device')
        finite(sigma, 'sigma')
        if (sigma <= 0).any():
            raise ValueError('sigma must be positive')
        if noise is None:
            noise = torch.randn(y.shape, device=y.device, generator=generator)
        if noise.shape != y.shape or noise.device != y.device or noise.dtype != torch.float32:
            raise ValueError('noise must match target schema')
        finite(noise, 'noise')
        condition = torch.cat((mean, x), dim=1) if self.hr_mean_conditioning else x
        latent = residual + sigma * noise
        finite(latent, 'noisy residual')
        weight = (sigma.square() + self.sigma_data**2) / (sigma * self.sigma_data).square()
        prediction = net(latent, condition, sigma)
        return masked_squared_error(prediction, residual, m, weight)
