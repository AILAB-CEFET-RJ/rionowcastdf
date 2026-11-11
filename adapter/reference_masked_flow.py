"""Small single-GPU/FP32 execution primitives, not a campaign launcher."""
import torch
from adapter.reference_masked_losses import finite


def move_batch(batch, device):
    return {**batch, **{k: batch[k].to(device=device, non_blocking=True)
                       for k in ('input', 'target', 'mask')}}


def train_step(model, optimizer, objective, batch):
    """objective(model, batch) -> LossResult. No gradient accumulation or DDP."""
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        raise RuntimeError('This step supports one process only; global support reduction is required for DDP')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = objective(model, batch)
    if result.loss is None:
        return dict(updated=False, valid_pixels=0, empty_samples=result.empty_samples, loss=None)
    finite(result.loss, 'loss')
    result.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        raise RuntimeError('No model gradients')
    for grad in grads:
        finite(grad, 'gradient')
    optimizer.step()
    for param in model.parameters():
        finite(param, 'parameter after update')
    return dict(updated=True, valid_pixels=result.valid_pixels,
                empty_samples=result.empty_samples, loss=result.loss.detach().item())


@torch.no_grad()
def validation_pass(model, objective, batches):
    """Aggregate sum(error)/sum(valid pixels), never mean(batch means).

    Caller supplies device-ready batches and reproducible diffusion RNG/noise.
    """
    previous = model.training
    total, count, empty, n_batches = 0.0, 0, 0, 0
    try:
        model.eval()
        for batch in batches:
            result = objective(model, batch)
            total += result.numerator.double().item()
            count += result.valid_pixels
            empty += result.empty_samples
            n_batches += 1
    finally:
        model.train(previous)
    if count == 0:
        raise ValueError('Validation has no valid pixels')
    return dict(loss=total/count, valid_pixels=count, empty_samples=empty, batches=n_batches)
