"""
Module for training Sparse Autoencoder (SAE) models.
"""

import time
from collections import defaultdict

import torch
from einops import rearrange

from .trackers import DeadCodeTracker

Epsilon = 1e-6


def l2(v, dims=None):
    """
    Compute the L2 norm, across 'dims'.

    Parameters
    ----------
    v : torch.Tensor
        Input tensor.
    dims : tuple, optional
        Dimensions over which to compute the L2 norm, by default None.

    Returns
    -------
    torch.Tensor
        L2 norm of v if dims=None else L2 norm across dims.
    """
    if dims is None:
        return v.square().sum().sqrt()
    return v.square().sum(dims).sqrt()

def r2_score(x, x_hat):
    """
    Compute the R^2 score (coefficient of determination) for the reconstruction.
    A score of 1 indicates a perfect reconstruction while a score of 0 indicates
    that the reconstruction is as good as the mean.

    Parameters
    ----------
    x : torch.Tensor
        Original input tensor of shape (batch_size, d).
    x_hat : torch.Tensor
        Reconstructed input tensor of shape (batch_size, d).

    Returns
    -------
    float
        R^2 score.
    """
    assert x.shape == x_hat.shape, "Input tensors must have the same shape"
    assert len(x.shape) == 2, "Input tensors must be 2D"

    ss_res = torch.mean((x - x_hat) ** 2)
    ss_tot = torch.mean((x - x.mean()) ** 2)

    r2 = 1 - (ss_res / (ss_tot + Epsilon))

    return torch.mean(r2)

def l0_eps(x, dims=None, threshold=1e-6):
    """
    Compute the l0 norm allowing for an epsilon tolerance.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dims : tuple, optional
        Dimensions over which to average the l0 norm, by default all dims.
    threshold : float, optional
        Epsilon tolerance, by default 1e-6.

    Returns
    -------
    torch.Tensor
        Average l0 if dims=None else l0 across dims.
    """
    if dims is None:
        return torch.mean((torch.abs(x) >= threshold).float())
    return torch.mean((torch.abs(x) >= threshold).float(), dims)

def extract_input(batch):
    """
    Extracts the input data from a batch. Supports different batch types:
    - If the batch is a tuple or list, returns the first element.
    - If the batch is a dict, tries to return the value with key 'data' (or 'input').
    - Otherwise, assumes the batch is the input data.
    """
    if isinstance(batch, (tuple, list)):
        return batch[0]
    elif isinstance(batch, dict):
        return batch.get('data')
    else:
        return batch


def _compute_reconstruction_error(x, x_hat):
    """
    Try to match the shapes of x and x_hat to compute the reconstruction error.

    If the input (x) shape is 4D assume it is (n, c, w, h), if it is 3D assume
    it is (n, t, c). Else, assume it is already flattened.

    Concerning the reconstruction error, we want a measure that could be compared
    across different input shapes, so we use the R2 score: 1 means perfect
    reconstruction, 0 means no reconstruction at all.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.

    Returns
    -------
    float
        Reconstruction error.
    """
    if len(x.shape) == 4 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, 'n c w h -> (n w h) c')
    elif len(x.shape) == 3 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, 'n t c -> (n t) c')
    else:
        assert x.shape == x_hat.shape, "Input and output shapes must match."
        x_flatten = x

    r2 = r2_score(x_flatten, x_hat)

    return r2.item()


def _log_metrics(monitoring, logs, model, z, loss, optimizer):
    """
    Log training metrics for the current training step.

    Parameters
    ----------
    monitoring : int
        Monitoring level, for 1 store only basic statistics, for 2 store more detailed statistics.
    logs : defaultdict
        Logs of training statistics.
    model : nn.Module
        The SAE model.
    z : torch.Tensor
        Encoded tensor.
    loss : torch.Tensor
        Loss value.
    optimizer : optim.Optimizer
        Optimizer for updating model parameters.
    """
    if monitoring == 0:
        return

    if monitoring > 0:
        logs['lr'].append(optimizer.param_groups[0]['lr'])
        logs['step_loss'].append(loss.item())

    if monitoring > 1:
        # store directly some z values
        # and the params / gradients norms
        logs['z'].append(z.detach()[::10])
        logs['z_l2'].append(l2(z).item())

        logs['dictionary_sparsity'].append(l0_eps(model.get_dictionary()).mean().item())
        logs['dictionary_norms'].append(l2(model.get_dictionary(), -1).mean().item())

        for name, param in model.named_parameters():
            if param.grad is not None:
                logs[f'params_norm_{name}'].append(l2(param).item())
                logs[f'params_grad_norm_{name}'].append(l2(param.grad).item())


def train_sae(model, dataloader, criterion, optimizer, scheduler=None,
              nb_epochs=20, clip_grad=1.0, monitoring=1, device="cpu"):
    """
    Train a Sparse Autoencoder (SAE) model.

    Parameters
    ----------
    model : nn.Module
        The SAE model to train.
    dataloader : DataLoader
        DataLoader providing the training data.
    criterion : callable
        Loss function.
    optimizer : optim.Optimizer
        Optimizer for updating model parameters.
    scheduler : callable, optional
        Learning rate scheduler. If None, no scheduler is used, by default None.
    nb_epochs : int, optional
        Number of training epochs, by default 20.
    clip_grad : float, optional
        Gradient clipping value, by default 1.0.
    monitoring : int, optional
        Whether to monitor and log training statistics, the options are:
         (0) silent.
         (1) monitor and log training losses.
         (2) monitor and log training losses and statistics about gradients norms and z statistics.
        By default 1.
    device : str, optional
        Device to run the training on, by default 'cpu'.

    Returns
    -------
    defaultdict
        Logs of training statistics.
    """
    logs = defaultdict(list)

    for epoch in range(nb_epochs):
        model.train()

        start_time = time.time()
        epoch_loss = 0.0
        epoch_error = 0.0
        epoch_sparsity = 0.0
        batch_count = 0

        dead_tracker = None

        for batch in dataloader:
            batch_count += 1
            x = extract_input(batch).to(device, non_blocking=True)
            optimizer.zero_grad()

            z_pre, z, x_hat = model(x)
            loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())

            # init dead code tracker with first batch
            if dead_tracker is None:
                dead_tracker = DeadCodeTracker(z.shape[1], device)
            dead_tracker.update(z)

            loss.backward()

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if monitoring:
                epoch_loss += loss.item()
                epoch_error += _compute_reconstruction_error(x, x_hat)
                epoch_sparsity += l0_eps(z, 0).sum().item()
                _log_metrics(monitoring, logs, model, z, loss, optimizer)

        if monitoring and batch_count > 0:
            avg_loss = epoch_loss / batch_count
            avg_error = epoch_error / batch_count
            avg_sparsity = epoch_sparsity / batch_count
            dead_ratio = dead_tracker.get_dead_ratio()
            epoch_duration = time.time() - start_time

            logs['avg_loss'].append(avg_loss)
            logs['r2'].append(avg_error)
            logs['time_epoch'].append(epoch_duration)
            logs['z_sparsity'].append(avg_sparsity)
            logs['dead_features'].append(dead_ratio)

            print(f"Epoch[{epoch+1}/{nb_epochs}], Loss: {avg_loss:.4f}, "
                  f"R2: {avg_error:.4f}, L0: {avg_sparsity:.4f}, "
                  f"Dead Features: {dead_ratio*100:.1f}%, "
                  f"Time: {epoch_duration:.4f} seconds")

    return logs


def train_sae_amp(model, dataloader, criterion, optimizer, scheduler=None,
                  nb_epochs=20, clip_grad=1.0, monitoring=1, device="cuda",
                  max_nan_fallbacks=5):
    """
    Train a Sparse Autoencoder (SAE) model with AMP and NaN fallback.
    Training in fp16 with AMP and fallback to fp32 for the rest of the current
    epoch if NaNs are detected 'max_nan_fallbacks' times. Next epoch will start
    again in fp16.

    Parameters
    ----------
    model : nn.Module
        The SAE model to train.
    dataloader : DataLoader
        DataLoader providing the training data.
    criterion : callable
        Loss function.
    optimizer : optim.Optimizer
        Optimizer for updating model parameters.
    scheduler : callable, optional
        Learning rate scheduler. If None, no scheduler is used, by default None.
    nb_epochs : int, optional
        Number of training epochs, by default 20.
    clip_grad : float, optional
        Gradient clipping value, by default 1.0.
    monitoring : int, optional
        Whether to monitor and log training statistics, the options are:
         (0) silent.
         (1) monitor and log training losses.
         (2) monitor and log training losses and statistics about gradients norms and z statistics.
        By default 1.
    device : str, optional
        Device to run the training on, by default 'cuda'.
    max_nan_fallbacks : int, optional
        Maximum number of NaN fallbacks per epoch before disabling AMP, by default 5.

    Returns
    -------
    defaultdict
        Logs of training statistics.
    """
    scaler = torch.amp.GradScaler(device=device, enabled=True)
    logs = defaultdict(list)

    for epoch in range(nb_epochs):
        model.train()

        start_time = time.time()
        epoch_loss = 0.0
        epoch_error = 0.0
        epoch_sparsity = 0.0
        batch_count = 0

        dead_tracker = None

        nan_fallback_count = 0

        for batch in dataloader:
            batch_count += 1
            x = extract_input(batch).to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device, enabled=True):
                z_pre, z, x_hat = model(x)
                loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())

            # init dead code tracker with first batch
            if dead_tracker is None:
                dead_tracker = DeadCodeTracker(z.shape[1], device)
            dead_tracker.update(z)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_fallback_count += 1
                if monitoring:
                    print(f"[Warning] NaN detected in loss at Epoch {epoch+1}, "
                          f"Iteration {len(logs['step_loss'])+1}. Switching to full precision.")

                with torch.amp.autocast(device_type=device, enabled=False):
                    z_pre, z, x_hat = model(x)
                    loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())
                    dead_tracker.update(z)

                if torch.isnan(loss) or torch.isinf(loss):
                    if monitoring:
                        print(f"[Error] Loss is NaN even in full precision at Epoch {epoch+1}, "
                              f"Iteration {len(logs['step_loss'])+1}. Skipping this batch.")
                    continue  # skip the current batch if NaN persists

                if nan_fallback_count >= max_nan_fallbacks:
                    if monitoring:
                        print(f"[Info] Maximum NaN fallbacks reached at Epoch {epoch+1}. "
                              f"Disabling AMP for the rest of this epoch.")
                    break

            scaler.scale(loss).backward()

            if clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            if monitoring:
                epoch_loss += loss.item()
                epoch_error += _compute_reconstruction_error(x, x_hat)
                epoch_sparsity += l0_eps(z, 0).sum().item()
                _log_metrics(monitoring, logs, model, z, loss, optimizer)

        if monitoring and batch_count > 0:
            avg_loss = epoch_loss / batch_count
            avg_error = epoch_error / batch_count
            avg_sparsity = epoch_sparsity / batch_count
            dead_ratio = dead_tracker.get_dead_ratio()
            epoch_duration = time.time() - start_time

            logs['avg_loss'].append(avg_loss)
            logs['r2'].append(avg_error)
            logs['z_sparsity'].append(avg_sparsity)
            logs['time_epoch'].append(epoch_duration)
            logs['epoch_nan_fallbacks'].append(nan_fallback_count)
            logs['dead_features'].append(dead_ratio)

            print(f"Epoch[{epoch+1}/{nb_epochs}], Loss: {avg_loss:.4f}, "
                  f"R2: {avg_error:.4f}, L0: {avg_sparsity:.4f}, "
                  f"Dead Features: {dead_ratio*100:.1f}%, "
                  f"Time: {epoch_duration:.4f} seconds")

    return logs
