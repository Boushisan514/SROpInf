import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable

from numpy.typing import ArrayLike, NDArray

from .custom_typing import Vector
from .model import SymmetryReducedQuadraticOpInfROM, duplicate_free_quadratic_vector
from .sample import TrajectoryData

__all__ = ["train_SROpInf"]

def full_to_latent_rhs(dxdt: torch.Tensor, test_basis: torch.Tensor) -> torch.Tensor:
    """
    Project full-state time derivative dx/dt to latent state's derivative da/dt.

    Args:
        dxdt: Full-state derivative. Shape (N_x,) or (N_x, N_t)

    Returns:
        Latent-state derivative. Shape (N_r,) or (N_r, N_t)
    """
    if dxdt.ndim not in (1, 2):
        raise ValueError("Input dxdt must be 1D or 2D.")
    
    return test_basis.T @ dxdt / test_basis.shape[0]

def train_SROpInf(model: SymmetryReducedQuadraticOpInfROM,
                  num_modes: int, data: TrajectoryData, batch_size: int = None, loss_fn: Callable = torch.nn.MSELoss(),
                  reg_coeffs: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  max_steps: int = 1000, grad_tol: float = 1e-6) -> SymmetryReducedQuadraticOpInfROM:
    """
    Trains SR-OpInf model using data from a TrajectoryData object.

    Args:
        model: Instance of SymmetryReducedQuadraticOpInfROM to be trained.
        num_modes: Number of modes for the reduced model.
        data: Instance of TrajectoryData.
        loss_fn: A callable computing the total loss from prediction and targets.
        max_steps: Max number of LBFGS iterations.
        grad_tol: Gradient norm threshold for convergence.
        batch_size: If None, use full-batch optimization.
        reg: Regularization weights (quad, linear, const).
        save_plot_path: Optional path to save the loss plot.
    """

    # Step 1: Extract & preprocess

    test_basis = torch.tensor(model.test_basis, dtype=torch.double)
    u_bias = torch.tensor(model.bias, dtype=torch.double)

    loader = DataLoader(data, batch_size=batch_size or len(data), shuffle=False)
    for batch in loader:
        u_fitted_batch, rhs_fitted_batch, cdot_batch = batch
        u_fitted_batch = u_fitted_batch.T
        rhs_fitted_batch = rhs_fitted_batch.T

        state_batch = test_basis.T @ (u_fitted_batch - u_bias[:, np.newaxis]) / test_basis.shape[0]
        state_quadratic_batch = duplicate_free_quadratic_vector(state_batch)
        rhs_fitted_latent_batch = full_to_latent_rhs(rhs_fitted_batch, test_basis)
        dudx_fitted_batch = model.derivative(u_fitted_batch, order=1)
        dudx_fitted_latent_batch = full_to_latent_rhs(dudx_fitted_batch, test_basis)

    # Convert to tensors (no need to do so perhaps, we already use DataLoader to get tensor data)

    num_quad = state_quadratic_batch.shape[0]

    # Step 2: Initialize trainable matrices

    d_rom = torch.nn.Parameter(torch.zeros(num_modes, dtype=torch.double))
    B_rom = torch.nn.Parameter(torch.zeros(num_modes, num_modes, dtype=torch.double))
    H_rom = torch.nn.Parameter(torch.zeros(num_modes, num_quad, dtype=torch.double))
    e_rom = torch.nn.Parameter(torch.zeros(1, dtype=torch.double))
    p_rom = torch.nn.Parameter(torch.zeros(num_modes, dtype=torch.double))
    q_rom = torch.nn.Parameter(torch.zeros(num_quad, dtype=torch.double))

    trainables = [d_rom, B_rom, H_rom, e_rom, p_rom, q_rom]

    optimizer = torch.optim.LBFGS(trainables, line_search_fn='strong_wolfe', max_iter=10, history_size=20)

    w_rom = torch.tensor(model.w_rom, dtype=torch.double)
    s_rom = torch.tensor(model.s_rom, dtype=torch.double)
    n_rom = torch.tensor(model.n_rom, dtype=torch.double)
    M_rom = torch.tensor(model.M_rom, dtype=torch.double)

    training_loss_history = []

    def closure():
        optimizer.zero_grad()

        rhs_fitted_rom = H_rom @ state_quadratic_batch + B_rom @ state_batch + d_rom.view(-1, 1)
        cdot_rom = - (q_rom @ state_quadratic_batch + p_rom @ state_batch + e_rom) / (s_rom @ state_batch + w_rom)
        dudx_fitted_latent_rom = M_rom @ state_batch + n_rom.view(-1, 1)

        rhs_fitted_loss = torch.sum((rhs_fitted_rom - rhs_fitted_latent_batch) ** 2) # this loss function is then averaged over the number of batchwise snapshots only
        advection_loss = torch.sum((cdot_rom * dudx_fitted_latent_rom - cdot_batch * dudx_fitted_latent_batch) ** 2)

        regularizer = (reg_coeffs[0] * torch.norm(d_rom) ** 2
                       + reg_coeffs[1] * torch.norm(B_rom) ** 2
                       + reg_coeffs[2] * torch.norm(H_rom) ** 2
                       + reg_coeffs[3] * torch.norm(e_rom) ** 2
                       + reg_coeffs[4] * torch.norm(p_rom) ** 2
                       + reg_coeffs[5] * torch.norm(q_rom) ** 2)
        
        loss = rhs_fitted_loss + advection_loss + regularizer
        loss.backward()
        print("velocity loss:", rhs_fitted_loss.item(), "advection loss", advection_loss.item(), "Regularizer loss:", regularizer.item())
        return loss

    # Step 3: Training loop
    for step in range(max_steps):
        loss = optimizer.step(closure)
        training_loss_history.append(loss.item())

        # Gradient norm
        grad_norm = sum(torch.norm(p.grad).item() ** 2 for p in trainables if p.grad is not None) ** 0.5
        print(f"[{step}] Loss = {loss.item():.6e}, Grad norm = {grad_norm:.6e}")
        if grad_norm < grad_tol:
            print(f"Converged at step {step}.")
            break

    # Step 4: Save learned model
    model.constant_vector = d_rom.detach().numpy()
    model.linear_matrix = B_rom.detach().numpy()
    model.quadratic_matrix = H_rom.detach().numpy()
    model.shifting_speed_numer_constant = e_rom.detach().item()
    model.shifting_speed_numer_linear_vector = p_rom.detach().numpy()
    model.shifting_speed_numer_quadratic_vector = q_rom.detach().numpy()

    # Plot loss

    plt.figure()
    plt.semilogy(training_loss_history)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return model, training_loss_history
