import torch
from torch.utils.data import DataLoader
from torch import Tensor
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple

from .model import SymmetryReducedBilinearROM
from .sample import TrajectoryData

__all__ = ["train_SROpInf"]

def reconstruct_symmetric_rank3matrix(H_unique: Tensor, nmodes: int) -> Tensor:
    """
    Reconstructs full H[i,j,k] tensor from H_unique[i,:], where each H[i,:,:] is symmetric.
    H_unique: shape (n, n(n+1)//2)
    Returns H: shape (n, n, n) with H[i,j,k] = H[i,k,j]
    """
    H = torch.zeros((nmodes, nmodes, nmodes), dtype=H_unique.dtype, device=H_unique.device)
    triu_i, triu_j = torch.triu_indices(nmodes, nmodes)  # indices of upper triangle

    for i in range(nmodes):
        H[i, triu_i, triu_j] = H_unique[i]
        H[i, triu_j, triu_i] = H_unique[i]  # fill symmetric lower triangle

    return H

def reconstruct_symmetric_matrix(Q_unique: Tensor, nmodes: int) -> Tensor:
    """
    Reconstruct symmetric Q of shape (n, n) from its duplicate-free form (n(n+1)/2,).
    """
    Q = torch.zeros((nmodes, nmodes), dtype=Q_unique.dtype, device=Q_unique.device)

    tril_indices = torch.triu_indices(nmodes, nmodes)  # upper triangle
    Q[tril_indices[0], tril_indices[1]] = Q_unique
    Q = Q + Q.T - torch.diag(Q.diag())  # symmetrize
    return Q

def train_SROpInf(model: SymmetryReducedBilinearROM,
                  num_modes: int, data: TrajectoryData, batch_size: int = None, loss_fn: Callable = torch.nn.MSELoss(),
                  reg_coeffs: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ratio: float = 1.0,
                  max_steps: int = 1000, grad_tol: float = 1e-6) -> SymmetryReducedBilinearROM:
    """
    Trains SR-OpInf model using data from a TrajectoryData object.

    Args:
        model: Instance of SymmetryReducedBilinearROM to be trained.
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
    model.cdot_denom_constant = torch.tensor(model.cdot_denom_constant, dtype=torch.double)
    model.cdot_denom_linear = torch.tensor(model.cdot_denom_linear, dtype=torch.double)
    model.udx_constant = torch.tensor(model.udx_constant, dtype=torch.double).view(-1, 1)
    model.udx_linear = torch.tensor(model.udx_linear, dtype=torch.double)

    loader = DataLoader(data, batch_size=batch_size or len(data), shuffle=False)

    for batch in loader:
        u_fitted_batch, rhs_fitted_batch, cdot_batch = batch
        u_fitted_batch = u_fitted_batch.T
        rhs_fitted_batch = rhs_fitted_batch.T

        # state_batch = test_basis.T @ (u_fitted_batch - u_bias[:, np.newaxis]) / test_basis.shape[0]
        state_batch = model.full_to_latent(u_fitted_batch)
        # state_quadratic_batch = duplicate_free_quadratic_vector(state_batch)
        # rhs_fitted_latent_batch = full_to_latent_rhs(rhs_fitted_batch, test_basis)
        rhs_fitted_latent_batch = model.full_to_latent_rhs(rhs_fitted_batch)
        dudx_fitted_batch = model.derivative(u_fitted_batch, order=1)
        dudx_fitted_latent_batch = model.full_to_latent_rhs(dudx_fitted_batch)

    # Convert to tensors (no need to do so perhaps, we already use DataLoader to get tensor data)

    num_quadratic_unique = num_modes * (num_modes + 1) // 2
    
    # Step 2: Initialize trainable matrices

    d_rom = torch.nn.Parameter(torch.zeros(num_modes, dtype=torch.double))
    A_rom = torch.nn.Parameter(torch.zeros(num_modes, num_modes, dtype=torch.double))
    B_rom_duplicate_free = torch.nn.Parameter(torch.zeros(num_modes, num_quadratic_unique, dtype=torch.double))
    e_rom = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.double))
    p_rom = torch.nn.Parameter(torch.zeros(num_modes, dtype=torch.double))
    Q_rom_duplicate_free = torch.nn.Parameter(torch.zeros(num_quadratic_unique, dtype=torch.double))

    trainables = [d_rom, A_rom, B_rom_duplicate_free, e_rom, p_rom, Q_rom_duplicate_free]

    def closure():
        for p in trainables:
            if p.grad is not None:
                p.grad.zero_()

        model.constant = d_rom.view(-1, 1)  # Ensure d_rom can be broadcasted correctly
        model.linear_mat = A_rom
        model.bilinear_mat = reconstruct_symmetric_rank3matrix(B_rom_duplicate_free, num_modes)
        model.cdot_numer_constant = e_rom
        model.cdot_numer_linear = p_rom
        model.cdot_numer_bilinear = reconstruct_symmetric_matrix(Q_rom_duplicate_free, num_modes)

        rhs_fitted_rom = model.velocity(state_batch)
        cdot_rom       = model.shifting_speed(state_batch)
        dudx_fitted_latent_rom = model.udx(state_batch)

        rhs_fitted_loss = torch.sum((rhs_fitted_rom - rhs_fitted_latent_batch) ** 2) # this loss function is then averaged over the number of batchwise snapshots only
        advection_loss = torch.sum((cdot_rom * dudx_fitted_latent_rom - cdot_batch * dudx_fitted_latent_batch) ** 2)

        regularizer = (reg_coeffs[0] * torch.norm(d_rom) ** 2
                       + reg_coeffs[1] * torch.norm(A_rom) ** 2
                       + reg_coeffs[2] * torch.norm(B_rom_duplicate_free) ** 2
                       + reg_coeffs[3] * torch.norm(e_rom) ** 2
                       + reg_coeffs[4] * torch.norm(p_rom) ** 2
                       + reg_coeffs[5] * torch.norm(Q_rom_duplicate_free) ** 2)

        loss = rhs_fitted_loss + ratio * advection_loss + regularizer
        loss.backward()
        print("velocity loss:", rhs_fitted_loss.item(),
              "advection loss", advection_loss.item(),
              "Regularizer loss:", regularizer.item())
        return loss
    
    optimizer = ConjugateGradientOptimizer(params=trainables, closure=closure, max_steps=max_steps, grad_tol=grad_tol)
    training_loss_history = optimizer.step()
    
    # Step 4: Save learned model
    model.constant = d_rom.detach().numpy()
    model.linear_mat = A_rom.detach().numpy()
    model.bilinear_mat = reconstruct_symmetric_rank3matrix(B_rom_duplicate_free.detach(), num_modes).numpy()
    model.cdot_numer_constant = e_rom.detach().item()
    model.cdot_numer_linear = p_rom.detach().numpy()
    model.cdot_numer_bilinear = reconstruct_symmetric_matrix(Q_rom_duplicate_free.detach(), num_modes).numpy()
    model.cdot_denom_constant = model.cdot_denom_constant.item()
    model.cdot_denom_linear = model.cdot_denom_linear.detach().numpy()
    model.udx_constant = model.udx_constant.detach().numpy().flatten()
    model.udx_linear = model.udx_linear.detach().numpy()

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

class ConjugateGradientOptimizer:
    def __init__(self,
                 params: List[torch.nn.Parameter],
                 closure: Callable[[], torch.Tensor],
                 max_steps: int = 1000,
                 grad_tol: float = 1e-6):
        
        self.params = params
        self.closure = closure
        self.max_steps = max_steps
        self.grad_tol = grad_tol

    def flatten_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.params])

    def flatten_grads(self) -> torch.Tensor:
        return torch.cat([
            p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1)
            for p in self.params
        ])

    def set_params_from_flat(self, flat_tensor: torch.Tensor):
        pointer = 0
        for p in self.params:
            numel = p.numel()
            p.data.copy_(flat_tensor[pointer:pointer + numel].view_as(p))
            pointer += numel

    def loss_and_grad(self, flat_params: torch.Tensor) -> Tuple[float, torch.Tensor]:
        self.set_params_from_flat(flat_params)
        loss = self.closure()
        grad = self.flatten_grads()
        return loss.item(), grad

    def step(self):
        # Flatten and detach current model parameters
        x_k = self.flatten_params().clone().detach()
        self.set_params_from_flat(x_k)

        # Evaluate initial gradient and residual
        loss, g_k = self.loss_and_grad(x_k)
        r_k = -g_k.clone()
        p_k = r_k.clone()
        training_loss_history = [loss]

        for k in range(self.max_steps):
            # Use closure again to evaluate matrix-vector product approximation
            # We compute A^T A p_k ≈ grad(x_k + ε p_k) - grad(x_k)
            epsilon = 1e-8
            self.set_params_from_flat(x_k + epsilon * p_k)
            _, grad_eps = self.loss_and_grad(x_k + epsilon * p_k)

            self.set_params_from_flat(x_k)
            _, grad_ref = self.loss_and_grad(x_k)

            # Approximate H p_k = (grad(x + εp) - grad(x)) / ε
            Hp_k = (grad_eps - grad_ref) / epsilon

            alpha_k = torch.dot(r_k, r_k) / torch.dot(p_k, Hp_k)
            x_k = x_k + alpha_k * p_k
            self.set_params_from_flat(x_k)

            loss, _ = self.loss_and_grad(x_k)
            training_loss_history.append(loss)

            r_next = r_k - alpha_k * Hp_k
            grad_norm = torch.norm(r_next).item()

            print(f"[{k}] Loss = {loss:.6e}, Grad norm = {grad_norm:.6e}")
            if grad_norm < self.grad_tol:
                print("Converged.")
                break

            beta_k = torch.dot(r_next, r_next) / torch.dot(r_k, r_k)
            p_k = r_next + beta_k * p_k
            r_k = r_next

        return training_loss_history
