import pickle
from typing import Any, Callable, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import Dataset

from .custom_typing import Vector, VectorField
from .model import BilinearModel, SymmetryReducedBilinearROM

__all__ = ["POD", "template_fitting", "sample"]

def POD(sol: ArrayLike, num_modes: int) -> Tuple[Vector, ArrayLike, Vector]:
    """
    Perform Proper Orthogonal Decomposition (POD) on the solution array.

    u = ubar + sum_{i=1}^{num_modes} a_i(t) * phi_i(x)

    Args:
        sol: Solution array of shape (Nx, Ns, Ntraj).
        num_modes: Number of modes to retain.

    Returns:
        Tuple containing:
        - sol_mean: Mean field (Nx,)
        - basis: POD modes (Nx, num_modes)
        - S: Singular values (num_modes,)
    """
    sol = np.reshape(sol, (sol.shape[0], -1))  # (Nx, Ns * Ntraj)
    sol_mean = np.mean(sol, axis=1)            # (Nx,)
    sol_centered = sol - sol_mean[:, np.newaxis]

    U, S, _ = np.linalg.svd(sol_centered, full_matrices=False)
    basis = U[:, :num_modes] * np.sqrt(sol.shape[0])  # (Nx, num_modes)

    return sol_mean, basis, S[:num_modes]

def template_fitting(sol: ArrayLike, sol_template: Vector, L: float, N: int, spatial_translation: Callable[[Vector, float], Vector], shifting_amount_old: float = None) -> ArrayLike:

        if sol.ndim == 1 or sol.shape[1] == 1:
            if shifting_amount_old is None: # this means we are fitting the initial condition, we need a larger search range
                shifting_amount_range = np.linspace(-L/2, L/2, 10000 * N, endpoint=False)
                minimal_error = 1e5
                for c in shifting_amount_range:
                    sol_tmp = spatial_translation(sol, c) # from u = uhat(x - c), then uhat = u(x + c)
                    error = np.linalg.norm(sol_tmp - sol_template) # take the initial condition as the template
                    # if this is a perfect TW, then the shifted state/rhs at the new timestep should equal to the previous state/rhs
                    if error < minimal_error:
                        minimal_error = error
                        shifting_amount = c
                return spatial_translation(sol, shifting_amount), shifting_amount
            else: # this means we are fitting the following snapshots, we only need to modify the old shifting amount by a small amount
                dx = L / N
                shifting_amount_step_range = np.linspace(-1 * dx, 1 * dx, 10 * N, endpoint=False)
                minimal_error = 1e5
                for c in shifting_amount_step_range:
                    sol_tmp = spatial_translation(sol, c + shifting_amount_old)
                    error = np.linalg.norm(sol_tmp - sol_template) # take the initial condition as the template
                    # if this is a perfect TW, then the shifted state/rhs at the new timestep should equal to the previous state/rhs
                    if error < minimal_error:
                        minimal_error = error
                        shifting_amount = c + shifting_amount_old
                return spatial_translation(sol, shifting_amount), shifting_amount
            
        else:
            num_snapshots = sol.shape[1]
            dx = L / N

            shifting_amount = np.zeros(num_snapshots)
            shifting_amount_range = np.linspace(-L/2, L/2, 10000 * N, endpoint=False)
            shifting_amount_step_range = np.linspace(-1 * dx, 1 * dx, 10 * N, endpoint=False)

            sol_fitted = np.zeros((N, num_snapshots))

            minimal_error = 1e5

            for c in shifting_amount_range:

                sol_snapshot_tmp = spatial_translation(sol[:, 0], c)

                error = np.linalg.norm(sol_snapshot_tmp - sol_template) # take the initial condition as the template
                # if this is a perfect TW, then the shifted state/rhs at the new timestep should equal to the previous state/rhs

                if error < minimal_error:
                    minimal_error = error
                    shifting_amount[0] = c

            sol_fitted[:, 0] = spatial_translation(sol[:, 0], shifting_amount[0])

            for time in range(1, num_snapshots):

                minimal_error = 1e5

                shifting_amount[time] = shifting_amount[time - 1]

                for c in shifting_amount_step_range:

                    sol_snapshot_tmp = spatial_translation(sol[:, time], shifting_amount[time - 1] + c)

                    error = np.linalg.norm(sol_snapshot_tmp - sol_template) # take the initial condition as the template
                    # if this is a perfect TW, then the shifted state/rhs at the new timestep should equal to the previous state/rhs

                    if error < minimal_error:
                        minimal_error = error
                        shifting_amount[time] = shifting_amount[time - 1] + c

                sol_fitted[:, time] = spatial_translation(sol[:, time], shifting_amount[time])

            return sol_fitted, shifting_amount

def compute_shifting_speed_FOM(rhs_fitted: Vector, u_fitted_dx: Vector, u_template_dx: Vector) -> float:
    """Compute the shifting speed for the FOM."""
    return - np.dot(rhs_fitted, u_template_dx) / np.dot(u_fitted_dx, u_template_dx)
class TrajectoryData(Dataset[Tuple[NDArray, NDArray, NDArray, float, float]]):
    """
    Container for time-flattened trajectory data for operator inference.

    Each sample is one time step from one trajectory.
    Returns:
        (sol[:, t, k], sol_fitted[:, t, k], rhs_fitted[:, t, k], cdot, c)

    Data shape expectations:
    - sol: (Nx, Ns, Ntraj)
    - sol_fitted: (Nx, Ns, Ntraj)
    - rhs_fitted: (Nx, Ns, Ntraj)
    - shifting_speed: (Ns, Ntraj)
    - shifting_amount: (Ns, Ntraj)
    """

    def __init__(
        self,
        sol: ArrayLike,
        sol_fitted: ArrayLike,
        rhs_fitted: ArrayLike,
        shifting_speed: ArrayLike,
        shifting_amount: ArrayLike,
    ):
        self.sol = np.array(sol)
        self.sol_fitted = np.array(sol_fitted)
        self.rhs_fitted = np.array(rhs_fitted)
        self.shifting_speed = np.array(shifting_speed)
        self.shifting_amount = np.array(shifting_amount)

        self.Nx, self.Ns, self.Ntraj = self.sol.shape
        self.total_num_samples = self.Ns * self.Ntraj

    def __len__(self) -> int:
        return self.total_num_samples

    def __getitem__(self, i: int) -> Tuple[Vector, Vector, float]:
        # Convert linear index i into (t, traj)
        t = i % self.Ns
        k = i // self.Ns
        return (self.sol_fitted[:, t, k], self.rhs_fitted[:, t, k], self.shifting_speed[t, k])

    def save(self, fname: str) -> None:
        with open(fname, "wb") as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, fname: str) -> "TrajectoryData":
        with open(fname, "rb") as fp:
            return pickle.load(fp)

def sample(num_traj: int, num_snapshots: int, u_init: ArrayLike, u_template: Vector,
    spatial_translation: Callable[[Vector, float], Vector],
    model: Union[BilinearModel, SymmetryReducedBilinearROM],
    timestepper: str, timespan: float, timestep: float, err_tol: float = 1e-6,
    re_proj_option = False, V: ArrayLike = None, W: ArrayLike = None, bias: Vector = None) -> TrajectoryData:
    """
    Sample num_traj trajectories each with length n from num_traj initial states
    Returns a TrajectoryList object

    x_init is of shape (Nx, num_traj)
    traj_list is of shape (Nx, num_snapshots, num_traj)
    """
    L = spatial_translation.Lx  # Access the custom attribute for the domain length
    Nx = u_init.shape[0]
    num_timesteps = int(timespan / timestep) + 1

    sample_interval = int((num_timesteps - 1) / (num_snapshots - 1))
    sol = np.zeros((Nx, num_snapshots, num_traj), dtype=u_init.dtype)
    sol_fitted = np.zeros_like(sol, dtype=u_init.dtype)
    rhs = np.zeros_like(sol, dtype=u_init.dtype)
    rhs_fitted = np.zeros_like(sol, dtype=u_init.dtype)
    shifting_speed = np.zeros((num_snapshots, num_traj), dtype=float)
    shifting_amount = np.zeros((num_snapshots, num_traj), dtype=float)

    if isinstance(model, SymmetryReducedBilinearROM):
        stepper = model.get_timestepper(method = timestepper, dt = timestep, err_tol = err_tol)
        for idx_traj in range(num_traj):
            print(f"Sampling trajectory {idx_traj + 1} of {num_traj} using {model.__class__.__name__} ...")
            u_init_fitted, shifting_amount[0, idx_traj] = template_fitting(u_init[:, idx_traj], u_template, L, Nx, spatial_translation)
            state = model.full_to_latent(u_init_fitted)  # convert the initial condition to the latent space
            u_init_fitted_proj = model.latent_to_full(state)  # re-project the initial condition to the latent space
            u_init_proj = spatial_translation(u_init_fitted_proj, -shifting_amount[0, idx_traj])  # apply the spatial translation to the re-projected initial condition

            sol[:, 0, idx_traj] = u_init_proj
            sol_fitted[:, 0, idx_traj] = u_init_fitted_proj
            shifting_speed[0, idx_traj] = model.shifting_speed(state)

            timestep_old = timestep
            state_old = state
            c_old = shifting_amount[0, idx_traj]
            cdot_old = shifting_speed[0, idx_traj]
            c_new = c_old
            cdot_new = cdot_old
            time = 0
            counter = 1
            dt_snapshots = timestep * sample_interval

            print(f"{model.__class__.__name__} trajectory {idx_traj + 1} of {num_traj}, snapshot {counter}/{num_snapshots} t = {time}, dt = {timestep}, cdot_numer = {model.cdot_numerator}, cdot_denom = {model.cdot_denominator}, cdot = {cdot_old}, c = {c_old}.")
            while time <= timespan:
                
                if int(time / dt_snapshots) >= counter and int(time / dt_snapshots) <= counter + 1:
                    
                    # if the timestep of state_old is closer to the sample moment than the timestep of state, then we keep state_old, otherwise we keep state
                    distance_old_moment = np.abs(time - counter * dt_snapshots)
                    distance_new_moment = np.abs(time - (counter + 1) * dt_snapshots)

                    if distance_old_moment <= distance_new_moment:
                        sol_fitted[:, counter, idx_traj] = model.latent_to_full(state_old)
                        sol[:, counter, idx_traj] = spatial_translation(sol_fitted[:, counter, idx_traj], -c_old)
                        shifting_amount[counter, idx_traj] = c_old
                        shifting_speed[counter, idx_traj] = cdot_old
                    else:
                        sol_fitted[:, counter, idx_traj] = model.latent_to_full(state)
                        sol[:, counter, idx_traj] = spatial_translation(sol_fitted[:, counter, idx_traj], -c_new)
                        shifting_amount[counter, idx_traj] = c_new
                        shifting_speed[counter, idx_traj] = cdot_new

                    print(f"{model.__class__.__name__} trajectory {idx_traj + 1} of {num_traj}, snapshot {counter + 1}/{num_snapshots}, t = {time}, dt = {timestep}, cdot_numer = {model.cdot_numerator}, cdot_denom = {model.cdot_denominator}, cdot = {cdot_new}, c = {c_new}.")

                    counter += 1

                timestep_old = timestep
                c_old = c_new
                cdot_old = cdot_new
                state_old = state

                c_new = c_old + cdot_old * timestep_old
                time += timestep_old

                state = stepper.step(state)
                cdot_new = model.shifting_speed(state)
                timestep = stepper.dt

                if not stepper.stability:
                    
                    print("The ROM simulation is terminated due to too small timestep and the latest recorded snapshot is set to infty.")
                    sol_fitted[:, counter:, idx_traj] = np.inf
                    sol[:, counter:, idx_traj] = np.inf
                    rhs_fitted[:, counter:, idx_traj] = np.inf
                    shifting_amount[counter:, idx_traj] = np.inf
                    shifting_speed[counter:, idx_traj] = np.inf
                    
                    return TrajectoryData(sol, sol_fitted, rhs_fitted, shifting_speed, shifting_amount)
            
            distance_old_moment = np.abs(time - counter * dt_snapshots)
            distance_new_moment = np.abs(time - (counter + 1) * dt_snapshots)

            if distance_old_moment <= distance_new_moment:
                sol_fitted[:, counter, idx_traj] = model.latent_to_full(state_old)
                sol[:, counter, idx_traj] = spatial_translation(sol_fitted[:, counter, idx_traj], -c_old)
                shifting_amount[counter, idx_traj] = c_old
                shifting_speed[counter, idx_traj] = cdot_old
            else:
                sol_fitted[:, counter, idx_traj] = model.latent_to_full(state)
                sol[:, counter, idx_traj] = spatial_translation(sol_fitted[:, counter, idx_traj], -c_new)
                shifting_amount[counter, idx_traj] = c_new
                shifting_speed[counter, idx_traj] = cdot_new

            print(f"{model.__class__.__name__} trajectory {idx_traj + 1} of {num_traj}, snapshot {counter + 1}/{num_snapshots}, t = {time}, dt = {timestep}, cdot_numer = {model.cdot_numerator}, cdot_denom = {model.cdot_denominator}, cdot = {cdot_new}, c = {c_new}.")

        return TrajectoryData(sol, sol_fitted, rhs_fitted, shifting_speed, shifting_amount)
    
    elif isinstance(model, BilinearModel):
        stepper = model.get_timestepper(method = timestepper, dt = timestep)
        for idx_traj in range(num_traj):
            if not re_proj_option: # collecting raw snapshots without re-projection
                print(f"Sampling non re-projected trajectory {idx_traj + 1} of {num_traj} using {model.__class__.__name__} FOM...")
                u = u_init[:, idx_traj]
                for time in range(num_timesteps):
                    if time % sample_interval == 0:
                        print(f"collecting non re-projected snapshot {1 + time // sample_interval}/{num_snapshots} for FOM trajectory {idx_traj + 1}...")
                        sol[:, time // sample_interval, idx_traj] = u
                        rhs[:, time // sample_interval, idx_traj] = model.rhs(u)
                        if time // sample_interval == 0:  # fitting the initial condition
                            u_fitted, shifting_amount[time // sample_interval, idx_traj] = template_fitting(u, u_template, L, Nx, spatial_translation)
                        else:  # fitting the following snapshots
                            u_fitted, shifting_amount[time // sample_interval, idx_traj] = template_fitting(u, u_template, L, Nx, spatial_translation, shifting_amount[time // sample_interval - 1, idx_traj])
                        sol_fitted[:, time // sample_interval, idx_traj] = u_fitted
                        rhs_fitted[:, time // sample_interval, idx_traj] = spatial_translation(rhs[:, time // sample_interval, idx_traj], shifting_amount[time // sample_interval, idx_traj])
                        shifting_speed[time // sample_interval, idx_traj] = compute_shifting_speed_FOM(rhs_fitted[:, time // sample_interval, idx_traj],
                                                                                            model.derivative(u_fitted, order = 1),
                                                                                            model.derivative(u_template, order = 1))
                    u = stepper.step(u)

            else:  # collecting re-projected snapshots

                trial_basis = V
                if W is None:  # if W is not provided, we use the trial basis as the test basis
                    W = V
                test_basis  = np.linalg.solve((V.T @ W).T, W.T).T * Nx

                def full_to_latent(x: Vector) -> Vector:
                    """Convert full state x to latent state a."""

                    return test_basis.T @ (x - bias) / test_basis.shape[0]
                
                def latent_to_full(a: Vector) -> Vector:
                    """Convert latent state a to full state x."""

                    return trial_basis @ a + bias

                print(f"Sampling re-projected trajectory {idx_traj + 1} of {num_traj} using {model.__class__.__name__} FOM...")
                u = u_init[:, idx_traj]
                u_fitted, c = template_fitting(u, u_template, L, Nx, spatial_translation)
                u_fitted = latent_to_full(full_to_latent(u_fitted))  # re-project the initial condition to the trial space
                # u = spatial_translation(u_fitted, -c)  # apply the spatial translation to the re-projected initial condition
                
                for time in range(num_timesteps):
                    if time % sample_interval == 0:
                        print(f"collecting re-projected snapshot {1 + time // sample_interval}/{num_snapshots} for trajectory {idx_traj + 1}...")
                        sol_fitted[:, time // sample_interval, idx_traj] = u_fitted
                        rhs_fitted[:, time // sample_interval, idx_traj] = model.rhs(u_fitted)
                        shifting_amount[time // sample_interval, idx_traj] = c
                        shifting_speed[time // sample_interval, idx_traj] = compute_shifting_speed_FOM(rhs_fitted[:, time // sample_interval, idx_traj],
                                                                                            model.derivative(u_fitted, order = 1),
                                                                                            model.derivative(u_template, order = 1))
                        
                    u = stepper.step(u_fitted)
                    u_fitted, c = template_fitting(u, u_template, L, Nx, spatial_translation, shifting_amount_old = c)
                    u_fitted = latent_to_full(full_to_latent(u_fitted))  # re-project the state to the trial space
                    # u = spatial_translation(u_fitted, -c)

        return TrajectoryData(sol, sol_fitted, rhs_fitted, shifting_speed, shifting_amount)