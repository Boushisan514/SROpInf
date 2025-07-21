import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def POD(sol, num_modes: int):
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
    sol_mean = np.mean(sol, axis=1)            # (Nx,)
    sol_centered = sol - sol_mean[:, np.newaxis]

    U, S, _ = np.linalg.svd(sol_centered, full_matrices=False)
    basis = U[:, :num_modes] * np.sqrt(sol.shape[0])  # (Nx, num_modes)

    return sol_mean, basis, S[:num_modes]

def full_to_proj(u, ubar, basis):
    
    N_space = u.shape[0]
        
    if u.ndim == 1:
        return basis.T @ (u - ubar) / N_space
    elif u.ndim == 2:
        return basis.T @ (u - ubar[:, np.newaxis]) / N_space
    else:
        raise ValueError("Input u must be 1D or 2D.")
        
def proj_to_full(a, ubar, basis):
        
    if a.ndim == 1:
        return ubar + basis @ a
    elif a.ndim == 2:
        return ubar[:, None] + basis @ a
    else:
        raise ValueError("Input a must be 1D or 2D.")
    
def freq_to_space(u_hat):
    
    N = len(u_hat) // 2
        
    return np.fft.ifft(np.fft.ifftshift(u_hat)).real * 2 * N

def space_to_freq(u):

    N = len(u) // 2

    u_freq = np.fft.fftshift(np.fft.fft(u)) / (2 * N)
    u_freq[0] = 0  # We filter out the Nyquist frequency mode to guarantee real-valued solutions

    return u_freq

def spatial_translation(u, c, Lx):

    # This function converts a given spatial function q(x) to q(x + c) by manipulating the Fourier coefficients
    
    u_hat = space_to_freq(u)
    
    N = len(u_hat) // 2

    mode_index = np.linspace(-N, N-1, 2 * N, dtype=int, endpoint=True)
    mode_index[0] = 0  # We filter out the Nyquist frequency mode

    u_shifted_hat = u_hat * np.exp(1j * c * (2*np.pi/Lx) * mode_index)

    return freq_to_space(u_shifted_hat)

def main():
    
    # Here we mainly want to plot figures as follows:
    # 1. the contourf plot of u(x, t) for a given grid of x and t
    # 2. the temporal evolution of the 2-norm of the reduced state a(t) with time
    # 3. the prediction of the traveling speed
    # 4. the comparison between the OpInf-learned operators and the Galerkin operators
    
    # region 0: set parameters and load data
    start_time = 120
    end_time = 130
    
    dt_sample = 0.01
    
    num_points = int((end_time - start_time) / dt_sample) + 1
    time_FOM = np.linspace(start_time, end_time, num_points, endpoint=True)
    
    N_space = 40
    L = 2 * np.pi
    x = np.linspace(0, L, N_space, endpoint=False)
    T = np.linspace(start_time, end_time, num_points, endpoint=True)
    
    # 1. load the FOM data
    
    filepath_prefix = r"C:\Users\ys5910\Desktop\SROpInf_working_on\data\dim_4"
    
    # endregion

    # region 1: training loss of SROpInf w/o reproj and w/ reproj

    # training_loss_SROpInf_wo_reproj = np.loadtxt(rf"{filepath_prefix}\SR_OpInf_ROM_wo_reproj_training_loss.txt")
    # training_loss_SROpInf_w_reproj = np.loadtxt(rf"{filepath_prefix}\SR_OpInf_ROM_w_reproj_training_loss.txt")
    # plt.figure(figsize=(6, 6))
    # plt.plot(training_loss_SROpInf_wo_reproj[0:4000], label="SR-OpInf ROM, w/o re-proj", color='red')
    # plt.plot(training_loss_SROpInf_w_reproj[0:4000], label="SR-OpInf ROM, w/ re-proj", color='blue', linestyle='--')

    # plt.xticks([0, 1000, 2000, 3000, 4000], fontsize = 15)

    # # plt.xticks([0, 1000, 2000, 3000, 4000], fontsize = 15)
    # plt.yscale("log")
    # plt.yticks([1e7, 1e5, 1e3, 1e1, 1e-1, 1e-3, 1e-5], fontsize = 15)

    # # plt.xlabel("Training iteration")
    # # plt.ylabel(r"Training loss $\mathcal{J}$")
    
    # plt.legend(fontsize = 15, loc='upper right')
    # plt.show()
    # plt.close()

    # print(f"Final training loss of SROpInf w/o reproj: {training_loss_SROpInf_wo_reproj[-1]:.4e}")
    # print(f"Final training loss of SROpInf w/ reproj: {training_loss_SROpInf_w_reproj[-1]:.4e}")
    
    # endregion 
    
    # region 2: contourf plots of FOM before and after template fitting
    
    # x = np.linspace(0, L, N_space + 1, endpoint=True)
    # x_grid, t_grid = np.meshgrid(x, T)
    # min_colorbar = -15
    # max_colorbar = 12
    # levels = np.linspace(-15, 12, 10)

    # sol_FOM = np.loadtxt(rf"{filepath_prefix}\ks_solution_FOM.txt")
    # sol_FOM = np.vstack([sol_FOM, sol_FOM[0]])  # Close the domain by repeating the first point
    # fig = plt.figure(figsize=(6, 6))
    # plt.contourf(x_grid, t_grid, sol_FOM.T, levels = levels, vmin = min_colorbar, vmax = max_colorbar, cmap='viridis')
    # plt.xticks([0, np.pi, 2 * np.pi], [r'$0$', r'$\pi$', r'$2\pi$'], fontsize = 15)
    # plt.yticks([120, 125, 130], fontsize = 15)
    # cbar = plt.colorbar(ticks=levels)
    # cbar.ax.tick_params(labelsize=15)
    # plt.show()
    # plt.close()

    # sol_fitted_FOM = np.loadtxt(rf"{filepath_prefix}\ks_solution_fitted_FOM.txt")
    # sol_fitted_FOM = np.vstack([sol_fitted_FOM, sol_fitted_FOM[0]])  # Close the domain by repeating the first point
    # fig = plt.figure(figsize=(6, 6))
    # plt.contourf(x_grid, t_grid, sol_fitted_FOM.T, levels = levels, vmin = min_colorbar, vmax = max_colorbar, cmap='viridis')
    # plt.xticks([0, np.pi, 2 * np.pi], [r'$0$', r'$\pi$', r'$2\pi$'], fontsize = 15)
    # plt.yticks([120, 125, 130], fontsize = 15)
    # cbar = plt.colorbar(ticks=levels)
    # cbar.ax.tick_params(labelsize=15)
    # plt.show()
    # plt.close()
    
    # endregion
    
    # region 3: contourf plots of SROpInf w/o reproj, w/ reproj, SRGalerkin and projected FOM at n = 4
    
    # x = np.linspace(0, L, N_space, endpoint=False)

    # sol_FOM = np.loadtxt(rf"{filepath_prefix}\ks_solution_FOM.txt")
    # sol_fitted_FOM = np.loadtxt(rf"{filepath_prefix}\ks_solution_fitted_FOM.txt")
    # sol_SRG = np.loadtxt(rf"{filepath_prefix}\ks_solution_SR_Galerkin_ROM.txt")
    # sol_SRO_wo_reproj = np.loadtxt(rf"{filepath_prefix}\ks_solution_SR_OpInf_ROM_wo_reproj.txt")
    # sol_SRO_w_reproj = np.loadtxt(rf"{filepath_prefix}\ks_solution_SR_OpInf_ROM_w_reproj.txt")
    # shifting_amount_FOM = np.loadtxt(rf"{filepath_prefix}\ks_solution_shifting_amount_FOM.txt")

    # sol_SRG_diff = sol_SRG - sol_FOM
    # sol_SRO_w_reproj_diff = sol_SRO_w_reproj - sol_FOM

    # ubar, basis, singular_values = POD(sol_fitted_FOM, num_modes=4)
    # u0   = np.cos(x)  # Initial condition for the FOM

    # sol_fitted_FOM_projected = proj_to_full(full_to_proj(sol_fitted_FOM, ubar, basis), ubar, basis)
    # sol_FOM_projected = np.zeros((N_space, num_points))
    # for time_index in range(num_points):
    #     sol_FOM_projected[:, time_index] = spatial_translation(sol_fitted_FOM_projected[:, time_index], -shifting_amount_FOM[time_index], L)

    # sol_FOM_projected_diff = sol_FOM_projected - sol_FOM

    # relative_error_SRO_w_reproj  = 100 * np.linalg.norm(sol_SRO_w_reproj_diff) / np.linalg.norm(sol_FOM)
    # relative_error_SRG           = 100 * np.linalg.norm(sol_SRG_diff) / np.linalg.norm(sol_FOM)
    # relative_error_FOM_projected = 100 * np.linalg.norm(sol_FOM_projected_diff) / np.linalg.norm(sol_FOM)

    # print(f"Relative error of SROpInf w/ reproj: {relative_error_SRO_w_reproj:.4f}")
    # print(f"Relative error of SRGalerkin: {relative_error_SRG:.4f}")
    # print(f"Relative error of FOM projected: {relative_error_FOM_projected:.4f}")

    # sol_SRO_wo_reproj = np.vstack([sol_SRO_wo_reproj, sol_SRO_wo_reproj[0]])  # Close the domain by repeating the first point
    # sol_SRO_w_reproj = np.vstack([sol_SRO_w_reproj, sol_SRO_w_reproj[0]])  # Close the domain by repeating the first point
    # sol_SRG = np.vstack([sol_SRG, sol_SRG[0]])  # Close the domain by repeating the first point
    # sol_FOM_projected = np.vstack([sol_FOM_projected, sol_FOM_projected[0]])  # Close the domain by repeating the first point

    # time = np.linspace(start_time, end_time, num_points, endpoint=True)
    # x    = np.linspace(0, L, N_space + 1, endpoint=True)

    # x_grid, t_grid = np.meshgrid(x, time)
    
    # min_colorbar = -15
    # max_colorbar = 12
    # levels = np.linspace(-15, 12, 10)
    
    # fig = plt.figure(figsize=(6, 6))
    # plt.contourf(x_grid, t_grid, sol_SRO_wo_reproj.T, levels = levels, vmin = min_colorbar, vmax = max_colorbar, cmap='viridis')
    # plt.xticks([0, np.pi, 2 * np.pi], [r'$0$', r'$\pi$', r'$2\pi$'], fontsize = 15)
    # plt.yticks([120, 125, 130], fontsize = 15)
    # cbar = plt.colorbar(ticks=levels)
    # cbar.ax.tick_params(labelsize=15)
    # plt.show()
    # plt.close()

    # fig = plt.figure(figsize=(6, 6))
    # plt.contourf(x_grid, t_grid, sol_SRO_w_reproj.T, levels = levels, vmin = min_colorbar, vmax = max_colorbar, cmap='viridis')
    # plt.xticks([0, np.pi, 2 * np.pi], [r'$0$', r'$\pi$', r'$2\pi$'], fontsize = 15)
    # plt.yticks([120, 125, 130], fontsize = 15)
    # cbar = plt.colorbar(ticks=levels)
    # cbar.ax.tick_params(labelsize=15)
    # plt.show()
    # plt.close()
    
    # fig = plt.figure(figsize=(6, 6))
    # plt.contourf(x_grid, t_grid, sol_SRG.T, levels = levels, vmin = min_colorbar, vmax = max_colorbar, cmap='viridis')
    # plt.xticks([0, np.pi, 2 * np.pi], [r'$0$', r'$\pi$', r'$2\pi$'], fontsize = 15)
    # plt.yticks([120, 125, 130], fontsize = 15)
    # cbar = plt.colorbar(ticks=levels)
    # cbar.ax.tick_params(labelsize=15)
    # plt.show()
    # plt.close()
    
    # fig = plt.figure(figsize=(6, 6))
    # plt.contourf(x_grid, t_grid, sol_FOM_projected.T, levels = levels, vmin = min_colorbar, vmax = max_colorbar, cmap='viridis')
    # plt.xticks([0, np.pi, 2 * np.pi], [r'$0$', r'$\pi$', r'$2\pi$'], fontsize = 15)
    # plt.yticks([120, 125, 130], fontsize = 15)
    # cbar = plt.colorbar(ticks=levels)
    # cbar.ax.tick_params(labelsize=15)
    # plt.show()
    # plt.close()
    
    # endregion
    
    # region 4: time-average L2 error of SROpInf w/ reproj and w/o reproj at n = 3, 4, 5, 6, 7, 8

    # time = np.linspace(start_time, end_time, num_points, endpoint=True)
    # x    = np.linspace(0, L, N_space, endpoint=False)  # Adjust to match the original code's x grid

    # filepath_prefix_dim_4 = r"C:\Users\ys5910\Desktop\SROpInf_working_on\data\dim_4"
    # sol_SRO_wo_reproj_dim_4 = np.loadtxt(rf"{filepath_prefix_dim_4}\ks_solution_SR_OpInf_ROM_wo_reproj.txt")
    # sol_SRO_w_reproj_dim_4 = np.loadtxt(rf"{filepath_prefix_dim_4}\ks_solution_SR_OpInf_ROM_w_reproj.txt")
    # sol_SRG_dim_4 = np.loadtxt(rf"{filepath_prefix_dim_4}\ks_solution_SR_Galerkin_ROM.txt")
    # sol_FOM = np.loadtxt(rf"{filepath_prefix_dim_4}\ks_solution_FOM.txt")
    # sol_fitted_FOM = np.loadtxt(rf"{filepath_prefix_dim_4}\ks_solution_fitted_FOM.txt")

    # print(f"Shape of sol_FOM: {sol_FOM.shape}")

    # ubar, basis, singular_values = POD(sol_fitted_FOM, num_modes=4)
    # sol_fitted_FOM_projected_dim_4 = proj_to_full(full_to_proj(sol_fitted_FOM, ubar, basis), ubar, basis)

    # filepath_prefix_dim_3 = r"C:\Users\ys5910\Desktop\SROpInf_working_on\data\dim_3"
    # sol_SRO_wo_reproj_dim_3 = np.loadtxt(rf"{filepath_prefix_dim_3}\ks_solution_SR_OpInf_ROM_wo_reproj.txt")
    # sol_SRO_w_reproj_dim_3 = np.loadtxt(rf"{filepath_prefix_dim_3}\ks_solution_SR_OpInf_ROM_w_reproj.txt")
    # sol_SRG_dim_3 = np.loadtxt(rf"{filepath_prefix_dim_3}\ks_solution_SR_Galerkin_ROM.txt")

    # ubar, basis, singular_values = POD(sol_fitted_FOM, num_modes=3)
    # sol_fitted_FOM_projected_dim_3 = proj_to_full(full_to_proj(sol_fitted_FOM, ubar, basis), ubar, basis)

    # filepath_prefix_dim_5 = r"C:\Users\ys5910\Desktop\SROpInf_working_on\data\dim_5"
    # sol_SRO_wo_reproj_dim_5 = np.loadtxt(rf"{filepath_prefix_dim_5}\ks_solution_SR_OpInf_ROM_wo_reproj.txt")
    # sol_SRO_w_reproj_dim_5 = np.loadtxt(rf"{filepath_prefix_dim_5}\ks_solution_SR_OpInf_ROM_w_reproj.txt")
    # sol_SRG_dim_5 = np.loadtxt(rf"{filepath_prefix_dim_5}\ks_solution_SR_Galerkin_ROM.txt")

    # ubar, basis, singular_values = POD(sol_fitted_FOM, num_modes=5)
    # sol_fitted_FOM_projected_dim_5 = proj_to_full(full_to_proj(sol_fitted_FOM, ubar, basis), ubar, basis)

    # filepath_prefix_dim_6 = r"C:\Users\ys5910\Desktop\SROpInf_working_on\data\dim_6"
    # sol_SRO_wo_reproj_dim_6 = np.loadtxt(rf"{filepath_prefix_dim_6}\ks_solution_SR_OpInf_ROM_wo_reproj.txt")
    # sol_SRO_w_reproj_dim_6 = np.loadtxt(rf"{filepath_prefix_dim_6}\ks_solution_SR_OpInf_ROM_w_reproj.txt")
    # sol_SRG_dim_6 = np.loadtxt(rf"{filepath_prefix_dim_6}\ks_solution_SR_Galerkin_ROM.txt")

    # ubar, basis, singular_values = POD(sol_fitted_FOM, num_modes=6)
    # sol_fitted_FOM_projected_dim_6 = proj_to_full(full_to_proj(sol_fitted_FOM, ubar, basis), ubar, basis)

    # filepath_prefix_dim_7 = r"C:\Users\ys5910\Desktop\SROpInf_working_on\data\dim_7"
    # sol_SRO_wo_reproj_dim_7 = np.loadtxt(rf"{filepath_prefix_dim_7}\ks_solution_SR_OpInf_ROM_wo_reproj.txt")
    # sol_SRO_w_reproj_dim_7 = np.loadtxt(rf"{filepath_prefix_dim_7}\ks_solution_SR_OpInf_ROM_w_reproj.txt")
    # sol_SRG_dim_7 = np.loadtxt(rf"{filepath_prefix_dim_7}\ks_solution_SR_Galerkin_ROM.txt")

    # ubar, basis, singular_values = POD(sol_fitted_FOM, num_modes=7)
    # sol_fitted_FOM_projected_dim_7 = proj_to_full(full_to_proj(sol_fitted_FOM, ubar, basis), ubar, basis)

    # filepath_prefix_dim_8 = r"C:\Users\ys5910\Desktop\SROpInf_working_on\data\dim_8"
    # sol_SRO_wo_reproj_dim_8 = np.loadtxt(rf"{filepath_prefix_dim_8}\ks_solution_SR_OpInf_ROM_wo_reproj.txt")
    # sol_SRO_w_reproj_dim_8 = np.loadtxt(rf"{filepath_prefix_dim_8}\ks_solution_SR_OpInf_ROM_w_reproj.txt")
    # sol_SRG_dim_8 = np.loadtxt(rf"{filepath_prefix_dim_8}\ks_solution_SR_Galerkin_ROM.txt")

    # ubar, basis, singular_values = POD(sol_fitted_FOM, num_modes=8)
    # sol_fitted_FOM_projected_dim_8 = proj_to_full(full_to_proj(sol_fitted_FOM, ubar, basis), ubar, basis)

    # relative_error_SRO_wo_reproj_dim_3 = np.linalg.norm(sol_SRO_wo_reproj_dim_3 - sol_FOM) / np.linalg.norm(sol_FOM)
    # relative_error_SRO_wo_reproj_dim_4 = np.linalg.norm(sol_SRO_wo_reproj_dim_4 - sol_FOM) / np.linalg.norm(sol_FOM)
    # relative_error_SRO_wo_reproj_dim_5 = np.linalg.norm(sol_SRO_wo_reproj_dim_5 - sol_FOM) / np.linalg.norm(sol_FOM)
    # relative_error_SRO_wo_reproj_dim_6 = np.linalg.norm(sol_SRO_wo_reproj_dim_6 - sol_FOM) / np.linalg.norm(sol_FOM)
    # relative_error_SRO_wo_reproj_dim_7 = np.linalg.norm(sol_SRO_wo_reproj_dim_7 - sol_FOM) / np.linalg.norm(sol_FOM)
    # relative_error_SRO_wo_reproj_dim_8 = np.linalg.norm(sol_SRO_wo_reproj_dim_8 - sol_FOM) / np.linalg.norm(sol_FOM)

    # relative_error_SRO_w_reproj_dim_3 = np.linalg.norm(sol_SRO_w_reproj_dim_3 - sol_FOM) / np.linalg.norm(sol_FOM)
    # relative_error_SRO_w_reproj_dim_4 = np.linalg.norm(sol_SRO_w_reproj_dim_4 - sol_FOM) / np.linalg.norm(sol_FOM)
    # relative_error_SRO_w_reproj_dim_5 = np.linalg.norm(sol_SRO_w_reproj_dim_5 - sol_FOM) / np.linalg.norm(sol_FOM)
    # relative_error_SRO_w_reproj_dim_6 = np.linalg.norm(sol_SRO_w_reproj_dim_6 - sol_FOM) / np.linalg.norm(sol_FOM)
    # relative_error_SRO_w_reproj_dim_7 = np.linalg.norm(sol_SRO_w_reproj_dim_7 - sol_FOM) / np.linalg.norm(sol_FOM)
    # relative_error_SRO_w_reproj_dim_8 = np.linalg.norm(sol_SRO_w_reproj_dim_8 - sol_FOM) / np.linalg.norm(sol_FOM)

    # relative_error_SRG_dim_3 = np.linalg.norm(sol_SRG_dim_3 - sol_FOM) / np.linalg.norm(sol_FOM)
    # relative_error_SRG_dim_4 = np.linalg.norm(sol_SRG_dim_4 - sol_FOM) / np.linalg.norm(sol_FOM)
    # relative_error_SRG_dim_5 = np.linalg.norm(sol_SRG_dim_5 - sol_FOM) / np.linalg.norm(sol_FOM)
    # relative_error_SRG_dim_6 = np.linalg.norm(sol_SRG_dim_6 - sol_FOM) / np.linalg.norm(sol_FOM)
    # relative_error_SRG_dim_7 = np.linalg.norm(sol_SRG_dim_7 - sol_FOM) / np.linalg.norm(sol_FOM)
    # relative_error_SRG_dim_8 = np.linalg.norm(sol_SRG_dim_8 - sol_FOM) / np.linalg.norm(sol_FOM)

    # relative_error_FOM_projected_dim_3 = np.linalg.norm(sol_fitted_FOM_projected_dim_3 - sol_fitted_FOM) / np.linalg.norm(sol_fitted_FOM)
    # relative_error_FOM_projected_dim_4 = np.linalg.norm(sol_fitted_FOM_projected_dim_4 - sol_fitted_FOM) / np.linalg.norm(sol_fitted_FOM)
    # relative_error_FOM_projected_dim_5 = np.linalg.norm(sol_fitted_FOM_projected_dim_5 - sol_fitted_FOM) / np.linalg.norm(sol_fitted_FOM)
    # relative_error_FOM_projected_dim_6 = np.linalg.norm(sol_fitted_FOM_projected_dim_6 - sol_fitted_FOM) / np.linalg.norm(sol_fitted_FOM)
    # relative_error_FOM_projected_dim_7 = np.linalg.norm(sol_fitted_FOM_projected_dim_7 - sol_fitted_FOM) / np.linalg.norm(sol_fitted_FOM)
    # relative_error_FOM_projected_dim_8 = np.linalg.norm(sol_fitted_FOM_projected_dim_8 - sol_fitted_FOM) / np.linalg.norm(sol_fitted_FOM)

    # print(f"Relative error of SROpInf w/ reproj at n = 3: {relative_error_SRO_w_reproj_dim_3:.6f}")
    # print(f"Relative error of SROpInf w/ reproj at n = 4: {relative_error_SRO_w_reproj_dim_4:.6f}")
    # print(f"Relative error of SROpInf w/ reproj at n = 5: {relative_error_SRO_w_reproj_dim_5:.6f}")
    # print(f"Relative error of SROpInf w/ reproj at n = 6: {relative_error_SRO_w_reproj_dim_6:.6f}")
    # print(f"Relative error of SROpInf w/ reproj at n = 7: {relative_error_SRO_w_reproj_dim_7:.6f}")
    # print(f"Relative error of SROpInf w/ reproj at n = 8: {relative_error_SRO_w_reproj_dim_8:.6f}")

    # print(f"Relative error of SRGalerkin at n = 3: {relative_error_SRG_dim_3:.6f}")
    # print(f"Relative error of SRGalerkin at n = 4: {relative_error_SRG_dim_4:.6f}")
    # print(f"Relative error of SRGalerkin at n = 5: {relative_error_SRG_dim_5:.6f}")
    # print(f"Relative error of SRGalerkin at n = 6: {relative_error_SRG_dim_6:.6f}")
    # print(f"Relative error of SRGalerkin at n = 7: {relative_error_SRG_dim_7:.6f}")
    # print(f"Relative error of SRGalerkin at n = 8: {relative_error_SRG_dim_8:.6f}")

    # print(f"Relative error of FOM projected at n = 3: {relative_error_FOM_projected_dim_3:.6f}")
    # print(f"Relative error of FOM projected at n = 4: {relative_error_FOM_projected_dim_4:.6f}")
    # print(f"Relative error of FOM projected at n = 5: {relative_error_FOM_projected_dim_5:.6f}")
    # print(f"Relative error of FOM projected at n = 6: {relative_error_FOM_projected_dim_6:.6f}")
    # print(f"Relative error of FOM projected at n = 7: {relative_error_FOM_projected_dim_7:.6f}")
    # print(f"Relative error of FOM projected at n = 8: {relative_error_FOM_projected_dim_8:.6f}")

    # relative_errors_SRO_wo_reproj = np.array([
    #     relative_error_SRO_wo_reproj_dim_3,
    #     relative_error_SRO_wo_reproj_dim_4,
    #     relative_error_SRO_wo_reproj_dim_5,
    #     relative_error_SRO_wo_reproj_dim_6,
    #     relative_error_SRO_wo_reproj_dim_7,
    #     relative_error_SRO_wo_reproj_dim_8
    # ])

    # relative_errors_SRO_w_reproj = np.array([
    #     relative_error_SRO_w_reproj_dim_3,
    #     relative_error_SRO_w_reproj_dim_4,
    #     relative_error_SRO_w_reproj_dim_5,
    #     relative_error_SRO_w_reproj_dim_6,
    #     relative_error_SRO_w_reproj_dim_7,
    #     relative_error_SRO_w_reproj_dim_8
    # ])

    # relative_errors_SRG = np.array([
    #     relative_error_SRG_dim_3,
    #     relative_error_SRG_dim_4,
    #     relative_error_SRG_dim_5,
    #     relative_error_SRG_dim_6,
    #     relative_error_SRG_dim_7,
    #     relative_error_SRG_dim_8
    # ])

    # relative_errors_FOM_projected = np.array([
    #     relative_error_FOM_projected_dim_3,
    #     relative_error_FOM_projected_dim_4,
    #     relative_error_FOM_projected_dim_5,
    #     relative_error_FOM_projected_dim_6,
    #     relative_error_FOM_projected_dim_7,
    #     relative_error_FOM_projected_dim_8
    # ])

    # plt.figure(figsize=(6, 6))
    # dim_array = np.array([3, 4, 5, 6, 7, 8])
    # plt.plot(dim_array, relative_errors_SRO_wo_reproj, marker='o', color='red')
    # plt.plot(dim_array, relative_errors_SRO_w_reproj, marker='o', color = 'blue')
    # plt.plot(dim_array, relative_errors_SRG, marker='o', color='orange')
    # plt.plot(dim_array, relative_errors_FOM_projected, marker='o', color='green')
    # plt.xticks(dim_array, fontsize=15)
    # plt.yscale("log")
    # plt.yticks([1e-3, 1e-2, 1e-1, 1], fontsize=15)
    # plt.legend(['SR-OpInf ROM w/o re-proj', 'SR-OpInf ROM, w/ re-proj', 'SR-Galerkin ROM', 'Projected FOM snapshots'], fontsize=15, loc='upper right')
    # plt.show()
    
    # endregion
    
    # region 5: plot the time evolution of No. 1, and No. 3 POD mode comparing FOM and ROM trajectories
    
    # plt.figure(figsize=(6, 6))

    # sol_fitted_FOM = np.loadtxt(rf"{filepath_prefix}\ks_solution_fitted_FOM.txt")
    # sol_fitted_SRG = np.loadtxt(rf"{filepath_prefix}\ks_solution_fitted_SR_Galerkin_ROM.txt")
    # sol_fitted_SRO_w_reproj = np.loadtxt(rf"{filepath_prefix}\ks_solution_fitted_SR_OpInf_ROM_w_reproj.txt")

    # ubar, basis, singular_values = POD(sol_fitted_FOM, num_modes=4)
    # state_FOM = full_to_proj(sol_fitted_FOM, ubar, basis)
    # state_SRO_w_reproj = full_to_proj(sol_fitted_SRO_w_reproj, ubar, basis)
    # state_SRG = full_to_proj(sol_fitted_SRG, ubar, basis)
    
    # time_span_POD_amplitudes = 2
    # num_snapshots_POD_amplitudes = int(time_span_POD_amplitudes / dt_sample) + 1
    
    # plt.plot(time_FOM[-num_snapshots_POD_amplitudes:], state_FOM[0, -num_snapshots_POD_amplitudes:], label="FOM", color='red')
    # plt.plot(time_FOM[-num_snapshots_POD_amplitudes:], state_SRG[0, -num_snapshots_POD_amplitudes:], label="SR-Galerkin ROM", color='blue')
    # plt.plot(time_FOM[-num_snapshots_POD_amplitudes:], state_SRO_w_reproj[0, -num_snapshots_POD_amplitudes:], label="SR-OpInf ROM, w/ re-proj", linestyle='--', color='orange')
    # plt.xticks([128, 129, 130], fontsize = 15)
    # plt.yticks([-5, 0, 5], fontsize = 15)
    
    # plt.legend(fontsize = 15, loc='upper right')
    # plt.show()
    # plt.close()
    
    # plt.figure(figsize=(6, 6))

    # plt.plot(time_FOM[-num_snapshots_POD_amplitudes:], state_FOM[2, -num_snapshots_POD_amplitudes:], label="FOM", color='red')
    # plt.plot(time_FOM[-num_snapshots_POD_amplitudes:], state_SRG[2, -num_snapshots_POD_amplitudes:], label="SR-Galerkin ROM", color='blue')
    # plt.plot(time_FOM[-num_snapshots_POD_amplitudes:], state_SRO_w_reproj[2, -num_snapshots_POD_amplitudes:], label="SR-OpInf ROM, w/ re-proj", linestyle='--', color='orange')
    
    # plt.xticks([128, 129, 130], fontsize = 15)
    # plt.yticks([-1, 0, 1], fontsize = 15)
    
    # plt.legend(fontsize = 15, loc='upper right')
    # plt.show()
    # plt.close()
    
    # endregion

    # region 6(old): plot the prediction of shifting speed and shifting amount, comparing different models
    
    # fig, main_ax = plt.subplots(figsize=(6, 6))

    # # === Main Plot ===
    # main_ax.plot(time_FOM, shifting_speed_FOM,
    #             label="FOM", color='red')

    # main_ax.plot(time_SRG, shifting_speed_SRG,
    #             label="SR-Galerkin ROM", color='blue')

    # main_ax.plot(time_SRO_wo_reproj, shifting_speed_SRO_wo_reproj,
    #             label="SR-OpInf ROM trained w/ non re-proj data", color='green', linestyle='--')

    # main_ax.plot(time_SRO_w_reproj, shifting_speed_SRO_w_reproj,
    #             label="SR-OpInf ROM trained w/ re-proj data", color='orange', linestyle='--')

    # # main_ax.set_ylabel(r"$\dot{c}(\mathbf{a}(t))$")
    # # main_ax.set_xlabel("t")
    # main_ax.set_xticks([120, 125, 130])
    # main_ax.set_ylim(-4, 4)
    # main_ax.set_yticks([-4, -2, 0, 2, 4])
    # main_ax.tick_params(axis='both', labelsize=15)
    # main_ax.legend(loc="lower right", fontsize=10)

    # # === Inset Panel ===
    # axins = inset_axes(main_ax,
    #                width="35%", height="25%",
    #                bbox_to_anchor=(0.6, 0.7, 1, 1),  # (x0, y0, width, height)
    #                bbox_transform=main_ax.transAxes,
    #                loc='lower left')

    # # Zoom region plots (same curves)
    # axins.plot(time_FOM, shifting_speed_FOM, color='red')
    # axins.plot(time_FOM, shifting_speed_FOM_w_reproj, color='purple',
    #         linestyle='-')
    # axins.plot(time_SRG, shifting_speed_SRG, color='blue',
    #         linestyle='-')
    # axins.plot(time_SRO_wo_reproj, shifting_speed_SRO_wo_reproj, color='green', linestyle='--')
    # axins.plot(time_SRO_w_reproj, shifting_speed_SRO_w_reproj, color='orange', linestyle='--')

    # # === Set zoom-in region (adjust if needed) ===
    # axins.set_xlim(124.5, 124.6)
    # axins.set_ylim(1, 1.5)
    # axins.set_xticks([124.5, 124.6])
    # axins.set_yticks([1, 1.5])
    # axins.tick_params(axis='both', labelsize=8, pad=1)

    # # === Connect with main plot ===
    # mark_inset(main_ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # # === Save and show ===
    # # plt.savefig(rf"{filepath_prefix}\comparison_L2_norm_reduced_state.png", dpi=300)
    # plt.show()
    # plt.close()
    
    # fig, main_ax = plt.subplots(figsize=(6, 6))

    # # === Main Plot ===
    # main_ax.plot(time_FOM, shifting_amount_FOM,
    #             label="FOM", color='red')

    # main_ax.plot(time_FOM, 2 * np.pi + shifting_amount_FOM_w_reproj,
    #             label="FOM along re-proj trajectory", color='purple')

    # main_ax.plot(time_SRG, 2 * np.pi + shifting_amount_SRG,
    #             label="SR-Galerkin ROM", color='blue')

    # main_ax.plot(time_SRO_wo_reproj, 2 * np.pi + shifting_amount_SRO_wo_reproj,
    #             label="SR-OpInf ROM trained w/ non re-proj data", color='green', linestyle='--')

    # main_ax.plot(time_SRO_w_reproj, 2 * np.pi + shifting_amount_SRO_w_reproj,
    #             label="SR-OpInf ROM trained w/ re-proj data", color='orange', linestyle='--')

    # main_ax.set_xticks([120, 125, 130])
    # main_ax.set_ylim(-1, 1)
    # main_ax.set_yticks([-1, 0, 1])
    # main_ax.tick_params(axis='both', labelsize=15)
    # main_ax.legend(loc="lower right", fontsize=10)

    # # === Inset Panel ===
    # axins = inset_axes(main_ax,
    #                width="35%", height="25%",
    #                bbox_to_anchor=(0.03, -0.02, 1, 1),   # upper-left of main_ax
    #                bbox_transform=main_ax.transAxes,
    #                loc='upper left')   # anchor point in bbox

    # # Zoom region plots (same curves)
    # axins.plot(time_FOM, shifting_amount_FOM, color='red')
    # axins.plot(time_FOM, 2 * np.pi + shifting_amount_FOM_w_reproj, color='purple',
    #         linestyle='-')
    # axins.plot(time_SRG, 2 * np.pi + shifting_amount_SRG, color='blue',
    #         linestyle='-')
    # axins.plot(time_SRO_wo_reproj, 2 * np.pi + shifting_amount_SRO_wo_reproj, color='green', linestyle='--')
    # axins.plot(time_SRO_w_reproj, 2 * np.pi + shifting_amount_SRO_w_reproj, color='orange', linestyle='--')

    # # === Set zoom-in region (adjust if needed) ===
    # axins.set_xlim(124.6, 124.7)
    # axins.set_ylim(0.4, 0.5)
    # axins.set_xticks([124.6, 124.7])
    # axins.set_yticks([0.4, 0.5])
    # axins.tick_params(axis='both', labelsize=8, pad=1)

    # # === Connect with main plot ===
    # mark_inset(main_ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    
    # plt.savefig(rf"{filepath_prefix}\comparison_shifting_amount.png", dpi=300)
    # # plt.legend(loc = "lower right", fontsize=10)
    # plt.show()
    # plt.close()

    # plt.figure(figsize=(6, 6))
    
    # plt.plot(time_FOM, shifting_amount_FOM, label="FOM", color='red')
    # plt.plot(time_FOM, 2 * np.pi + shifting_amount_FOM_w_reproj, label="FOM along re-proj trajectory", color='purple', linestyle='-')
    # plt.plot(time_SRG, 2 * np.pi + shifting_amount_SRG, label="SR-Galerkin ROM", color='blue', linestyle='-')
    # plt.plot(time_SRO_wo_reproj, 2 * np.pi + shifting_amount_SRO_wo_reproj, label="SR-OpInf ROM trained w/ non re-proj data", color='green')
    # plt.plot(time_SRO_w_reproj, 2 * np.pi + shifting_amount_SRO_w_reproj, label="SR-OpInf ROM trained w/ re-proj data", color='orange', linestyle='--')
    
    # plt.xticks([120, 125, 130], fontsize = 15)
    # plt.ylim(-1, 1)
    # plt.yticks([-1, 0, 1], fontsize = 15)
    
    # plt.ylabel(r"$c(t)$")
    # plt.xlabel("t")
    # plt.ylim(-1, 1)
    # plt.legend()
    
    # plt.show()
    # plt.close()
    
    # endregion
    
    # region 7: plot the perdiction of shifting speed and amount comparing FOM, SRG and SRO w/ re-proj
    
    # plt.figure(figsize=(6, 6))

    # time_span_shifting_speed = 2
    # num_snapshots_shifting_speed = int(time_span_shifting_speed / dt_sample) + 1
    
    # filepath_prefix = r"C:\Users\ys5910\Desktop\SROpInf_working_on\data\dim_4"
    
    # shifting_speed_FOM = np.loadtxt(rf"{filepath_prefix}\ks_solution_shifting_speed_FOM.txt")
    # shifting_speed_SRG = np.loadtxt(rf"{filepath_prefix}\ks_solution_shifting_speed_SR_Galerkin_ROM.txt")
    # shifting_speed_SRO_w_reproj = np.loadtxt(rf"{filepath_prefix}\ks_solution_shifting_speed_SR_OpInf_ROM_w_reproj.txt")
    
    # shifting_amount_FOM = np.loadtxt(rf"{filepath_prefix}\ks_solution_shifting_amount_FOM.txt")
    # shifting_amount_SRG = np.loadtxt(rf"{filepath_prefix}\ks_solution_shifting_amount_SR_Galerkin_ROM.txt")
    # shifting_amount_SRO_w_reproj = np.loadtxt(rf"{filepath_prefix}\ks_solution_shifting_amount_SR_OpInf_ROM_w_reproj.txt")

    # plt.plot(time_FOM[-num_snapshots_shifting_speed:], shifting_speed_FOM[-num_snapshots_shifting_speed:], label="FOM", color='red')
    # plt.plot(time_FOM[-num_snapshots_shifting_speed:], shifting_speed_SRG[-num_snapshots_shifting_speed:], label="SR-Galerkin ROM", color='blue')
    # plt.plot(time_FOM[-num_snapshots_shifting_speed:], shifting_speed_SRO_w_reproj[-num_snapshots_shifting_speed:], label="SR-OpInf ROM, w/ re-proj", linestyle='--', color='orange')
    
    # plt.xticks([128, 129, 130], fontsize = 15)
    # plt.yticks([-2, -1, 0, 1, 2], fontsize = 15)
    
    # plt.legend(fontsize = 15, loc='upper right')
    # plt.show()
    # plt.close()
    
    # plt.figure(figsize=(6, 6))
    
    # plt.plot(time_FOM, shifting_amount_FOM, label="FOM", color='red')
    # plt.plot(time_FOM, shifting_amount_SRG, label="SR-Galerkin ROM", color='blue')
    # plt.plot(time_FOM, shifting_amount_SRO_w_reproj, label="SR-OpInf ROM, w/ re-proj", linestyle='--', color='orange')
    
    # plt.xticks([120, 125, 130], fontsize = 15)
    # plt.yticks([-2.4, -2.7, -3.0, -3.3], fontsize = 15)
    
    # plt.legend(fontsize = 15, loc='upper right')
    # plt.show()
    # plt.close()
    
    # endregion
    
    # region 8: plot the contour plots of testing snapshots, comparing FOMs with ROMs
    
    filepath_prefix = r"C:\Users\ys5910\Desktop\SROpInf_working_on\data\dim_4"

    testing_start_time = 30
    testing_end_time   = 40
    
    num_points = int((testing_end_time - testing_start_time) / dt_sample) + 1
    time_FOM = np.linspace(testing_start_time, testing_end_time, num_points, endpoint=True)

    sol_fitted_FOM = np.loadtxt(rf"{filepath_prefix}\ks_solution_fitted_FOM.txt")
    ubar, basis, singular_values = POD(sol_fitted_FOM, num_modes=4)

    sol_FOM_testing = np.loadtxt(rf"{filepath_prefix}\ks_solution_FOM_testing.txt")
    sol_fitted_FOM_testing = np.loadtxt(rf"{filepath_prefix}\ks_solution_fitted_FOM_testing.txt")
    sol_SRO_w_reproj_testing = np.loadtxt(rf"{filepath_prefix}\ks_solution_SR_OpInf_ROM_w_reproj_testing.txt")
    sol_fitted_SRO_w_reproj_testing = np.loadtxt(rf"{filepath_prefix}\ks_solution_fitted_SR_OpInf_ROM_w_reproj_testing.txt")

    relative_error_testing = np.linalg.norm(sol_SRO_w_reproj_testing - sol_FOM_testing) / np.linalg.norm(sol_FOM_testing)

    sol_FOM_testing = np.vstack([sol_FOM_testing, sol_FOM_testing[0]])
    sol_SRO_w_reproj_testing = np.vstack([sol_SRO_w_reproj_testing, sol_SRO_w_reproj_testing[0]])

    x = np.linspace(0, L, N_space + 1, endpoint=True)
    
    x_grid, t_grid = np.meshgrid(x, time_FOM)
        
    min_colorbar = -15
    max_colorbar = 12
    levels = np.linspace(-15, 12, 10)
    
    fig = plt.figure(figsize=(6, 6))
    plt.contourf(x_grid, t_grid, sol_SRO_w_reproj_testing.T, levels = levels, vmin = min_colorbar, vmax = max_colorbar, cmap='viridis')
    plt.xticks([0, np.pi, 2 * np.pi], [r'$0$', r'$\pi$', r'$2\pi$'], fontsize = 15)
    plt.yticks([30, 35, 40], fontsize = 15)
    cbar = plt.colorbar(ticks=levels)
    cbar.ax.tick_params(labelsize=15)
    # plt.title(f"SR-OpInf ROM w/ re-proj data \n relative error $={NRMSE_prediction_u_SRO_w_reproj_test:.2f}\%$", fontsize=20)
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(6, 6))
    plt.contourf(x_grid, t_grid, sol_FOM_testing.T, levels = levels, vmin = min_colorbar, vmax = max_colorbar, cmap='viridis')
    plt.xticks([0, np.pi, 2 * np.pi], [r'$0$', r'$\pi$', r'$2\pi$'], fontsize = 15)
    plt.yticks([30, 35, 40], fontsize = 15)
    cbar = plt.colorbar(ticks=levels)
    cbar.ax.tick_params(labelsize=15)
    # plt.title(f"FOM \n", fontsize=20)
    plt.show()
    plt.close()

    state_FOM_testing = full_to_proj(sol_fitted_FOM_testing, ubar, basis)
    state_SRO_w_reproj_testing = full_to_proj(sol_fitted_SRO_w_reproj_testing, ubar, basis)

    plt.figure(figsize=(6, 6))
    
    time_span_POD_amplitudes = 2
    num_snapshots_POD_amplitudes = int(time_span_POD_amplitudes / dt_sample) + 1

    plt.plot(time_FOM[:num_snapshots_POD_amplitudes], state_FOM_testing[0, :num_snapshots_POD_amplitudes], label="FOM", color='red')
    plt.plot(time_FOM[:num_snapshots_POD_amplitudes], state_SRO_w_reproj_testing[0, :num_snapshots_POD_amplitudes], label="SR-OpInf ROM, w/ re-proj", linestyle='--', color='blue',)

    plt.xticks([30, 31, 32], fontsize = 15)
    plt.yticks([-5, 0, 5], fontsize = 15)
    
    plt.legend(fontsize = 15, loc='upper right')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.plot(time_FOM[:num_snapshots_POD_amplitudes], state_FOM_testing[2, :num_snapshots_POD_amplitudes], label="FOM", color='red')
    plt.plot(time_FOM[:num_snapshots_POD_amplitudes], state_SRO_w_reproj_testing[2, :num_snapshots_POD_amplitudes], label="SR-OpInf ROM, w/ re-proj", linestyle='--', color='blue',)

    plt.xticks([30, 31, 32], fontsize = 15)
    plt.yticks([-2, 0, 2], fontsize = 15)
    
    plt.legend(fontsize = 15, loc='upper right')
    plt.show()
    plt.close()
    
    # endregion
    
    """
    # region 8: plot the comparison of the reduced operators between SROpInf w/ reproj and SRGalerkin
    
    # filepath_prefix = r"D:\Documents\Journal Publishments\2025\ACM 2025 Symmetry-reduced model reduction of shift-equivariant systems via operator inference\Data\raw_data_dim_3"
    
    # reduced_operators_SRO_w_reproj = np.load(rf"{filepath_prefix}\reduced_operators_SR_OpInf_POD_ROM_w_reproj.npz")
    # reduced_operators_SRG          = np.load(rf"{filepath_prefix}\reduced_operators_SR_Galerkin_POD_ROM.npz")

    # SRO_w_reproj_d_vec = reduced_operators_SRO_w_reproj["d_vec"]
    # SRO_w_reproj_B_mat = reduced_operators_SRO_w_reproj["B_mat"]
    # SRO_w_reproj_H_mat = reduced_operators_SRO_w_reproj["H_mat"]
    # SRO_w_reproj_e     = reduced_operators_SRO_w_reproj["e"]
    # SRO_w_reproj_P_mat = reduced_operators_SRO_w_reproj["P_mat"]
    # SRO_w_reproj_Q_mat = reduced_operators_SRO_w_reproj["Q_mat"]
    
    # SRG_d_vec = reduced_operators_SRG["d_vec"]
    # SRG_B_mat = reduced_operators_SRG["B_mat"]
    # SRG_H_mat = reduced_operators_SRG["H_mat"]
    # SRG_e     = reduced_operators_SRG["e"]
    # SRG_P_mat = reduced_operators_SRG["P_mat"]
    # SRG_Q_mat = reduced_operators_SRG["Q_mat"]
    
    # ROM_ndim = 3
    
    # SRG_H_mat_kron = np.zeros((ROM_ndim, ROM_ndim*(ROM_ndim + 1) // 2))
    
    # for i in range(ROM_ndim):
    #     for j in range(ROM_ndim):
    #         SRG_H_mat_kron[i, j * ROM_ndim - j*(j - 1)//2] = SRG_H_mat[i, j, j]
    #         for k in range(j + 1, ROM_ndim):
    #             SRG_H_mat_kron[i, j * ROM_ndim - j*(j - 1)//2 + (k - j)] = SRG_H_mat[i, j, k] + SRG_H_mat[i, k, j]
    
    # H_diff = np.abs(SRG_H_mat_kron - SRO_w_reproj_H_mat)
    # B_diff = np.abs(SRG_B_mat - SRO_w_reproj_B_mat)
    # d_diff = np.abs(SRG_d_vec - SRO_w_reproj_d_vec)
    
    # # # Create a figure with subplots.
    # # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # # # Plot heatmap for B_diff.
    # # im0 = axs[0].imshow(H_diff, aspect='auto', cmap='viridis')
    # # axs[0].set_xlabel('Column index j')
    # # axs[0].set_ylabel('Row index i')
    # # fig.colorbar(im0, ax=axs[0])
    
    # # # Plot heatmap for C_diff.
    # # im1 = axs[1].imshow(B_diff, aspect='auto', cmap='viridis')
    # # axs[1].set_xlabel('Column index j')
    # # axs[1].set_ylabel('Row index i')
    # # fig.colorbar(im1, ax=axs[1])
    
    # # # Plot heatmap for D_diff.
    # # # If D_diff is a 1D vector, we reshape it into a 2D column matrix for visualization.
    # # if d_diff.ndim == 1:
    # #     d_diff_plot = d_diff.reshape(-1, 1)
    # # else:
    # #     d_diff_plot = d_diff
    
    # # im2 = axs[2].imshow(d_diff_plot, aspect='auto', cmap='viridis')
    # # axs[2].set_ylabel('Row index i')
    # # fig.colorbar(im2, ax=axs[2])
    
    # # plt.tight_layout()
    # # plt.show()
    # # plt.close()
    
    # # Create a figure with subplots.
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # # Plot heatmap for B_diff.
    # im0 = axs[0].imshow(H_diff/np.linalg.norm(SRG_H_mat_kron), aspect='auto', cmap='viridis')
    # axs[0].set_xticks(np.arange(0, ROM_ndim*(ROM_ndim + 1) // 2))
    # axs[0].set_yticks(np.arange(0, ROM_ndim))
    # axs[0].set_xlabel('Column index j')
    # axs[0].set_ylabel('Row index i')
    # fig.colorbar(im0, ax=axs[0])
    
    # # Plot heatmap for C_diff.
    # im1 = axs[1].imshow(B_diff/np.linalg.norm(SRG_B_mat), aspect='auto', cmap='viridis')
    # axs[1].set_xticks(np.arange(0, ROM_ndim))
    # axs[1].set_yticks(np.arange(0, ROM_ndim))
    # axs[1].set_xlabel('Column index j')
    # axs[1].set_ylabel('Row index i')
    # fig.colorbar(im1, ax=axs[1])
    
    # # Plot heatmap for D_diff.
    # # If D_diff is a 1D vector, we reshape it into a 2D column matrix for visualization.
    # if d_diff.ndim == 1:
    #     d_diff_plot = d_diff.reshape(-1, 1)
    # else:
    #     d_diff_plot = d_diff
    
    # im2 = axs[2].imshow(d_diff_plot/np.linalg.norm(SRG_d_vec), aspect='auto', cmap='viridis')
    # axs[2].set_xticks([])       # removes tick marks
    # axs[2].set_xticklabels([])  # removes tick labels
    # axs[2].set_yticks(np.arange(0, ROM_ndim))
    # axs[2].set_ylabel('Row index i')
    # fig.colorbar(im2, ax=axs[2])
    
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # im0 = axs[0].imshow(SRO_w_reproj_H_mat, aspect='auto', cmap='viridis')
    # axs[0].set_xticks(np.arange(0, ROM_ndim*(ROM_ndim + 1) // 2))
    # axs[0].set_yticks(np.arange(0, ROM_ndim))
    # axs[0].set_xlabel('Column index j')
    # axs[0].set_ylabel('Row index i')
    # fig.colorbar(im0, ax=axs[0])
    
    # im1 = axs[1].imshow(SRO_w_reproj_B_mat, aspect='auto', cmap='viridis')
    # axs[1].set_xticks(np.arange(0, ROM_ndim))
    # axs[1].set_yticks(np.arange(0, ROM_ndim))
    # axs[1].set_xlabel('Column index j')
    # axs[1].set_ylabel('Row index i')
    # fig.colorbar(im1, ax=axs[1])
    
    # if SRO_w_reproj_d_vec.ndim == 1:
    #     SRO_w_reproj_d_vec_plot = SRO_w_reproj_d_vec.reshape(-1, 1)
    # else:
    #     SRO_w_reproj_d_vec_plot = SRO_w_reproj_d_vec
    
    # im2 = axs[2].imshow(SRO_w_reproj_d_vec_plot, aspect='auto', cmap='viridis')
    # axs[2].set_xticks([])       # removes tick marks
    # axs[2].set_xticklabels([])  # removes tick labels
    # axs[2].set_yticks(np.arange(0, ROM_ndim))
    # axs[2].set_ylabel('Row index i')
    # fig.colorbar(im2, ax=axs[2])
    
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # im0 = axs[0].imshow(SRG_H_mat_kron, aspect='auto', cmap='viridis')
    # axs[0].set_xticks(np.arange(0, ROM_ndim*(ROM_ndim + 1) // 2))
    # axs[0].set_yticks(np.arange(0, ROM_ndim))
    # axs[0].set_xlabel('Column index j')
    # axs[0].set_ylabel('Row index i')
    # fig.colorbar(im0, ax=axs[0])
    
    # im1 = axs[1].imshow(SRG_B_mat, aspect='auto', cmap='viridis')
    # axs[1].set_xticks(np.arange(0, ROM_ndim))
    # axs[1].set_yticks(np.arange(0, ROM_ndim))
    # axs[1].set_xlabel('Column index j')
    # axs[1].set_ylabel('Row index i')
    # fig.colorbar(im1, ax=axs[1])

    # if SRG_d_vec.ndim == 1:
    #     SRG_d_vec_plot = SRG_d_vec.reshape(-1, 1)
    # else:
    #     SRG_d_vec_plot = SRG_d_vec
    
    # im2 = axs[2].imshow(SRG_d_vec_plot, aspect='auto', cmap='viridis')
    # axs[2].set_xticks([])       # removes tick marks
    # axs[2].set_xticklabels([])  # removes tick labels
    # axs[2].set_yticks(np.arange(0, ROM_ndim))
    # axs[2].set_ylabel('Row index i')
    # fig.colorbar(im2, ax=axs[2])
    
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    
    # endregion
    
    # # below are possibly to be retired results that are proposed but can't be realized in our current framework
    
    # region 9: contourf plots of ROM starting from u_init = u_0(x + 1) for n = 4. SROpInf w/ reproj, SRGalerkin and projected FOM (and FOM)
    
    # filepath_prefix = r"D:\Documents\Journal Publishments\2025\ACM 2025 Symmetry-reduced model reduction of shift-equivariant systems via operator inference\Data\raw_data_dim_4_new_init"
    
    # prediction_u_FOM_new_init = np.loadtxt(rf"{filepath_prefix}\KSE_FOM_prediction_u.txt")
    # prediction_u_SRO_w_reproj_new_init = np.loadtxt(rf"{filepath_prefix}\prediction_u_SRO_w_reproj.txt")
    
    # # prediction_u_SRG_new_init = np.loadtxt(rf"{filepath_prefix}\prediction_u_SRG.txt")
    
    # prediction_u_SRO_w_reproj_new_init_diff = prediction_u_SRO_w_reproj_new_init - prediction_u_FOM_new_init
    # # prediction_u_SRG_new_init_diff = prediction_u_SRG_new_init - prediction_u_FOM_new_init
    
    # NRMSE_prediction_u_SRO_w_reproj_new_init = 100 * np.sqrt(np.mean(prediction_u_SRO_w_reproj_new_init_diff**2)) / np.sqrt(np.mean(prediction_u_FOM_new_init**2))
    # # NRMSE_prediction_u_SRG_new_init = 100 * np.sqrt(np.mean(prediction_u_SRG_new_init_diff**2)) / np.sqrt(np.mean(prediction_u_FOM_new_init**2))
    
    # time_contourf_SRO_w_reproj_new_init = dt_sample * np.arange(prediction_u_SRO_w_reproj_new_init.shape[1])
    # x_grid, t_grid = np.meshgrid(x, time_contourf_SRO_w_reproj_new_init)
    
    # min_colorbar = -15
    # max_colorbar = 12
    # levels = np.linspace(-15, 12, 10)
    
    # fig = plt.figure(figsize=(6, 6))
    # plt.contourf(x_grid, t_grid, prediction_u_SRO_w_reproj_new_init.T, levels = levels, vmin = min_colorbar, vmax = max_colorbar, cmap='viridis')
    # plt.xlabel("x")
    # plt.ylabel("t")
    # plt.colorbar(ticks = levels)
    # plt.title(f"SR-OpInf ROM w/ re-proj data \n relative error $={NRMSE_prediction_u_SRO_w_reproj_new_init:.2f}\%$", fontsize=20)
    # plt.show()
    # plt.close()
    
    # time_contourf_SRG_new_init = dt_sample * np.arange(prediction_u_SRG_new_init.shape[1])
    # x_grid, t_grid = np.meshgrid(x, time_contourf_SRG_new_init)
    
    # fig = plt.figure(figsize=(6, 6))
    # plt.contourf(x_grid, t_grid, prediction_u_SRG_new_init.T, levels = levels, vmin = min_colorbar, vmax = max_colorbar, cmap='viridis')
    # plt.xlabel("x")
    # plt.ylabel("t")
    # plt.colorbar(ticks = levels)
    # plt.title(rf"NRMSE $={NRMSE_prediction_u_SRG_new_init:.2f}\%$", fontsize=15)
    # plt.show()
    # plt.close()
        
    # fig = plt.figure(figsize=(6, 6))
    # plt.contourf(x_grid, t_grid, prediction_u_FOM_new_init.T, levels = levels, vmin = min_colorbar, vmax = max_colorbar, cmap='viridis')
    # plt.xlabel("x")
    # plt.ylabel("t")
    # plt.colorbar(ticks = levels)
    # plt.title(f"FOM \n", fontsize=20)
    # plt.show()
    # plt.close()
    
    # endregion
    
    # region 9: contourf plots of ROM starting from u_init = ubar not u0 for n = 3. SROpInf w/ reproj, SRGalerkin and projected FOM (and FOM)
    
    # filepath_prefix = r"D:\Documents\Journal Publishments\2025\ACM 2025 Symmetry-reduced model reduction of shift-equivariant systems via operator inference\Data\raw_data_dim_3_init_ubar"
    
    # prediction_u_FOM_new_init = np.loadtxt(rf"{filepath_prefix}\prediction_u_FOM.txt")
    # prediction_u_SRO_w_reproj_new_init = np.loadtxt(rf"{filepath_prefix}\prediction_u_SRO.txt")
    # prediction_u_SRG_new_init = np.loadtxt(rf"{filepath_prefix}\prediction_u_SRG.txt")
    
    # min_colorbar = -15
    # max_colorbar = 12
    # levels = np.linspace(-15, 12, 10)
    
    # time_contourf_SRO_w_reproj_new_init = dt_sample * np.arange(prediction_u_SRO_w_reproj_new_init.shape[1])
    # x_grid, t_grid = np.meshgrid(x, time_contourf_SRO_w_reproj_new_init)
    
    # fig = plt.figure(figsize=(6, 6))
    # plt.contourf(x_grid, t_grid, prediction_u_SRO_w_reproj_new_init.T, levels = levels, vmin = min_colorbar, vmax = max_colorbar, cmap='viridis')
    # plt.xlabel("x")
    # plt.ylabel("t")
    # plt.colorbar(ticks = levels)
    # # plt.title(rf"NRMSE $={NRMSE_prediction_u_SRO_w_reproj:.2f}\%$", fontsize=15)
    # plt.show()
    # plt.close()
    
    # time_contourf_SRG = dt_sample * np.arange(prediction_u_SRG_new_init.shape[1])
    # x_grid, t_grid = np.meshgrid(x, time_contourf_SRG)
    
    # fig = plt.figure(figsize=(6, 6))
    # plt.contourf(x_grid, t_grid, prediction_u_SRG_new_init.T, levels = levels, vmin = min_colorbar, vmax = max_colorbar, cmap='viridis')
    # plt.xlabel("x")
    # plt.ylabel("t")
    # plt.colorbar(ticks = levels)
    # # plt.title(rf"NRMSE $={NRMSE_prediction_u_SRO_w_reproj:.2f}\%$", fontsize=15)
    # plt.show()
    # plt.close()
    
    # time_contourf_FOM = dt_sample * np.arange(prediction_u_FOM_new_init.shape[1])
    # x_grid, t_grid = np.meshgrid(x, time_contourf_FOM)
    
    # fig = plt.figure(figsize=(6, 6))
    # plt.contourf(x_grid, t_grid, prediction_u_FOM_new_init.T, levels = levels, vmin = min_colorbar, vmax = max_colorbar, cmap='viridis')
    # plt.xlabel("x")
    # plt.ylabel("t")
    # plt.colorbar(ticks = levels)
    # # plt.title(rf"NRMSE $={NRMSE_prediction_u_SRO_w_reproj:.2f}\%$", fontsize=15)
    # plt.show()
    # plt.close()

    # endregion
    
    # # # region 10: contourf plots of ROM of nu = 2.25/87 interpolated from nu = 2/87, 2.5/87, 3/87, 3.5/87, 4/87, 4.5/87, 5/87, 5.5/87, 6/87, SROpInf w/ reproj, SRGalerkin and projected FOM (and FOM)
    
    # # # endregion
    
    # # # region 11: contourf plots of ROM of nu = 8/87 interpolated from nu = 2/87, 2.5/87, 3/87, 3.5/87, 4/87, 4.5/87, 5/87, 5.5/87, 6/87, SROpInf w/ reproj, SRGalerkin and projected FOM (and FOM)
    
    # # # endregion
    
    """
    
    
if __name__ == "__main__":
    main()