#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

from SROpInf.models.ks import KuramotoSivashinsky, spatial_translation
from SROpInf.sample import POD, sample, TrajectoryData
from SROpInf.train import train_SROpInf

def main():

    # region 0: initialization
    nmodes = 20
    L = 2 * np.pi
    nu = 4 / 87
    Nx = 2 * nmodes  # number of collocation points (must be even, better be a factor of 4)
    x = np.linspace(0, L, Nx, endpoint=False)
    dx = x[1] - x[0]
    start_time = 120
    end_time = 130
    dt = 1e-3
    dt_snapshots = 0.01
    N_snapshots = int((end_time - start_time) / dt_snapshots) + 1
    T = np.arange(N_snapshots) * dt_snapshots + start_time

    num_traj = 1

    if start_time != 0:
        print("Loading initial condition...")
        # compute the initial condition at t = start_time
        u_init = np.loadtxt(f"ks_initial_condition_{start_time}.txt")

    elif FileNotFoundError or start_time == 0:
        print("Taking default initial condition")
        # compute the initial condition at t = 0
        u_init = -np.sin(x) + 2 * np.cos(2 * x) + 3 * np.cos(3 * x) - 4 * np.sin(4 * x)

    u_init = u_init.reshape((Nx, num_traj))

    u_template = np.cos(x)

    shifting_operation = partial(spatial_translation, Lx = L)
    shifting_operation.Lx = L  # Add custom attribute for later access

    # endregion

    # region 1: compute the FOM solution

    FOM = KuramotoSivashinsky(nu, Nx, L)

    FOM_data = sample(num_traj = num_traj, num_snapshots = N_snapshots, u_init = u_init, 
                      u_template = u_template, spatial_translation = shifting_operation,
                      model = FOM, timestepper = "rk3cn", timespan = end_time - start_time, timestep = dt)
    
    sol_FOM = FOM_data.sol[:, :, 0]
    sol_fitted_FOM = FOM_data.sol_fitted[:, :, 0]
    shifting_amount_FOM = FOM_data.shifting_amount[:, 0]
    shifting_speed_FOM = FOM_data.shifting_speed[:, 0]

    # np.savetxt("ks_initial_condition_120.txt", sol_FOM[:, int(120/dt_snapshots)])

    # fig, ax = plt.subplots()
    # ax.contourf(x, T, sol_FOM.T)
    # ax.set_xlim(0, L)
    # ax.set_xlabel("x")
    # ax.set_ylabel("t")
    # ax.set_title(f"Kuramoto-Sivashinsky solution, L = {L}, ({nmodes} Fourier modes)")
    # plt.show()

    np.savetxt("ks_solution_FOM.txt", sol_FOM)
    np.savetxt("ks_solution_fitted_FOM.txt", sol_fitted_FOM)
    np.savetxt("ks_solution_shifting_amount_FOM.txt", shifting_amount_FOM)
    np.savetxt("ks_solution_shifting_speed_FOM.txt", shifting_speed_FOM)

    FOM_data.save("ks_solution_FOM.npz")

    # endregion

    # region 2: compute the SR-Galerkin ROM solution

    num_modes = 4

    FOM_data = TrajectoryData.load("ks_solution_FOM.npz")
 
    sol_fitted_FOM = np.loadtxt("ks_solution_fitted_FOM.txt")
    sol_FOM        = np.loadtxt("ks_solution_FOM.txt")

    ubar, POD_basis, POD_singular_values = POD(sol_fitted_FOM, num_modes)

    # also, we first compute the projection error of FOM snapshots onto the POD basis

    SR_Galerkin_ROM = FOM.symmetry_reduced_project(V = POD_basis, template = u_template, W = None, bias = ubar)

    SR_Galerkin_ROM_data = sample(num_traj = num_traj, num_snapshots = N_snapshots, u_init = u_init,
                                  u_template = u_template, spatial_translation = shifting_operation,
                                  model = SR_Galerkin_ROM, timestepper = "rkf45", timespan = end_time - start_time,
                                  timestep = dt, err_tol = 1e-6)
    
    sol_SR_Galerkin_ROM = SR_Galerkin_ROM_data.sol[:, :, 0]
    sol_fitted_SR_Galerkin_ROM = SR_Galerkin_ROM_data.sol_fitted[:, :, 0]
    shifting_amount_SR_Galerkin_ROM = SR_Galerkin_ROM_data.shifting_amount[:, 0]
    shifting_speed_SR_Galerkin_ROM = SR_Galerkin_ROM_data.shifting_speed[:, 0]

    # fig, ax = plt.subplots()
    # ax.contourf(x, T, sol_SR_Galerkin_ROM.T)
    # ax.set_xlim(0, L)
    # ax.set_xlabel("x")
    # ax.set_ylabel("t")
    # ax.set_title(f"Kuramoto-Sivashinsky solution, L = {L}, ({num_modes} POD modes)")
    # plt.show()

    relative_error_SRG = np.linalg.norm(sol_SR_Galerkin_ROM - sol_FOM) / np.linalg.norm(sol_FOM)

    sol_fitted_FOM_projected = SR_Galerkin_ROM.latent_to_full(SR_Galerkin_ROM.full_to_latent(sol_fitted_FOM))
    projection_error = np.linalg.norm(sol_fitted_FOM - sol_fitted_FOM_projected) / np.linalg.norm(sol_fitted_FOM)

    np.savetxt("ks_solution_SR_Galerkin_ROM.txt", sol_SR_Galerkin_ROM)
    np.savetxt("ks_solution_fitted_SR_Galerkin_ROM.txt", sol_fitted_SR_Galerkin_ROM)
    np.savetxt("ks_solution_shifting_amount_SR_Galerkin_ROM.txt", shifting_amount_SR_Galerkin_ROM)
    np.savetxt("ks_solution_shifting_speed_SR_Galerkin_ROM.txt", shifting_speed_SR_Galerkin_ROM)

    # endregion

    # region 3: compute the SR-OpInf ROM solution without re-projection technique

    # num_modes = 3
    sol_fitted_FOM = np.loadtxt("ks_solution_fitted_FOM.txt")
    sol_FOM        = np.loadtxt("ks_solution_FOM.txt")
    FOM_data       = TrajectoryData.load("ks_solution_FOM.npz")
    max_steps      = 1000

    ubar, POD_basis, POD_singular_values = POD(sol_fitted_FOM, num_modes)

    SR_OpInf_ROM = FOM.symmetry_reduced_OpInf_initialization(V = POD_basis, template = u_template, W = None, bias = ubar)
    SR_OpInf_ROM, training_loss = train_SROpInf(model = SR_OpInf_ROM, num_modes = num_modes, data = FOM_data,
                                                max_steps = max_steps, grad_tol = 1e-6)
    
    SR_OpInf_ROM_data = sample(num_traj = num_traj, num_snapshots = N_snapshots, u_init = u_init,
                               u_template = u_template, spatial_translation = shifting_operation,
                               model = SR_OpInf_ROM, timestepper = "rkf45", timespan = end_time - start_time,
                               timestep = dt, err_tol = 1e-6)

    sol_SR_OpInf_ROM = SR_OpInf_ROM_data.sol[:, :, 0]
    sol_fitted_SR_OpInf_ROM = SR_OpInf_ROM_data.sol_fitted[:, :, 0]
    shifting_amount_SR_OpInf_ROM = SR_OpInf_ROM_data.shifting_amount[:, 0]
    shifting_speed_SR_OpInf_ROM = SR_OpInf_ROM_data.shifting_speed[:, 0]

    # fig, ax = plt.subplots()
    # ax.contourf(x, T, sol_SR_OpInf_ROM.T)
    # ax.set_xlim(0, L)
    # ax.set_xlabel("x")
    # ax.set_ylabel("t")
    # ax.set_title(f"Kuramoto-Sivashinsky solution, L = {L}, ({num_modes} POD modes)")
    # plt.show()

    np.savetxt("SR_OpInf_ROM_wo_reproj_training_loss.txt", training_loss)
    np.savetxt("ks_solution_SR_OpInf_ROM_wo_reproj.txt", sol_SR_OpInf_ROM)
    np.savetxt("ks_solution_fitted_SR_OpInf_ROM_wo_reproj.txt", sol_fitted_SR_OpInf_ROM)
    np.savetxt("ks_solution_shifting_amount_SR_OpInf_ROM_wo_reproj.txt", shifting_amount_SR_OpInf_ROM)
    np.savetxt("ks_solution_shifting_speed_SR_OpInf_ROM_wo_reproj.txt", shifting_speed_SR_OpInf_ROM)

    relative_error_SRO_wo_reproj = np.linalg.norm(sol_SR_OpInf_ROM - sol_FOM) / np.linalg.norm(sol_FOM)

    # endregion

    # region 4: compute the SR-OpInf ROM solution with re-projection technique

    # num_modes = 3
    sol_fitted_FOM = np.loadtxt("ks_solution_fitted_FOM.txt")
    sol_FOM        = np.loadtxt("ks_solution_FOM.txt")
    FOM_data       = TrajectoryData.load("ks_solution_FOM.npz")

    ubar, POD_basis, POD_singular_values = POD(sol_fitted_FOM, num_modes)

    # FOM_data_re_proj = TrajectoryData.load("ks_solution_FOM_re_proj.npz")

    FOM_data_re_proj = sample(num_traj = num_traj, num_snapshots = N_snapshots, u_init = u_init, 
                      u_template = u_template, spatial_translation = shifting_operation,
                      model = FOM, timestepper = "rk3cn", timespan = end_time - start_time, timestep = dt,
                      re_proj_option = True, V = POD_basis, W = None, bias = ubar)
    
    FOM_data_re_proj.save("ks_solution_FOM_re_proj.npz")
    
    SR_OpInf_ROM = FOM.symmetry_reduced_OpInf_initialization(V = POD_basis, template = u_template, W = None, bias = ubar)
    SR_OpInf_ROM, training_loss = train_SROpInf(model = SR_OpInf_ROM, num_modes = num_modes, data = FOM_data_re_proj,
                                                max_steps = max_steps, grad_tol = 1e-6)
    
    SR_OpInf_ROM_data = sample(num_traj = num_traj, num_snapshots = N_snapshots, u_init = u_init,
                               u_template = u_template, spatial_translation = shifting_operation,
                               model = SR_OpInf_ROM, timestepper = "rkf45", timespan = end_time - start_time,
                               timestep = dt, err_tol = 1e-6)

    sol_SR_OpInf_ROM = SR_OpInf_ROM_data.sol[:, :, 0]
    sol_fitted_SR_OpInf_ROM = SR_OpInf_ROM_data.sol_fitted[:, :, 0]
    shifting_amount_SR_OpInf_ROM = SR_OpInf_ROM_data.shifting_amount[:, 0]
    shifting_speed_SR_OpInf_ROM = SR_OpInf_ROM_data.shifting_speed[:, 0]

    # fig, ax = plt.subplots()
    # ax.contourf(x, T, sol_SR_OpInf_ROM.T)
    # ax.set_xlim(0, L)
    # ax.set_xlabel("x")
    # ax.set_ylabel("t")
    # ax.set_title(f"Kuramoto-Sivashinsky solution, L = {L}, ({num_modes} POD modes)")
    # plt.show()

    relative_error_SRO_w_reproj = np.linalg.norm(sol_SR_OpInf_ROM - sol_FOM) / np.linalg.norm(sol_FOM)
    np.savetxt("SR_OpInf_ROM_w_reproj_training_loss.txt", training_loss)

    np.savetxt("ks_solution_SR_OpInf_ROM_w_reproj.txt", sol_SR_OpInf_ROM)
    np.savetxt("ks_solution_fitted_SR_OpInf_ROM_w_reproj.txt", sol_fitted_SR_OpInf_ROM)
    np.savetxt("ks_solution_shifting_amount_SR_OpInf_ROM_w_reproj.txt", shifting_amount_SR_OpInf_ROM)
    np.savetxt("ks_solution_shifting_speed_SR_OpInf_ROM_w_reproj.txt", shifting_speed_SR_OpInf_ROM)

    with open("relative error.txt", "w") as f:
        f.write(f"Relative error of SR-OpInf ROM w/o re-proj: {relative_error_SRO_wo_reproj:.6e}\n")
        f.write(f"Relative error of SR-OpInf ROM w/ re-proj: {relative_error_SRO_w_reproj:.6e}\n")
        f.write(f"Relative error of SR-Galerkin ROM: {relative_error_SRG:.6e}\n")
        f.write(f"Relative error of projected FOM snapshots: {projection_error:.6e}\n")

    # endregion

    # region 5: testing dataset simulate FOM and SROpInf ROM w/ reproj from t = 30 to t = 40

    # u_init = -np.sin(x) + 2 * np.cos(2 * x) + 3 * np.cos(3 * x) - 4 * np.sin(4 * x)

    # u_init = u_init.reshape((Nx, num_traj))

    # start_time = 0
    # end_time = 30

    # N_snapshots = int((end_time - start_time) / dt_snapshots) + 1

    # FOM_data = sample(num_traj = num_traj, num_snapshots = N_snapshots, u_init = u_init, 
    #                   u_template = u_template, spatial_translation = shifting_operation,
    #                   model = FOM, timestepper = "rk3cn", timespan = end_time - start_time, timestep = dt)

    # u_init = FOM_data.sol[:, -1, 0]  # take the last snapshot as the new initial condition

    # np.savetxt("ks_initial_condition_30.txt", u_init)

    # start_time = 30
    # end_time = 40
    # N_snapshots = int((end_time - start_time) / dt_snapshots) + 1
    # T = np.arange(N_snapshots) * dt_snapshots + start_time

    # u_init = np.loadtxt(f"ks_initial_condition_{start_time}.txt").reshape((Nx, num_traj))

    # FOM_data = sample(num_traj = num_traj, num_snapshots = N_snapshots, u_init = u_init, 
    #                   u_template = u_template, spatial_translation = shifting_operation,
    #                   model = FOM, timestepper = "rk3cn", timespan = end_time - start_time, timestep = dt)
    
    # sol_FOM = FOM_data.sol
    # fig, ax = plt.subplots()
    # ax.contourf(x, T, sol_FOM[:,:,0].T)
    # ax.set_xlim(0, L)
    # ax.set_xlabel("x")
    # ax.set_ylabel("t")
    # ax.set_title(f"Kuramoto-Sivashinsky solution, L = {L}, ({nmodes} Fourier modes)")
    # plt.show()

    # np.savetxt("ks_solution_FOM_testing.txt", sol_FOM[:, :, 0])
    # np.savetxt("ks_solution_fitted_FOM_testing.txt", FOM_data.sol_fitted[:, :, 0])

    # SR_OpInf_ROM_data = sample(num_traj = num_traj, num_snapshots = N_snapshots, u_init = u_init,
    #                            u_template = u_template, spatial_translation = shifting_operation,
    #                            model = SR_OpInf_ROM, timestepper = "rkf45", timespan = end_time - start_time,
    #                            timestep = dt, err_tol = 1e-6)
    
    # sol_SR_OpInf_ROM = SR_OpInf_ROM_data.sol
    # fig, ax = plt.subplots()
    # ax.contourf(x, T, sol_SR_OpInf_ROM[:,:,0].T)
    # ax.set_xlim(0, L)
    # ax.set_xlabel("x")
    # ax.set_ylabel("t")
    # ax.set_title(f"Kuramoto-Sivashinsky solution, L = {L}, ({num_modes} POD modes)")
    # plt.show()

    # np.savetxt("ks_solution_SR_OpInf_ROM_w_reproj_testing.txt", sol_SR_OpInf_ROM[:, :, 0])
    # np.savetxt("ks_solution_fitted_SR_OpInf_ROM_testing.txt", SR_OpInf_ROM_data.sol_fitted[:, :, 0])

if __name__ == "__main__":
    main()
