#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import ArrayLike
from SROpInf.custom_typing import Vector

from SROpInf.models.ks import KuramotoSivashinsky, spatial_translation

def template_fitting(sol: ArrayLike, sol_template: Vector, L: float, N: int, dx: float) -> ArrayLike:

        if sol.ndim == 1 or sol.shape[1] == 1:

            pull_back_range = np.linspace(-L, L, 10000 * N, endpoint=False)

            minimal_error = 1e5
            
            for c in pull_back_range:

                sol_tmp = spatial_translation(sol, -c, L)

                error = np.linalg.norm(sol_tmp - sol_template) # take the initial condition as the template
                # if this is a perfect TW, then the shifted state/rhs at the new timestep should equal to the previous state/rhs

                if error < minimal_error:
                    minimal_error = error
                    pull_back_amount =  -c

            return spatial_translation(sol, pull_back_amount, L), -pull_back_amount

        else:
            num_snapshots = sol.shape[1]

            pull_back_amount = np.zeros(num_snapshots)
            pull_back_range = np.linspace(-L, L, 10000 * N, endpoint=False)
            pull_back_step_range = np.linspace(-1 * dx, 1 * dx, 10 * N, endpoint=False)

            sol_fitted = np.zeros((N, num_snapshots))

            minimal_error = 1e5
            
            for c in pull_back_range:

                sol_snapshot_tmp = spatial_translation(sol[:, 0], - c, L)

                error = np.linalg.norm(sol_snapshot_tmp - sol_template) # take the initial condition as the template
                # if this is a perfect TW, then the shifted state/rhs at the new timestep should equal to the previous state/rhs

                if error < minimal_error:
                    minimal_error = error
                    pull_back_amount[0] =  -c

            sol_fitted[:, 0] = spatial_translation(sol[:, 0], pull_back_amount[0], L)

            for time in range(1, num_snapshots):

                minimal_error = 1e5

                pull_back_amount[time] = pull_back_amount[time - 1]

                for c in pull_back_step_range:

                    sol_snapshot_tmp = spatial_translation(sol[:, time], pull_back_amount[time - 1] - c, L)

                    error = np.linalg.norm(sol_snapshot_tmp - sol_template) # take the initial condition as the template
                    # if this is a perfect TW, then the shifted state/rhs at the new timestep should equal to the previous state/rhs

                    if error < minimal_error:
                        minimal_error = error
                        pull_back_amount[time] = pull_back_amount[time - 1] - c

                sol_fitted[:, time] = spatial_translation(sol[:, time], pull_back_amount[time], L)

            shifting_amount = -pull_back_amount

            return sol_fitted, shifting_amount

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
    dt_snapshots = 0.1
    N_snapshots = int((end_time - start_time) / dt_snapshots) + 1
    T = np.arange(N_snapshots) * dt_snapshots + start_time
    Nt = int((end_time - start_time) / dt) + 1

    # endregion

    # region 1: compute the FOM solution

    FOM = KuramotoSivashinsky(nu, Nx, L)
    stepper_FOM = FOM.get_timestepper(method = "rk3cn", dt = dt)

    # initial condition
    x = np.linspace(0, L, Nx, endpoint=False)

    if start_time != 0:
        print("Loading initial condition...")
        # compute the initial condition at t = start_time
        u_init = np.loadtxt(f"ks_initial_condition_{start_time}.txt")

    elif FileNotFoundError or start_time == 0:
        print("Taking default initial condition")
        # compute the initial condition at t = 0
        u_init = -np.sin(x) + 2 * np.cos(2 * x) + 3 * np.cos(3 * x) - 4 * np.sin(4 * x)

    sol = np.zeros((Nx, N_snapshots))
    
    print("Computing solution...")

    training_start_time = 120
    training_initial_condition_recorded = True

    sol = np.loadtxt("ks_solution.txt")

    # u = u_init
    
    # for t in range(Nt):
    #     if t % int(dt_snapshots / dt) == 0:
    #         sol[:, t // int(dt_snapshots / dt)] = u
    #         print("snapshot %4d / %4d" % (t // int(dt_snapshots / dt), N_snapshots))
    #         if not training_initial_condition_recorded and T[t // int(dt_snapshots / dt)] >= training_start_time:
    #             print("recording training initial condition")
    #             np.savetxt(f"ks_initial_condition_{training_start_time}.txt", u)
    #             training_initial_condition_recorded = True
    #     u = stepper_FOM.step(u)

    # fig, ax = plt.subplots()
    # ax.contourf(x, T, sol.T)
    # ax.set_xlim(0, L)
    # ax.set_xlabel("x")
    # ax.set_ylabel("t")
    # ax.set_title(f"Kuramoto-Sivashinsky solution, L = {L}, ({nmodes} modes)")
    # plt.show()

    # np.savetxt("ks_solution.txt", sol)

    # endregion
    
    # region 2: compute the standard Galerkin ROM solution
        
    # N_rom = 20
    
    # ubar = np.mean(sol, axis=1)
    
    # sol_centered = sol - ubar[:, np.newaxis]
    
    # U, S, Vh = np.linalg.svd(sol_centered, full_matrices=False)
    
    # basis = U[:, :N_rom]  # normalize the basis vectors
    
    # POD_Galerkin_ROM = FOM.project(V = basis, W = None, bias = ubar)
    
    # stepper_rom = POD_Galerkin_ROM.get_timestepper(method = "rkf45", dt = dt, err_tol = 1e-6)

    # sol_rom = np.zeros((Nx, N_snapshots))
    # state = POD_Galerkin_ROM.full_to_latent(u_init)
    # sol_rom[:, 0] = POD_Galerkin_ROM.latent_to_full(state)
    # counter = 1
    # time = start_time
    # timestep = stepper_rom.dt
    # state_old = state
    
    # print("Computing standard Galerkin ROM solution...")

    # while time <= end_time:
        
    #     if int((time - start_time)/dt_snapshots) >= counter and int((time - start_time)/dt_snapshots) <= counter + 1:
            
    #         # if the timestep of state_old is closer to the sample moment than the timestep of state, then we keep state_old, otherwise we keep state

    #         distance_old_moment = np.abs(time - timestep - start_time - counter * dt_snapshots)
    #         distance_new_moment = np.abs(time - start_time - counter * dt_snapshots)
            
    #         if distance_old_moment <= distance_new_moment:
    #             sol_rom[:, counter] = POD_Galerkin_ROM.latent_to_full(state_old)
    #         else:
    #             sol_rom[:, counter] = POD_Galerkin_ROM.latent_to_full(state)
    #         counter = counter + 1
        
    #     state_old = state
    #     state = stepper_rom.step(state)
    #     timestep = stepper_rom.dt
    #     time += timestep

    #     print(f"t = {time}, dt = {timestep}.")
        
    #     if not stepper_rom.stability:
            
    #         print("The ROM simulation is terminated due to too small timestep and the latest recorded snapshot is set to infty.")
    #         sol_rom[:, counter:] = np.inf
    #         break
            
    #     distance_old_moment = np.abs(time - timestep - start_time - counter * dt_snapshots)
    #     distance_new_moment = np.abs(time - start_time - counter * dt_snapshots)
        
    #     if distance_old_moment <= distance_new_moment:
    #         sol_rom[:, counter] = POD_Galerkin_ROM.latent_to_full(state_old)
    #     else:
    #         sol_rom[:, counter] = POD_Galerkin_ROM.latent_to_full(state)
        
    # fig, ax = plt.subplots()
    # ax.contourf(x, T, sol_rom.T)
    # ax.set_xlim(0, L)
    # ax.set_xlabel("x")
    # ax.set_ylabel("t")
    # ax.set_title(f"Kuramoto-Sivashinsky solution, L = {L}, ({N_rom} POD modes)")
    # plt.show()

    # endregion

    # region 3: compute the symmetry-reduced Galerkin ROM solution

    # try to formulate a projection matrix
    N_rom = 4

    u_template = np.cos(x)

    sol = np.loadtxt("ks_solution.txt")

    sol_fitted, sol_shifting_amount_FOM = template_fitting(sol, u_template, L, Nx, dx)

    print(f"last shifting amount: {sol_shifting_amount_FOM[-1]}")

    plt.contourf(x, T, sol_fitted.T)
    plt.xlim(0, L)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(f"Kuramoto-Sivashinsky solution, L = {L}, ({N_rom} POD modes)")
    plt.show()

    ubar = np.mean(sol_fitted, axis=1)

    sol_fitted_centered = sol_fitted - ubar[:, np.newaxis]

    C = sol_fitted_centered @ sol_fitted_centered.T

    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    basis = eigenvectors[:, :N_rom] * np.sqrt(Nx)

    plt.plot(x, basis[:, 0], label='Mode 1')
    plt.plot(x, basis[:, 1], label='Mode 2')
    plt.plot(x, basis[:, 2], label='Mode 3')
    plt.plot(x, basis[:, 3], label='Mode 4')
    plt.xlabel('x')
    plt.ylabel('Basis Functions')
    plt.title('POD Basis Functions')
    plt.legend()
    plt.grid()
    plt.show()

    SR_POD_Galerkin_ROM = FOM.symmetry_reduced_project(V = basis, template = u_template, W = None, bias = ubar)
    
    sol_fitted_projection = SR_POD_Galerkin_ROM.latent_to_full(SR_POD_Galerkin_ROM.full_to_latent(sol_fitted))

    projection_error = np.linalg.norm(sol_fitted - sol_fitted_projection) / np.linalg.norm(sol_fitted)
    print(f"Projection error: {projection_error:.4e}")

    stepper_SR_rom = SR_POD_Galerkin_ROM.get_timestepper(method = "rkf45", dt = dt, err_tol = 1e-6)

    sol_SR_rom = np.zeros((Nx, N_snapshots))
    sol_SR_rom_fitted = np.zeros((Nx, N_snapshots))
    sol_shifting_amount_rom = np.zeros(N_snapshots)
    sol_shifting_speed_rom = np.zeros(N_snapshots)

    u_init_fitted, shifting_amount = template_fitting(u_init, u_template, L, Nx, dx)
    state = SR_POD_Galerkin_ROM.full_to_latent(u_init_fitted)
    shifting_speed = SR_POD_Galerkin_ROM.compute_shifting_speed(state)

    sol_SR_rom_fitted[:, 0] = SR_POD_Galerkin_ROM.latent_to_full(state)
    sol_SR_rom[:, 0] = spatial_translation(sol_SR_rom_fitted[:, 0], shifting_amount, L)
    sol_shifting_amount_rom[0] = shifting_amount
    sol_shifting_speed_rom[0] = shifting_speed

    state_old = state
    timestep = stepper_SR_rom.dt
    time = start_time
    counter = 1

    # inner_product = np.array([SR_POD_Galerkin_ROM.test_inner_product(sol_SR_rom_fitted[:, 0], u_template)])

    print("Computing SR-Galerkin ROM solution...")
    # print(f"t = {time}, dt = {timestep}, cdot_numer = {SR_POD_Galerkin_ROM.cdot_numerator}, cdot_denom = {SR_POD_Galerkin_ROM.cdot_denominator}, cdot = {shifting_speed}, c = {shifting_amount}, inner_product = {inner_product[-1]}.")
    print(f"t = {time}, dt = {timestep}, cdot_numer = {SR_POD_Galerkin_ROM.cdot_numerator}, cdot_denom = {SR_POD_Galerkin_ROM.cdot_denominator}, cdot = {shifting_speed}, c = {shifting_amount}.")
  

    while time <= end_time:
        
        if int((time - start_time)/dt_snapshots) >= counter and int((time - start_time)/dt_snapshots) <= counter + 1:
            
            # if the timestep of state_old is closer to the sample moment than the timestep of state, then we keep state_old, otherwise we keep state

            distance_old_moment = np.abs(time - timestep - start_time - counter * dt_snapshots)
            distance_new_moment = np.abs(time - start_time - counter * dt_snapshots)
            
            if distance_old_moment <= distance_new_moment:
                sol_SR_rom_fitted[:, counter] = SR_POD_Galerkin_ROM.latent_to_full(state_old)
                sol_SR_rom[:, counter] = spatial_translation(sol_SR_rom_fitted[:, counter], shifting_amount_old, L)
                sol_shifting_amount_rom[counter] = shifting_amount_old
                sol_shifting_speed_rom[counter] = shifting_speed_old
            else:
                sol_SR_rom_fitted[:, counter] = SR_POD_Galerkin_ROM.latent_to_full(state)
                sol_SR_rom[:, counter] = spatial_translation(sol_SR_rom_fitted[:, counter], shifting_amount, L)
                sol_shifting_amount_rom[counter] = shifting_amount
                sol_shifting_speed_rom[counter] = shifting_speed
            counter = counter + 1
        
        timestep_old = timestep
        shifting_amount_old = shifting_amount
        shifting_speed_old = shifting_speed
        state_old = state

        state = stepper_SR_rom.step(state)
        shifting_speed = SR_POD_Galerkin_ROM.compute_shifting_speed(state)
        timestep = stepper_SR_rom.dt
        shifting_amount = shifting_speed * timestep_old + shifting_amount_old
        time += timestep
        # inner_product = np.append(inner_product, SR_POD_Galerkin_ROM.test_inner_product(SR_POD_Galerkin_ROM.latent_to_full(state), u_template))

        # print(f"t = {time}, dt = {timestep}, cdot_numer = {SR_POD_Galerkin_ROM.cdot_numerator}, cdot_denom = {SR_POD_Galerkin_ROM.cdot_denominator}, cdot = {shifting_speed}, c = {shifting_amount}, inner_product = {inner_product[-1]}.")
        print(f"t = {time}, dt = {timestep}, cdot_numer = {SR_POD_Galerkin_ROM.cdot_numerator}, cdot_denom = {SR_POD_Galerkin_ROM.cdot_denominator}, cdot = {shifting_speed}, c = {shifting_amount}.")

        if not stepper_SR_rom.stability:
            
            print("The ROM simulation is terminated due to too small timestep and the latest recorded snapshot is set to infty.")
            sol_SR_rom[:, counter:] = np.inf
            break
            
    distance_old_moment = np.abs(time - timestep - start_time - counter * dt_snapshots)
    distance_new_moment = np.abs(time - start_time - counter * dt_snapshots)
    
    if distance_old_moment <= distance_new_moment:
        sol_SR_rom_fitted[:, counter] = SR_POD_Galerkin_ROM.latent_to_full(state_old)
        sol_SR_rom[:, counter] = spatial_translation(sol_SR_rom_fitted[:, counter], shifting_amount_old, L)
        sol_shifting_amount_rom[counter] = shifting_amount_old
        sol_shifting_speed_rom[counter] = shifting_speed_old
    else:
        sol_SR_rom_fitted[:, counter] = SR_POD_Galerkin_ROM.latent_to_full(state)
        sol_SR_rom[:, counter] = spatial_translation(sol_SR_rom_fitted[:, counter], shifting_amount, L)
        sol_shifting_amount_rom[counter] = shifting_amount
        sol_shifting_speed_rom[counter] = shifting_speed

    fig, ax = plt.subplots()
    ax.contourf(x, T, sol_SR_rom.T)
    ax.set_xlim(0, L)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(f"Kuramoto-Sivashinsky solution, L = {L}, ({N_rom} POD modes)")
    plt.show()

    plt.contourf(x, T, sol_SR_rom_fitted.T)
    plt.xlim(0, L)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(f"Kuramoto-Sivashinsky solution, L = {L}, ({N_rom} POD modes)")
    plt.show()

    # plt.plot(inner_product, label='Shifting Amount')
    # plt.xlabel('Time')
    # plt.ylabel('Shifting Amount')
    # plt.title('Shifting Amount Over Time')
    # plt.grid()
    # plt.legend()
    # plt.show()

    relative_error = np.linalg.norm(sol_SR_rom - sol) / np.linalg.norm(sol)
    print(f"Relative error of the symmetry-reduced ROM: {relative_error:.4e}")

    # endregion

if __name__ == "__main__":
    main()
