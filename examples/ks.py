#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np

from romnet.models import KuramotoSivashinsky
from romnet.models.ks import fft_multiply, freq_to_space, space_to_freq


def test_to_space():
    nmodes = 16
    uf = np.random.rand(nmodes + 1) + 1j * np.random.rand(nmodes + 1)
    uf[0] = np.real(uf[0])
    u = np.zeros(2 * nmodes, dtype=float)
    x = np.arange(2 * nmodes) * 2 * np.pi / (2 * nmodes)
    u[:] = np.real(uf[0])
    for k in range(1, nmodes):
        u += 2 * np.real(uf[k] * (np.cos(k * x) + 1j * np.sin(k * x)))
    # last Fourier mode is real and appears only once in sum
    u += np.real(uf[-1]) * np.cos(nmodes * x)
    u2 = freq_to_space(uf)
    # plt.plot(x, u)
    # plt.plot(x, u2)
    # plt.show()
    assert np.allclose(u, u2)


def test_to_freq():
    nmodes = 16
    uf = np.random.rand(nmodes + 1) + 1j * np.random.rand(nmodes + 1)
    uf[0] = np.real(uf[0])
    uf[-1] = np.real(uf[-1])
    u = freq_to_space(uf)
    uf2 = space_to_freq(u)
    assert np.allclose(uf, uf2)


def test_fft_mult():
    nmodes = 32
    uf = np.random.rand(nmodes + 1) + 1j * np.random.rand(nmodes + 1)
    vf = np.random.rand(nmodes + 1) + 1j * np.random.rand(nmodes + 1)
    u = freq_to_space(uf)
    v = freq_to_space(vf)
    uv = u * v
    uv2 = freq_to_space(fft_multiply(uf, vf))
    # plt.plot(uv)
    # plt.plot(uv2)
    # plt.show()
    assert np.allclose(uv, uv2)


def main():
    nmodes = 512
    L = 400
    ks = KuramotoSivashinsky(nmodes, L)
    dt = 0.1
    # ks.set_stepper(dt, "rk3cn")
    step = ks.get_stepper(dt, "rk3cn")

    # initial condition
    u = np.zeros(nmodes + 1, dtype=complex)
    u[45] = 0.9
    u[46] = 1j

    nt = 5000
    nx = 2 * nmodes
    sol = np.zeros((nt, nx))
    print("Computing solution...")
    for t in range(nt):
        u = step(u)
        sol[t] = freq_to_space(u)
        if t % 1000 == 0:
            print("  step %4d / %4d" % (t, nt))
    T = np.arange(nt) * dt
    dx = L / (2 * nmodes)
    x = np.arange(2 * nmodes) * dx

    fig, ax = plt.subplots()
    ax.contourf(x, T, sol)
    ax.set_xlim(0, L)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(f"Kuramoto-Sivashinsky solution, L = {L}, ({nmodes} modes)")

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, nmodes + 1), np.abs(u[1:]))
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Fourier coefficient magnitude")
    ax.set_title("Energy spectrum")
    plt.show()


if __name__ == "__main__":
    main()
