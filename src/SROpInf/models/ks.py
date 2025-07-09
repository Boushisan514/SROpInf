"""Kuramoto-Sivashinsky equation"""

from typing import Callable

import numpy as np
import scipy as sp

from ..model import BilinearModel
from ..custom_typing import Vector

__all__ = ["KuramotoSivashinsky", "freq_to_space", "space_to_freq"]

def freq_to_space(u_hat):
    
    N = len(u_hat) // 2
        
    return np.fft.ifft(np.fft.ifftshift(u_hat)).real * 2 * N

def space_to_freq(u):

    N = len(u) // 2

    return np.fft.fftshift(np.fft.fft(u)) / (2 * N)

def spatial_translation(u, c, Lx):
    
    # This function converts a given spatial function q(x) to q(x + c) by manipulating the Fourier coefficients
    
    u_hat = space_to_freq(u)
    
    N = len(u_hat) // 2
    
    u_shifted_hat = u_hat * np.exp(1j * c * (2*np.pi/Lx) * np.linspace(-N, N-1, 2 * N, dtype = int, endpoint = True))
    
    return freq_to_space(u_shifted_hat)

class KuramotoSivashinsky(BilinearModel):
    r"""Kuramoto-Sivashinsky equation

    u_t + u u_x + u_xx + u_xxxx = 0

    with periodic boundary conditions, for 0 <= x <= L
    The equation is solved using a dealiased pseudo-spectral method
    """

    def __init__(self, nu: float, N: int, L: float):
        """summary here

        Args:
            N: Number of collocation points (must be even)
            L: Length of domain
        """
        self.nmodes = N // 2 # number of Fourier modes: from 0, 1, ..., to N//2 (Nyquist frequency)
        self.L = L
        self.nu = nu
        self.mode_index = np.linspace(-self.nmodes, self.nmodes-1, 2 * self.nmodes, dtype = int, endpoint = True)
        self.k = 2 * np.pi / L * self.mode_index
        self._deriv_factor = 1j * self.k
        self._linear_factor = - self._deriv_factor ** 2 - self.nu * self._deriv_factor ** 4
        self._linear_factor[0] = 0
        
    def derivative(self, u: Vector, order: int = 1) -> Vector:
        """Compute the nth derivative of u(x) via FFT"""
        return freq_to_space(self._deriv_factor ** order * space_to_freq(u))

    def get_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        def solver(rhs: Vector) -> Vector:
            return freq_to_space(space_to_freq(rhs) / (1 - alpha * self._linear_factor))

        return solver
    
    @property
    def constant(self) -> Vector:
        """Return the constant term of the model."""
        return np.zeros(2 * self.nmodes)

    def linear(self, u: Vector) -> Vector:
        return freq_to_space(self._linear_factor * space_to_freq(u))

    def bilinear(self, u: Vector, v: Vector) -> Vector:
        # a * b_x
        bilinear_output = np.zeros(2 * self.nmodes, dtype=complex)

        state_u = space_to_freq(u)
        state_v = space_to_freq(v)

        for p in range(2 * self.nmodes):
            p_freq = self.mode_index[p]
            for m in range(2 * self.nmodes):
                m_freq = self.mode_index[m]
                n_freq = p_freq - m_freq
                n = n_freq + self.nmodes
                if n_freq <= self.nmodes - 1 and n_freq >= -self.nmodes:
                    bilinear_output[p] += - self._deriv_factor[m] * state_u[n] * state_v[m]

        bilinear_output[0] = 0

        return freq_to_space(bilinear_output)

    def nonlinear(self, u: Vector) -> Vector:
        return self.constant + self.bilinear(u, u)
    
    def rhs(self, u: Vector) -> Vector:
        """Return the right-hand side of the Kuramoto-Sivashinsky equation."""
        return self.linear(u) + self.nonlinear(u)

