"""model - Define how a given state evolves in time."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import lu_factor, lu_solve

from .timestepper import Timestepper, AdaptiveTimestepper, SemiImplicit
from .custom_typing import Vector, Matrix, Rank3Tensor, VectorField, VectorList

__all__ = ["LUSolver", "Model", "SemiLinearModel", "BilinearModel", "BilinearReducedOrderModel"]

class LUSolver:
    """A class for solving linear systems A x = b

    Args:
        mat(array): the matrix A

    When instantiated, an LU factorization of A is computed, and this is
    used when solving the system for a given right-hand side b.
    """

    def __init__(self, mat: ArrayLike):
        self.LU = lu_factor(mat)

    def __call__(self, rhs: Vector) -> Vector:
        """Solve the system A x = rhs for x

        Args:
            rhs(array): the right-hand side of the equation to be solved

        Returns:
            The solution x
        """
        return lu_solve(self.LU, rhs)

class Model(ABC):
    """
    Abstract base class for an ODE dx/dt = f(x)
    """
    @abstractmethod
    def rhs(self, x: Vector) -> Vector:
        """Return the right-hand-side of the ODE x' = f(x)."""

    def get_timestepper(self, method: str = "rk4", dt: float = 0.001, err_tol: float = None) -> Union[Timestepper, AdaptiveTimestepper]:
        """Return a discrete-time model, for the given timestep."""
        
        # First we search for the timestepper in the (explicit) timestepper subclass.
        try:
            cls = Timestepper.lookup(method)
            stepper = cls(dt, self.rhs)
            return stepper
        # If that fails, we search in the (explicit) adaptive timestepper subclass.
        except NotImplementedError:
            try:
                cls = AdaptiveTimestepper.lookup(method)
                if err_tol is None:
                    raise ValueError("Adaptive timesteppers require an error tolerance (err_tol).")
                else:
                    stepper = cls(dt, err_tol, self.rhs)
                return stepper
            # If that still fails, we search in the semi-implicit subclass.
            except NotImplementedError:
                available_explicit = Timestepper.methods() + AdaptiveTimestepper.methods()
                available_semiimplicit = SemiImplicit.methods()
                raise NotImplementedError(
                    f"Unknown explicit timestepper '{method}' for the abstract Model class. "
                    f"Available explicit timesteppers: {available_explicit}. "
                    "Alternatively, try the subclass 'SemiLinearModel' with semi-implicit timesteppers:"
                    f"{available_semiimplicit}."
                )
                
class SemiLinearModel(Model):
    """Abstract base class for semi-linear models.

    Subclasses describe a model of the form
        x' = A x + N(x)
    """

    @abstractmethod
    def linear(self, x: Vector) -> Vector:
        """Return the linear part A x."""
        ...

    @abstractmethod
    def nonlinear(self, x: Vector) -> Vector:
        """Return the nonlinear part N(x)."""
        ...

    @abstractmethod
    def get_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        """Return a solver for the linear part

        The returned solver is a callable object that, when called with
        argument b, returns a solution of the system

            x - alpha * linear(x) = b
        """
        ...

    def rhs(self, x: Vector) -> Vector:
        return self.linear(x) + self.nonlinear(x)

    def get_timestepper(self, method: str = "rk2cn", dt: float = 0.001, err_tol: float = None) -> SemiImplicit:
        try:
            cls = SemiImplicit.lookup(method)
            stepper = cls(dt, self.linear, self.nonlinear, self.get_solver)
            return stepper
        except NotImplementedError:
            return super().get_timestepper(method, dt, err_tol)

class BilinearModel(SemiLinearModel):
    """Model where the right-hand side is a bilinear function of the state

    Models have the form

        x' = d + B x + H(x, x)

    where H is bilinear

    Args:
        d(array_like): vector containing the constant terms d
        B(array_like): matrix containing the linear map B
        H(array_like): rank-3 tensor describing the bilinear map H
    """

    def __init__(self, d: ArrayLike, B: ArrayLike, H: ArrayLike):
        self._constant = np.array(d)
        self._linear = np.array(B)
        self._bilinear = np.array(H)
        self.state_dim = self._linear.shape[0]

    @property
    def constant(self) -> Vector:
        """Return the constant term d."""
        return self._constant

    def linear(self, x: Vector) -> Vector:
        """Evaluate the linear term B x"""
        return self._linear.dot(x)

    def get_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        mat = np.eye(self.state_dim) - alpha * self._linear
        return LUSolver(mat)

    def bilinear(self, a: Vector, b: Vector) -> Vector:
        """Evaluate the bilinear term H(a, b)"""
        return self._bilinear.dot(a).dot(b)

    def nonlinear(self, x: Vector) -> Vector:
        return self.constant + self.bilinear(x, x)

    def project(self, V: VectorList, W: Optional[VectorList] = None, bias: Optional[Vector] = None) -> "BilinearReducedOrderModel":
        """
        Project the full-order model onto reduced bases defined by columns of V and W.

        Parameters
        ----------
        V : shape (N, n)
            Trial basis: each column is a mode to project onto.
        W : shape (N, n), optional
            Test basis: each column is an adjoint mode.
            If None, uses Galerkin projection (W = V).
            
        Computations
        -----------

        Psi : test basis, shape (N, n), such that a = Psi^T (x - bias)
        Psi^T: the encoder weights
        Phi : trial basis, shape (N, n), such that x = bias + Phi a, Psi^T Phi = I
        Phi: the decoder weights

        Phi = V,
        Psi = W(V^T W)^{-1}

        Returns
        -------
        BilinearReducedOrderModel
            The reduced-order model with precomputed d_proj, B_proj, H_proj

            a' = d_proj + B_proj a + H_proj(a, a)
        """
        if W is None:
            W = V
            
        assert V.shape == W.shape
        N, n = V.shape
            
        if bias is None:
            bias = np.zeros(N)

        Phi = V
        Psi = np.linalg.solve((V.T @ W).T, W.T).T
          # Normalize the basis vectors

        # Project constant term
        d_proj = np.zeros(n)
        B_proj = np.zeros((n, n))
        H_proj = np.zeros((n, n, n))

        constant_component = self.constant + self.linear(bias) + self.bilinear(bias, bias)

        linear_component = np.zeros((N, n))
        for j in range(n):
            phi_j = Phi[:, j]
            linear_component[:, j] = self.linear(phi_j) + self.bilinear(bias, phi_j) + self.bilinear(phi_j, bias)

        bilinear_component = np.zeros((N, n, n))
        for j in range(n):
            phi_j = Phi[:, j]
            for k in range(n):
                phi_k = Phi[:, k]
                bilinear_component[:, j, k] = self.bilinear(phi_j, phi_k)

        for i in range(n):
            psi_i = Psi[:, i]
            d_proj[i] = np.dot(constant_component, psi_i)
            for j in range(n):
                B_proj[i, j] = np.dot(linear_component[:, j], psi_i)
                for k in range(n):
                    H_proj[i, j, k] = np.dot(bilinear_component[:, j, k], psi_i)

        return BilinearReducedOrderModel(d_proj, B_proj, H_proj, Phi, Psi, bias)

    def symmetry_reduced_project(self, V: VectorList, template: Vector, W: Optional[VectorList] = None,
                                bias: Optional[Vector] = None) -> "SymmetryReducedBilinearGalerkinROM":
        """
        Project the full-order model onto reduced bases defined by columns of V and W.

        Parameters
        ----------
        V : shape (N, n)
            Trial basis: each column is a mode to project onto.
        W : shape (N, n), optional
            Test basis: each column is an adjoint mode.
            If None, uses Galerkin projection (W = V).
            
        Computations
        -----------
        
        Psi : test basis, shape (N, n), such that a = Psi^T (x - bias)
        Psi^T: the encoder weights
        Phi : trial basis, shape (N, n), such that x = bias + Phi a, Psi^T Phi = I
        Phi: the decoder weights

        Phi = V,
        Psi = W(V^T W)^{-1}

        Returns
        -------
        SymmetryReducedBilinearReducedOrderModel: with precomputed d, B, H, e, P, Q, w, S, n, M

        a' = d_proj + B_proj a + H_proj(a, a) - cdot * (n_proj + M_proj a)

        cdot : float, the shifting speed

        cdot = (e_proj + P_proj a + Q_proj(a, a)) / (w_proj + S_proj a)

        """
        
        # First, check that whether the spatial derivative is provided in the FOM

        if W is None:
            W = V
            
        assert V.shape == W.shape

        N, n = V.shape

        if hasattr(self, 'derivative'):
            template_dx = self.derivative(template, order = 1)
            V_dx = np.zeros_like(V)
            for i in range(V.shape[1]):
                V_dx[:, i] = self.derivative(V[:, i], order = 1)
            if bias is None:
                bias = np.zeros(N)
                bias_dx = np.zeros(N)
            else:
                bias_dx = self.derivative(bias, order = 1)
        else:
            raise NotImplementedError("Can't do symmetry reduction because the spatial derivative method is not implemented in the FOM.")
            
        Phi = V
        Phi_dx = V_dx
        Psi = np.linalg.solve((V.T @ W).T, W.T).T * N

        d_proj = np.zeros(n)
        B_proj = np.zeros((n, n))
        H_proj = np.zeros((n, n, n))
        p_proj = np.zeros((n))
        Q_proj = np.zeros((n, n))
        s_proj = np.zeros((n))
        n_proj = np.zeros(n)
        M_proj = np.zeros((n, n))

        constant_component = self.constant + self.linear(bias) + self.bilinear(bias, bias)

        linear_component = np.zeros((N, n))
        for j in range(n):
            phi_j = Phi[:, j]
            linear_component[:, j] = self.linear(phi_j) + self.bilinear(bias, phi_j) + self.bilinear(phi_j, bias)

        bilinear_component = np.zeros((N, n, n))
        for j in range(n):
            phi_j = Phi[:, j]
            for k in range(n):
                phi_k = Phi[:, k]
                bilinear_component[:, j, k] = self.bilinear(phi_j, phi_k)

        for i in range(n):
            psi_i = Psi[:, i]
            d_proj[i] = np.dot(constant_component, psi_i) / len(psi_i)
            for j in range(n):
                B_proj[i, j] = np.dot(linear_component[:, j], psi_i) / len(psi_i)
                for k in range(n):
                    H_proj[i, j, k] = np.dot(bilinear_component[:, j, k], psi_i) / len(psi_i)

        e_proj = np.dot(constant_component, template_dx) / len(template_dx)
        w_proj = np.dot(bias_dx, template_dx) / len(template_dx)

        for i in range(n):
            psi_i = Psi[:, i]
            p_proj[i] = np.dot(linear_component[:, i], template_dx) / len(template_dx)
            n_proj[i] = np.dot(bias_dx, psi_i) / len(psi_i)
            s_proj[i] = np.dot(Phi_dx[:, i], template_dx) / len(template_dx)
            for j in range(n):
                Q_proj[i, j] = np.dot(bilinear_component[:, i, j], template_dx) / len(template_dx)
                M_proj[i, j] = np.dot(Phi_dx[:, i], Psi[:, j]) / len(Phi_dx[:, i])

        SRBilinearGalerkinROM = SymmetryReducedBilinearGalerkinROM(d_proj, B_proj, H_proj, Phi, Psi, bias,
               e_proj, p_proj, Q_proj, w_proj, s_proj, n_proj, M_proj)
        
        for attr in ['derivative']:  # add others here
            if hasattr(self, attr):
                setattr(SRBilinearGalerkinROM, attr, getattr(self, attr))
        return SRBilinearGalerkinROM

class BilinearReducedOrderModel(BilinearModel):
    """ROMs for the BilinearModel where the right-hand side is a bilinear function of the reduced state

    Models have the form

        a' = d_proj + B_proj a + H_proj(a, a)

    where B is bilinear

    Args:
        d(array_like): vector containing the constant terms d
        L(array_like): matrix containing the linear map L
        B(array_like): rank-3 tensor describing the bilinear map B
    """

    def __init__(self, d_proj: ArrayLike, B_proj: ArrayLike, H_proj: ArrayLike,
                 Phi: ArrayLike, Psi: ArrayLike, bias: ArrayLike):
        super().__init__(d_proj, B_proj, H_proj)
        self.Phi = np.array(Phi)
        self.Psi = np.array(Psi)
        self._bias = np.array(bias) if bias is not None else np.zeros(self._linear.shape[0])
        self.state_dim = self._linear.shape[0]
    
    def full_to_latent(self, x: ArrayLike) -> ArrayLike:
        """Convert full state x to latent state a."""

        if x.ndim == 1:
            return self.test_basis.T @ (x - self.bias) / self.test_basis.shape[0]
        elif x.ndim == 2:
            return self.test_basis.T @ (x - self.bias[:, np.newaxis]) / self.test_basis.shape[0]
        else:
            raise ValueError("Input x must be 1D or 2D.")
        
    def latent_to_full(self, a: ArrayLike) -> ArrayLike:
        """Convert latent state a to full state x."""
        
        if a.ndim == 1:
            return self.bias + self.trial_basis @ a
        elif a.ndim == 2:
            return self.bias[:, np.newaxis] + self.trial_basis @ a
        else:
            raise ValueError("Input a must be 1D or 2D.")
    
    @property
    def constant_vector(self) -> Vector:
        """Return the constant term d."""
        return self._constant
    
    @property
    def linear_matrix(self) -> Matrix:
        """Return the linear matrix L."""
        return self._linear
    
    @property
    def bilinear_tensor(self) -> Rank3Tensor:
        """Return the bilinear tensor B."""
        return self._bilinear
    
    @property
    def trial_basis(self) -> Matrix:
        """Return the trial basis Phi, x = bias + Phi z"""
        return self.Phi
    
    @property
    def test_basis(self) -> Matrix:
        """Return the test basis Psi., z = Psi^T (x - bias)"""
        return self.Psi

    @property
    def bias(self) -> Vector:
        """Return the bias term."""
        return self._bias
    
class SymmetryReducedBilinearGalerkinROM(BilinearReducedOrderModel):
    """ROMs for the BilinearModel where the right-hand side is a bilinear function of the reduced state

    Models have the form

        a' = d_hat + L_hat a + B_hat(a, a) - cdot * (n_hat + M_hat a)

    where B is bilinear

    Args:
        d(array_like): vector containing the constant terms d
        L(array_like): matrix containing the linear map L
        B(array_like): rank-3 tensor describing the bilinear map B
    """

    def __init__(self, d_proj: ArrayLike, B_proj: ArrayLike, H_proj: ArrayLike, Phi: ArrayLike, Psi: ArrayLike, bias: ArrayLike,
                 e_proj: ArrayLike, P_proj: ArrayLike, Q_proj: ArrayLike,
                 w_proj: ArrayLike, S_proj: ArrayLike, n_proj: ArrayLike, M_proj: ArrayLike):
        super().__init__(d_proj, B_proj, H_proj, Phi, Psi, bias)
        self.e_proj = np.array(e_proj)
        self.P_proj = np.array(P_proj)
        self.Q_proj = np.array(Q_proj)
        self.w_proj = np.array(w_proj)
        self.S_proj = np.array(S_proj)
        self.n_proj = np.array(n_proj)
        self.M_proj = np.array(M_proj)

        self.shifting_speed_denom_threshold = 1e-6

        self.cdot = 0.0  # Initialize shifting speed

    @property
    def shifting_speed_numer_constant(self) -> float:
        """Return the numerator constant term e_proj in the numerator of the shifting speed"""
        return self.e_proj
    
    @property
    def shifting_speed_numer_linear_vector(self) -> Vector:
        """Return the linear term P_proj"""
        return self.P_proj
    
    @property
    def shifting_speed_numer_bilinear_matrix(self) -> Matrix:
        """Return the bilinear term Q_proj"""
        return self.Q_proj

    @property
    def shifting_speed_denom_constant(self) -> float:
        """Return the denominator constant term w_proj in the denominator of the shifting speed"""
        return self.w_proj

    @property
    def shifting_speed_denom_linear_vector(self) -> Vector:
        """Return the linear term S_proj"""
        return self.S_proj

    @property
    def spatial_derivative_constant_vector(self) -> Vector:
        """Return the spatial derivative constant term n_proj"""
        return self.n_proj
    
    @property
    def spatial_derivative_linear_matrix(self) -> Matrix:
        """Return the spatial derivative linear term M_proj"""
        return self.M_proj
    
    def compute_shifting_speed(self, a: Vector) -> float:
        self.cdot_numerator = self.e_proj + np.dot(self.P_proj, a) + np.dot(a, np.dot(self.Q_proj, a))
        self.cdot_denominator = self.w_proj + np.dot(self.S_proj, a)
        if np.abs(self.cdot_denominator) < self.shifting_speed_denom_threshold:
            raise ValueError(f"Denominator in shifting speed is less than {self.shifting_speed_denom_threshold}, cannot compute cdot.")
        else:
            return self.cdot_numerator / self.cdot_denominator

    def nonlinear(self, a: Vector) -> Vector:
        """Return the nonlinear part N(a) = d_proj + B_proj a + H_proj(a, a) - cdot * (n_proj + M_proj a)"""
        cdot = self.compute_shifting_speed(a)
        return self.constant_vector + self.bilinear(a, a) - cdot * (self.spatial_derivative_constant_vector + np.dot(self.spatial_derivative_linear_matrix, a))
    
    def test_inner_product(self, u: Vector, u_template: Vector) -> float:
        """According to the symmetry reduction, the inner product is constrained to be:
        <u_dx, u_template_dx> = 0, where u is the template-fitted solution"""

        u_template_dx = self.derivative(u_template, order=1)
        return np.dot(u, u_template_dx) / len(u)