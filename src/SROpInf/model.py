"""model - Define how a given state evolves in time."""

from numbers import Number
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
from torch import Tensor, einsum
from numpy.typing import ArrayLike
from scipy.linalg import lu_factor, lu_solve

from .timestepper import Timestepper, AdaptiveTimestepper, SemiImplicit
from .custom_typing import Vector, Matrix, VectorField, VectorList

__all__ = ["LUSolver", "Model", "SemiLinearModel", "BilinearModel", "duplicate_free_quadratic_vector"]

def inner_product(a: Vector, b: Vector) -> float:
    """Compute the inner product of two vectors."""
    return np.dot(a, np.conj(b)) / len(a)

def is_scalar(x):
    return isinstance(x, (Number, np.generic)) or (isinstance(x, Tensor) and x.ndim == 0)

def duplicate_free_quadratic_vector(v: Union[Vector, ArrayLike]) -> Union[Vector, ArrayLike]:   
    """Return a vector with duplicate entries removed, keeping the first occurrence.

    Args:
        v(Vector): input vector

    Returns:
        Vector: a vector (v \kron v) with duplicate entries removed
    """
    if v.ndim == 1:
        N = len(v)
        # get the i,j pairs for i ≤ j
        i, j = np.triu_indices(N)
        return v[i] * v[j]
    elif v.ndim == 2:
        N, _ = v.shape
        # get the i,j pairs for i ≤ j
        i, j = np.triu_indices(N)
        return v[i, :] * v[j, :]
    else:
        raise ValueError("Input v must be a 1D or 2D array-like object.")
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

    def __init__(self, d: Vector, A: Matrix, B: Matrix):
        self._constant = d
        self._linear = A
        self._bilinear = B

    @property
    def constant(self) -> Vector:
        """Return the constant term d."""
        return self._constant
    
    def linear(self, x: Vector) -> Vector:
        """Evaluate the linear term B x"""
        return self._linear.dot(x)

    def get_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        mat = np.eye(self._linear.shape[0]) - alpha * self._linear
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
        Psi = np.linalg.solve((V.T @ W).T, W.T).T * N
          # Normalize the basis vectors

        # Project constant term
        d_proj = np.zeros(n)
        A_proj = np.zeros((n, n))
        B_proj = np.zeros((n, n, n))

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
            d_proj[i] = inner_product(constant_component, psi_i)
            for j in range(n):
                A_proj[i, j] = inner_product(linear_component[:, j], psi_i)
                for k in range(n):
                    B_proj[i, j, k] = inner_product(bilinear_component[:, j, k], psi_i)

        return BilinearReducedOrderModel(Phi, Psi, bias, d_proj, A_proj, B_proj)

    def symmetry_reduced_project(self, V: VectorList, template: Vector, W: Optional[VectorList] = None,
                                bias: Optional[Vector] = None) -> "SymmetryReducedBilinearROM":
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
            V_dx = self.derivative(V, order = 1)
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
        A_proj = np.zeros((n, n))
        B_proj = np.zeros((n, n, n))
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
            d_proj[i] = inner_product(constant_component, psi_i)
            for j in range(n):
                A_proj[i, j] = inner_product(linear_component[:, j], psi_i)
                for k in range(n):
                    B_proj[i, j, k] = inner_product(bilinear_component[:, j, k], psi_i)

        e_proj = inner_product(constant_component, template_dx)
        w_proj = inner_product(bias_dx, template_dx)

        for i in range(n):
            psi_i = Psi[:, i]
            p_proj[i] = inner_product(linear_component[:, i], template_dx)
            n_proj[i] = inner_product(bias_dx, psi_i)
            s_proj[i] = inner_product(Phi_dx[:, i], template_dx)
            for j in range(n):
                Q_proj[i, j] = inner_product(bilinear_component[:, i, j], template_dx)
                M_proj[i, j] = inner_product(Phi_dx[:, j], psi_i)

        SRBilinearGalerkinROM = SymmetryReducedBilinearROM(Phi, Psi, bias, w_proj, s_proj, n_proj, M_proj, 
                                d_proj, A_proj, B_proj, e_proj, p_proj, Q_proj)

        for attr in ['derivative']:  # add others here
            if hasattr(self, attr):
                setattr(SRBilinearGalerkinROM, attr, getattr(self, attr))
        return SRBilinearGalerkinROM
    
    def symmetry_reduced_OpInf_initialization(self, V: VectorList, template: Vector, W: Optional[VectorList] = None,
                                bias: Optional[Vector] = None) -> "SymmetryReducedQuadraticOpInfROM":
        """
        Create the template for the symmetry-reduced OpInf ROM,
        with all known reduced matrices precomputed.

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
            V_dx = self.derivative(V, order = 1)
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

        w_proj = np.dot(bias_dx, template_dx) / len(template_dx)
        s_proj = np.zeros(n)
        n_proj = np.zeros(n)
        M_proj = np.zeros((n, n))

        for i in range(n):
            psi_i = Psi[:, i]
            n_proj[i] = inner_product(bias_dx, psi_i)
            s_proj[i] = inner_product(Phi_dx[:, i], template_dx)
            for j in range(n):
                M_proj[i, j] = inner_product(Phi_dx[:, j], psi_i)

        SRBilinearOpInfROM = SymmetryReducedQuadraticOpInfROM(Phi, Psi, bias, w_proj, s_proj, n_proj, M_proj)
        
        for attr in ['derivative']:  # add others here
            if hasattr(self, attr):
                setattr(SRBilinearOpInfROM, attr, getattr(self, attr))
        return SRBilinearOpInfROM

class BilinearReducedOrderModel(BilinearModel):
    """ROMs for the BilinearModel where the right-hand side is a bilinear function of the reduced state

    Models have the form

        a' = d_rom + B_rom a + H_rom(a, a)

    where B is bilinear

    Args:
        d(array_like): vector containing the constant terms d
        L(array_like): matrix containing the linear map L
        B(array_like): rank-3 tensor describing the bilinear map B
    """

    def __init__(self, Phi: ArrayLike, Psi: ArrayLike, bias: Vector, 
                 d_rom: Vector, A_rom: Matrix, B_rom: Matrix):
        super().__init__(d_rom, A_rom, B_rom)
        self.Phi = np.array(Phi)
        self.Psi = np.array(Psi)
        self._bias = np.array(bias) if bias is not None else np.zeros(self._linear.shape[0])
        self.state_dim = self.Phi.shape[1]
    
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
    def trial_basis(self) -> Matrix:
        """Return the trial basis Phi, x = bias + Phi z"""
        return self.Phi
    
    @trial_basis.setter
    def trial_basis(self, value: Matrix):
        """Set the trial basis Phi."""
        if value.shape[0] != self.state_dim:
            raise ValueError(f"Trial basis must have shape ({self.state_dim}, n), got {value.shape}.")
        self.Phi = value
    
    @property
    def test_basis(self) -> Matrix:
        """Return the test basis Psi., z = Psi^T (x - bias)"""
        return self.Psi
    
    @test_basis.setter
    def test_basis(self, value: Matrix):
        """Set the test basis Psi."""
        if value.shape[0] != self.state_dim:
            raise ValueError(f"Test basis must have shape ({self.state_dim}, n), got {value.shape}.")
        self.Psi = value

    @property
    def bias(self) -> Vector:
        """Return the bias term."""
        return self._bias
    
    @bias.setter
    def bias(self, value: Vector):
        """Set the bias term."""
        if value.shape[0] != self.state_dim:
            raise ValueError(f"Bias must have shape ({self.state_dim},), got {value.shape}.")
        self._bias = value
    
class SymmetryReducedBilinearROM(BilinearReducedOrderModel):
    """ROMs for the BilinearModel where the right-hand side is a bilinear function of the reduced state

    Models have the form

        a' = d_hat + L_hat a + B_hat(a, a) - cdot * (n_hat + M_hat a)

    where B is bilinear

    Args:
        d(array_like): vector containing the constant terms d
        L(array_like): matrix containing the linear map L
        B(array_like): rank-3 tensor describing the bilinear map B
    """

    def __init__(self, Phi: ArrayLike, Psi: ArrayLike, bias: Vector,
                 w_rom: float, s_rom: Vector, n_rom: Vector, M_rom: Matrix,
                 d_rom: Vector, A_rom: Matrix, B_rom: Matrix,
                 e_rom: float, p_rom: Vector, Q_rom: Matrix):
        super().__init__(Phi, Psi, bias, d_rom, A_rom, B_rom)
        self._cdot_numer_constant = e_rom
        self._cdot_numer_linear = p_rom
        self._cdot_numer_bilinear = Q_rom
        self._cdot_denom_constant = w_rom
        self._cdot_denom_linear = s_rom
        self._udx_constant = n_rom
        self._udx_linear = M_rom

        self.shifting_speed_denom_threshold = 1e-6

    @property
    def constant(self) -> Union[Vector, Tensor]:
        """Return the constant term d_rom."""
        return self._constant

    @constant.setter
    def constant(self, value: Union[Vector, Tensor]):
        """Set the constant term d_rom."""
        if value.shape[0] != self.state_dim:
            raise ValueError(f"Constant term must have shape ({self.state_dim},), got {value.shape}.")
        self._constant = value

    @property
    def linear_mat(self) -> Union[Matrix, Tensor]:
        """Return the linear term p_rom."""
        return self._linear

    @linear_mat.setter
    def linear_mat(self, value: Union[Matrix, Tensor]):
        """Set the linear term p_rom."""
        if value.shape[0] != self.state_dim:
            raise ValueError(f"Linear term must have shape ({self.state_dim}, {self.state_dim}), got {value.shape}.")
        self._linear = value

    def linear(self, a: Union[Vector, Tensor]) -> Union[Vector, Tensor]:
        """Evaluate the linear term L_rom x,
        First we check the shapes, then we check the types"""
        if self.linear_mat.shape[1] != a.shape[0]:
                raise ValueError(f"Incompatible shapes: {self.linear_mat.shape} @ {a.shape}")
        if isinstance(a, Vector) and isinstance(self.linear_mat, Matrix):
            return self.linear_mat.dot(a)
        elif isinstance(a, Tensor) and isinstance(self.linear_mat, Tensor):
            return self.linear_mat @ a
        else:
            raise ValueError(f"Incompatible types for Aa: got a: {type(a)} and A: {type(self.linear_mat)}.")

    @property
    def bilinear_mat(self) -> Union[Matrix, Tensor]:
        """Return the bilinear term B_rom"""
        return self._bilinear

    @bilinear_mat.setter
    def bilinear_mat(self, value: Union[Matrix, Tensor]):
        """Set the bilinear term B_rom"""
        if value.shape != (self.state_dim, self.state_dim, self.state_dim):
            raise ValueError(f"Bilinear term must have shape ({self.state_dim}, {self.state_dim}, {self.state_dim}), got {value.shape}.")
        self._bilinear = value

    def bilinear(self, a: Union[Vector, Tensor], b: Union[Vector, Tensor]) -> Union[Vector, Tensor]:
        """Evaluate the bilinear term B_rom(a, b) = einsum('ijk,j,k->i', B, a, b)"""
        if self.bilinear_mat.shape[1] != a.shape[0] or self.bilinear_mat.shape[2] != b.shape[0]:
                raise ValueError(f"Incompatible shapes: {self.bilinear_mat.shape} with a: {a.shape}, b: {b.shape}")
        if isinstance(a, Vector) and isinstance(b, Vector) and isinstance(self.bilinear_mat, Matrix):
            return self.bilinear_mat.dot(a).dot(b)
        elif isinstance(a, Tensor) and isinstance(b, Tensor) and isinstance(self.bilinear_mat, Tensor):
            return einsum('ijk,j,k->i', self.bilinear_mat, a, b)
        else:
            raise ValueError(f"Incompatible types for B(a, b): got a: {type(a)}, b: {type(b)}, B: {type(self.bilinear_mat)}.")

    @property
    def cdot_numer_constant(self) -> Union[float, Tensor]:
        """Return the numerator constant term e_rom in the numerator of the shifting speed"""
        return self._cdot_numer_constant
    
    @cdot_numer_constant.setter
    def cdot_numer_constant(self, value: Union[float, Tensor]):
        """Set the numerator constant term e_rom in the numerator of the shifting speed"""
        if not np.isscalar(value):
            raise ValueError(f"Numerator constant must be a scalar, got {value}.")
        self._cdot_numer_constant = value

    @property
    def cdot_numer_linear(self) -> Union[Vector, Tensor]:
        """Return the numerator linear term p_rom in the numerator of the shifting speed"""
        return self._cdot_numer_linear

    @cdot_numer_linear.setter
    def cdot_numer_linear(self, value: Union[Vector, Tensor]):
        """Set the numerator linear term p_rom in the numerator of the shifting speed"""
        if value.shape[0] != self.state_dim:
            raise ValueError(f"Numerator linear vector must have shape ({self.state_dim},), got {value.shape}.")
        self._cdot_numer_linear = value

    @property
    def cdot_numer_bilinear(self) -> Union[Matrix, Tensor]:
        """Return the numerator bilinear term Q_rom in the numerator of the shifting speed"""
        return self._cdot_numer_bilinear

    @cdot_numer_bilinear.setter
    def cdot_numer_bilinear(self, value: Union[Matrix, Tensor]):
        """Set the numerator bilinear term Q_rom in the numerator of the shifting speed"""
        if value.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"Numerator bilinear matrix must have shape ({self.state_dim}, {self.state_dim}), got {value.shape}.")
        self._cdot_numer_bilinear = value
    
    @property
    def cdot_denom_constant(self) -> Union[float, Tensor]:
        """Return the denominator constant term w_rom in the denominator of the shifting speed"""
        return self._cdot_denom_constant
    
    @property
    def cdot_denom_linear(self) -> Union[Vector, Tensor]:
        """Return the denominator linear term s_rom in the denominator of the shifting speed"""
        return self._cdot_denom_linear

    @property
    def udx_constant(self) -> Union[Vector, Tensor]:
        """Return the spatial derivative constant term n_rom"""
        return self._udx_constant
    
    @property
    def udx_linear(self) -> Union[Matrix, Tensor]:
        """Return the spatial derivative linear term M_rom"""
        return self._udx_linear

    def udx(self, a: Union[Vector, Tensor]) -> Union[Vector, Tensor]:
        """Evaluate the spatial derivative term: n_rom + M_rom @ a, supporting both NumPy and Torch."""
        if self.udx_linear.shape[1] != a.shape[0] or self.udx_constant.shape[0] != a.shape[0]:
                raise ValueError(f"Incompatible shapes: {self.udx_linear.shape} @ {a.shape} + {self.udx_constant.shape}")
        if isinstance(a, Vector) and isinstance(self.udx_constant, Vector) and isinstance(self.udx_linear, Matrix):
            return self.udx_constant + self.udx_linear @ a
        elif isinstance(a, Tensor) and isinstance(self.udx_linear, Tensor) and isinstance(self.udx_constant, Tensor):
            return self.udx_constant + self.udx_linear @ a
        else:
            raise TypeError(f"Incompatible types for (n + Ma): got a: {type(a)}, M: {type(self.udx_linear)}, n: {type(self.udx_constant)}.")

    def shifting_speed(self, a: Union[Vector, Tensor]) -> Union[float, Tensor]:
        if self.cdot_numer_linear.shape[0] != a.shape[0] or self.cdot_numer_bilinear.shape[0] != a.shape[0]:
            raise ValueError(f"Incompatible shapes for cdot_numerator: {self.cdot_numer_linear.shape} @ {a.shape} + {self.cdot_numer_bilinear.shape}({a.shape}, {a.shape})")
        if self.cdot_denom_linear.shape[0] != a.shape[0]:
            raise ValueError(f"Incompatible shapes for cdot_denominator: {self.cdot_denom_linear.shape} @ {a.shape}")
        
        if isinstance(a, Vector) and isinstance(self.cdot_numer_constant, float) and isinstance(self.cdot_numer_linear, Vector) and isinstance(self.cdot_numer_bilinear, Matrix):
            self.cdot_numerator   = self.cdot_numer_constant + self.cdot_numer_linear.dot(a) + self.cdot_numer_bilinear.dot(a).dot(a)
        elif isinstance(a, Tensor) and isinstance(self.cdot_numer_constant, Tensor) and isinstance(self.cdot_numer_linear, Tensor) and isinstance(self.cdot_numer_bilinear, Tensor):
            self.cdot_numerator   = self.cdot_numer_constant + self.cdot_numer_linear @ a + einsum('ij,j,k->i', self.cdot_numer_bilinear, a, a)
        self.cdot_numerator   = self.cdot_numer_constant + self.cdot_numer_linear.dot(a) + self.cdot_numer_bilinear.dot(a).dot(a)
        self.cdot_denominator = self.cdot_denom_constant + self.cdot_denom_linear.dot(a)
        if np.abs(self.cdot_denominator) > self.shifting_speed_denom_threshold:
            return - self.cdot_numerator / self.cdot_denominator
        else:
            raise ValueError(f"Denominator in shifting speed is less than {self.shifting_speed_denom_threshold}, cannot compute the shifting speed.")

    def velocity(self, a: Union[Vector, Tensor]) -> Union[Vector, Tensor]:
        # compute the velocity of the reduced state a without the additional symmetry-reducing term
        return self.constant + self.linear(a) + self.bilinear(a, a)

    def nonlinear(self, a: Union[Vector, Tensor]) -> Union[Vector, Tensor]:
        """Return the nonlinear part N(a) = d_rom + B_rom a + H_rom(a, a) + cdot * (n_rom + M_rom a)"""
        return self.constant + self.bilinear(a, a) + self.shifting_speed(a) * self.udx(a)

    # def test_inner_product(self, u: Vector, u_template: Vector) -> float:
    #     """According to the symmetry reduction, the inner product is constrained to be:
    #     <u_dx, u_template_dx> = 0, where u is the template-fitted solution"""

    #     u_template_dx = self.derivative(u_template, order=1)
    #     return np.dot(u, u_template_dx) / len(u)

class SymmetryReducedQuadraticOpInfROM(SymmetryReducedBilinearROM):
    """ROMs for the SRBilinearModel where the right-hand side is a quadratic function of the reduced state

    Models have the form

        a' = d_rom + B_rom a + H_rom a^2 - cdot * (n_rom + M_rom a)

    """
    def __init__(self, Phi: ArrayLike, Psi: ArrayLike, bias: Vector,
                 w_rom: float, s_rom: Vector, n_rom: Vector, M_rom: Matrix,
                 d_rom: Vector = None, B_rom: Matrix = None, H_rom: Matrix = None,
                 e_rom: float = None, p_rom: Vector = None, q_rom: Vector = None):
        
        super().__init__(Phi, Psi, bias, w_rom, s_rom, n_rom, M_rom,
                         d_rom, B_rom, H_rom, e_rom, p_rom, q_rom)
        
        self._quadratic = H_rom  # H_rom is now the quadratic term
        self.q_rom = q_rom       # q_rom is the quadratic vector in the numerator of the shifting speed
        self.quadratic_dim = self.state_dim * (self.state_dim + 1) // 2

    def bilinear(self, *args, **kwargs) -> None:
        """
        Deprecated method.

        This model uses a quadratic form instead of a bilinear form.
        Calling this method is not supported. Use the `quadratic` method
        for evaluating the nonlinear quadratic term.
        """
        raise NotImplementedError(
            "The SR-OpInf model uses a quadratic term instead of a bilinear one. "
            "Use the `quadratic` method to evaluate the nonlinear term."
        )
    
    def shifting_speed_numer_bilinear_matrix(self, *args, **kwargs) -> None:
        """
        Deprecated method.

        This model uses a quadratic form instead of a bilinear form.
        Calling this method is not supported. Use the `quadratic` method
        for evaluating the nonlinear quadratic term in the numerator of the shifting speed.
        """
        raise NotImplementedError(
            "The SR-OpInf model uses a quadratic term instead of a bilinear one. "
            "Use the `quadratic` method to evaluate the nonlinear term in the numerator of the shifting speed."
        )

    @property
    def quadratic_matrix(self) -> Matrix:
        """Return the quadratic matrix B."""
        return self._quadratic
    
    @quadratic_matrix.setter
    def quadratic_matrix(self, value: Matrix):
        """Set the quadratic matrix B."""
        if value.shape != (self.state_dim, self.quadratic_dim):
            raise ValueError(f"Quadratic matrix must have shape ({self.state_dim}, {self.quadratic_dim}), got {value.shape}.")
        self._quadratic = value

    @property
    def shifting_speed_numer_quadratic_vector(self) -> Vector:
        """Return the quadratic term q_rom in the numerator of the shifting speed"""
        return self.q_rom
    
    @shifting_speed_numer_quadratic_vector.setter
    def shifting_speed_numer_quadratic_vector(self, value: Vector):
        """Set the quadratic term q_rom in the numerator of the shifting speed"""
        if value.shape != (self.quadratic_dim,):
            raise ValueError(f"Numerator quadratic vector must have shape ({self.quadratic_dim},), got {value.shape}.")
        self.q_rom = value

    def quadratic(self, a: Vector) -> Vector:
        """Evaluate the quadratic term H_rom a^2"""
        a_quadratic = duplicate_free_quadratic_vector(a)
        return self._quadratic.dot(a_quadratic)
    
    def compute_shifting_speed(self, a: Vector) -> float:
        self.cdot_numerator = self.shifting_speed_numer_constant + np.dot(self.shifting_speed_numer_linear_vector, a) + np.dot(self.shifting_speed_numer_quadratic_vector, duplicate_free_quadratic_vector(a))
        self.cdot_denominator = self.shifting_speed_denom_constant + np.dot(self.shifting_speed_denom_linear_vector, a)
        if np.abs(self.cdot_denominator) < self.shifting_speed_denom_threshold:
            raise ValueError(f"Denominator in shifting speed is less than {self.shifting_speed_denom_threshold}, cannot compute cdot.")
        else:
            return - self.cdot_numerator / self.cdot_denominator

    def nonlinear(self, a: Vector) -> Vector:
        """Return the nonlinear part N(a) = d_rom + B_rom a + H_rom(a, a) - cdot * (n_rom + M_rom a)"""
        cdot = self.compute_shifting_speed(a)
        return self.constant_vector + self.quadratic(a) + cdot * (self.spatial_derivative_constant_vector + self.spatial_derivative_linear_matrix.dot(a))
    