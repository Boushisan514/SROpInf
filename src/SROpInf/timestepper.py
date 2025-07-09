"""timestepper - step ordinary differential equations forward in time"""

import abc
import numpy as np
from typing import Callable, Dict, List, Type, Tuple

from .custom_typing import Vector, VectorField

__all__ = ["Timestepper", "AdaptiveTimestepper", "SemiImplicit"]

class Timestepper(abc.ABC):
    """Abstract base class for timesteppers."""

    # registry for subclasses, mapping names to constructors
    __registry: Dict[str, Type["Timestepper"]] = {}

    def __init_subclass__(cls) -> None:
        name = cls.__name__.lower()
        cls.__registry[name] = cls

    @classmethod
    def lookup(cls, method: str) -> Type["Timestepper"]:
        """Return the subclass corresponding to the string in `method`."""
        try:
            return cls.__registry[method.lower()]
        except KeyError as exc:
            raise NotImplementedError from exc

    def __init__(self, dt: float, rhs: VectorField):
        self.dt = dt
        self.rhs = rhs

    @abc.abstractmethod
    def step(self, x: Vector, rhs: VectorField) -> Vector:
        """Advance the state x by one timestep, for the ODE x' = rhs(x)."""

    @classmethod
    def methods(cls) -> List[str]:
        return list(cls.__registry.keys())
    
class AdaptiveTimestepper(abc.ABC):
    """Abstract base class for adaptive timesteppers."""

    # registry for subclasses, mapping names to constructors
    __registry: Dict[str, Type["AdaptiveTimestepper"]] = {}

    def __init_subclass__(cls) -> None:
        name = cls.__name__.lower()
        cls.__registry[name] = cls

    @classmethod
    def lookup(cls, method: str) -> Type["AdaptiveTimestepper"]:
        """Return the subclass corresponding to the string in `method`."""
        try:
            return cls.__registry[method.lower()]
        except KeyError as exc:
            raise NotImplementedError from exc

    def __init__(self, dt: float, err_tol: float, rhs: VectorField):
        self.dt_initial = dt
        self._dt = dt
        self.err_tol = err_tol
        self.rhs = rhs
        self._convergence = False  # Convergence = True means that the new timestep is converged and can be used for the next step
        self._stability = True    # Stability = True means that the new timestep is smaller than the minimum threshold
        
    @property
    def dt(self) -> float:
        return self._dt
    
    @dt.setter
    def dt(self, value: float) -> None:
        self._dt = value
        
    @abc.abstractmethod
    def dt_update(self, err: float) -> float:
        """Update the timestep to a new value based on the value of error"""
    
    @property
    def convergence(self) -> bool:
        """Return the current convergence indicator."""
        return self._convergence
    
    @convergence.setter
    def convergence(self, value: bool) -> None:
        self._convergence = value
        
    @property
    def stability(self) -> bool:
        """Return the current stability indicator."""
        return self._stability
    
    @stability.setter
    def stability(self, value: bool) -> None:
        self._stability = value
        
    @abc.abstractmethod
    def temporal_step(self, x: Vector, rhs: VectorField) -> Tuple[Vector, float]:
        """Advance the state x by a tentative timestep, for the ODE x' = rhs(x).
           Returns the temporal new state and the truncation error.
           The error is used to update the timestep size.
        """

    @abc.abstractmethod
    def step(self, x: Vector, rhs: VectorField) -> Vector:
        """Advance the state x by the updated timestep, for the ODE x' = rhs(x)."""

    @classmethod
    def methods(cls) -> List[str]:
        return list(cls.__registry.keys())

class SemiImplicit(abc.ABC):
    """Abstract base class for semi-implicit timesteppers."""

    # registry for subclasses, mapping names to constructors
    __registry: Dict[str, Type["SemiImplicit"]] = {}

    def __init_subclass__(cls) -> None:
        name = cls.__name__.lower()
        cls.__registry[name] = cls

    @classmethod
    def lookup(cls, method: str) -> Type["SemiImplicit"]:
        """Return the subclass corresponding to the string in `method`."""
        try:
            return cls.__registry[method.lower()]
        except KeyError as exc:
            raise NotImplementedError from exc

    def __init__(
        self,
        dt: float,
        linear: VectorField,
        nonlinear: VectorField,
        solver_factory: Callable[[float], Callable[[Vector], Vector]],
    ):
        self._dt = dt
        self.linear = linear
        self.nonlinear = nonlinear
        self.get_solver = solver_factory
        self.update()

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, value: float) -> None:
        self._dt = value
        self.update()

    @abc.abstractmethod
    def update(self) -> None:
        """Update quantities used in the semi-implicit solve.

        This routine is called when the timestepper is created, and whenever
        the timestep is changed
        """

    @abc.abstractmethod
    def step(self, x: Vector, nonlinear: VectorField) -> Vector:
        """Advance the state forward by one step"""

    @classmethod
    def methods(cls) -> List[str]:
        return list(cls.__registry.keys())

class Euler(Timestepper):
    """Explicit Euler timestepper."""

    def step(self, x: Vector) -> Vector:
        return x + self.dt * self.rhs(x)

class RK2(Timestepper):
    """Second-order Runge-Kutta timestepper."""

    def step(self, x: Vector) -> Vector:
        k1 = self.dt * self.rhs(x)
        k2 = self.dt * self.rhs(x + k1)
        return x + (k1 + k2) / 2.0

class RK4(Timestepper):
    """Fourth-order Runge-Kutta timestepper."""

    def step(self, x: Vector) -> Vector:
        k1 = self.dt * self.rhs(x)
        k2 = self.dt * self.rhs(x + k1 / 2.0)
        k3 = self.dt * self.rhs(x + k2 / 2.0)
        k4 = self.dt * self.rhs(x + k3)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    
class RKF45(AdaptiveTimestepper):
    
    # The standard Runge-Kutta-Fehlberg method for adaptive time-stepping
    # Please see Mathews & Fink Numerical Methods using MATLAB 4ed for more details about adaptive timestep control
    # Also see the Table III (the most commonly used version of RKF45) in wikipedia page of RKF45 method
    
    def __init__(self, dt: float, err_tol: float, rhs: VectorField):
        
        # First call the parent's constructor to set up shared variables
        super().__init__(dt, err_tol, rhs)
        self.A  = np.array([0, 1/4, 3/8, 12/13, 1, 1/2]) # A(k)
        self.B  = np.array([[0, 0, 0, 0, 0],
                           [1/4, 0, 0, 0, 0],
                           [3/32, 9/32, 0, 0, 0],
                           [1932/2197, -7200/2197, 7296/2197, 0, 0],
                           [439/216, -8, 3680/513, -845/4104, 0],
                           [-8/27, 2, -3544/2565, 1859/4104, -11/40]])
        self.C  = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0]) # this is the 4th order RK method
        self.CH = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]) # this is the 5th order RK method that we used
        
        self.dt_max = 100 * self.dt_initial
        self.dt_min = 1e-2 * self.dt_initial
        
    def temporal_step(self, x: Vector) -> Tuple[Vector, float]:
        
        A = self.A
        B = self.B
        C = self.C
        CH = self.CH
        dt = self.dt
        
        k1 = dt * self.rhs(x)
        k2 = dt * self.rhs(x + B[1,0] * k1)
        k3 = dt * self.rhs(x + B[2,0] * k1 + B[2,1] * k2)
        k4 = dt * self.rhs(x + B[3,0] * k1 + B[3,1] * k2 + B[3,2] * k3)
        k5 = dt * self.rhs(x + B[4,0] * k1 + B[4,1] * k2 + B[4,2] * k3 + B[4,3] * k4)
        k6 = dt * self.rhs(x + B[5,0] * k1 + B[5,1] * k2 + B[5,2] * k3 + B[5,3] * k4 + B[5,4] * k5)
        
        state_4th = x + C[0] * k1 + C[2] * k3 + C[3] * k4 + C[4] * k5  # the new state given by 4th order RK method
        state_5th = x + CH[0] * k1 +  CH[2] * k3 + CH[3] * k4 + CH[4] * k5 + CH[5] * k6 # the new state given by 5th order RK method
        
        err = np.linalg.norm(state_5th - state_4th, ord = 1)
        
        return state_5th, err
    
    def step(self, x: Vector) -> Vector:
        
        state_5th, err = self.temporal_step(x)
        self.dt = self.dt_update(err)
        
        while not self.convergence and self.stability:
                
            state_5th, err = self.temporal_step(x)
            self.dt = self.dt_update(err)
            if not self.stability:
                break
        
        return state_5th
    
    def dt_update(self, err: float) -> float:
        
        self.dt = self.dt * (self.err_tol * self.dt / (2 * err)) ** 0.25
        
        if self.dt < self.dt_min:
            # The timestep search is not converged, and the timestep is too small indicating unstable simulation
            self.convergence = False
            self.stability = False
            return self.dt_min
        
        elif self._dt > self.dt_max:
            # The timestep search is not converged, but the timestep is large enough indicating a stable simulation
            # to make the output not so sparse, we set the timestep to the maximum value
            self.convergence = True
            self.stability = True
            return self.dt_max
        
        else:
            
            if err > self.err_tol:
                # The timestep search is not converged yet but overall the simulation is still stable
                self.convergence = False
                self.stability = True
                
            else:
                # The timestep search is converged, and the simulation is stable
                self.convergence = True
                self.stability = True
            
            return self.dt

class RK2CN(SemiImplicit):
    """Semi-implicit timestepper: Crank-Nicolson + 2nd-order Runge-Kutta.

    See Peyret p148-149
    """

    def update(self) -> None:
        self.solve = self.get_solver(0.5 * self.dt)

    def step(self, x: Vector) -> Vector:
        rhs_linear = x + 0.5 * self.dt * self.linear(x)
        Nx = self.nonlinear(x)

        rhs1 = rhs_linear + self.dt * Nx
        x1 = self.solve(rhs1)

        rhs2 = rhs_linear + 0.5 * self.dt * (Nx + self.nonlinear(x1))
        x2 = self.solve(rhs2)
        return x2

class RK3CN(SemiImplicit):
    """Semi-implicit timestepper: Crank-Nicolson + 3rd-order Runge-Kutta.

    Peyret, p.146 and 149
    """

    A = [0, -5.0 / 9, -153.0 / 128]
    B = [1.0 / 3, 15.0 / 16, 8.0 / 15]
    Bprime = [1.0 / 6, 5.0 / 24, 1.0 / 8]

    def update(self) -> None:
        self.solvers = [self.get_solver(b * self.dt) for b in self.Bprime]

    def step(self, x: Vector) -> Vector:
        A = self.A
        B = self.B
        Bprime = self.Bprime

        Q1 = self.dt * self.nonlinear(x)
        rhs1 = x + B[0] * Q1 + Bprime[0] * self.dt * self.linear(x)
        x1 = self.solvers[0](rhs1)

        Q2 = A[1] * Q1 + self.dt * self.nonlinear(x1)
        rhs2 = x1 + B[1] * Q2 + Bprime[1] * self.dt * self.linear(x1)
        x2 = self.solvers[1](rhs2)

        Q3 = A[2] * Q2 + self.dt * self.nonlinear(x2)
        rhs3 = x2 + B[2] * Q3 + Bprime[2] * self.dt * self.linear(x2)
        x3 = self.solvers[2](rhs3)
        return x3