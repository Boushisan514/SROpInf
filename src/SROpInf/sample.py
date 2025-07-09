import pickle
from typing import Any, Callable, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import Dataset

from .custom_typing import Vector, VectorField

__all__ = ["sample", "sample_gradient", "load", "sample_gradient_long_traj"]

class TrajectoryList(Dataset[Vector]):
    """
    Container for samples of trajectories

    Suppose traj is a numpy array of dimension (num_traj, n, *)
    That is, there are num_traj trajectories, each of length n

    dataset = TrajectoryList(traj)
    dataset.traj[i] is trajectory i (an array with n samples)
    dataset[j] is sample j (from all trajectories, concatenated together)

    This class is compatible with torch.DataLoader:

    training = DataLoader(dataset, batch_size=64, shuffle=True)
    """

    def __init__(self, traj: ArrayLike):
        self.traj = np.array(traj)
        self.num_traj = self.traj.shape[0]
        self.n = self.traj.shape[1]
        newshape = list(self.traj.shape)
        newshape[1] *= newshape[0]
        newshape.pop(0)
        self.data = self.traj.view()
        self.data.shape = tuple(newshape)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Vector:
        return self.data[i]

    def save(self, fname: str) -> None:
        with open(fname, "wb") as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)


class GradientDataset(Dataset[Tuple[Vector, Vector]]):
    def __init__(self, X: ArrayLike, G: ArrayLike):
        self.X = np.array(X)
        self.G = np.array(G)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> Tuple[Vector, Vector]:
        return self.X[i], self.G[i]

    def save(self, fname: str) -> None:
        with open(fname, "wb") as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)


def load(fname: str) -> Any:
    with open(fname, "rb") as fp:
        data = pickle.load(fp)
    return data


def sample(
    step: VectorField, random_state: Callable[[], Vector], num_traj: int, n: int
) -> TrajectoryList:
    """
    Sample num_traj trajectories each with length n

    random_state() generates a random initial state x
    step(x) advances the state forward in time

    Returns a TrajectoryList object
    """
    traj_list = list()
    for _ in range(num_traj):
        traj = list()
        x = random_state()
        traj.append(x)
        for _ in range(n - 1):
            x = step(x)
            x = np.array(x)  # NetworkROM in torch
            traj.append(x)
        traj_list.append(traj)
    return TrajectoryList(traj_list)


def sample_gradient(
    traj_list: TrajectoryList,
    adj_step: Callable[[Vector, Vector], Vector],
    adj_output: Callable[[Vector, Vector], Vector],
    num_outputs: int,
    samples_per_traj: int,
    L: int,
) -> GradientDataset:
    """Sample the gradient using the standard method discussed in Section 3 of
    [1].

    Args:
        traj_list (TrajectoryList): data structure that is used to define a
            list of trajectories.
        model (Model): dynamical system being sampled.
        samples_per_traj (int): number of gradient samples calculated
            per trajectory.
        L (int): time horizon used for advancing the adjoint variable.

    Returns:
        GradientDataset: State and gradient data structure
        compatible with PyTorch's dataloader. GradientDataset.X[i] is the ith
        state sample and GradientDataset.G[i] is the ith gradient sample.

    References:
        [1] Otto, S.E., Padovan, A. and Rowley, C.W., 2022. Model Reduction
        for Nonlinear Systems by Balanced Truncation of State and
        Gradient Covariance.
    """
    X = list()
    G = list()
    N = traj_list.n  # num pts in each trajectory
    for x in traj_list.traj:
        for _ in range(samples_per_traj):
            # choose a time t in [0..N-1-L]
            t = np.random.randint(N - L)
            # choose a tau in [0..L]
            tau = np.random.randint(L + 1)
            # choose random direction eta for gradient
            eta = np.sqrt(L + 1) * np.random.randn(num_outputs)
            lam = adj_output(x[t + tau], eta)
            for i in range(1, tau):
                lam = adj_step(x[t + tau - i], lam)
            X.append(x[t])
            G.append(lam)
    return GradientDataset(X, G)


def sample_gradient_long_traj(
    traj_list: TrajectoryList,
    adj_step: Callable[[Vector, Vector], Vector],
    adj_output: Callable[[Vector, Vector], Vector],
    num_outputs: int,
    samples_per_traj: int,
    L: int,
) -> Tuple[GradientDataset, NDArray[np.float64]]:
    """Sample the gradient using the method of long trajectories discussed in
    Algorithm 3.1 of [1].

    Args:
        traj_list (TrajectoryList): data structure that is used to define a
            list of trajectories.
        model (Model): dynamical system being sampled.
        samples_per_traj (int): number of gradient samples calculated
            per trajectory.
        L (int): time horizon used for advancing the adjoint variable.

    Returns:
        tuple:
            GradientDataset (GradientDataset): State and gradient data
            structure compatible with PyTorch's dataloader.
            GradientDataset.X[i] is the ith state sample and
            GradientDataset.G[i] is the ith gradient sample.

            D (ndarray): Vector of scaling factors, D, taking the form

            .. math:: \\frac{1}{\\sqrt{1 - \\tau_{max} - \\tau_{min}}}

            given in Algorithm 3.1 of [1]. The
            matrix Y in Algorithm 3.1 can be computed using
            Y = D * GradientDataset.G.

    References:
        [1] Otto, S.E., Padovan, A. and Rowley, C.W., 2022. Model Reduction
        for Nonlinear Systems by Balanced Truncation of State and
        Gradient Covariance.
    """
    X = list()
    G = list()
    D = list()
    N = traj_list.n  # num pts in each trajectory
    for x in traj_list.traj:
        for _ in range(samples_per_traj):
            t = np.random.randint(N - L)
            tau = np.random.randint(L + 1)
            eta = np.sqrt(L + 1) * np.random.randn(num_outputs)
            tau_min = np.max((0, t + tau - (N - L - 1)))
            tau_max = np.min((L, t + tau))
            nu = 1 + tau_max - tau_min
            X_ = list()
            Lam = list()
            X_.append(x[t + tau])
            Lam.append(adj_output(x[t + tau], eta))
            for i in range(1, tau_max):
                X_.append(x[t + tau - i])
                Lam.append(adj_step(x[t + tau - i], Lam[i - 1]))
            X.extend(X_[tau_min:tau_max])
            G.extend(Lam[tau_min:tau_max])
            D.extend([1 / np.sqrt(nu)] * len(Lam[tau_min:tau_max]))
    return GradientDataset(X, G), np.array(D).reshape(-1, 1)
