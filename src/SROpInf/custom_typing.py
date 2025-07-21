from typing import Any, Callable, List, Union

import numpy as np
import numpy.typing as npt
import torch

Vector = npt.NDArray[Any]
Matrix = npt.NDArray[Any]
Tensor = torch.Tensor
VectorField = Callable[[Vector], Vector]
VectorList = Union[List[Vector], npt.NDArray[np.float64]]
