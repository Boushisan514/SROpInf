from typing import Any, Callable, List, Union

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[Any]
Matrix = npt.NDArray[Any]
Rank3Tensor = npt.NDArray[Any]
VectorField = Callable[[Vector], Vector]
VectorList = Union[List[Vector], npt.NDArray[np.float64]]
