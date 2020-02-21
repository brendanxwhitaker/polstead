import os
from asta import Tensor, typechecked, check, dims
DIM = dims.DIM

@typechecked
def identity(ob: Tensor[float, DIM, DIM, DIM]) -> Tensor[float, DIM, DIM, DIM]:
    return ob
