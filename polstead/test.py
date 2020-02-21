import os
import config
from asta import Tensor, typechecked, check, shapes
DIM = shapes.DIM

@typechecked
def identity(ob: Tensor[float, DIM, DIM, DIM]) -> Tensor[float, DIM, DIM, DIM]:
    return ob
