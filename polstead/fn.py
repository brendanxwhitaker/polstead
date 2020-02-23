""" Example function for shapechecking. """
from asta import Tensor, typechecked, check, dims

check.on()
DIM = dims.DIM


@typechecked
def identity(ob: Tensor[float, DIM, DIM, DIM]) -> Tensor[float, DIM, DIM, DIM]:
    """ Identity function on an RL observation. """
    return ob
