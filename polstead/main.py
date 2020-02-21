import importlib

import torch
from asta import Tensor, typechecked, dims

import test


def main() -> None:
    global DIM

    # Before we set ``DIM``, typecheck fails.
    ob = torch.ones((5,5,5))
    try:
        res = test.identity(ob)
    except TypeError as err:
        print("TYPECHECK FAILED.")
        
    # Set ``DIM`` to the correct size, and reload any typechecked functions.
    dims.DIM = 5
    importlib.reload(test)
    res = test.identity(ob)


if __name__ == "__main__":
    main()
