""" Example script for testing typecheck toggling. """
import importlib

import torch
from asta import dims

import fn
from fn import identity


def main() -> None:
    """ Test asta dims functionality. """
    # Before we set ``DIM``, typecheck fails.
    ob = torch.ones((5, 5, 5))
    try:
        identity(ob)
    except TypeError as _err:
        print("TYPECHECK FAILED.")

    # Set ``DIM`` to the correct size, and reload any typechecked functions.
    dims.DIM = 5
    importlib.reload(fn)
    fn.identity(ob)


if __name__ == "__main__":
    main()
