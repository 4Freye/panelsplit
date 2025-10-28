from typing import Union
from .typing import CVIndices

from tqdm import tqdm


__pdoc__ = {"_split_wrapper": False}


def _split_wrapper(
    indices: CVIndices, progress_bar: bool = False
) -> Union[CVIndices, tqdm]:
    """Wraps indices with tqdm if progress_bar is True, else returns indices."""
    if progress_bar:
        return tqdm(indices)
    else:
        return indices
