from typing import Union, Iterable
from .typing import CVIndices

from tqdm import tqdm


__pdoc__ = {"_split_wrapper": False}


def _split_wrapper(
    indices: Union[Iterable, CVIndices], progress_bar: bool = False
) -> Union[Iterable, CVIndices, tqdm]:
    """Wraps indices with tqdm if progress_bar is True, else returns indices."""
    if progress_bar:
        return tqdm(indices)
    else:
        return indices
