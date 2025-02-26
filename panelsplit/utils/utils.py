from tqdm import tqdm


__pdoc__ = {
    '_split_wrapper': False}

def _split_wrapper(indices, progress_bar = False):
    if progress_bar:
        return tqdm(indices)
    else:
        return indices