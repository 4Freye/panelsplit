from tqdm import tqdm
def _split_wrapper(indices, progress_bar = False):
    if progress_bar:
        return tqdm(indices)
    else:
        return indices