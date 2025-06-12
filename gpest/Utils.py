import contextlib
import pickle
import random
import numpy as np


@contextlib.contextmanager
def seeded_random_state(seed):
    pre_rd_state = random.getstate()
    pre_np_state = np.random.get_state()

    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(pre_rd_state)
        np.random.set_state(pre_np_state)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
