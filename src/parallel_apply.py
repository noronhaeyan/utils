from threading import Thread
from functools import partial
import numpy as np
import pandas as pd


def parallelize(data, func, n_jobs=8):
    data_split = np.array_split(data, n_jobs)

    t = [None] * n_jobs

    output = [pd.DataFrame()] * n_jobs

    for i in range(n_jobs):
        t[i] = Thread(target=func, args=(data_split[i], output, i, ))

    for i in range(n_jobs):
        t[i].start()

    for i in range(n_jobs):
        t[i].join()

    return pd.concat(output, axis=0)


def run_on_subset(func, data_subset, output, index):
    output[index] = data_subset.apply(func, axis=1)


def parallelize_on_rows(data, func, n_jobs=8):
    return parallelize(data, partial(run_on_subset, func), n_jobs=n_jobs)

# Usage: Instead of dataFrame.apply(my_func, axis = 1) use parallelize_on_rows(dataFrame, my_func, 8)
