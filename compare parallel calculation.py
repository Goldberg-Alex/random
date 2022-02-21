import itertools
import multiprocessing as mp
import time

import numba
import numpy as np
from tqdm import tqdm

matrix_size = 1000
secondary_size = 8


def basic_calc():
    mat = np.zeros((matrix_size, matrix_size, secondary_size))
    start_time = time.time()
    for i in tqdm(range(matrix_size)):
        mat[i, :, :] = iterate_rows(i)
    assert np.all(mat)
    print(f"basic took {time.time() - start_time}")


def parallel_calc():
    start_time = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        args = list(range(matrix_size))
        result = pool.map(iterate_rows, args)

    mat = np.stack(result)

    assert np.all(mat)
    print(f"parallel took {time.time() - start_time}")


def very_parallel_calc():
    start_time = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        args = list(itertools.combinations_with_replacement(range(matrix_size), 2))
        result = pool.starmap(iterate_single_index, args)

    mat = np.stack(result)

    assert np.all(mat)
    print(f"very parallel took {time.time() - start_time}")


def main():
    basic_calc()
    parallel_calc()
    very_parallel_calc()


@numba.jit(nopython=True)
def iterate_single_index(i, j):
    mat = np.zeros(secondary_size)
    for k in range(secondary_size):
        mat[k] = assign_single(i, j, k)

    return mat


@numba.jit(nopython=True)
def iterate_rows(i):
    mat = np.zeros((matrix_size, secondary_size))

    for j in range(matrix_size):
        for k in range(secondary_size):
            mat[j][k] = assign_single(i, j, k)
    return mat


@numba.jit(nopython=True)
def assign_single(i, j, k):
    return np.random.rand() * np.random.rand() * np.random.rand()


if __name__ == "__main__":
    main()
