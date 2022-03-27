import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def concatenate_lists(list_of_lists: [list]) -> list:
    """
    Concatenate a list of lists into a single list.
    """
    return [item for sublist in list_of_lists for item in sublist]


def argmax(array: list) -> int:
    """
    Return the index of the maximum value in a list.
    """
    index = -1
    for i, value in enumerate(array):
        if value > array[index]:
            index = i
    return index
