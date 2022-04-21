from typing import Callable
from uuid import uuid1

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


def argmax(array: list, func: Callable = lambda x: x) -> int:
    """
    Return the index of the maximum value in a list.
    """
    index = -1
    max_value = None
    for i, each in enumerate(array):
        value = func(each)
        if max_value is None or value > max_value:
            index = i
            max_value = value
    return index


def trace_id() -> str:
    """
    Generate a trace id.
    """
    return str(uuid1()).replace('-', '')
