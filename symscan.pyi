import numpy as np
from numpy.typing import NDArray
from typing import Iterable, Literal

def get_neighbors_within(
    query: Iterable[str],
    max_distance: int = 1,
    distance_type: Literal["levenshtein", "hamming"] = "levenshtein",
) -> tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint8]]: ...
def get_neighbors_across(
    query: Iterable[str],
    reference: Iterable[str],
    max_distance: int = 1,
    distance_type: Literal["levenshtein", "hamming"] = "levenshtein",
) -> tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint8]]: ...

class CachedRef:
    def __init__(
        self,
        reference: Iterable[str],
        max_distance: int = 1,
        distance_type: Literal["levenshtein", "hamming"] = "levenshtein",
    ) -> None: ...
    def get_neighbors_within(
        self,
        max_distance: int = 1,
    ) -> tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint8]]: ...
    def get_neighbors_across(
        self,
        query: Iterable[str] | "CachedRef",
        max_distance: int = 1,
    ) -> tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint8]]: ...
