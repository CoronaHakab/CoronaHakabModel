from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, List, Tuple, Optional

import numpy as np


class SparseBase(Protocol):
    @abstractmethod
    def batch_set(self, row, columns: np.ndarray, probs: np.ndarray, vals: np.ndarray):
        """
        Set the probability and value of the matrix at the specified row and columns.
        It is an error to call this method twice on the same row
        It is an error to call this method with columns unsorted
        """

    @abstractmethod
    def row_set_prob_coff(self, row: int, coff: float):
        """
        set a multiplier on the probability across an entire row
        """

    @abstractmethod
    def col_set_prob_coff(self, col: int, coff: float):
        """
        set a multiplier on the probability across an entire column
        """

    @abstractmethod
    def row_set_value_offset(self, row: int, offs: float):
        """
        set an offset on the value across an entire row
        """

    @abstractmethod
    def col_set_value_offset(self, col: int, offs: float):
        """
        set an offset on the value across an entire column
        """

    @abstractmethod
    def __getitem__(self, item: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """
        Get a probability/value pair at the index, not including coefficients and offsets.
        Returns None if the index is zero
        """

    @abstractmethod
    def manifest(self, sample: np.ndarray = None) -> ManifestBase:
        """
        Create a manifested sparse matrix from sample rolls.
        if sample is None, a new rolls array is created
        """


class ManifestBase(Protocol):
    @abstractmethod
    def I_POA(self, v: np.ndarray, magic) -> np.ndarray:
        """
        calculate the inverse probability of any using vector v and a magic operator
        """

    @abstractmethod
    def nz_rows(self) -> List[List[int]]:
        """
        create and return a list of all indices for which the manifest matrix is non-zero, per row
        """

    @abstractmethod
    def __getitem__(self, item: Tuple[int, int])->float:
        """
        get the actual value of the manifest at indexes
        if the cell does not exists, or is not manifest, return 0
        """
