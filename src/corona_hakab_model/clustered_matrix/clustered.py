from math import prod
from typing import Collection, Iterable

import numpy as np

# todo test
# todo benchmark
from consts import generator
from sparse_base import ManifestBase, SparseBase


class Cluster:
    __slots__ = (
        "indices",
        "size",
        "probs",
        "vals",
        "probs_row_mul",
        "probs_col_mul",
        "vals_row_offs",
        "vals_col_offs",
        "_probs_actual",
        "_vals_actual",
    )

    def __init__(self, indices: Iterable[int]):
        self.indices = np.array(sorted(indices))
        self.size = len(self.indices)
        self.probs = np.zeros((self.size, self.size), dtype=np.float32)
        self.vals = np.zeros((self.size, self.size), dtype=np.float32)

        self.probs_row_mul = np.ones(self.size, dtype=np.float32)
        self.probs_col_mul = np.ones(self.size, dtype=np.float32)

        self.vals_row_offs = np.zeros(self.size, dtype=np.float32)
        self.vals_col_offs = np.zeros(self.size, dtype=np.float32)

        self._probs_actual = None
        self._vals_actual = None

    def batch_set(self, row, columns, probs, vals):
        self.probs[row, columns] = probs
        self.vals[row, columns] = vals
        self._vals_actual = self._probs_actual = None

    def row_set_prob_coff(self, row, coff):
        self.probs_row_mul[row] = coff
        self._probs_actual = None

    def col_set_prob_coff(self, col, coff):
        self.probs_col_mul[col] = coff
        self._probs_actual = None

    def row_set_value_offset(self, row, offs):
        self.vals_row_offs[row] = offs
        self._vals_actual = None

    def col_set_value_offset(self, col, offs):
        self.vals_col_offs[col] = offs
        self._vals_actual = None

    def actual_probs(self):
        if self._probs_actual:
            return self._probs_actual
        ret = np.copy(self.probs)
        ret *= self.probs_col_mul
        ret *= self.probs_row_mul.reshape(-1, 1)
        self._probs_actual = ret
        return ret

    def actual_vals(self):
        if self._vals_actual:
            return self._vals_actual
        ret = np.copy(self.vals)
        ret += self.vals_row_offs
        ret += self.vals_col_offs.reshape(-1, 1)
        self._vals_actual = ret
        return ret

    def __getitem__(self, item):
        p = self.probs[item]
        if p == 0:
            return None
        return p, self.vals[item]


class ClusteredSparseMatrix(SparseBase):
    __slots__ = ("size", "cluster_index", "local_indices", "clusters")

    def __init__(self, clusters: Iterable[Collection[int]]):
        self.size: int = sum(len(c) for c in clusters)
        self.cluster_index = np.empty(self.size, dtype=int)
        self.local_indices = np.empty(self.size, dtype=int)
        self.clusters = []
        for i, c in enumerate(clusters):
            self.clusters.append(Cluster(c))
            for j, a in enumerate(c):
                self.cluster_index[a] = i
                self.local_indices[a] = j

    def non_zero_columns(self):
        ret = [None] * self.size
        for cluster in self.clusters:
            for index, row in zip(cluster.indices, cluster.probs):
                ret[index] = list(cluster.indices[np.flatnonzero(row)])
        return ret

    def batch_set(self, row, columns, probs, vals):
        c_i = self.cluster_index[row]
        l_i = self.local_indices[row]
        local_columns = self.local_indices[columns]
        self.clusters[c_i].batch_set(l_i, local_columns, probs, vals)

    def row_set_prob_coff(self, row, coff):
        c_i = self.cluster_index[row]
        l_i = self.local_indices[row]
        self.clusters[c_i].row_set_prob_coff(l_i, coff)

    def col_set_prob_coff(self, col, coff):
        c_i = self.cluster_index[col]
        l_i = self.local_indices[col]
        self.clusters[c_i].col_set_prob_coff(l_i, coff)

    def row_set_value_offset(self, row, offs):
        c_i = self.cluster_index[row]
        l_i = self.local_indices[row]
        self.clusters[c_i].row_set_value_offset(l_i, offs)

    def col_set_value_offset(self, col, offs):
        c_i = self.cluster_index[col]
        l_i = self.local_indices[col]
        self.clusters[c_i].col_set_value_offset(l_i, offs)

    def __getitem__(self, item):
        i, j = item
        c_i = self.cluster_index[i]
        c_j = self.cluster_index[j]
        if c_i != c_j:
            return None
        l_i = self.local_indices[i]
        l_j = self.local_indices[j]
        return self.clusters[c_i][l_i, l_j]

    def manifest(self, sample=None, global_prob_factor: float = 1):
        if sample is None:
            sample = generator.random(sum(c.size ** 2 for c in self.clusters), dtype=np.float32)
        if global_prob_factor != 1:
            sample /= global_prob_factor
        s_next = 0
        ret = ManifestClusters(self)
        for i, c in enumerate(self.clusters):
            bunch_size = c.size ** 2
            is_manifest = sample[s_next: s_next + bunch_size].reshape(c.size, -1) < c.actual_probs()
            s_next += bunch_size
            ret.inners[i][is_manifest] = c.actual_vals()[is_manifest]
        return ret


class ManifestClusters(ManifestBase):
    __slots__ = ("original", "inners")

    def __init__(self, original: ClusteredSparseMatrix):
        self.original = original
        self.inners = [np.zeros_like(c.vals) for c in self.original.clusters]

    def I_POA(self, v: np.ndarray, magic):
        ret = np.empty(self.original.size, dtype=np.float32)
        for cluster, manifest in zip(self.original.clusters, self.inners):
            zipped_v = v[cluster.indices]
            ret[cluster.indices] = [prod(magic.operate(m, v) for (m, v) in zip(row, zipped_v)) for row in manifest]
        return ret

    def nz_rows(self):
        ret = [None] * self.original.size
        for cluster, manifest in zip(self.original.clusters, self.inners):
            for i, row in zip(cluster.indices, manifest):
                ret[i] = cluster.indices[np.flatnonzero(row)]
        return ret

    def __getitem__(self, item):
        i, j = item
        c_i = self.original.cluster_index[i]
        c_j = self.original.cluster_index[j]
        if c_i != c_j:
            return 0
        l_i = self.original.local_indices[i]
        l_j = self.original.local_indices[j]
        return self.inners[c_i][l_i, l_j]
