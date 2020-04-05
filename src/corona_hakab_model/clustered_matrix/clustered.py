from math import prod
from typing import Collection, Iterable

import numpy as np


# todo test
# todo benchmark

class Cluster:
    def __init__(self, indices: Iterable[int]):
        self.indices = sorted(indices)
        size = len(self.indices)
        self.inv_indices = {v: i for (i, v) in enumerate(self.indices)}
        self.probs = np.zeros((size, size), dtype=np.float32)
        self.vals = np.zeros((size, size), dtype=np.float32)

        self.probs_row_mul = np.ones(size, dtype=np.float32)
        self.probs_col_mul = np.ones(size, dtype=np.float32)

        self.vals_row_offs = np.zeros(size, dtype=np.float32)
        self.vals_col_offs = np.zeros(size, dtype=np.float32)

    def batch_set(self, row, columns, probs, vals):
        self.probs[row, columns] = probs
        self.vals[row, columns] = vals

    def nz_count(self):
        return np.count_nonzero(self.probs)

    def row_set_prob_coff(self, row, coff):
        self.probs_row_mul[row] = coff

    def col_set_prob_coff(self, col, coff):
        self.probs_col_mul[col] = coff

    def row_set_value_offset(self, row, offs):
        self.vals_row_offs[row] = offs

    def col_set_value_offset(self, col, offs):
        self.vals_col_offs[col] = offs

    def __getitem__(self, item):
        p = self.probs[item]
        if p == 0:
            return None
        return p, self.vals[item]


class ClusteredSparseMatrix:
    def __init__(self, clusters: Iterable[Collection[int]]):
        self.size = sum(len(c) for c in clusters)
        self.cluster_index = np.empty(self.size, dtype=int)
        self.local_indices = np.empty(self.size, dtype=int)
        self.clusters = []
        for i, c in enumerate(clusters):
            self.clusters.append(Cluster(c))
            for j, a in enumerate(c):
                self.cluster_index[a] = i
                self.local_indices[a] = j

    def batch_set(self, row, columns, probs, vals):
        c_i = self.cluster_index[row]
        l_i = self.local_indices[row]
        local_columns = self.local_indices[columns]
        self.clusters[c_i].batch_set(l_i, local_columns, probs, vals)

    def nz_count(self):
        return sum(c.nz_count() for c in self.clusters)

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


class ManifestClusters:
    def __init__(self, original: ClusteredSparseMatrix):
        self.original = original
        self.inners = [
            np.zeros_like(c.vals) for c in self.original.clusters
        ]

    def I_POA(self, v: np.ndarray, magic):
        ret = np.empty(self.original.size, dtype=np.float32)
        for cluster, manifest in zip(self.original.clusters, self.inners):
            zipped_v = v[cluster.indices]
            ret[cluster.indices] = [
                prod(magic.operate(m, v) for (m,v) in zip(row, zipped_v))
                for row in manifest
            ]
        return ret

    def nz_rows(self):
        ret = [None] * self.original.size
        for cluster, manifest in zip(self.original.clusters, self.inners):
            for i, row in zip(cluster.indices, manifest):
                ret[i] = np.flatnonzero(row)
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
