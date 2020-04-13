from typing import Dict

from generation.connection_types import ConnectionTypes
from sparse_base import SparseBase
from sparse_matrix import MagicOperator


class EncounterLayer:
    def __init__(self, kind: ConnectionTypes, magic_op: MagicOperator, matrix: SparseBase):
        self.kind = kind
        self.magic_op = magic_op
        self.matrix = matrix
        self.global_prob_factor: float = 1.0

    def manifest(self):
        return self.matrix.manifest(global_prob_factor=self.global_prob_factor)


class EncounterLayerSet:
    def __init__(self):
        self.layers: Dict[ConnectionTypes, EncounterLayer] = {}

    def add_layer(self, layer: EncounterLayer):
        if self.layers.setdefault(layer.kind, layer) is not layer:
            raise Exception(f"layer of kind {layer.kind} already added")

    def POA(self, v):
        manifests = {
            n: m.manifest() for (n, m) in self.layers.items() if m.global_prob_factor
        }
        m_iter = iter(manifests.items())
        first_label, first_manifest = next(m_iter)
        c = first_manifest.I_POA(v, self.layers[first_label].magic_op)
        for label, manifest in m_iter:
            c *= manifest.I_POA(v, self.layers[label].magic_op)
        return manifests, (1 - c)
