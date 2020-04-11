from typing import Dict

from sparse_base import SparseBase


class EncounterLayer:
    def __init__(self, name: str, magic_op, matrix: SparseBase):
        self.name = name
        self.magic_op = magic_op
        self.matrix = matrix


class EncounterLayerSet:
    def __init__(self):
        self.layers: Dict[str, EncounterLayer] = {}

    def add_layer(self, layer: EncounterLayer):
        self.layers[layer.name] = layer

    def POA(self, v):
        manifests = {
            n: m.matrix.manifest() for (n, m) in self.layers.items()
        }
        m_iter = iter(manifests.items())
        first_label, first_manifest = next(m_iter)
        c = first_manifest.I_POA(v, self.layers[first_label].magic_op)
        for label, manifest in m_iter:
            c *= manifest.I_POA(v, self.layers[label].magic_op)
        return manifests, (1 - c)
