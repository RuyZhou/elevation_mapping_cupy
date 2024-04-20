import cupy as cp
import numpy as np
from typing import List
import re

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class ScientificFilter(PluginBase):
    def __init__(self, classes: list = ["_background_, normal_rock, scientific_rock"], **kwargs):
        super().__init__()
        self.classes = classes
        self.value_encoding = self.value_map(classes)

    def value_map(self, classes):
        vmap = np.zeros((len(classes)), dtype=np.float32)
        for i, cls in enumerate(classes):
            if cls == "_background_":
                vmap[i] = 0.0
            elif cls == "normal_rock":
                vmap[i] = 0.0
            elif cls == "scientific_rock":
                vmap[i] = 1.0
        return cp.asarray(vmap)

    def get_layer_indices(self, layer_names: List[str]) -> List[int]:
        indices = []
        for i, layer_name in enumerate(layer_names):
            if any(re.match(pattern, layer_name) for pattern in self.classes):
                indices.append(i)
        return indices
    
    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        # get indices of all layers that contain semantic class information
        data = []
        for m, layer_names in zip(
            [elevation_map, plugin_layers, semantic_map], [layer_names, plugin_layer_names, semantic_layer_names]
        ):
            layer_indices = self.get_layer_indices(layer_names)
            if len(layer_indices) > 0:
                data.append(m[layer_indices])
        if len(data) > 0:
            data = cp.concatenate(data, axis=0)
            class_map_id = cp.argmax(data, axis=0)
        else:
            class_map_id = cp.zeros_like(elevation_map[0], dtype=cp.int32)
        mask = (class_map_id != cp.nan)
        scientific_value = cp.where(mask, self.value_encoding[class_map_id], cp.nan)
        return scientific_value