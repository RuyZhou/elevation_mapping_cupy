import cupy as cp
import numpy as np
from typing import List
import re

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class SemanticColorFilter(PluginBase):
    def __init__(self, channels: List = ["semantic_r", "semantic_g", "semantic_b"], **kwargs):
        super().__init__()
        self.channels = channels
        
    def tranform_color(self, semantic_r, semantic_g, semantic_b):
        r = np.asarray(semantic_r, dtype=np.uint32)
        g = np.asarray(semantic_g, dtype=np.uint32)
        b = np.asarray(semantic_b, dtype=np.uint32)
        rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)
        rgb_arr.dtype = np.float32
        return cp.asarray(rgb_arr)

    def get_layer_indice(self, layer_names: List[str], target_layer_name) -> List[int]:
        """ Get the indices of the layers that are to be processed using regular expressions.
        Args:
            layer_names (List[str]): List of layer names.
        Returns:
            List[int]: List of layer indices.
        """
        indices = None
        for i, layer_name in enumerate(layer_names):
            print('layer_name: ', layer_name)
            print('target_layer_name: ', target_layer_name)
            if any(re.match(target_layer_name, layer_name)):
                indices = i
                break
        return indices
    
    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        rotation,
        elements_to_shift,
        *args,
    ) -> cp.ndarray:
        """
        Args:
            elevation_map (cupy._core.core.ndarray):
            layer_names (List[str]):
            plugin_layers (cupy._core.core.ndarray):
            plugin_layer_names (List[str]):
            semantic_map (elevation_mapping_cupy.semantic_map.SemanticMap):
            *args ():

        Returns:
            cupy._core.core.ndarray:
        """
        # get indices of all layers that contain semantic class information
        semantic_r_layer = None
        semantic_g_layer = None
        semantic_b_layer = None
        semantic_rgb = None
        for m, layer_names in zip(
            [elevation_map, plugin_layers, semantic_map], [layer_names, plugin_layer_names, semantic_layer_names]
        ):
            r_layer_indice = self.get_layer_indice(layer_names, "semantic_r")
            semantic_r_layer = m[r_layer_indice]
            g_layer_indice = self.get_layer_indice(layer_names, "semantic_g")
            semantic_g_layer = m[g_layer_indice]
            b_layer_indice = self.get_layer_indice(layer_names, "semantic_b")
            semantic_b_layer = m[b_layer_indice]
        if semantic_r_layer is not None and semantic_g_layer is not None and semantic_b_layer is not None:
            semantic_rgb = self.tranform_color(semantic_r_layer, semantic_g_layer, semantic_b_layer)
        return semantic_rgb
