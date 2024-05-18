import cupy as cp
import numpy as np
from typing import List
import re

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class RGBColorFilter(PluginBase):
    def __init__(self, channels: List = ["r", "g", "b"], **kwargs):
        super().__init__()
        self.channels = channels
        
    def tranform_color(self, rgb_r, rgb_g, rgb_b):
        r = rgb_r.get().astype(np.uint32)
        g = rgb_g.get().astype(np.uint32)
        b = rgb_b.get().astype(np.uint32)
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
            # print('layer_name: ', layer_name)
            # print('target_layer_name: ', target_layer_name)
            if re.match(target_layer_name, layer_name):
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
        rgb_r_layer = None
        rgb_g_layer = None
        rgb_b_layer = None
        color_rgb = None
        for m, layer_names in zip(
            [elevation_map, plugin_layers, semantic_map], [layer_names, plugin_layer_names, semantic_layer_names]
        ):
            r_layer_indice = self.get_layer_indice(layer_names, "r")
            rgb_r_layer = m[r_layer_indice]
            g_layer_indice = self.get_layer_indice(layer_names, "g")
            rgb_g_layer = m[g_layer_indice]
            b_layer_indice = self.get_layer_indice(layer_names, "b")
            rgb_b_layer = m[b_layer_indice]
        if rgb_r_layer is not None and rgb_g_layer is not None and rgb_b_layer is not None:
            color_rgb = self.tranform_color(rgb_r_layer, rgb_g_layer, rgb_b_layer)
        return color_rgb
