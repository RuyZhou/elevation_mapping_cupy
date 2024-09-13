#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import cupy as cp
from typing import List
import re

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class GeometryTraversability(PluginBase):
    def __init__(
        self,
        input_layer_name,
        **kwargs,
    ):
        super().__init__()
        self.input_layer_name = input_layer_name

    def get_layer_indice(self, layer_names: List[str], target_layer_name) -> List[int]:
        """ Get the indices of the layers that are to be processed using regular expressions.
        Args:
            layer_names (List[str]): List of layer names.
        Returns:
            List[int]: List of layer indices.
        """
        indices = None
        for i, layer_name in enumerate(layer_names):
            if re.match(target_layer_name, layer_name):
                indices = i
                break
        return indices
    
    def transform_traversability(self, height_layer, is_valid):
        w, h = height_layer.shape
        center_height = height_layer[int(w/2-1), int(h/2-1)]
        traversability_layer = cp.ones_like(height_layer, dtype=cp.float32)  # traversable - 1, non-traversable - 0
        low_value = cp.where(height_layer-center_height > 0.05, 0, 1)
        not_valid_value = cp.where(is_valid, 0, 1)
        traversability_layer = traversability_layer * low_value * not_valid_value
        return traversability_layer
    
    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        height_layer = None
        traversability_layer = None
        is_valid_layer = None
        for m, layer_names in zip(
            [elevation_map, plugin_layers], [layer_names, plugin_layer_names]
        ):
            layer_index = self.get_layer_indice(layer_names, self.input_layer_name)
            if layer_index is not None:
                height_layer = m[layer_index]
            layer_index = self.get_layer_indice(layer_names, 'is_valid')
            if layer_index is not None:
                is_valid_layer = m[layer_index]
        if (height_layer is not None) and (is_valid_layer is not None):
            traversability_layer = self.transform_traversability(height_layer, is_valid_layer)
        else:
            raise ValueError(f"Layer {self.input_layer_name} not found.")
        # print('traversability: ', cp.unique(traversability_layer))
        # print('-----------------------')
        # print('traversability shape: ', traversability_layer.shape)
        # print('traversability: ', cp.sum(traversability_layer == 1))
        # print('non traversability: ', cp.sum(traversability_layer == 0))
        # print('sum up: ', cp.sum(traversability_layer == 1) + cp.sum(traversability_layer == 0))
        return traversability_layer