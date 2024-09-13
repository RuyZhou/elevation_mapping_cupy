#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import cupy as cp
from typing import List
import re

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class TraversabilityFilter(PluginBase):
    def __init__(
        self,
        input_layer_name,
        traversability_threshold,
        **kwargs,
    ):
        super().__init__()
        self.input_layer_name = input_layer_name
        self.traversability_threshold = cp.asarray(traversability_threshold)

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
    
    def transform_traversability(self, raw_traversability_layer, is_valid):
        traversability_layer = cp.ones_like(raw_traversability_layer, dtype=cp.float32)
        low_value = cp.where(raw_traversability_layer < self.traversability_threshold, 0, 1)
        not_valid_value = cp.where(is_valid, 0, 1)
        # traversability_layer = traversability_layer * low_value * nan_value
        # traversability_layer = traversability_layer * low_value
        traversability_layer = traversability_layer * low_value + not_valid_value
        traversability_layer = cp.where(traversability_layer > 0.5, 1, 0)
        # traversability_layer = traversability_layer * valid_value
        return traversability_layer
    
    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        raw_traversability_layer = None
        is_valid_layer = None
        traversability_layer = None
        for m, layer_names in zip(
            [elevation_map], [layer_names]
        ):
            layer_index = self.get_layer_indice(layer_names, self.input_layer_name)
            if layer_index is not None:
                raw_traversability_layer = m[layer_index]
            layer_index = self.get_layer_indice(layer_names, 'is_valid')
            if layer_index is not None:
                is_valid_layer = m[layer_index]
        if (raw_traversability_layer is not None) and (is_valid_layer is not None):
            traversability_layer = self.transform_traversability(raw_traversability_layer, is_valid_layer)
        else:
            raise ValueError(f"Layer {self.input_layer_name} not found.")
        return traversability_layer