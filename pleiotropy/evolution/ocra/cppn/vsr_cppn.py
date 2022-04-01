from __future__ import annotations

import itertools
from typing import Tuple, List, Dict, Union, Optional

import numpy as np
import torch
from torch import tensor

from pleiotropy.evolution.ocra.neat.genome import DefaultGenome
from pleiotropy.evolution.ocra.pytorch_neat.cppn import Node, create_cppn, Leaf
from pleiotropy.evolution.utils.utils import get_voxel_locations

INPUT_NODES = ["x", "y", "r"]  # x-coordinate, y-coordinate, radial polar coordinate
MORPH_OUTPUT_NODES = ["p", "m"]  # presence, muscle (> 0 is muscle, < 0 is passive)
CONTROLLER_OUTPUT_NODES = ["o", "f"]  # phase offset, frequency
OUTPUT_NODES = MORPH_OUTPUT_NODES + CONTROLLER_OUTPUT_NODES


class VSRCPPN:
    """
    Class that represents a Compositional Pattern Producing Network (CPPN).
    """

    def __init__(self, params: Dict, genome: List[DefaultGenome], pt_cppn: List[Node],
                 nodes: List[Optional[Dict]] = None) -> None:
        self.params = params
        self.genome = genome
        self.pt_cppn = pt_cppn
        self.nodes = nodes

        self.__num_nodes_per_sub_genome = None
        self.__num_nodes = None
        self.__num_connections = None

        self.__shared_pathness = None
        self.__morph_shared_pathness = None
        self.__control_shared_pathness = None

        self.__shared_ratio = None
        self.__morph_shared_ratio = None
        self.__control_shared_ratio = None

    def is_valid(self, generation) -> bool:
        valid = True
        if self.params["pleiotropy"]["force_shared_path"]["enabled"]:
            valid &= self.shared_pathness > 0

            pathness_params = self.params["pleiotropy"]["force_shared_path"]
            if pathness_params["increasing"]:
                increase_per_gen = (pathness_params["stop"] - pathness_params["start"]) / \
                                   pathness_params["time"]
                min_pathness_threshold = min(generation * increase_per_gen + pathness_params["start"],
                                             pathness_params["stop"])
                valid &= self.shared_pathness >= min_pathness_threshold

        return valid

    @property
    def num_nodes_per_sub_genome(self):
        if self.__num_nodes_per_sub_genome is None:
            self.__num_nodes_per_sub_genome = [len(sub_genome.nodes) - len(sub_genome.output_labels) for sub_genome in
                                               self.genome]
        return self.__num_nodes_per_sub_genome

    @property
    def num_nodes(self):
        if self.__num_nodes is None:
            self.__num_nodes = sum(self.num_nodes_per_sub_genome)
        return self.__num_nodes

    @property
    def num_connections(self):
        if self.__num_connections is None:
            self.__num_connections = sum([len(sub_genome.connections) for sub_genome in self.genome])
        return self.__num_connections

    @property
    def morphology_nodes(self):
        return self._subtree(MORPH_OUTPUT_NODES)

    @property
    def controller_nodes(self):
        return self._subtree(CONTROLLER_OUTPUT_NODES)

    @property
    def presence_nodes(self):
        return self._subtree(['p'])

    @property
    def muscle_nodes(self):
        return self._subtree(['m'])

    @property
    def offset_nodes(self):
        return self._subtree(['o'])

    @property
    def frequency_nodes(self):
        return self._subtree(['f'])

    @property
    def shared_pathness(self):
        if self.__shared_pathness is None:
            if self.num_nodes == 0 or len(self.genome) != 1:
                self.__shared_pathness = 0.
            else:
                connections = list(itertools.chain(*[n.weights for n in self.nodes[0].values() if isinstance(n, Node)]))
                self.__shared_pathness = self._mutual_influence_similarity(connections, MORPH_OUTPUT_NODES,
                                                                           CONTROLLER_OUTPUT_NODES)
        return self.__shared_pathness

    @property
    def morph_shared_pathness(self):
        if self.__morph_shared_pathness is None:
            if self.num_nodes_per_sub_genome[0] == 0 or len(self.genome) > 2:
                self.__morph_shared_pathness = 0.
            else:
                connections = list(itertools.chain(*[n.weights for n in self.nodes[0].values() if isinstance(n, Node)]))
                self.__morph_shared_pathness = self._mutual_influence_similarity(connections, ['p'], ['m'])
        return self.__morph_shared_pathness

    @property
    def control_shared_pathness(self):
        if self.__control_shared_pathness is None:
            nodes_idx = len(self.genome) - 1
            if self.num_nodes_per_sub_genome[nodes_idx] == 0 or len(self.genome) > 2:
                self.__control_shared_pathness = 0.
            else:
                connections = list(
                    itertools.chain(*[n.weights for n in self.nodes[nodes_idx].values() if isinstance(n, Node)]))
                self.__control_shared_pathness = self._mutual_influence_similarity(connections, ['o'], ['f'])
        return self.__control_shared_pathness

    @property
    def shared_ratio(self):
        if self.__shared_ratio is None:
            if self.num_nodes == 0 or len(self.genome) != 1:
                self.__shared_ratio = 0.
            else:
                morphology_nodes = set(self.morphology_nodes)
                controller_nodes = set(self.controller_nodes)
                n_shared_nodes = len(list(morphology_nodes & controller_nodes))
                n_total_nodes = len(list(morphology_nodes | controller_nodes))
                self.__shared_ratio = n_shared_nodes / n_total_nodes
        return self.__shared_ratio

    @property
    def morph_shared_ratio(self):
        if self.__morph_shared_ratio is None:
            if self.num_nodes_per_sub_genome[0] == 0 or len(self.genome) > 2:
                self.__morph_shared_ratio = 0.
            else:
                presence_nodes = set(self.presence_nodes)
                muscle_nodes = set(self.muscle_nodes)
                n_shared_nodes = len(list(presence_nodes & muscle_nodes))
                n_total_nodes = len(list(presence_nodes | muscle_nodes))
                self.__morph_shared_ratio = n_shared_nodes / n_total_nodes
        return self.__morph_shared_ratio

    @property
    def control_shared_ratio(self):
        if self.__control_shared_ratio is None:
            nodes_idx = len(self.genome) - 1
            if self.num_nodes_per_sub_genome[nodes_idx] == 0 or len(self.genome) > 2:
                self.__control_shared_ratio = 0.
            else:
                offset_nodes = set(self.offset_nodes)
                frequency_nodes = set(self.frequency_nodes)
                n_shared_nodes = len(list(offset_nodes & frequency_nodes))
                n_total_nodes = len(list(offset_nodes | frequency_nodes))
                self.__control_shared_ratio = n_shared_nodes / n_total_nodes
        return self.__control_shared_ratio

    def _mutual_influence_similarity(self, connections, group1, group2):
        """
        Measure the similarity between the degree that connections influence both morphology and controller
        """
        x, y, z = get_voxel_locations(self.params)
        with torch.no_grad():
            group1_outputs = torch.cat(self.query(x, y, z, group1, return_tensors=True), dim=0)
            group2_outputs = torch.cat(self.query(x, y, z, group2, return_tensors=True), dim=0)

            group1_influence = np.zeros((len(connections),), dtype=float)
            group2_influence = np.zeros((len(connections),), dtype=float)

            for ii, connection in enumerate(connections):
                data = connection.data.clone()
                connection.data = tensor(0.)

                group1_outputs_without_connection = torch.cat(
                    self.query(x, y, z, group1, return_tensors=True), dim=0)
                group2_outputs_without_connection = torch.cat(
                    self.query(x, y, z, group2, return_tensors=True), dim=0)

                group1_df = np.sum(np.abs((group1_outputs - group1_outputs_without_connection).numpy()))
                group2_df = np.sum(np.abs((group2_outputs - group2_outputs_without_connection).numpy()))

                group1_influence[ii] = group1_df
                group2_influence[ii] = group2_df

                connection.data = data

        if np.linalg.norm(group1_influence) == 0 or np.linalg.norm(group2_influence) == 0:
            return 0.
        else:
            cosine_similarity = np.dot(group1_influence, group2_influence) \
                                / (np.linalg.norm(group1_influence) * np.linalg.norm(group2_influence))
            return cosine_similarity

    def _subtree(self, root_node_ids):
        queue = [self.pt_cppn[OUTPUT_NODES.index(node_id)] for node_id in root_node_ids]
        subtree = set()
        while len(queue) > 0:
            node = queue.pop(0)
            if type(node) != Leaf:
                subtree.add(node)
                for child_node in node.children:
                    if child_node not in subtree:  # Avoid infinite loops due to recurrence
                        queue.append(child_node)

        subtree = list(subtree)
        return subtree

    def query(self, x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray], target_nodes: List[str],
              return_tensors: bool = False) -> Tuple[Union[torch.Tensor, np.ndarray], ...]:
        x = torch.tensor(x)
        y = torch.tensor(y)

        if z is not None:
            z = torch.tensor(z)
            r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
            inputs = dict(x=x, y=y, z=z, r=r)
        else:
            r = torch.sqrt(x ** 2 + y ** 2)
            inputs = dict(x=x, y=y, r=r)

        assert all([self.pt_cppn[OUTPUT_NODES.index(node_id)].name == node_id for node_id in target_nodes])
        outputs: List[torch.Tensor] = [self.pt_cppn[OUTPUT_NODES.index(node_id)](inputs) for node_id in target_nodes]

        if not return_tensors:
            outputs = [ou.cpu().detach().numpy() for ou in outputs]

        return tuple(outputs)

    @staticmethod
    def create_cppn(genome, config, params) -> VSRCPPN:
        if isinstance(genome, DefaultGenome):
            genome = [genome]

        pt_cppn = []
        nodes = []
        for sub_genome in genome:
            sub_pt_cppn, sub_nodes = create_cppn(sub_genome, config, INPUT_NODES,
                                                 sub_genome.output_labels, output_activation="bss", return_nodes=True)
            pt_cppn += sub_pt_cppn
            nodes.append(sub_nodes)

        return VSRCPPN(params=params, pt_cppn=pt_cppn, nodes=nodes, genome=genome)
