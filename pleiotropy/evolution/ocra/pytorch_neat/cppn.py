# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
import torch
from torch import nn

from pleiotropy.evolution.ocra.neat.graphs import required_for_output
from .activations import str_to_activation
from .aggregations import str_to_aggregation
from ...utils.utils import tarjans_strongly_connected_components_algorithm


class Node(nn.Module):
    def __init__(self,
                 children,
                 weights,
                 response,
                 bias,
                 activation,
                 activation_name,
                 aggregation,
                 name=None, leaves=None):
        """
        children: list of Nodes
        weights: list of floats
        response: float
        bias: float
        activation: torch function from .activations
        aggregation: torch function from .aggregations
        name: str
        leaves: dict of Leaves
        """
        super().__init__()
        self.children = children
        self.leaves = leaves
        # todo: don't need to be tensors anymore
        self.weights = [torch.tensor(w, requires_grad=False) for w in weights]
        self.recurrent = [False for w in weights]
        self.response = response
        self.bias = torch.tensor(bias, requires_grad=False)
        self.activation = activation
        self.activation_name = activation_name
        self.aggregation = aggregation
        self.aggregation_name = aggregation
        self.name = name
        if leaves is not None:
            assert isinstance(leaves, dict)
        self.leaves = leaves

        self.activating = False
        self.activs = None
        self.activs_previous = 0.
        self.is_reset = None

    def __repr__(self):
        child_reprs = []
        for w, child in zip(self.weights, self.children):
            child_reprs.append(
                "    <- {} * ".format(w) + child.header
            )
        return self.header + "\n" + "\n".join(child_reprs)

    @property
    def header(self):
        return "Node({}, response={}, bias={}, activation={}, aggregation={})".format(
            self.name,
            self.response,
            self.bias,
            self.activation_name,
            self.aggregation_name,
        )

    def get_cmp_str(self) -> str:
        header = "Node({})".format(
            self.activation_name,
        )
        child_reprs = sorted([child.get_cmp_str() for child in self.children])
        return header + "$" + "|".join(child_reprs)

    def activate(self, xs, shape):
        """
        xs: list of torch tensors
        """
        if not xs:
            return torch.ones(shape) * self.bias
        inputs = [w * x for w, x in zip(self.weights, xs)]
        try:
            pre_activs = self.aggregation(inputs)
            activs = self.activation(self.response * pre_activs + self.bias)
            assert activs.shape == shape, "Wrong shape for node {}".format(self.name)
        except Exception:
            raise Exception("Failed to activate node {}".format(self.name))

        return activs

    def get_activs(self, shape):
        if self.activating:
            return self.activs_previous
        else:
            if self.activs is None:
                self.activating = True
                xs = [None for _ in range(len(self.children))]
                for ii, child in enumerate(self.children):
                    if child.activating:
                        self.recurrent[ii] = True
                    xs[ii] = child.get_activs(shape)
                    # if rec and False:
                    #     if child.activating:
                    #         xs[ii] = child.activs_previous
                    #     else:
                    #         xs[ii] = child.get_activs(shape)
                    # else:
                    #     xs[ii] = child.get_activs(shape)
                # xs = [child.get_activs(shape) for child in self.children]

                self.activs = self.activate(xs, shape)
                self.activating = False
            return self.activs

    def __call__(self, inputs, n_iters=1):
        assert self.leaves is not None
        assert inputs
        assert n_iters > 0
        shape = list(inputs.values())[0].shape
        self.reset()
        for name in self.leaves.keys():
            assert (
                    inputs[name].shape == shape
            ), "Wrong activs shape for leaf {}, {} != {}".format(
                name, inputs[name].shape, shape
            )
            self.leaves[name].set_activs(inputs[name])

        for _ in range(n_iters):
            self.prepare_next_timestep(shape)
            ou = self.get_activs(shape)

        return ou

    def _prereset(self):
        if self.is_reset is None:
            self.is_reset = False
            for child in self.children:
                child._prereset()  # pylint: disable=protected-access

    def _postreset(self):
        if self.is_reset is not None:
            self.is_reset = None
            for child in self.children:
                child._postreset()  # pylint: disable=protected-access

    def _reset(self):
        if not self.is_reset:
            self.is_reset = True
            self.activs = None
            self.activs_previous = None
            self.activating = False
            for child in self.children:
                child._reset()  # pylint: disable=protected-access

    def reset(self):
        self._prereset()  # pylint: disable=protected-access
        self._reset()  # pylint: disable=protected-access
        self._postreset()  # pylint: disable=protected-access

    def _next_timestep(self, shape):
        if not self.is_reset:
            self.is_reset = True
            self.activs_previous = torch.zeros(shape) if self.activs is None else self.activs.clone()
            self.activs = None
            self.activating = False
            for child in self.children:
                child._next_timestep(shape)

    def prepare_next_timestep(self, shape):
        self._prereset()
        self._next_timestep(shape)
        self._postreset()


class Leaf:
    def __init__(self, name=None):
        self.activs = None
        self.name = name
        self.activating = False

    def __repr__(self):
        return "Leaf({})".format(self.name)

    @property
    def header(self):
        return repr(self)

    def get_cmp_str(self) -> str:
        return "Leaf({})".format(self.name)

    def set_activs(self, activs):
        self.activs = activs

    def get_activs(self, shape):
        assert self.activs is not None, "Missing activs for leaf {}".format(self.name)
        assert (
                self.activs.shape == shape
        ), "Wrong activs shape for leaf {}, {} != {}".format(
            self.name, self.activs.shape, shape
        )
        return self.activs

    def _prereset(self):
        pass

    def _postreset(self):
        pass

    def _reset(self):
        self.activs = None

    def _next_timestep(self, shape):
        pass

    def next_timestep(self, shape):
        pass

    def reset(self):
        self._reset()


def create_cppn(genome, config, leaf_names, node_names, output_activation=None, return_nodes=False):
    genome_config = config.genome_config
    required = required_for_output(
        genome_config.input_keys, genome_config.output_keys, genome.connections
    )

    # Gather inputs and expressed connections.
    node_inputs = {i: [] for i in genome_config.output_keys}
    for cg in genome.connections.values():
        if not cg.enabled:
            continue

        i, o = cg.key
        if o not in required and i not in required:
            continue

        # if i in genome_config.output_keys:
        #     continue

        if o not in node_inputs:
            node_inputs[o] = [(i, cg.weight)]
        else:
            node_inputs[o].append((i, cg.weight))

        if i not in node_inputs:
            node_inputs[i] = []

    nodes = {i: Leaf() for i in genome_config.input_keys}

    assert len(leaf_names) == len(genome_config.input_keys)
    leaves = {name: nodes[i] for name, i in zip(leaf_names, genome_config.input_keys)}

    def build_node(idx):
        if idx in nodes:
            return nodes[idx]
        node = genome.nodes[idx]
        conns = node_inputs[idx]
        weights = [w for i, w in conns]
        if idx in genome_config.output_keys and output_activation is not None:
            node.activation = output_activation
        activation = str_to_activation[node.activation]
        aggregation = str_to_aggregation[node.aggregation]
        # Add new node to dict before adding children to prevent unbreakable loops with reccurrency
        nodes[idx] = Node(
            [],
            weights,
            node.response,
            node.bias,
            activation,
            node.activation,
            aggregation,
            leaves=leaves,
        )
        nodes[idx].children += [build_node(i) for i, w in conns]

        return nodes[idx]

    for idx in genome_config.output_keys:
        build_node(idx)

    outputs = [nodes[i] for i in genome_config.output_keys]

    for name in leaf_names:
        leaves[name].name = name

    for i, name in zip(genome_config.output_keys, node_names):
        nodes[i].name = name

    # check recurrency
    # label_recurrent_connections([nodes[ii] for ii in nodes if ii > max(genome_config.output_keys)])
    # label_recurrent_connections([nodes[ii] for ii in nodes if ii > 0])

    if return_nodes:
        return outputs, nodes
    else:
        return outputs


def label_recurrent_connections(nodes):
    V = [{'index': None} for _ in range(len(nodes))]
    E = []
    for node in nodes:
        for child in node.children:
            if child in nodes:
                E.append((nodes.index(child), nodes.index(node)))

    SCCs = tarjans_strongly_connected_components_algorithm(V, E)  # indices
    SCCs = [[nodes[ii] for ii in c] for c in SCCs]

    for component in SCCs:
        for node1 in component:
            for node2 in component:
                try:
                    ii = node1.children.index(node2)
                    node1.recurrent[ii] = True
                except ValueError:
                    continue


def clamp_weights_(weights, weight_threshold=0.2, weight_max=3.0):
    # TODO: also try LEO
    low_idxs = weights.abs() < weight_threshold
    weights[low_idxs] = 0
    weights[weights > 0] -= weight_threshold
    weights[weights < 0] += weight_threshold
    weights[weights > weight_max] = weight_max
    weights[weights < -weight_max] = -weight_max


def get_coord_inputs(in_coords, out_coords, batch_size=None):
    n_in = len(in_coords)
    n_out = len(out_coords)

    if batch_size is not None:
        in_coords = in_coords.unsqueeze(0).expand(batch_size, n_in, 2)
        out_coords = out_coords.unsqueeze(0).expand(batch_size, n_out, 2)

        x_out = out_coords[:, :, 0].unsqueeze(2).expand(batch_size, n_out, n_in)
        y_out = out_coords[:, :, 1].unsqueeze(2).expand(batch_size, n_out, n_in)
        x_in = in_coords[:, :, 0].unsqueeze(1).expand(batch_size, n_out, n_in)
        y_in = in_coords[:, :, 1].unsqueeze(1).expand(batch_size, n_out, n_in)
    else:
        x_out = out_coords[:, 0].unsqueeze(1).expand(n_out, n_in)
        y_out = out_coords[:, 1].unsqueeze(1).expand(n_out, n_in)
        x_in = in_coords[:, 0].unsqueeze(0).expand(n_out, n_in)
        y_in = in_coords[:, 1].unsqueeze(0).expand(n_out, n_in)

    return (x_out, y_out), (x_in, y_in)
