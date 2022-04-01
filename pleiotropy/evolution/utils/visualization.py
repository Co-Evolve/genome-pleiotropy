import itertools
from collections import defaultdict
from typing import Union

import graphviz

from pleiotropy.evolution.ocra.cppn.vsr_cppn import CPPN, INPUT_NODES, OUTPUT_NODES, MORPH_QUERY, CONTROLLER_QUERY
from pleiotropy.evolution.ocra.pytorch_neat.cppn import Node, Leaf

try:
    import cPickle as pickle
except:
    import pickle


def draw_cppn(net: CPPN, filename=None):
    """
    Uses graphviz to draw a CPPN with arbitrary topology.
    """
    try:
        config = [
            ((INPUT_NODES, OUTPUT_NODES), filename),
            (MORPH_QUERY, filename + '_morph_query'),
            (CONTROLLER_QUERY, filename + '_controller_query'),
        ]
        for (input_node_names, output_node_names), name in config:
            node_idx = itertools.count(0)

            node_attrs = {
                'shape': 'circle',
                'fontsize': '9',
                'height': '0.2',
                'width': '0.2'}

            dot = graphviz.Digraph('svg', node_attr=node_attrs)

            connections = defaultdict(set)
            node_to_idx = dict()

            def rec_traverse(node: Union[Node, Leaf]):
                if node not in node_to_idx:
                    # Create dot node
                    if node.name is None:
                        # Hidden node
                        node_to_idx[node] = node_to_idx.get(node, str(next(node_idx)))
                        idx = node_to_idx[node]
                        label = node.activation_name
                        color = 'white'
                        shape = 'oval'
                    else:
                        # Input or output node
                        node_to_idx[node] = node_to_idx.get(node, str(next(node_idx)))
                        idx = node_to_idx[node]
                        if node.name in input_node_names:
                            label = f"{node.name}"
                            color = 'lightgray'
                        elif node.name in output_node_names:
                            label = f"{node.name}:{node.activation_name}"
                            color = 'lightblue'
                        else:
                            return None
                        shape = 'box'

                    input_attrs = {'style': 'filled',
                                   'shape': shape,
                                   'fillcolor': color}
                    dot.node(idx, label=label, _attributes=input_attrs)

                # Iterate over children
                try:
                    for child, weight in zip(node.children, node.weights):
                        connections[node_to_idx[node]].add((rec_traverse(child), weight))
                except AttributeError as ex:
                    pass

                return node_to_idx[node]

            for node in net.pt_cppn:
                if node.name in output_node_names:
                    rec_traverse(node)
            for node in net.pt_cppn[0].leaves.values():
                if node.name in input_node_names:
                    rec_traverse(node)

            for node_idx, cons in connections.items():
                for i, w in cons:
                    if i is not None:
                        input_attrs = {'style': 'solid',
                                       'color': 'green' if w > 0.0 else 'red',
                                       'penwidth': str(0.1 + abs(w / 3.0))}
                        dot.edge(i, node_idx, _attributes=input_attrs)

            dot.render(name)
    except Exception as ex:
        print(ex)
