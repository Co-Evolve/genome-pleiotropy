from typing import Dict

import numpy as np


def get_voxel_locations(params: Dict):
    resolution = params["morphology"]["resolution"]

    if len(resolution) == 2:
        dim_x, dim_y = resolution
        x, y = np.mgrid[-1:1:complex(dim_x), -1:1:complex(dim_y)]
        x, y = x.flatten(), y.flatten()
        z = None
    elif len(resolution) == 3:
        dim_x, dim_y, dim_z = resolution
        x, y, z = np.mgrid[-1:1:complex(dim_x), -1:1:complex(dim_y), -1:1:complex(dim_z)]
        x, y, z = x.flatten(), y.flatten(), z.flatten()
    else:
        raise ValueError(f"Expected 2-D or 3-D resolution, but got {resolution}-D.")

    return x, y, z


def tarjans_strongly_connected_components_algorithm(V, E):
    """
    Translated from pseudocode:
    https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
    """
    index = 0
    S = []
    components = []

    def strongconnect(v):
        nonlocal index, S
        v["index"] = index
        v["lowlink"] = index
        index += 1
        S.insert(0, v)
        v["onStack"] = True

        for w in [V[e[1]] for e in E if V[e[0]] == v]:
            if w["index"] is None:
                strongconnect(w)
                v["lowlink"] = min(v["lowlink"], w["lowlink"])
            else:
                v["lowlink"] = min(v["lowlink"], w["index"])

        if v["lowlink"] == v["index"]:
            components.append([])
            while True:
                w = S.pop(0)
                w["onStack"] = False
                components[-1].append(w)
                if w == v:
                    break

    for v in V:
        if v["index"] is None:
            strongconnect(v)

    return [[v["index"] for v in c] for c in components]
