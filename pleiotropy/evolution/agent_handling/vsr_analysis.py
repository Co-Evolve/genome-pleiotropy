from typing import Dict, List, Set

import numpy as np
import torch
from torch.nn import functional as F

from pleiotropy.evolution.agent_handling.vsr_morphology import VSRMorphology
from pleiotropy.evolution.ocra.cppn.vsr_cppn import VSRCPPN
from pleiotropy.evolution.ocra.neat import Config


def behavioral_differential_entropy(behaviors, params):
    # replace invalid behaviours with default [0, 0] (no movement)
    ndim = len(params["morphology"]["resolution"])
    dummy = np.zeros(ndim)

    behaviors = [b if b is not None else dummy for b in behaviors]
    num_behaviors = len(behaviors)
    if num_behaviors < 2:  # no diversity with only one agent
        behaviors = np.zeros((ndim, ndim))

    sigma = np.cov(behaviors, rowvar=0)
    differential_entropy = .5 * np.log(np.abs(np.linalg.det(sigma))) \
                           + num_behaviors / 2 * (1 + np.log(2 * np.pi))

    return differential_entropy


def behavioral_sparsity(behaviors, params) -> Dict:
    # sparsity: Lehman, J., & Stanley, K. O. (2011). Novelty search and the problem with objectives. In Genetic programming theory and practice IX (pp. 37-56). Springer, New York, NY.

    # replace invalid behaviours with default [0, 0] (no movement)
    ndim = len(params["morphology"]["resolution"])
    dummy = np.zeros(ndim)

    behaviors = [b if b is not None else dummy for b in behaviors]

    num_behaviors = len(behaviors)
    if num_behaviors < 2:  # no diversity with only one agent
        return {"k_100": [0], "k_50": [0], "k_25": [0], "k_10": [0], "k_0": [0]}

    X = np.stack(behaviors, axis=0)
    distance_matrix = X[None, :] - X[:, None]
    distance_matrix = np.sum(distance_matrix ** 2, axis=-1) ** .5
    distance_matrix = distance_matrix[~np.eye(num_behaviors, dtype=bool)].reshape(num_behaviors, -1)
    distances = np.sort(distance_matrix, axis=1)

    sparsity = {}
    for k_percentage in (100, 50, 25, 10, 0):
        k = max(1, int((num_behaviors - 1) / 100 * k_percentage))
        sparsity[f"k_{k_percentage}"] = np.mean(distances[:, :k], axis=1).tolist()

    return sparsity


def morphological_diversity(morphologies, params) -> List[float]:
    # replace invalid morphologies with default 1 x 1 (x 1) voxel
    morphologies = [m if m is not None
                    else {'muscle': np.array([True]), 'x': np.array([-1.]),
                          'y': np.array([-1.]), 'z': np.array([-1.])}
                    for m in morphologies]

    num_morphologies, num_dims = len(morphologies), len(params["morphology"]["resolution"])
    if num_morphologies < 2:  # there is no diversity
        return [0.]

    if num_dims == 2:
        width, height = params["morphology"]["resolution"]
        grid_unactuated = np.zeros((num_morphologies, height, width), dtype=int)
        grid_muscles = np.zeros_like(grid_unactuated)

        for ii, morphology in enumerate(morphologies):
            muscle_mask = morphology["muscle"]

            x = ((morphology['x'] + 1) / 2 * (width - 1)).astype(int)
            y = ((morphology['y'] + 1) / 2 * (height - 1)).astype(int)

            grid_unactuated[ii, y[~muscle_mask], x[~muscle_mask]] = 1
            grid_muscles[ii, y[muscle_mask], x[muscle_mask]] = 1

    elif num_dims == 3:
        width, height, length = params["morphology"]["resolution"]
        grid_unactuated = np.zeros((num_morphologies, height, width, length), dtype=int)
        grid_muscles = np.zeros_like(grid_unactuated)

        for ii, morphology in enumerate(morphologies):
            muscle_mask = morphology["muscle"]
            x = ((morphology['x'] + 1) / 2 * (width - 1)).astype(int)
            y = ((morphology['y'] + 1) / 2 * (height - 1)).astype(int)
            z = ((morphology['z'] + 1) / 2 * (length - 1)).astype(int)

            grid_unactuated[ii, z[~muscle_mask], y[~muscle_mask], x[~muscle_mask]] = 1
            grid_muscles[ii, z[muscle_mask], y[muscle_mask], x[muscle_mask]] = 1
    else:
        raise ValueError(f"Invalid number of dimensions. Expected 2 or 3, but got {num_dims}")

    grid_unactuated = torch.tensor(grid_unactuated)
    grid_muscles = torch.tensor(grid_muscles)

    # convolve grids to find maximum overlap
    if num_dims == 2:
        intersection_unactuated = F.conv2d(
            grid_unactuated[:, None, ...], grid_unactuated[:, None, ...],
            padding=(height - 1, width - 1))
        intersection_muscles = F.conv2d(
            grid_muscles[:, None, ...], grid_muscles[:, None, ...],
            padding=(height - 1, width - 1))
    elif num_dims == 3:
        intersection_unactuated = F.conv3d(
            grid_unactuated[:, None, ...], grid_unactuated[:, None, ...],
            padding=(length - 1, height - 1, width - 1))
        intersection_muscles = F.conv3d(
            grid_muscles[:, None, ...], grid_muscles[:, None, ...],
            padding=(length - 1, height - 1, width - 1))
    else:
        raise ValueError(f"Invalid number of dimensions. Expected 2 or 3, but got {num_dims}")
    reduction_dims = tuple(range(-1, -num_dims - 1, -1))
    intersection = torch.amax(intersection_muscles + intersection_unactuated,
                              dim=reduction_dims)

    # Union is the sum of the number of nodes in both morphologies minus intersection
    # (because intersection is counted twice when taking the sum)
    grid_voxels = grid_muscles + grid_unactuated
    union = torch.sum(grid_voxels, dim=reduction_dims)[:, None, ...] \
            + torch.sum(grid_voxels, dim=reduction_dims)[None, :, ...] \
            - intersection

    # Similarity is intersection over union
    similarity_matrix = (intersection / union).numpy()

    # the similarity matrix is symmetric, so we take the upper triangle, ignoring the diagonal
    triu_indices = np.argwhere(np.triu(np.ones(similarity_matrix.shape, dtype=bool), k=1))

    dissimilarities = 1 - similarity_matrix[triu_indices[:, 0], triu_indices[:, 1]]

    return dissimilarities.tolist()


# todo move this to
def morphology2grid(morphology, params):
    # todo: clean up this code
    num_dims = len(params["morphology"]["resolution"])

    if num_dims == 2:
        width, height = params["morphology"]["resolution"]
        grid_unactuated = np.zeros((height, width), dtype=int)
        grid_muscles = np.zeros_like(grid_unactuated)
        x = ((morphology['x'] + 1) / 2 * (width - 1)).astype(int)
        y = ((morphology['y'] + 1) / 2 * (height - 1)).astype(int)
        muscle_mask = morphology["muscle"]
        grid_unactuated[y[~muscle_mask], x[~muscle_mask]] = 1
        grid_muscles[y[muscle_mask], x[muscle_mask]] = 1
    elif num_dims == 3:
        width, height, length = params["morphology"]["resolution"]
        grid_unactuated = np.zeros((height, width, length), dtype=int)
        grid_muscles = np.zeros_like(grid_unactuated)

        muscle_mask = morphology["muscle"]
        x = ((morphology['x'] + 1) / 2 * (width - 1)).astype(int)
        y = ((morphology['y'] + 1) / 2 * (height - 1)).astype(int)
        z = ((morphology['z'] + 1) / 2 * (length - 1)).astype(int)

        grid_unactuated[z[~muscle_mask], y[~muscle_mask], x[~muscle_mask]] = 1
        grid_muscles[z[muscle_mask], y[muscle_mask], x[muscle_mask]] = 1
    else:
        raise ValueError(f"Invalid number of dimensions. Expected 2 or 3, but got {num_dims}")

    if np.all((grid_unactuated == 0) & (grid_muscles == 0)):
        return grid_muscles, grid_unactuated

    # shift morphology as close to origin as possible
    if num_dims == 2:
        while np.all((grid_unactuated[:, 0] == 0) & (grid_muscles[:, 0] == 0)):
            grid_unactuated = np.roll(grid_unactuated, shift=-1, axis=1)
            grid_muscles = np.roll(grid_muscles, shift=-1, axis=1)
        while np.all((grid_unactuated[0, :] == 0) & (grid_muscles[0, :] == 0)):
            grid_unactuated = np.roll(grid_unactuated, shift=-1, axis=0)
            grid_muscles = np.roll(grid_muscles, shift=-1, axis=0)
    elif num_dims == 3:
        while np.all((grid_unactuated[:, :, 0] == 0) & (grid_muscles[:, :, 0] == 0)):
            grid_unactuated = np.roll(grid_unactuated, shift=-1, axis=2)
            grid_muscles = np.roll(grid_muscles, shift=-1, axis=2)
        while np.all((grid_unactuated[:, 0, :] == 0) & (grid_muscles[:, 0, :] == 0)):
            grid_unactuated = np.roll(grid_unactuated, shift=-1, axis=1)
            grid_muscles = np.roll(grid_muscles, shift=-1, axis=1)
        while np.all((grid_unactuated[0, :, :] == 0) & (grid_muscles[0, :, :] == 0)):
            grid_unactuated = np.roll(grid_unactuated, shift=-1, axis=0)
            grid_muscles = np.roll(grid_muscles, shift=-1, axis=0)

    return grid_muscles, grid_unactuated


def add_morphology_to_global_tracker_if_novel(morphologies: List[VSRMorphology], global_dict: Set[str], params: Dict):
    # replace invalid morphologies with default 1 x 1 (x 1) voxel
    morphologies = [m if m is not None
                    else {'muscle': np.array([True]), 'x': np.array([-1.]),
                          'y': np.array([-1.]), 'z': np.array([-1.])}
                    for m in morphologies]

    for morphology in morphologies:
        grid_muscles, grid_unactuated = morphology2grid(morphology, params)
        morphology_hash_key = ''.join(grid_muscles.reshape(-1).astype(str).tolist()) \
                              + ''.join(grid_unactuated.reshape(-1).astype(str).tolist())
        global_dict.add(morphology_hash_key)


def plot_morphology(m, params):
    v_musc = np.zeros(params["morphology"]["resolution"], dtype=int)
    v_noac = np.zeros_like(v_musc)
    musc = m['muscle']
    x = ((m['x'] + 1) / 2 * (params["morphology"]["resolution"][0] - 1)).astype(int)
    y = ((m['y'] + 1) / 2 * (params["morphology"]["resolution"][1] - 1)).astype(int)

    v_musc[y[musc], x[musc]] = 1
    v_noac[y[~musc], x[~musc]] = 1


class VSRAnalysis:
    def __init__(self, genome_id: int, params: Dict, config: Config):
        self.genome_id = genome_id
        self.params = params
        self.config = config

        self.balance = None
        self.distance_x_axis = None
        self.distance_euclidean = None
        self.distance_xz_euclidean = None
        self.behavior = None
        self.behavior_xz = None
        self.morphology = None
        self.morphology_size = None
        self.morphology_muscle_ratio = None
        self.morphology_voxels = None

        self.cppn_shared_ratio = None
        self.cppn_morph_shared_ratio = None
        self.cppn_control_shared_ratio = None

        self.cppn_shared_pathness = None
        self.cppn_morph_shared_pathness = None
        self.cppn_control_shared_pathness = None

        self.cppn_num_nodes = None
        self.cppn_num_connections = None

    def analyze_trial(self, trial: Dict):
        self.balance = trial["balance"]
        self.distance_x_axis = trial["dist"] / self.params["environment"]["voxel_size"]
        self.behavior = trial["endpos"] / self.params["environment"]["voxel_size"]
        self.behavior_xz = trial["endpos"][[0, 2]] / self.params["environment"]["voxel_size"]
        self.distance_euclidean = np.sum(self.behavior ** 2) ** .5
        self.distance_xz_euclidean = np.sum(self.behavior_xz ** 2) ** .5

    def analyze_morphology(self, morphology: VSRMorphology):
        self.morphology = morphology
        self.morphology_size = morphology.x_voxels.size
        self.morphology_muscle_ratio = np.sum(morphology.muscle_voxels) / morphology.muscle_voxels.size
        self.morphology_voxels = {"muscle": morphology.muscle_voxels, "x": morphology.x_voxels,
                                  "y": morphology.y_voxels, "z": morphology.z_voxels}

    def analyze_cppn(self, cppn: VSRCPPN):
        self.cppn_num_nodes = cppn.num_nodes
        self.cppn_num_connections = cppn.num_connections

        self.cppn_shared_ratio = cppn.shared_ratio
        self.cppn_morph_shared_ratio = cppn.morph_shared_ratio
        self.cppn_control_shared_ratio = cppn.control_shared_ratio

        self.cppn_shared_pathness = cppn.shared_pathness
        self.cppn_morph_shared_pathness = cppn.morph_shared_pathness
        self.cppn_control_shared_pathness = cppn.control_shared_pathness
