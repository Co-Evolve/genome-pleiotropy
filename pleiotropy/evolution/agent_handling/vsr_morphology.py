from typing import Dict

import numpy as np
from scipy.ndimage import label

from pleiotropy.evolution.ocra.cppn.vsr_cppn import VSRCPPN
from pleiotropy.evolution.utils.utils import get_voxel_locations


class VSRMorphology:
    def __init__(self, genome_id: int, params: Dict, cppn: VSRCPPN):
        self.genome_id = genome_id
        self.params = params
        self.cppn = cppn

        self.x_voxels, self.y_voxels, self.z_voxels, self.muscle_voxels = None, None, None, None
        self._binary = None

    @property
    def binary(self):
        if self._binary is None:
            grid = np.zeros((2, *self.params["morphology"]["resolution"]), dtype=bool)

            muscle_mask = self.muscle_voxels
            x = np.rint(((self.x_voxels + 1) / 2 * (grid.shape[-1] - 1))).astype(int)
            y = np.rint(((self.y_voxels + 1) / 2 * (grid.shape[-2] - 1))).astype(int)
            if self.z_voxels is None:
                grid[0, y[~muscle_mask], x[~muscle_mask]] = True
                grid[1, y[muscle_mask], x[muscle_mask]] = True
            else:
                z = np.rint(((self.z_voxels + 1) / 2 * (grid.shape[-3] - 1))).astype(int)
                grid[0, z[~muscle_mask], y[~muscle_mask], x[~muscle_mask]] = 1
                grid[1, z[muscle_mask], y[muscle_mask], x[muscle_mask]] = 1

            grid = grid.reshape(-1).tolist()
            self._binary = '0b' + "".join('1' if b else '0' for b in grid)

        return self._binary

    def _get_biggest_cluster_mask(self, presences) -> np.ndarray:
        presences_grid = presences.reshape(self.params["morphology"]["resolution"])
        clusters, n_clusters = label(presences_grid)

        biggest_cluster = np.argmax([np.sum(clusters == cluster) for cluster in range(1, n_clusters + 1)]) + 1

        return clusters.flatten() == biggest_cluster

    def build(self) -> bool:
        # x, y, z = self._get_voxel_locations()
        x, y, z = get_voxel_locations(self.params)

        presences, types = self.cppn.query(x, y, z, target_nodes=['p', 'm'])
        presences = presences > 0.0
        muscle = types > 0.0

        if np.sum(presences) < self.params["morphology"]["min_n_muscle_voxels"]:
            # Number of presences is already smaller than required number of muscle voxels
            return False

        mask = self._get_biggest_cluster_mask(presences)

        self.x_voxels, self.y_voxels = x[mask], y[mask]
        if z is not None:
            self.z_voxels = z[mask]

        self.muscle_voxels = muscle[mask]

        return np.sum(self.muscle_voxels) >= self.params["morphology"]["min_n_muscle_voxels"]

    def reset(self) -> None:
        self.x_voxels, self.y_voxels, self.z_voxels, self.muscle_voxels = None, None, None, None
