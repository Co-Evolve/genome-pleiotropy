from typing import Dict

import numpy as np

from pleiotropy.evolution.agent_handling.vsr_morphology import VSRMorphology
from pleiotropy.evolution.ocra.cppn.vsr_cppn import VSRCPPN


class VSRController:
    def __init__(self, genome_id: int, params: Dict, cppn: VSRCPPN):
        self.genome_id = genome_id
        self.params = params
        self.cppn = cppn

        self.offsets, self.frequency_multipliers, self.global_frequency, self.period = None, None, None, None

    def build(self, morphology: VSRMorphology) -> bool:
        muscle_x = morphology.x_voxels[morphology.muscle_voxels]
        muscle_y = morphology.y_voxels[morphology.muscle_voxels]
        if morphology.z_voxels is not None:
            muscle_z = morphology.z_voxels[morphology.muscle_voxels]
        else:
            muscle_z = None

        offsets, frequencies = self.cppn.query(muscle_x, muscle_y, muscle_z, target_nodes=['o', 'f'])

        # Scale to intervals from params
        frequencies = np.interp(frequencies, (-1, 1), self.params["controller"]["frequency_range"])
        self.global_frequency = frequencies.min()
        self.frequency_multipliers = np.rint(frequencies / self.global_frequency)

        average_frequency = max(
            np.sum(self.global_frequency * self.frequency_multipliers) / len(morphology.x_voxels),
            self.params["controller"]["frequency_range"][0])
        self.period = 2 * np.pi / abs(average_frequency + 1e-8)

        global_period = 2 * np.pi / abs(self.global_frequency + 1e-8)
        self.offsets = global_period * offsets

        return True

    def reset(self) -> None:
        self.offsets, self.frequency_multipliers, self.global_frequency = None, None, None

    def __eq__(self, other):
        return all(self.offsets == other.offsets) and all(self.frequency_multipliers == other.frequency_multipliers) \
               and self.global_frequency == other.global_frequency and self.period == other.period
