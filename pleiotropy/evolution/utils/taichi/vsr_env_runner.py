import gc
import logging
import signal
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import numpy as np
import psutil
import ray
import torch
from ray.util import ActorPool

from pleiotropy.environment.vsr_2d import VSR2D
from pleiotropy.environment.vsr_3d import VSR3D
from pleiotropy.evolution.agent_handling.vsr_agent import VSRAgent, AgentEvaluationResult
from pleiotropy.evolution.agent_handling.vsr_analysis import morphological_diversity
from pleiotropy.evolution.evolutionary_algorithms.base.evolution_handler import BaseEvolutionHandler
from pleiotropy.evolution.ocra.neat.config import Config
from pleiotropy.evolution.ocra.neat.genome import DefaultGenome, GenomeType
from pleiotropy.evolution.utils.tools import pyout


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def vsr_env_runner_factory(params):
    @ray.remote(num_cpus=params["execution"]["n_cores_per_process"],
                num_gpus=0.2 if torch.cuda.is_available() and params["execution"]["hardware"] == 'gpu' else 0)
    class VSREnvRunner:
        def __init__(self, worker_id: int, config: Config, params: Dict, allow_saving: Optional[bool] = True) -> None:
            self.worker_id = worker_id
            self.config = config
            self.params = params
            self.allow_saving = allow_saving

            self.time_limit = self.params["evaluation"]["time_limit"]
            self._num_evalations = self.params["evaluation"]["num_evaluations"]

            self.evaluated_morphologies = set()
            self.num_evaluations_done = 0
            self.env = None
            self._prep_taichi()

        def _eval_agent(self, agent) -> float:
            try:
                with time_limit(3600 if self.params["experiment"]["debug"] else self.time_limit):
                    if agent.morphology.z_voxels is not None:
                        # trial_result = VSR3D(self.params).run(agent)
                        trial_result = self.env.run(agent)
                    else:
                        trial_result = VSR2D(self.params).run(agent)
                agent.analysis.analyze_trial(trial_result)

                reward = max(agent.analysis.distance_x_axis, 0)
                if self.params["experiment"]["balanced_locomotion"]:
                    reward *= agent.analysis.balance

                return reward
            except TimeoutException:
                logging.warning(f"Agent evaluation error: Exceeded time threshold of {self.time_limit}s")
                return 0.0

        def _prep_taichi(self) -> None:
            import taichi as ti
            real = ti.f32
            if self.num_evaluations_done > 0:
                ti.reset()
                gc.collect()

            if self.params["execution"]["hardware"] == 'gpu' and torch.cuda.is_available():
                ti.init(default_fp=real, arch=ti.gpu,
                        device_memory_GB=self.params["execution"]["gpu_memory"])
            elif self.params["execution"]["hardware"] == 'metal':
                ti.init(default_fp=real, arch=ti.metal)
            else:
                # Will automatically fall back to appropriate cpu architecture
                ti.init(default_fp=real, arch=ti.cpu)

            self.env = VSR3D(self.params)

        def eval_genome(self, generation: int,
                        genome: Tuple[int, GenomeType]) -> AgentEvaluationResult:
            agent = VSRAgent(genome, self.config, self.params, generation=generation)

            if all(agent.valid):
                evaluation_results = [self._eval_agent(agent) for _ in range(self._num_evalations)]
            else:
                evaluation_results = [0]

            evaluation_result = agent.finish_evaluation(generation, evaluation_results, self.allow_saving)

            return evaluation_result

        def visualize_genome(self,
                             genome: Tuple[int, GenomeType],
                             path: str,
                             color_t0: np.array([255, 255, 255], dtype=np.uint8),
                             skip_morphology_replicates=False):
            agent = VSRAgent(genome, self.config, self.params, genome_path=path)
            assert all(agent.valid)

            # check if this morphology has already been evaluated with a different controller
            if skip_morphology_replicates:
                if agent.morphology.binary in self.evaluated_morphologies:
                    return
                else:
                    self.evaluated_morphologies.add(agent.morphology.binary)

            try:
                with time_limit(3600 if self.params["experiment"]["debug"] else self.time_limit):
                    if agent.morphology.z_voxels is None:
                        env = VSR2D(self.params, use_graphics=True)
                    else:
                        env = VSR3D(self.params, use_graphics=True, color_t0=color_t0)
                    trial_result = env.run(agent)
                    agent.analysis.analyze_trial(trial_result)
                    return agent.finish_evaluation(0, [max(agent.analysis.distance_x_axis, 0)], self.allow_saving)
            except AttributeError as ex:
                logging.warning(f"Agent evaluation error: AttributeError -> {ex}")
            except TimeoutException:
                logging.warning(f"Agent visualization error: Exceeded time threshold of {self.time_limit}s")
            pyout()

        def eval_compliant_mutation_single(self,
                                           genomes: Tuple[Tuple[int, DefaultGenome], Tuple[int, DefaultGenome]]) -> \
                Optional[Tuple[float, int]]:
            child_genome, parent_genome = genomes

            child = VSRAgent(child_genome, self.config, self.params)
            parent = VSRAgent(parent_genome, self.config, self.params)

            if all(child.valid) and all(parent.valid) and morphological_diversity(
                    [child.analysis.morphology_voxels, parent.analysis.morphology_voxels], params)[0] > 0:

                child_morph_parent_control = VSRAgent(parent_genome, self.config, self.params,
                                                      morphology=child.morphology)
                if child.controller != child_morph_parent_control.controller:
                    child_morph_parent_control_fitness = self._eval_agent(child_morph_parent_control)
                    return child_morph_parent_control_fitness, child_genome[0]

        def eval_compliant_mutation_distinct(self,
                                             genomes: Tuple[Tuple[int, Tuple[DefaultGenome, DefaultGenome]], Tuple[
                                                 int, Tuple[DefaultGenome, DefaultGenome]]]) -> \
                Optional[Tuple[float, int]]:
            child_genome, parent_genome = genomes
            child_id, (child_morph, child_control) = child_genome
            parent_id, _ = parent_genome

            child = VSRAgent(child_genome, self.config, self.params, skip_controller=True)
            parent = VSRAgent(parent_genome, self.config, self.params, skip_controller=True)

            if all(child.valid) and all(parent.valid) and morphological_diversity(
                    [child.analysis.morphology_voxels, parent.analysis.morphology_voxels], params)[0] > 0:

                _, mutated_child_controller = BaseEvolutionHandler.create_new_child_from_parent(self.config,
                                                                                                self.params,
                                                                                                (child_morph,
                                                                                                 child_control),
                                                                                                child_id)
                child_morph_mutated_control = VSRAgent((child_id, (child_morph, mutated_child_controller)), self.config,
                                                       self.params,
                                                       morphology=child.morphology)

                import taichi as ti
                real = ti.f32

                if self.params["execution"]["hardware"] == 'gpu' and torch.cuda.is_available():
                    ti.init(default_fp=real, arch=ti.gpu, flatten_if=True,
                            device_memory_GB=self.params["execution"]["gpu_memory"])
                elif self.params["execution"]["hardware"] == 'metal':
                    ti.init(default_fp=real, arch=ti.metal, flatten_if=True)
                else:
                    # Will automatically fall back to appropriate cpu architecture
                    ti.init(default_fp=real, arch=ti.cpu, flatten_if=True)

                child_morph_mutated_control_fitness = self._eval_agent(child_morph_mutated_control)
                return child_morph_mutated_control_fitness, child_id

    return VSREnvRunner


def get_worker_pool(params: Dict, config: Config, allow_saving: bool = True) -> ActorPool:
    num_slaves = params["execution"]["n_processes"]
    if num_slaves == -1:
        num_slaves = psutil.cpu_count()

    env_runners = [
        vsr_env_runner_factory(params).remote(worker_id=i + 1, config=config, params=params,
                                              allow_saving=allow_saving) for
        i in range(num_slaves)]

    return ActorPool(env_runners)
