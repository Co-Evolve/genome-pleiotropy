from typing import List, Union, Dict, Set

import numpy as np
import wandb

from pleiotropy.evolution.agent_handling.vsr_agent import AgentEvaluationResult
from pleiotropy.evolution.agent_handling.vsr_analysis import morphological_diversity, \
    add_morphology_to_global_tracker_if_novel


class CMAArchive(object):
    pass


class TBLogger:
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.generation = None
        self.total_morph_overtakes = 0

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        wandb.log({tag: value}, step=step)

    def log_evaluation_results(self, evaluation_results: List[AgentEvaluationResult], num_evals_done: int) -> None:
        fitnesses = [er.total_reward for er in evaluation_results]
        self.log_scalar("Performance/mean_fitness", np.mean(fitnesses), self.generation)
        self.log_scalar("Performance/median_fitness", np.median(fitnesses), self.generation)
        self.log_scalar("Performance/max_fitness", np.max(fitnesses), self.generation)
        self.log_scalar("Evolution/Num_Evals", num_evals_done, self.generation)

    def log_analysis(self, evaluation_results: List[AgentEvaluationResult],
                     inter_generational_morphologies: Set[str], params):
        analyses = [er.analysis for er in evaluation_results]
        parent_analyses = [er.analysis for er in evaluation_results if er.is_parent]

        def log_list(array_: List[Union[float, int]], prefix: str):
            self.log_scalar(f"{prefix}_mean", np.mean(array_), self.generation)
            self.log_scalar(f"{prefix}_median", np.median(array_), self.generation)
            self.log_scalar(f"{prefix}_std", np.std(array_), self.generation)

        def log_metric(metric: str, default_value: float = 0.):
            # valid = [eval(f"a.{metric}") for a in analyses if eval(f"a.{metric}") is not None]
            # valid = valid if len(valid) > 0 else [default_value]
            # log_list(valid, f"Analysis/valid/{metric}")

            all_ = [eval(f"a.{metric}") if eval(f"a.{metric}") is not None else default_value for a in analyses]
            log_list(all_, f"Analysis/all/{metric}")

        # PERFORMANCE
        log_metric("balance")

        # MORPHOLOGY RELATED
        log_metric("morphology_size", default_value=1.)
        log_metric("morphology_muscle_ratio")

        # CPPN RELATED
        log_metric("cppn_num_nodes")
        log_metric("cppn_num_connections")
        log_metric("cppn_shared_ratio")
        log_metric("cppn_shared_pathness")
        log_metric("cppn_morph_shared_ratio")
        log_metric("cppn_morph_shared_pathness")
        log_metric("cppn_control_shared_ratio")
        log_metric("cppn_control_shared_pathness")

        # DIVERSITY

        morphologies_parents = [a.morphology_voxels for a in parent_analyses]
        morph_diversity_parents = morphological_diversity(morphologies_parents, params)
        log_list(morph_diversity_parents, "Analysis/all/parent_morph_diversity")

        add_morphology_to_global_tracker_if_novel(morphologies_parents, inter_generational_morphologies, params)

        self.log_scalar("Analysis/total_nr_of_different_parent_morphologies", len(inter_generational_morphologies),
                        self.generation)

    def log_morphological_overtakes(self, evaluation_results: List[AgentEvaluationResult],
                                    child_id_to_parent_er: Dict[int, AgentEvaluationResult], params):
        """
        Cumulative sum of the number of genomes that improved their performance while
        representing a different morphology than their parent genome.
        """
        for er in evaluation_results:
            if all(er.valid) and er.genome_id in child_id_to_parent_er:
                parent_er = child_id_to_parent_er[er.genome_id]
                if er.total_reward > parent_er.total_reward:
                    if not all(parent_er.valid) or morphological_diversity(
                            [er.analysis.morphology_voxels, parent_er.analysis.morphology_voxels], params)[0] > 0:
                        self.total_morph_overtakes += 1

        self.log_scalar('Analysis/morphological_overtakes', self.total_morph_overtakes, self.generation)

    def log_children_validity(self, children_evaluation_results: List[AgentEvaluationResult]):
        n_valid = 0
        for er in children_evaluation_results:
            if all(er.valid):
                n_valid += 1
        self.log_scalar('Analysis/children_validity', n_valid / len(children_evaluation_results), self.generation)
