import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

import numpy as np

from pleiotropy.evolution.agent_handling.vsr_analysis import VSRAnalysis
from pleiotropy.evolution.agent_handling.vsr_controller import VSRController
from pleiotropy.evolution.agent_handling.vsr_morphology import VSRMorphology
from pleiotropy.evolution.ocra.cppn.vsr_cppn import VSRCPPN
from pleiotropy.evolution.ocra.neat.config import Config
from pleiotropy.evolution.ocra.neat.genome import GenomeType


class AgentEvaluationResult(object):
    """
    Helper class to hold agent evaluation results.
    """

    def __init__(self, genome_id: int, genome: GenomeType,
                 total_reward: float,
                 valid: Tuple[bool, bool, bool], analysis: VSRAnalysis) -> None:
        self.genome_id = genome_id
        self.genome = genome
        self.total_reward = total_reward
        self.valid = valid
        self.analysis: VSRAnalysis = analysis
        self.is_parent = False


class VSRAgent:
    """
    Class that encapsulates everything an agent needs to be created and has to be able to do during its life.
    """

    def __init__(self, genome: Tuple[int, GenomeType],
                 config: Config, params: Dict, genome_path: str = None, generation: int = 0,
                 morphology: Optional[VSRMorphology] = None, skip_controller: bool = False) -> None:
        self.config = config
        self.params = params
        self.genome_path = genome_path
        self.generation = generation
        self.skip_controller = skip_controller
        self.genome_pleiotropy = params["pleiotropy"]["genome_pleiotropy"]
        self.genome_id, self.genome = genome

        try:
            self.parent_id = self.genome.parent_key
        except AttributeError:
            self.parent_id = self.genome[0].parent_key

        self.cppn = VSRCPPN.create_cppn(genome=self.genome, config=self.config, params=params)

        if morphology is not None:
            self.morphology = morphology
        else:
            self.morphology = VSRMorphology(genome_id=self.genome_id, params=params, cppn=self.cppn)
        self.controller = VSRController(genome_id=self.genome_id, params=params, cppn=self.cppn)

        self.analysis = VSRAnalysis(genome_id=self.genome_id, params=params, config=config)
        self.analysis.analyze_cppn(self.cppn)

        # Evaluation
        self.reward = 0

        # Genotype-Phenotype mapping
        self.valid = self.create_phenotype()

    def create_phenotype(self) -> Tuple[bool, bool, bool]:
        """
        Build agent morphology and it's brain.
        :return: tuple of bools indicating respectively if the created genome, morphology and brain are valid
        """
        valid_genome = True

        if not valid_genome:
            return False, False, False

        valid_morph = self.morphology.build()
        if not valid_morph:
            return valid_genome, False, False
        self.analysis.analyze_morphology(self.morphology)

        if self.skip_controller:
            valid_brain = True
        else:
            valid_brain = self.controller.build(self.morphology)

        return valid_genome, valid_morph, valid_brain

    def save(self, generation: int, forced: bool = False) -> None:
        solved = self.reward >= self.params["fitness"]["solved"]

        if forced or solved:
            base_path = Path(self.params["experiment"]["current_results_dir"])

            if forced:
                base_path = base_path / "forced"
            elif solved:
                base_path = base_path / "solved"

            def write(write_path):
                name = f"gen_{generation}_genome_{self.genome_id}_parent_{self.parent_id}_reward_{round(self.reward, 4)}.pkl"
                output_path = write_path / name
                write_path.mkdir(exist_ok=True, parents=True)

                # Save genome
                if self.genome:
                    with open(str(output_path), "wb") as output:
                        pickle.dump(self.genome, output, pickle.HIGHEST_PROTOCOL)

            try:
                write(base_path)
            except OSError as ex:
                print("#" * 100)
                print(f"AGENT COULD NOT BE SAVED (saving to backup instead): {ex}")
                write(Path(self.params["experiment"]["backup_current_results_dir"]))
                print("#" * 100)

    def reset(self) -> None:
        self.reward = 0
        self.morphology.reset()
        self.controller.reset()

    def finish_evaluation(self, generation: int,
                          evaluation_results: List[Any],
                          allow_saving: bool) -> AgentEvaluationResult:
        self.reward = float(np.mean(evaluation_results))
        if allow_saving:
            self.save(generation)

        return AgentEvaluationResult(genome_id=self.genome_id,
                                     genome=self.genome,
                                     total_reward=self.reward,
                                     valid=self.valid,
                                     analysis=self.analysis)
