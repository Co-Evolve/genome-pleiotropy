"""Implements the core evolution algorithm."""
from __future__ import print_function

import copy
from itertools import count
from pathlib import Path
from random import sample
from typing import List, Dict

import numpy as np

from pleiotropy.evolution.agent_handling.vsr_agent import AgentEvaluationResult
from pleiotropy.evolution.ocra.cppn.vsr_cppn import VSRCPPN
from pleiotropy.evolution.ocra.neat import DefaultGenome
from pleiotropy.evolution.ocra.neat.config import Config
from pleiotropy.evolution.utils.logging.tensorboard_logger import TBLogger


class BaseEvolutionHandler(object):
    def __init__(self, config: Config, params: Dict, tb_logger: TBLogger) -> None:
        self.config = config
        self.params = params
        self.tb_logger = tb_logger

        self.elites_path = Path(self.params["experiment"]["current_results_dir"]) / 'elites'

        # NEAT setup
        self.generation = 0
        self.num_evals_done = 0
        stagnation = config.stagnation_type(config.stagnation_config, None)
        reproduction = config.reproduction_type(config.reproduction_config,
                                                None,
                                                stagnation)

        self.mu, self.la = self.params["evolution"]["base_ea"]["mu"], self.params["evolution"]["base_ea"]["lambda"]
        self.pop_size = self.mu + self.la

        # Create a population from scratch
        self.genome_pleiotropy = self.params["pleiotropy"]["genome_pleiotropy"]
        if self.genome_pleiotropy == 'pleiotropic':
            population = reproduction.create_new(config.genome_type,
                                                 config.genome_config,
                                                 self.pop_size)
            self.population_children = [(key, genome) for key, genome in population.items()]

            for key, genome in self.population_children:
                genome.key = key
                genome.output_labels = ['p', 'm', 'o', 'f']

        elif self.genome_pleiotropy == 'disconnected':
            morph_population = reproduction.create_new(config.genome_type,
                                                       config.genome_config,
                                                       self.pop_size)

            controller_population = reproduction.create_new(config.genome_type,
                                                            config.genome_config,
                                                            self.pop_size)
            self.population_children = [
                (key, (morph_genome, controller_genome)) for key, (morph_genome, controller_genome) in
                enumerate(zip(morph_population.values(), controller_population.values()))
            ]

            for key, (morph_genome, controller_genome) in self.population_children:
                morph_genome.key = key
                morph_genome.output_labels = ['p', 'm']

                controller_genome.key = key
                controller_genome.output_labels = ['o', 'f']

        elif self.genome_pleiotropy == 'fully disconnected':
            morph_presence_population = reproduction.create_new(config.genome_type,
                                                                config.genome_config,
                                                                self.pop_size)
            morph_muscle_population = reproduction.create_new(config.genome_type,
                                                              config.genome_config,
                                                              self.pop_size)
            controller_offset_population = reproduction.create_new(config.genome_type,
                                                                   config.genome_config,
                                                                   self.pop_size)
            controller_frequency_population = reproduction.create_new(config.genome_type,
                                                                      config.genome_config,
                                                                      self.pop_size)
            self.population_children = [
                (key, (morph_pres, morph_muscle, control_offset, control_freq)) for
                key, (morph_pres, morph_muscle, control_offset, control_freq) in
                enumerate(zip(morph_presence_population.values(), morph_muscle_population.values(),
                              controller_offset_population.values(), controller_frequency_population.values()))
            ]

            for key, (morph_pres, morph_muscle, control_offset, control_freq) in self.population_children:
                morph_pres.key = key
                morph_pres.output_labels = ['p']

                morph_muscle.key = key
                morph_muscle.output_labels = ['m']

                control_offset.key = key
                control_offset.output_labels = ['o']

                control_freq.key = key
                control_freq.output_labels = ['f']
        else:
            raise NotImplementedError(f'Given genome_pleiotropy not known: {self.genome_pleiotropy}')

        self.genome_indexer = count(len(self.population_children))
        self.child_id_to_parent_er: Dict[int: AgentEvaluationResult] = dict()
        self.parent_evaluation_results = []

        self.all_morphologies_found = set()

    @staticmethod
    def mutate_genome_until_valid(generation, config, params, genome: DefaultGenome):
        genome_copy = None
        for _ in range(100):
            genome_copy = copy.deepcopy(genome)
            genome_copy.mutate(config.genome_config)
            if VSRCPPN.create_cppn(genome_copy, config, params).is_valid(generation):
                break

        return genome_copy

    @staticmethod
    def create_new_child_from_parent(config, params, parent_genome, child_id,
                                     generation=0):
        genome_pleiotropy = params["pleiotropy"]["genome_pleiotropy"]
        if genome_pleiotropy == 'pleiotropic':
            child_genome = BaseEvolutionHandler.mutate_genome_until_valid(generation, config, params, parent_genome)
            child_genome.is_elite = False
            child_genome.key = child_id
            child_genome.parent_key = parent_genome.key
        else:
            if len(parent_genome) == 4:
                num_to_mutate = np.random.choice([1, 2, 3, 4], p=[8 / 15, 4 / 15, 2 / 15, 1 / 15])
                sub_genome_indices = sample([0, 1, 2, 3], num_to_mutate)
            elif len(parent_genome) == 2:
                num_to_mutate = np.random.choice([1, 2], p=[2 / 3, 1 / 3])
                sub_genome_indices = sample([0, 1], num_to_mutate)
            else:
                raise NotImplementedError('Create new child from parent failed')

            child_sub_genomes = []
            for i in range(len(parent_genome)):
                sub_genome = parent_genome[i]
                if i in sub_genome_indices:
                    child_sub_genome = BaseEvolutionHandler.mutate_genome_until_valid(generation, config, params,
                                                                                      sub_genome)
                else:
                    child_sub_genome = copy.deepcopy(sub_genome)

                child_sub_genome.key = child_id
                child_sub_genome.is_elite = False
                child_sub_genome.parent_key = sub_genome.key
                child_sub_genome.output_labels = sub_genome.output_labels
                child_sub_genomes.append(child_sub_genome)

            child_genome = tuple(child_sub_genomes)

        return child_genome

    def generation_step(self, children_evaluation_results: List[AgentEvaluationResult]) -> None:
        self.num_evals_done += len(children_evaluation_results)

        population_sorted = self.get_sorted_population(children_evaluation_results)

        # Gather and report statistics.
        self.tb_logger.generation = self.generation
        self.tb_logger.log_evaluation_results(population_sorted, self.num_evals_done)
        self.tb_logger.log_analysis(population_sorted, self.all_morphologies_found, self.params)
        self.tb_logger.log_morphological_overtakes(children_evaluation_results, self.child_id_to_parent_er, self.params)
        self.child_id_to_parent_er = dict()
        self.population_children = []

        # Create the next generation from the current generation.
        #   Get mu parents -> best ones in population
        self.parent_evaluation_results = population_sorted[:self.mu]
        for er in self.parent_evaluation_results:
            er.is_parent = True

        #   Create lambda children from parents
        self.population_children = []
        for parent_er in self.parent_evaluation_results[:self.la]:
            child_id = next(self.genome_indexer)

            child_genome = self.create_new_child_from_parent(config=self.config, params=self.params,
                                                             parent_genome=parent_er.genome,
                                                             child_id=child_id, generation=self.generation)

            self.child_id_to_parent_er[child_id] = parent_er
            self.population_children.append((child_id, child_genome))

        self.generation += 1

    def get_sorted_population(self, evaluation_results: List[AgentEvaluationResult]) -> List[AgentEvaluationResult]:
        evaluation_results = self.parent_evaluation_results + evaluation_results
        return sorted(evaluation_results, key=lambda er: -er.total_reward)
