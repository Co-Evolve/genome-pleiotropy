from typing import Dict

from pleiotropy.evolution.ocra import neat


def create_config(params: Dict) -> neat.config.Config:
    """
    Loads a NEAT config from the given experiment parameter file.
    """
    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                                neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                params["experiment"]["config_path"])

    return config
