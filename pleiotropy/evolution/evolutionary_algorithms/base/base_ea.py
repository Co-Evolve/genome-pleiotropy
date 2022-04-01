from typing import Dict, List

from tqdm import tqdm

from pleiotropy.evolution.agent_handling.vsr_agent import AgentEvaluationResult
from pleiotropy.evolution.evolutionary_algorithms.base.evolution_handler import BaseEvolutionHandler
from pleiotropy.evolution.ocra.neat.config import Config
from pleiotropy.evolution.utils.logging.tensorboard_logger import TBLogger
from pleiotropy.evolution.utils.taichi.vsr_env_runner import get_worker_pool


def base_ea(params: Dict, config: Config, logger: TBLogger) -> None:
    """
    Basic evolutionary algorithm.
    """

    # Setup evolution
    pool = get_worker_pool(params=params, config=config)
    evolution_handler = BaseEvolutionHandler(config=config, params=params, tb_logger=logger)

    # Start evolution
    for _ in tqdm(range(params["evolution"]["base_ea"]["generations"]), desc=f"Generations", leave=False):
        generation = evolution_handler.generation
        population = evolution_handler.population_children

        evaluation_results: List[AgentEvaluationResult] = \
            list(pool.map_unordered(
                lambda env_runner, genome: env_runner.eval_genome.remote(generation, genome),
                population
            ))

        evolution_handler.generation_step(evaluation_results)
