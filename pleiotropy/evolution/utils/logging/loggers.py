import json
import logging
import random
from pathlib import Path
from shutil import copy2
from typing import Dict

import numpy as np
import torch
import wandb

from pleiotropy.evolution.utils.logging.tensorboard_logger import TBLogger

wandb_run = None


def setup_loggers_and_dirs(cluster: bool, params: Dict, experiment_name: str) -> TBLogger:
    """
    Sets up the tensorboard logging and logging / result directories.
    Also copies the given experiment parameters and NEAT config to the logging directory.
    """
    global wandb_run

    seed = random.randrange(1000000)
    random.seed(seed)
    params["experiment"]["seed"] = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup logger and environment.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    results_path = Path(params["experiment"]["results_dir"]) / experiment_name
    results_path.mkdir(exist_ok=True, parents=True)

    if wandb_run is not None:
        wandb_run.finish()

    if cluster:
        wandb_run = wandb.init(project="pleiotropy", entity="co-evolve", group=experiment_name, config=params)
    else:
        wandb_run = wandb.init(project="pleiotropy", entity="co-evolve", group=experiment_name, config=params,
                               dir=str(results_path))

    run_results_path = Path(params["experiment"]["results_dir"]) / experiment_name / wandb_run.name
    run_backup_results_path = Path(params["experiment"]["backup_results_dir"]) / experiment_name / wandb_run.name
    params["experiment"]["current_results_dir"] = str(run_results_path)
    params["experiment"]["backup_current_results_dir"] = str(run_backup_results_path)

    Path(params["experiment"]["current_results_dir"]).mkdir(exist_ok=True, parents=True)
    Path(params["experiment"]["backup_current_results_dir"]).mkdir(exist_ok=True, parents=True)

    print(f'Results will be saved to {params["experiment"]["current_results_dir"]}')

    tb_logger = TBLogger(results_path)

    # Save params to log dir
    config_path = results_path / "configs"
    if not config_path.exists():
        config_path.mkdir(exist_ok=True, parents=True)
        with open(str(config_path / "settings.json"), "w") as f:
            json.dump(params, f)
        copy2(params["experiment"]["config_path"], str(config_path / "neat_config"))

    return tb_logger
