import argparse
import json
import logging
import os
import pickle
import shutil
import warnings
from os.path import join
from typing import Union

import psutil

from pleiotropy.environment.visualization_utils.cmap import cmap
from pleiotropy.evolution.agent_handling.vsr_agent import AgentEvaluationResult
from pleiotropy.evolution.evolutionary_algorithms.base.base_ea import base_ea
from pleiotropy.evolution.ocra import neat
from pleiotropy.evolution.utils.input_parsing.config import create_config
from pleiotropy.evolution.utils.input_parsing.params import get_params
from pleiotropy.evolution.utils.logging.loggers import setup_loggers_and_dirs
from pleiotropy.evolution.utils.taichi.vsr_env_runner import vsr_env_runner_factory
from pleiotropy.evolution.utils.tools import pyout, poem

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tqdm import tqdm
import ray
import torch


def evolve(args: argparse.Namespace) -> None:
    PARAMS_PATH = args.params
    params = get_params(PARAMS_PATH)
    config = create_config(params)

    if params["experiment"]["debug"]:
        warnings.warn("You have launched in debug mode! Make sure to turn off debugging before "
                      "running experiments.")

    # Initialize cluster setup
    ray.init(num_cpus=psutil.cpu_count(),
             num_gpus=torch.cuda.device_count(),
             log_to_driver=False,
             logging_level=logging.WARNING if params["experiment"]["debug"] else logging.ERROR,
             local_mode=params["experiment"]["debug"])

    for _ in tqdm(range(params["experiment"]["trials"]), desc="Trial", disable=params["experiment"]["debug"]):
        # Setup loggers and output directories
        logging.info("Setting up loggers and output directories...")
        logger = setup_loggers_and_dirs(args.cluster, params, args.name)
        logging.info("\tSetting up loggers and output directories done!")

        base_ea(params=params, config=config, logger=logger)


def evaluate(args: argparse.Namespace) -> None:
    PARAMS_PATH = args.params
    params = get_params(PARAMS_PATH)
    config = create_config(params)

    import sys
    sys.modules["neat"] = neat

    ray.init(num_cpus=psutil.cpu_count())

    env_runner = vsr_env_runner_factory(params).remote(0, config=config, params=params, allow_saving=False)

    genome_files = [f for f in os.listdir(args.eval_path) if "DS_Store" not in f]

    # Filter
    genome_files = [f for f in genome_files]
    genome_files.sort(key=lambda x: float(x.split('_')[-1].split('.')[0]), reverse=True)

    for i, genome_file in enumerate(genome_files):
        with open(join(args.eval_path, genome_file), "rb") as genome_f:
            genome = pickle.load(genome_f)

        print('-' * 100)
        print(f"Evaluating {genome_file}")
        er: AgentEvaluationResult = ray.get(env_runner.eval_genome.remote(0, (0, genome)))
        # er: AgentEvaluationResult = env_runner.eval_genome(0, (0, genome))
        print(f'Reward: {er.total_reward}')
        print("-" * 100)


def visualize(args: argparse.Namespace) -> None:
    PARAMS_PATH = args.params
    params = get_params(PARAMS_PATH)
    config = create_config(params)

    import sys
    sys.modules["neat"] = neat

    ray.init(local_mode=True, num_cpus=1)
    env_runner = vsr_env_runner_factory(params).remote(0, config=config, params=params, allow_saving=False)

    if os.path.isdir(args.eval_path):
        genome_folders = [f for f in os.listdir(args.eval_path) if "DS_Store" not in f]
        genome_folders = sorted(genome_folders, key=lambda x: -float(x.split('_')[-1].split('.')[0]))

        for ii, genome_file in tqdm(enumerate(genome_folders), desc=poem("Visualizing"),
                                    leave=False, total=len(genome_folders)):

            path = join(args.eval_path, genome_file)
            with open(path, "rb") as genome_f:
                genome = pickle.load(genome_f)

            er: Union[None, AgentEvaluationResult] = ray.get(
                env_runner.visualize_genome.remote((0, genome), path, color_t0=cmap(50)))
            if er is not None:
                tqdm.write(f"file: {genome_file} || reward: {er.total_reward} || balance: {er.analysis.balance}")
    else:
        os.makedirs(params['evaluation']['visualization']['save_dir'], exist_ok=True)

        assert args.eval_path.endswith('.json')
        with open(args.eval_path, 'r') as f:
            D = json.load(f)
        genome_folders = {}
        for run in D:
            for gen in D[run]:
                genome_folders[f"{run}_gen_{gen}_reward_{D[run][gen]['reward']}"] = D[run][gen]['path']

        for ou_name, genome_file in tqdm(genome_folders.items(), desc=poem("Visualizing progression"), leave=False):
            with open(genome_file, "rb") as genome_f:
                genome = pickle.load(genome_f)

            if os.path.isdir(f"{params['evaluation']['visualization']['save_dir']}/{ou_name}"):
                continue

            r = float(ou_name.split('_')[-1]) / params['evaluation']['visualization']['fitness_range'][1]
            er: Union[None, AgentEvaluationResult] = ray.get(
                env_runner.visualize_genome.remote((0, genome), genome_file, color_t0=cmap(r)))

            pyout(params['evaluation']['visualization']['save_dir'])
            pyout(genome_file.split('/')[-1])
            pyout(ou_name)
            pyout(er)

            in_path = f"{params['evaluation']['visualization']['save_dir']}/{genome_file.split('/')[-1]}"
            ou_path = f"{params['evaluation']['visualization']['save_dir']}/{ou_name}"
            shutil.rmtree(ou_path, ignore_errors=True)
            os.rename(in_path, ou_path)

            abspth = os.path.abspath(ou_path)

            os.system(f"gio set {abspth} metadata::custom-icon file:///{abspth}/still.jpg")
