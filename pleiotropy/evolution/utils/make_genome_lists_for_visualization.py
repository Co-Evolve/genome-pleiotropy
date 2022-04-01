import json
import os
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

from pleiotropy.evolution.utils.tools import pyout, bash, poem


def progression(folders):
    max_R = 0.

    for root in folders:

        # Unpack tars
        tars = [f for f in os.listdir(root) if f.endswith('.tar')]
        for tar in tqdm(tars, desc=poem(f"Unpacking {root.name}"), leave=False):
            if not os.path.exists(f"{root / tar.replace('.tar', '')}"):
                bash(f"tar -xvf {root / tar} -C {root}")

        json_ou = {}

        runs = [f for f in os.listdir(root) if os.path.isdir(root / f)]
        for run in tqdm(runs, desc=poem(f"Scanning {root.name}"), leave=False):
            solved = f"{root}/{run}/solved_{run.replace('run_', '')}"
            if not os.path.isdir(solved):
                solved = f"{root}/{run}/solved"

            genomes = {}
            last_gen = []
            for f in os.listdir(solved):
                _, gen, _, genome, _, parent, _, reward = tuple(f.split('.')[0].split('_'))
                gen, genome, parent, reward = int(gen), int(genome), \
                                              int(parent.replace("None", "-1")), float(reward)
                genomes[int(genome)] = {'gen': gen, 'genome': genome, 'parent': parent,
                                        'reward': reward,
                                        'path': os.path.abspath(f"{solved}/{f}")}

                if gen == 999:
                    last_gen.append(genome)
                    last_gen.append(parent)

            last_gen = [x for x in last_gen if x in genomes]

            checkpoints = {1: None, 10: None, 50: None, 100: None, 500: None, 1000: None}

            idx = sorted(last_gen, key=lambda x: genomes[x]['reward'], reverse=True)[0]

            g = genomes[idx]

            checkpoints[1000] = g
            max_R = max(max_R, g['reward'])
            for gen_ii in range(998, -1, -1):
                if (gen_ii + 1) in checkpoints:
                    checkpoints[gen_ii + 1] = g

                if gen_ii == g['gen']:
                    if g['parent'] >= 0:
                        g = genomes[g['parent']]
                    else:
                        break

            json_ou[run] = checkpoints

        with open(f"{root}/progression.json", 'w+') as f:
            json.dump(json_ou, f, indent=2)

    pyout(f"Max reward: {max_R:.2f}")


def gen_0(folders):
    for root in folders:

        # Unpack tars
        tars = [f for f in os.listdir(root) if f.endswith('.tar')]
        for tar in tqdm(tars, desc=poem(f"Unpacking {root.name}"), leave=False):
            if not os.path.exists(f"{root / tar.replace('.tar', '')}"):
                bash(f"tar -xvf {root / tar} -C {root}")

        json_ou = {}

        runs = [f for f in os.listdir(root) if os.path.isdir(root / f)]
        for run in tqdm(runs, desc=poem(f"Scanning {root.name}"), leave=False):
            solved = f"{root}/{run}/solved_{run.replace('run_', '')}"
            if not os.path.isdir(solved):
                solved = f"{root}/{run}/solved"

            genomes = {}
            first_gen = []
            for f in os.listdir(solved):
                _, gen, _, genome, _, parent, _, reward = tuple(f.split('.')[0].split('_'))
                gen, genome, parent, reward = int(gen), int(genome), int(
                    parent.replace("None", "-1")), float(reward)
                genomes[int(genome)] = {'gen': gen, 'genome': genome, 'parent': parent,
                                        'reward': reward,
                                        'path': os.path.abspath(f"{solved}/{f}")}

                if gen == 0:
                    first_gen.append(genome)

            g = {gen: genomes[gen] for gen in first_gen}
            json_ou[run] = g

        with open(f"{root}/gen_0.json", 'w+') as f:
            json.dump(json_ou, f, indent=2)


if __name__ == '__main__':
    F: Tuple[Path, ...] = (Path("../res/run_example"),)

    progression(F)
    gen_0(F)
