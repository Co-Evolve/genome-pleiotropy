import os
import random
import subprocess
import traceback

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

bcolors = {'PINK': '\033[95m',
           'BLUE': '\033[94m',
           'CYAN': '\033[96m',
           'GREEN': '\033[92m',
           'YELLOW': '\033[93m',
           'RED': '\033[91m', }


def bash(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error


def set_seed(seed):
    """
    Set rng seed for all sources of randomness

    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)


def pretty_string(message: str, color=None, bold=False, underline=False):
    """
    add color and effects to string
    :param message:
    :param color:
    :param bold:
    :param underline:
    :return:
    """
    ou = message
    if color:
        ou = bcolors[color] + message + '\033[0m'
    if bold:
        ou = '\033[1m' + ou + '\033[0m'
    if underline:
        ou = '\033[4m' + ou + '\033[0m'
    return ou


def poem(string):
    if len(string) > 20:
        return string[:20] + '...'
    else:
        return string + ' ' * (23 - len(string))


def pyout(*message):
    """
    Print message preceded by traceback. I use this method to prevent rogue "print" statements
    during debugging
    :param message:
    :return:
    """

    message = ' '.join(str(m) for m in message)

    trace = traceback.extract_stack()[-2]

    fname = trace.filename.replace(os.path.abspath(os.curdir), "...")

    trace = f"{fname}: {trace.name}(...) - ln{trace.lineno}"

    tqdm.write(pretty_string(trace, 'PINK', bold=True))
    if message is not None:
        tqdm.write(message)


pseudo_random_state = 49


def prng(decimals=4):
    global pseudo_random_state

    ou = 0
    for ii in range(1, decimals + 1):
        pseudo_random_state = (7 * pseudo_random_state) % 101

        ou += (pseudo_random_state % 10) * 10 ** -ii
    ou = str(ou)[:decimals + 2]

    return float(ou)


def plot(A: np.ndarray):
    fig = plt.figure()
    fig.patch.set_facecolor('xkcd:navy')

    plt.imshow(A, interpolation='nearest', cmap='magma')
    plt.show()
