# SPDX-FileCopyrightText: 2026 Franz Rohrhofer <frohrhofer@know-center.at>
# SPDX-License-Identifier: MIT
"""Trains and evaluates one scenario.

Scenario k corresponds to Step k in the paper. Results (weights, log and
figures) are written to results/<version>/, with <version> from the config.

Usage:
    python main.py -c config/scenario_3_physics.yaml          # train and evaluate
    python main.py -c config/scenario_3_physics.yaml --load   # evaluate saved weights only
"""

import argparse
from time import time

import matplotlib
matplotlib.use('Agg')       # figures are only saved, not shown

from src.config import load_config
from src.data_loader import DataLoader
from src.model import NeuralNet
from src.pinn import PINN
from src.plotter import Plotter
from src.utils import set_seed, get_device


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='config/scenario_3_physics.yaml',
                        help="Path to scenario configuration file")
    parser.add_argument('-t', '--threads', type=int, default=None,
                        help="Number of CPU threads (default: half the cores, at most 8)")
    parser.add_argument('--load', action='store_true',
                        help="Load the trained weights from results/<version>/ instead of training")
    return parser.parse_args()


def train_and_test(config, n_threads=None, load=False):

    configs = config['configs']
    set_seed(config['seed'])
    device = get_device(n_threads)

    # Initializing DataLoader instance
    data = DataLoader(load_config(configs['problem']),
                      load_config(configs['thermocouples']),
                      file=configs['reference'], seed=config['seed'])
    print(data.problem.summary(), end='\n\n')

    # Initializing network and PINN instance
    hard_ic = config.get('constraints', {}).get('hard_ic', False)
    model = NeuralNet(load_config(configs['network']), data.problem, hard_ic=hard_ic)
    pinn = PINN(config, model, data, device)

    # Training (or loading), evaluation and saving
    if load:
        pinn.load()
        metrics = pinn.evaluate()
    else:
        pinn.train()
        metrics = pinn.evaluate()
        pinn.save()

    # Figures
    Plotter(pinn).all()
    return metrics


if __name__ == '__main__':

    args = parse_arguments()
    time_start = time()

    config = load_config(args.config)
    train_and_test(config, args.threads, args.load)

    print(f"Finished in {time() - time_start:.0f} seconds!")
