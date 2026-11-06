# SPDX-FileCopyrightText: 2026 Franz Rohrhofer <frohrhofer@know-center.at>
# SPDX-License-Identifier: MIT
"""Generates the finite-difference reference solution used by all scenarios.

Usage:
    python generate_reference_data.py            # solve and save
    python generate_reference_data.py --verify   # additionally run a grid-convergence study
"""

import argparse

from src.config import load_config
from src.physics import WeldingProblem
from src.reference_solver import ReferenceSolver


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', type=str,
                        default='config/problem.yaml',
                        help="Path to problem configuration file")
    parser.add_argument('-s', '--solver', type=str,
                        default='config/reference_solver.yaml',
                        help="Path to reference solver configuration file")
    parser.add_argument('--verify', action='store_true',
                        help="Run a grid-convergence study")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    # Load configuration files
    problem_config = load_config(args.problem)
    solver_config = load_config(args.solver)

    problem = WeldingProblem.from_config(problem_config)
    print(problem.summary(), end='\n\n')

    # Solve and save reference solution
    solver = ReferenceSolver(problem, solver_config)
    solution = solver.solve()
    file = solver.save(solution, problem_config)
    print(f"Peak temperature: {solution.T.max():.0f} K (melting temperature {problem.T_melt:.0f} K)")
    print(f"Saved reference solution to {file} ({file.stat().st_size / 1e6:.1f} MB)")

    # Optional grid-convergence study
    if args.verify:
        print("\nGrid-convergence study:")
        rows = solver.grid_convergence()
        print(" | ".join(f"{key:>22}" for key in rows[0]))
        for row in rows:
            print(" | ".join(f"{val:>22.4g}" for val in row.values()))
