# SPDX-FileCopyrightText: 2026 Franz Rohrhofer <frohrhofer@know-center.at>
# SPDX-License-Identifier: MIT
"""Finite-difference reference ("ground-truth") solver for the welding problem.

Scheme
------
* uniform grid x_i = i dx, y_j = j dy whose nodes lie on the plate edges
* explicit Euler in time + 2nd-order central differences in space (FTCS)
* convective edges -k dT/dn = h (T - T_inf) imposed with ghost nodes
  (h = 0 gives adiabatic edges)
* stability condition: alpha dt (1/dx^2 + 1/dy^2) <= 1/2

The scheme is deliberately simple, so that every line can be checked by hand.
Its accuracy is verified by a global energy balance and a grid-convergence
study.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.physics import WeldingProblem


@dataclass
class Solution:
    """Temperature field T[n, j, i] = T(t_n, y_j, x_i) and solver diagnostics."""

    x: np.ndarray               # (nx,) node coordinates [m]
    y: np.ndarray               # (ny,) node coordinates [m]
    t: np.ndarray               # (nt,) snapshot times [s]
    T: np.ndarray               # (nt, ny, nx) temperature [K]
    dt: float                   # internal time step [s]
    fourier: float              # alpha dt (1/dx^2 + 1/dy^2)
    energy: dict = field(default_factory=dict)  # energy-balance time series [J]
    runtime: float = 0.0        # wall-clock time [s]

    @property
    def energy_balance_error(self) -> float:
        """Relative mismatch |stored - (input - loss)| / input at the final time."""
        e = self.energy
        return abs(e['stored'][-1] - (e['input'][-1] - e['loss'][-1])) / e['input'][-1]

    def coarsen(
            self,
            spacing: float,
            time_step: float,
        ) -> 'Solution':
        """Returns the solution on a coarser grid (subset of nodes and snapshots).

        Args:
            spacing (float): Grid spacing of the coarse grid [m].
            time_step (float): Time between coarse snapshots [s].

        Returns:
            solution (Solution): Coarsened solution.
        """
        sx = _integer_ratio(spacing, self.x[1] - self.x[0], 'output spacing')
        st = _integer_ratio(time_step, self.t[1] - self.t[0], 'output time step')
        return Solution(x=self.x[::sx], y=self.y[::sx], t=self.t[::st],
                        T=self.T[::st, ::sx, ::sx], dt=self.dt, fourier=self.fourier,
                        energy=self.energy, runtime=self.runtime)


class ReferenceSolver:
    """FTCS solver configured by config/reference_solver.yaml."""

    def __init__(
            self,
            problem: WeldingProblem,
            config: dict,
        ) -> None:
        """Initialize solver with the physical problem and solver settings.

        Args:
            problem (WeldingProblem): Physical problem.
            config (dict): Solver configuration.
        """
        self.problem = problem
        self.config = config
        self.spacing = config['grid']['spacing']
        self.fourier_target = config['time']['fourier_number']
        self.output_spacing = config['output']['spacing']
        self.output_time_step = config['output']['time_step']
        self.file = Path(config['output']['file'])

        if self.fourier_target > 0.5:
            raise ValueError(f"Fourier number {self.fourier_target} > 0.5 is unstable")

    def solve(
            self,
            spacing: float | None = None,
            verbose: bool = True,
        ) -> Solution:
        """Integrates the heat equation from t = 0 to t_end.

        Snapshots are stored every output time step on the full solver grid.

        Args:
            spacing (float, None): Grid spacing [m], default: from config.
            verbose (bool): Whether to print a summary.

        Returns:
            solution (Solution): Temperature field and diagnostics.
        """
        tic = time.perf_counter()
        prob = self.problem
        spacing = self.spacing if spacing is None else spacing

        # ---- grid ----
        nx = _integer_ratio(prob.L, spacing, 'plate length') + 1
        ny = _integer_ratio(prob.W, spacing, 'plate width') + 1
        x = np.linspace(0.0, prob.L, nx)
        y = np.linspace(0.0, prob.W, ny)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y)                       # (ny, nx)

        # ---- time step: below stability limit and dividing the output step ----
        alpha, k, h = prob.alpha, prob.k, prob.h_eff
        rho_cp = prob.rho * prob.cp
        dt_max = self.fourier_target / (alpha * (1 / dx**2 + 1 / dy**2))
        steps_per_output = int(np.ceil(self.output_time_step / dt_max))
        dt = self.output_time_step / steps_per_output
        fourier = alpha * dt * (1 / dx**2 + 1 / dy**2)
        n_output = _integer_ratio(prob.t_end, self.output_time_step, 'end time')

        # ---- weights for the energy balance ----
        # control-volume area of each node (half cells at edges, quarter at corners)
        wx = np.full(nx, dx); wx[[0, -1]] /= 2
        wy = np.full(ny, dy); wy[[0, -1]] /= 2
        area = wy[:, None] * wx[None, :]
        # boundary length belonging to each edge node
        edge = np.zeros((ny, nx))
        edge[:, [0, -1]] += wy[:, None]
        edge[[0, -1], :] += wx[None, :]

        # ---- time integration ----
        T = np.full((ny, nx), prob.T0)
        Tg = np.empty((ny + 2, nx + 2))                # T with ghost nodes
        snapshots = [T.copy()]
        E_in = E_loss = 0.0
        energy = {'stored': [0.0], 'input': [0.0], 'loss': [0.0]}

        step = 0
        for _ in range(n_output):
            for _ in range(steps_per_output):
                t = step * dt
                # ghost nodes from  -k dT/dn = h (T - T_inf)
                Tg[1:-1, 1:-1] = T
                Tg[1:-1, 0] = T[:, 1] - 2 * dx * h / k * (T[:, 0] - prob.T_inf)     # x = 0
                Tg[1:-1, -1] = T[:, -2] - 2 * dx * h / k * (T[:, -1] - prob.T_inf)  # x = L
                Tg[0, 1:-1] = T[1, :] - 2 * dy * h / k * (T[0, :] - prob.T_inf)     # y = 0
                Tg[-1, 1:-1] = T[-2, :] - 2 * dy * h / k * (T[-1, :] - prob.T_inf)  # y = W

                laplace = ((Tg[1:-1, 2:] - 2 * T + Tg[1:-1, :-2]) / dx**2
                           + (Tg[2:, 1:-1] - 2 * T + Tg[:-2, 1:-1]) / dy**2)
                Q = prob.heat_source(X, Y, t)

                E_in += dt * prob.d * np.sum(area * Q)
                E_loss += dt * prob.d * h * np.sum(edge * (T - prob.T_inf))

                T = T + dt * (alpha * laplace + Q / rho_cp)
                step += 1

            snapshots.append(T.copy())
            energy['stored'].append(rho_cp * prob.d * np.sum(area * (T - prob.T0)))
            energy['input'].append(E_in)
            energy['loss'].append(E_loss)

        solution = Solution(
            x=x, y=y,
            t=np.arange(n_output + 1) * self.output_time_step,
            T=np.asarray(snapshots), dt=dt, fourier=fourier,
            energy={key: np.asarray(val) for key, val in energy.items()},
            runtime=time.perf_counter() - tic,
        )
        if verbose:
            print(f"FD solver: grid {nx} x {ny} (dx = {dx*1e3:.3f} mm), "
                  f"dt = {dt*1e3:.3f} ms, {step} steps, Fourier number = {fourier:.3f}, "
                  f"energy balance error = {solution.energy_balance_error:.1e}, "
                  f"runtime {solution.runtime:.1f} s")
        return solution

    def save(
            self,
            solution: Solution,
            problem_config: dict,
            file: str | Path | None = None,
        ) -> Path:
        """Stores the solution on the output grid as compressed .npz file.

        The problem and solver configurations are stored alongside, so that
        the data can later be checked against the configuration in use.

        Args:
            solution (Solution): Solution on the solver grid.
            problem_config (dict): Content of config/problem.yaml.
            file (str | Path, None): Output file, default: from config.

        Returns:
            file (Path): Path of the written file.
        """
        file = Path(self.file if file is None else file)
        file.parent.mkdir(parents=True, exist_ok=True)
        out = solution.coarsen(self.output_spacing, self.output_time_step)
        np.savez_compressed(
            file,
            x=out.x, y=out.y, t=out.t, T=out.T.astype(np.float32),
            dt=out.dt, fourier=out.fourier,
            **{f'energy_{key}': val for key, val in out.energy.items()},
            problem_config=json.dumps(problem_config),
            solver_config=json.dumps(self.config),
        )
        return file

    def grid_convergence(
            self,
            spacings: list[float] | None = None,
        ) -> list[dict]:
        """Solves on successively refined grids and compares with the finest one.

        All solutions are compared on the nodes of the coarsest grid at all
        snapshot times.

        Args:
            spacings (list[float], None): Grid spacings [m], default: from config.

        Returns:
            rows (list[dict]): One row of diagnostics per grid.
        """
        spacings = sorted(spacings or self.config['verification']['spacings'], reverse=True)
        solutions = [self.solve(s) for s in spacings]
        coarsest, finest = spacings[0], solutions[-1].coarsen(spacings[0], self.output_time_step)
        rows = []
        for s, sol in zip(spacings, solutions):
            diff = np.abs(sol.coarsen(coarsest, self.output_time_step).T - finest.T)
            rows.append({
                'dx [mm]': s * 1e3,
                'dt [ms]': sol.dt * 1e3,
                'Fourier': sol.fourier,
                'T_max [K]': sol.T.max(),
                'max |T - T_finest| [K]': diff.max(),
                'energy error': sol.energy_balance_error,
                'runtime [s]': sol.runtime,
            })
        return rows


def _integer_ratio(
        a: float,
        b: float,
        name: str,
    ) -> int:
    """Returns a / b as integer, raises an error if it is not one.

    Args:
        a (float): Numerator.
        b (float): Denominator.
        name (str): Quantity name used in the error message.

    Returns:
        n (int): Integer ratio.
    """
    n = int(round(a / b))
    if n < 1 or abs(n * b - a) > 1e-9 * max(abs(a), abs(b)):
        raise ValueError(f"{name} ({a:g}) is not an integer multiple of {b:g}")
    return n
