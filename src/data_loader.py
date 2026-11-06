# SPDX-FileCopyrightText: 2026 Franz Rohrhofer <frohrhofer@know-center.at>
# SPDX-License-Identifier: MIT
"""Access to the reference solution and the synthetic thermocouple data.

All point data is returned as NumPy arrays in SI units with the layout

    X : (N, 3)  columns (x [m], y [m], t [s])
    T : (N, 1)  temperature [K]
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from src.physics import WeldingProblem


class DataLoader:
    """Loads the reference solution and extracts thermocouple measurements."""

    def __init__(
            self,
            problem_config: dict,
            thermocouple_config: dict,
            file: str | Path = 'data/reference_solution.npz',
            seed: int = 0,
        ) -> None:
        """Initialize DataLoader with problem and thermocouple configuration.

        Loads the reference solution, checks it against the configuration and
        creates the (noisy) thermocouple signals.

        Args:
            problem_config (dict): Content of config/problem.yaml.
            thermocouple_config (dict): Content of config/thermocouples.yaml.
            file (str | Path): Reference solution created by
                generate_reference_data.py.
            seed (int): Seed for sampling the training points.
        """
        self.problem = WeldingProblem.from_config(problem_config)
        self.rng = np.random.default_rng(seed)
        self._load_reference(file, problem_config)
        self._set_thermocouples(thermocouple_config)

    # ---------------- reference solution ----------------

    def _load_reference(
            self,
            file: str | Path,
            problem_config: dict,
        ) -> None:
        """Loads the reference solution and checks that it was generated with
        the current problem configuration.

        Args:
            file (str | Path): Reference solution file.
            problem_config (dict): Current problem configuration.
        """
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(
                f"{file} not found. Run 'python generate_reference_data.py' first.")
        data = np.load(file)
        if json.loads(str(data['problem_config'])) != problem_config:
            raise ValueError(
                f"{file} was generated with a different problem configuration. "
                "Run 'python generate_reference_data.py' to regenerate it.")

        self.x = data['x']                        # (nx,) [m]
        self.y = data['y']                        # (ny,) [m]
        self.t = data['t']                        # (nt,) [s]
        self.T = data['T'].astype(np.float64)     # (nt, ny, nx) [K]

    def mesh(
            self,
            stride_xy: int = 1,
            stride_t: int = 1,
        ) -> tuple[np.ndarray, np.ndarray]:
        """Reference solution on the full space-time grid (e.g. for evaluation).

        Args:
            stride_xy (int): Use every n-th grid node in x and y.
            stride_t (int): Use every n-th snapshot.

        Returns:
            X_mesh (np.ndarray): Grid points (N, 3).
            T_mesh (np.ndarray): Reference temperature (N, 1).
        """
        x, y, t = self.x[::stride_xy], self.y[::stride_xy], self.t[::stride_t]
        tt, yy, xx = np.meshgrid(t, y, x, indexing='ij')         # (nt, ny, nx)
        X_mesh = np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=1)
        T_mesh = self.T[::stride_t, ::stride_xy, ::stride_xy].reshape(-1, 1)
        return X_mesh, T_mesh

    def snapshot(
            self,
            t: float,
        ) -> tuple[np.ndarray, np.ndarray]:
        """Reference solution at the stored time closest to t.

        Args:
            t (float): Time [s].

        Returns:
            X_snap (np.ndarray): Grid points (nx * ny, 3) at that time.
            T_snap (np.ndarray): Reference temperature (ny, nx).
        """
        n = int(np.argmin(np.abs(self.t - t)))
        yy, xx = np.meshgrid(self.y, self.x, indexing='ij')      # (ny, nx)
        X_snap = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, self.t[n])], axis=1)
        return X_snap, self.T[n]

    # ---------------- thermocouples ----------------

    def _set_thermocouples(
            self,
            config: dict,
        ) -> None:
        """Checks the thermocouple layout and creates the measurement signals.

        Every sensor must lie on a node of the stored grid and the sampling
        interval must be an integer multiple of the stored time step, so that
        measurements are read directly from the reference solution without
        interpolation.

        Args:
            config (dict): Thermocouple configuration.
        """
        self.sensors = config['sensors']
        frequency = config['sampling']['frequency']
        noise_std, seed = config['noise']['std'], config['noise']['seed']

        # sampling interval -> every n-th stored snapshot
        dt_ref = self.t[1] - self.t[0]
        stride = int(round(1.0 / frequency / dt_ref))
        if stride < 1 or not np.isclose(stride * dt_ref, 1.0 / frequency):
            raise ValueError(
                f"Sampling interval 1/{frequency:g} Hz is not an integer multiple "
                f"of the stored time step {dt_ref:g} s (max. frequency {1 / dt_ref:g} Hz)")
        self.t_sensor = self.t[::stride]                          # (n_samples,)

        # sensor positions -> grid indices
        names = [s['name'] for s in self.sensors]
        if len(set(names)) != len(names):
            raise ValueError("Thermocouple names must be unique")
        signals = []
        for s in self.sensors:
            if s['role'] not in ('train', 'test'):
                raise ValueError(f"{s['name']}: role must be 'train' or 'test'")
            i, j = self._grid_index(s['x'], self.x), self._grid_index(s['y'], self.y)
            if i is None or j is None:
                raise ValueError(
                    f"{s['name']} at ({s['x']:g}, {s['y']:g}) m does not lie on a "
                    f"node of the stored grid")
            signals.append(self.T[::stride, j, i])
        T_true = np.stack(signals)                                # (n_sensors, n_samples)

        # additive Gaussian noise, drawn once for all sensors -> reproducible
        rng = np.random.default_rng(seed)
        self.T_sensor_true = T_true
        self.T_sensor = T_true + noise_std * rng.standard_normal(T_true.shape)

    @staticmethod
    def _grid_index(
            value: float,
            ticks: np.ndarray,
        ) -> int | None:
        """Index of the grid node at value, or None if value is not on a node."""
        i = int(np.argmin(np.abs(ticks - value)))
        return i if np.isclose(ticks[i], value, rtol=0, atol=1e-9) else None

    def sensor_names(
            self,
            role: str | None = None,
        ) -> list[str]:
        """Names of all thermocouples, optionally only 'train' or 'test' ones."""
        return [s['name'] for s in self.sensors if role in (None, s['role'])]

    def thermocouples(
            self,
            role: str | None = 'train',
            noise: bool = True,
        ) -> tuple[np.ndarray, np.ndarray]:
        """Thermocouple measurements as point data.

        Args:
            role (str, None): 'train', 'test' or None for all sensors.
            noise (bool): Whether to return the noisy measurements (default)
                or the exact reference values at the sensors.

        Returns:
            X_tc (np.ndarray): Measurement points (n_sensors * n_samples, 3).
            T_tc (np.ndarray): Measured temperature (n_sensors * n_samples, 1).
        """
        signals = self.T_sensor if noise else self.T_sensor_true
        X_tc, T_tc = [], []
        for s, signal in zip(self.sensors, signals):
            if role not in (None, s['role']):
                continue
            n = len(self.t_sensor)
            X_tc.append(np.stack([np.full(n, s['x']), np.full(n, s['y']), self.t_sensor], axis=1))
            T_tc.append(signal[:, None])
        return np.concatenate(X_tc), np.concatenate(T_tc)

    def thermocouple_signal(
            self,
            name: str,
            noise: bool = True,
        ) -> tuple[np.ndarray, np.ndarray]:
        """Time series of a single thermocouple (e.g. for plotting).

        Args:
            name (str): Thermocouple name, e.g. 'TC1'.
            noise (bool): Whether to return the noisy measurement.

        Returns:
            t (np.ndarray): Sample times (n_samples,) [s].
            T (np.ndarray): Temperature (n_samples,) [K].
        """
        k = self.sensor_names().index(name)
        signals = self.T_sensor if noise else self.T_sensor_true
        return self.t_sensor, signals[k]

    # ---------------- training points for the physics losses ----------------

    def _lhs(
            self,
            N: int,
            d: int,
        ) -> np.ndarray:
        """Latin-hypercube sample of N points in the unit cube [0, 1]^d."""
        return qmc.LatinHypercube(d=d, rng=self.rng).random(N)

    def sample_domain(
            self,
            N: int,
        ) -> np.ndarray:
        """Collocation points in the space-time domain [0, L] x [0, W] x [0, t_end].

        Args:
            N (int): Number of points.

        Returns:
            X_col (np.ndarray): Collocation points (N, 3).
        """
        p = self.problem
        return self._lhs(N, 3) * np.array([p.L, p.W, p.t_end])

    def sample_initial(
            self,
            N: int,
        ) -> np.ndarray:
        """Points on the plate at t = 0.

        Args:
            N (int): Number of points.

        Returns:
            X_ic (np.ndarray): Initial-condition points (N, 3).
        """
        p = self.problem
        X_ic = np.zeros((N, 3))
        X_ic[:, :2] = self._lhs(N, 2) * np.array([p.L, p.W])
        return X_ic

    def sample_boundary(
            self,
            N: int,
        ) -> tuple[np.ndarray, np.ndarray]:
        """Points on the four plate edges for 0 <= t <= t_end.

        Points are distributed over the edges proportionally to their length.

        Args:
            N (int): Number of points.

        Returns:
            X_bc (np.ndarray): Boundary points (N, 3).
            n_bc (np.ndarray): Outward unit normals (N, 2).
        """
        p = self.problem
        u = self._lhs(N, 2)                         # (position along edge, time)
        # edges: 0: y = 0, 1: y = W, 2: x = 0, 3: x = L
        lengths = np.array([p.L, p.L, p.W, p.W])
        edge = self.rng.choice(4, size=N, p=lengths / lengths.sum())
        s = u[:, 0] * lengths[edge]
        x = np.select([edge == 2, edge == 3], [0.0, p.L], s)
        y = np.select([edge == 0, edge == 1], [0.0, p.W], s)
        X_bc = np.stack([x, y, u[:, 1] * p.t_end], axis=1)
        normals = np.array([[0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]])
        return X_bc, normals[edge]

    def sample_source(
            self,
            N: int,
            spread: float = 2.5,
        ) -> np.ndarray:
        """Additional collocation points concentrated around the moving arc.

        Times are sampled uniformly; positions are normally distributed around
        the arc centre (x_s(t), y_s) with standard deviation spread * sigma and
        clipped to the plate.

        Args:
            N (int): Number of points.
            spread (float): Standard deviation in units of the source width sigma.

        Returns:
            X_src (np.ndarray): Collocation points (N, 3).
        """
        p = self.problem
        t = self._lhs(N, 1)[:, 0] * p.t_end
        x = p.source_position(t) + spread * p.sigma * self.rng.standard_normal(N)
        y = p.y_s + spread * p.sigma * self.rng.standard_normal(N)
        return np.stack([np.clip(x, 0.0, p.L), np.clip(y, 0.0, p.W), t], axis=1)
