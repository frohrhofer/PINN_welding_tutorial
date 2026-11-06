# SPDX-FileCopyrightText: 2026 Franz Rohrhofer <frohrhofer@know-center.at>
# SPDX-License-Identifier: MIT
"""Training of the neural network with data and/or physics losses.

All losses are computed in non-dimensional form (see src/physics.py):

    data :  theta_NN - (T_meas - T_0) / T_s                       at thermocouples
    PDE  :  theta_t - D_x theta_xx - D_y theta_yy - S q           in the domain
    IC   :  theta                                                 at t = 0
    BC   :  n_x theta_x + n_y theta_y + Bi (theta - theta_inf)    on the edges

Derivatives are taken with respect to the scaled inputs X_hat, so the
chain-rule factors are contained in the coefficients D_x, D_y, S, Bi.
Which terms are used is set by their weights in the scenario configuration
(weight 0 switches a term off). Training uses Adam, optionally followed by
L-BFGS refinement.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data_loader import DataLoader
from src.model import NeuralNet

LOSS_TERMS = ('data', 'pde', 'ic', 'bc')


class PINN:
    """Trains a NeuralNet according to a scenario configuration."""

    def __init__(
            self,
            config: dict,
            model: NeuralNet,
            data: DataLoader,
            device: torch.device | None = None,
        ) -> None:
        """Initialize with scenario configuration, network and data.

        Args:
            config (dict): Scenario configuration.
            model (NeuralNet): Network to be trained.
            data (DataLoader): Reference solution, thermocouples and sampling.
            device (torch.device, None): Device for training, default: CPU.
        """
        self.config = config
        self.device = device or torch.device('cpu')
        self.model = model.to(self.device)
        self.data = data
        self.problem = data.problem
        self.coeff = model.coeff

        self.weights = {term: float(config['losses'][term]) for term in LOSS_TERMS}
        self.active = [term for term in LOSS_TERMS if self.weights[term] > 0]
        if not self.active:
            raise ValueError("At least one loss term needs a positive weight")
        self.sampling = config['sampling']
        self.training = config['training']
        self.logging = config['logging']
        self.path = Path('results') / config['version']

        # fixed data: training and test thermocouples, validation mesh
        self.X_tc, self.T_tc = map(self._tensor, data.thermocouples('train'))
        self.X_test, self.T_test = map(self._tensor, data.thermocouples('test'))
        self.X_val, self.T_val = map(self._tensor, data.mesh(stride_xy=2, stride_t=4))

        self.log = {'config': config}

    def _tensor(
            self,
            a: np.ndarray,
        ) -> torch.Tensor:
        """NumPy array -> float32 tensor on the training device."""
        return torch.tensor(a, dtype=torch.float32, device=self.device)

    def _theta(
            self,
            T: torch.Tensor,
        ) -> torch.Tensor:
        """Temperature [K] -> scaled temperature rise theta."""
        return (T - self.problem.T0) / self.coeff.T_scale

    # ---------------- loss terms ----------------

    def loss_data(
            self,
            X: torch.Tensor,
            T: torch.Tensor,
        ) -> torch.Tensor:
        """Data loss: mean squared error of theta at measurement points.

        Args:
            X (torch.Tensor): Measurement points (N, 3) [m, m, s].
            T (torch.Tensor): Measured temperature (N, 1) [K].

        Returns:
            loss (torch.Tensor): Scalar loss.
        """
        theta = self.model(self.model.scale_inputs(X))
        return torch.mean((theta - self._theta(T))**2)

    def _gradients(
            self,
            X: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Network output and its gradient with respect to the scaled inputs.

        Args:
            X (torch.Tensor): Physical points (N, 3).

        Returns:
            X_hat (torch.Tensor): Scaled points (N, 3), part of the graph.
            theta (torch.Tensor): Network output (N, 1).
            dtheta (torch.Tensor): (theta_x, theta_y, theta_t) w.r.t. X_hat (N, 3).
        """
        X_hat = self.model.scale_inputs(X).requires_grad_(True)
        theta = self.model(X_hat)
        dtheta = torch.autograd.grad(theta, X_hat, torch.ones_like(theta), create_graph=True)[0]
        return X_hat, theta, dtheta

    def pde_residuals(
            self,
            X: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
        """Residual of the non-dimensional heat equation and its components.

        Args:
            X (torch.Tensor): Collocation points (N, 3) [m, m, s].

        Returns:
            terms (dict): theta, theta_t, theta_xx, theta_yy, source and
                residual, each (N, 1).
        """
        X_hat, theta, dtheta = self._gradients(X)
        theta_x, theta_y, theta_t = dtheta[:, 0:1], dtheta[:, 1:2], dtheta[:, 2:3]
        ones = torch.ones_like(theta_x)
        theta_xx = torch.autograd.grad(theta_x, X_hat, ones, create_graph=True)[0][:, 0:1]
        theta_yy = torch.autograd.grad(theta_y, X_hat, ones, create_graph=True)[0][:, 1:2]
        # source evaluated at the physical coordinates
        source = self.coeff.S * self.problem.heat_source_shape(X[:, 0:1], X[:, 1:2], X[:, 2:3])
        residual = theta_t - self.coeff.D_x * theta_xx - self.coeff.D_y * theta_yy - source
        return {'theta': theta, 'theta_t': theta_t, 'theta_xx': theta_xx,
                'theta_yy': theta_yy, 'source': source, 'residual': residual}

    def loss_pde(
            self,
            X: torch.Tensor,
        ) -> torch.Tensor:
        """PDE loss: mean squared residual at collocation points."""
        return torch.mean(self.pde_residuals(X)['residual']**2)

    def loss_ic(
            self,
            X: torch.Tensor,
        ) -> torch.Tensor:
        """Initial-condition loss: T = T_0, i.e. theta = 0, at t = 0."""
        theta = self.model(self.model.scale_inputs(X))
        return torch.mean(theta**2)

    def bc_residuals(
            self,
            X: torch.Tensor,
            n: torch.Tensor,
        ) -> torch.Tensor:
        """Residual of the convective boundary condition.

        Args:
            X (torch.Tensor): Boundary points (N, 3) [m, m, s].
            n (torch.Tensor): Outward unit normals (N, 2).

        Returns:
            residual (torch.Tensor): n . grad(theta) + Bi (theta - theta_inf), (N, 1).
        """
        _, theta, dtheta = self._gradients(X)
        n_x, n_y = n[:, 0:1], n[:, 1:2]
        dtheta_dn = n_x * dtheta[:, 0:1] + n_y * dtheta[:, 1:2]
        Bi = n_x.abs() * self.coeff.Bi_x + n_y.abs() * self.coeff.Bi_y
        return dtheta_dn + Bi * (theta - self.coeff.theta_inf)

    def loss_bc(
            self,
            X: torch.Tensor,
            n: torch.Tensor,
        ) -> torch.Tensor:
        """Boundary-condition loss: mean squared BC residual."""
        return torch.mean(self.bc_residuals(X, n)**2)

    def sample_points(
            self,
            terms: list[str],
        ) -> dict[str, tuple[torch.Tensor, ...]]:
        """Samples fresh training points for the requested physics terms.

        Collocation points consist of N_col points spread uniformly over the
        domain plus (optionally) N_source points concentrated around the arc.

        Args:
            terms (list[str]): Subset of 'pde', 'ic', 'bc' ('data' is ignored).

        Returns:
            points (dict): Tensors per term, e.g. points['bc'] = (X_bc, n_bc).
        """
        points = {}
        if 'pde' in terms:
            X_col = self.data.sample_domain(self.sampling['N_col'])
            N_source = self.sampling.get('N_source', 0)
            if N_source > 0:
                X_src = self.data.sample_source(N_source, self.sampling.get('source_spread', 2.5))
                X_col = np.concatenate([X_col, X_src])
            points['pde'] = (self._tensor(X_col),)
        if 'ic' in terms:
            points['ic'] = (self._tensor(self.data.sample_initial(self.sampling['N_ic'])),)
        if 'bc' in terms:
            points['bc'] = tuple(map(self._tensor, self.data.sample_boundary(self.sampling['N_bc'])))
        return points

    def losses(
            self,
            terms: list[str],
            points: dict | None = None,
        ) -> dict[str, torch.Tensor]:
        """Evaluates the requested loss terms.

        Args:
            terms (list[str]): Subset of 'data', 'pde', 'ic', 'bc'.
            points (dict, None): Training points from sample_points(),
                default: freshly sampled points.

        Returns:
            losses (dict): Loss value per term.
        """
        points = self.sample_points(terms) if points is None else points
        out = {}
        if 'data' in terms:
            out['data'] = self.loss_data(self.X_tc, self.T_tc)
        if 'pde' in terms:
            out['pde'] = self.loss_pde(*points['pde'])
        if 'ic' in terms:
            out['ic'] = self.loss_ic(*points['ic'])
        if 'bc' in terms:
            out['bc'] = self.loss_bc(*points['bc'])
        return out

    def _total(
            self,
            losses: dict[str, torch.Tensor],
        ) -> torch.Tensor:
        """Weighted sum of the active loss terms."""
        return sum(self.weights[term] * losses[term] for term in self.active)

    def _out_of_time(self) -> bool:
        """Whether the optional time budget training.max_time [s] is used up."""
        max_time = self.training.get('max_time')
        return max_time is not None and time.perf_counter() - self._tic > max_time

    # ---------------- training ----------------

    def train(self) -> None:
        """Adam optimization, optionally followed by L-BFGS refinement.

        Training stops early once the optional time budget is used up.
        """
        print(f"Training started (active loss terms: {', '.join(self.active)}) ...")
        self._tic = time.perf_counter()
        epoch = self._train_adam()
        if self.config.get('lbfgs') and not self._out_of_time():
            self._train_lbfgs(epoch)
        print(f"Training finished in {time.perf_counter() - self._tic:.0f} s")

    def _train_adam(self) -> int:
        """Adam with exponential learning-rate decay, new points in every epoch.

        Returns:
            epoch (int): Last epoch.
        """
        n_epochs = self.training['N_epochs']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training['learning_rate'])
        # lr(epoch) = lr_0 * decay_rate^(epoch / decay_steps)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.training['decay_rate']**(1 / self.training['decay_steps']))

        for epoch in range(n_epochs + 1):
            losses = self.losses(self.active)
            loss_train = self._total(losses)

            last = epoch == n_epochs or self._out_of_time()
            if epoch % self.logging['freq_log'] == 0 or last:
                self._write_logs(epoch, losses, loss_train, optimizer.param_groups[0]['lr'], last)
            if last:
                return epoch

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            scheduler.step()

    def _train_lbfgs(
            self,
            epoch: int,
        ) -> None:
        """L-BFGS refinement after Adam.

        Every outer step runs up to max_iter L-BFGS iterations on a fixed set
        of points; new points are sampled for the next outer step. The number
        of points can be changed for this phase (lbfgs.sampling). If the loss
        becomes NaN, the weights of the previous step are restored.

        Args:
            epoch (int): Last Adam epoch (logging continues from there).
        """
        cfg = self.config['lbfgs']
        self.sampling = {**self.sampling, **(cfg.get('sampling') or {})}
        optimizer = torch.optim.LBFGS(
            self.model.parameters(), lr=cfg.get('learning_rate', 1.0),
            max_iter=cfg['max_iter'], history_size=cfg.get('history_size', 50),
            line_search_fn='strong_wolfe')
        print(f"L-BFGS refinement ({cfg['N_steps']} steps x {cfg['max_iter']} iterations) ...")

        for step in range(1, cfg['N_steps'] + 1):
            points = self.sample_points(self.active)

            def closure():
                optimizer.zero_grad()
                loss = self._total(self.losses(self.active, points))
                loss.backward()
                return loss

            backup = {k: v.clone() for k, v in self.model.state_dict().items()}
            optimizer.step(closure)
            losses = self.losses(self.active, points)
            loss_train = self._total(losses)
            if not torch.isfinite(loss_train):
                print("L-BFGS diverged -> restoring previous weights and stopping")
                self.model.load_state_dict(backup)
                break

            last = step == cfg['N_steps'] or self._out_of_time()
            self._write_logs(epoch + step * cfg['max_iter'], losses, loss_train,
                             cfg.get('learning_rate', 1.0), force_print=True)
            if last:
                break

    def _write_logs(
            self,
            epoch: int,
            losses: dict[str, torch.Tensor],
            loss_train: torch.Tensor,
            lr: float,
            force_print: bool = False,
        ) -> None:
        """Records (and prints) training and monitoring quantities.

        Inactive loss terms are evaluated as well, for monitoring only.

        Args:
            epoch (int): Current epoch (or L-BFGS iteration count).
            losses (dict): Active loss terms of the current step.
            loss_train (torch.Tensor): Weighted total loss.
            lr (float): Current learning rate.
            force_print (bool): Print regardless of freq_print.
        """
        inactive = [term for term in LOSS_TERMS if term not in self.active]
        losses = {**losses, **self.losses(inactive)}
        with torch.no_grad():
            T_val = self.model.temperature(self.X_val)
            entries = {
                'epoch': epoch,
                'time': time.perf_counter() - self._tic,
                'lr': lr,
                'loss_train': loss_train.item(),
                **{f'loss_{term}': val.item() for term, val in losses.items()},
                'loss_test': self.loss_data(self.X_test, self.T_test).item(),
                'rmse_val': torch.sqrt(torch.mean((T_val - self.T_val)**2)).item(),
            }
        for key, val in entries.items():
            self.log.setdefault(key, []).append(val)

        if epoch % self.logging['freq_print'] == 0 or force_print:
            s = f"{epoch:6d} | {entries['time']:6.0f} s"
            for key in self.logging['keys_print']:
                s += f" | {key}: {entries[key]:.2e}"
            print(s, flush=True)

    # ---------------- evaluation ----------------

    def predict(
            self,
            X: np.ndarray,
            batch_size: int = 65536,
        ) -> np.ndarray:
        """Temperature prediction at physical points.

        Args:
            X (np.ndarray): Physical points (N, 3).
            batch_size (int): Points per forward pass.

        Returns:
            T (np.ndarray): Predicted temperature (N, 1) [K].
        """
        return np.concatenate([
            self.model.temperature(self._tensor(X[i:i + batch_size])).cpu().numpy()
            for i in range(0, len(X), batch_size)
        ])

    def evaluate(self) -> dict[str, float]:
        """Error metrics on the full reference grid.

        The relative L2 error is computed on the temperature rise T - T_0,
        because the constant offset T_0 would otherwise make every model look good.

        Returns:
            metrics (dict): RMSE [K], relative L2 error [-], max. error [K].
        """
        X_mesh, T_ref = self.data.mesh()
        error = self.predict(X_mesh) - T_ref
        metrics = {
            'rmse': float(np.sqrt(np.mean(error**2))),
            'rel_l2': float(np.linalg.norm(error) / np.linalg.norm(T_ref - self.problem.T0)),
            'max_error': float(np.abs(error).max()),
        }
        self.log['evaluation'] = metrics
        print(f"RMSE = {metrics['rmse']:.1f} K | rel. L2 = {metrics['rel_l2']:.3f} | "
              f"max. error = {metrics['max_error']:.0f} K")
        return metrics

    # ---------------- saving ----------------

    def load(self) -> None:
        """Loads network weights and training log from results/<version>/."""
        file = self.path / 'model.pt'
        if not file.exists():
            raise FileNotFoundError(f"{file} not found. Train the scenario first.")
        self.model.load_state_dict(torch.load(file, map_location=self.device))
        with open(self.path / 'log.json') as f:
            self.log = json.load(f)
        print(f"Loaded weights and log from {self.path}")

    def save(self) -> None:
        """Saves log (JSON) and network weights to results/<version>/."""
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / 'log.json', 'w') as f:
            json.dump(self.log, f, indent=2)
        torch.save(self.model.state_dict(), self.path / 'model.pt')
        print(f"Saved log and weights to {self.path}")
