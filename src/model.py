# SPDX-FileCopyrightText: 2026 Franz Rohrhofer <frohrhofer@know-center.at>
# SPDX-License-Identifier: MIT
"""Fully connected neural network (x, y, t) -> T used in all scenarios.

The network itself works only with non-dimensional quantities:

    physical X = (x, y, t)  --scale_inputs-->  X_hat  --forward-->  theta

    T = T_0 + T_s theta

so that the same object can be trained with data, physics or both.
"""

import torch
import torch.nn as nn

from src.physics import WeldingProblem


class Sine(nn.Module):
    """Sine activation (not built into PyTorch)."""

    def forward(self, z):
        return torch.sin(z)


ACTIVATIONS = {
    'tanh': nn.Tanh,
    'sin': Sine,
    'silu': nn.SiLU,
    'gelu': nn.GELU,
}

INITIALIZATIONS = {
    'xavier_normal': nn.init.xavier_normal_,
    'xavier_uniform': nn.init.xavier_uniform_,
}


class NeuralNet(nn.Module):
    """MLP with input scaling, configured by config/network.yaml."""

    def __init__(
            self,
            config: dict,
            problem: WeldingProblem,
            hard_ic: bool = False,
            verbose: bool = True,
        ) -> None:
        """Initialize network: scaling from problem and config, then layers.

        Args:
            config (dict): Network configuration.
            problem (WeldingProblem): Physical problem (domain size, T_0).
            hard_ic (bool): Enforce the initial condition exactly via
                theta = (t_hat - a) / (b - a) * NN, so theta = 0 at t = 0.
            verbose (bool): Whether to print the architecture.
        """
        super().__init__()
        arch, scaling = config['architecture'], config['scaling']
        if arch['inputs'] != ['x', 'y', 't'] or arch['outputs'] != ['T']:
            raise ValueError("Network must map inputs [x, y, t] to outputs [T]")

        # ---- scaling (buffers: moved with .to(device), saved with the weights) ----
        coeff = problem.nondimensional(scaling['input_range'], scaling['temperature_scale'])
        self.coeff = coeff
        self.hard_ic = hard_ic
        self.register_buffer('a', torch.tensor(float(scaling['input_range'][0])))
        self.register_buffer('b', torch.tensor(float(scaling['input_range'][1])))
        self.register_buffer('s', torch.tensor([[coeff.s_x, coeff.s_y, coeff.s_t]]))
        self.register_buffer('T0', torch.tensor(problem.T0))
        self.register_buffer('T_scale', torch.tensor(coeff.T_scale))

        # ---- layers ----
        activation = ACTIVATIONS[arch['activation']]
        init = INITIALIZATIONS[arch['initialization']]
        layers, n_in = [], len(arch['inputs'])
        for _ in range(arch['hidden_layers']):
            layers += [nn.Linear(n_in, arch['neurons_per_layer']), activation()]
            n_in = arch['neurons_per_layer']
        layers.append(nn.Linear(n_in, len(arch['outputs'])))
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init(layer.weight)
                nn.init.zeros_(layer.bias)

        if verbose:
            n_params = sum(p.numel() for p in self.parameters())
            print(self.net)
            print(f"Trainable parameters: {n_params}")

    def scale_inputs(
            self,
            X: torch.Tensor,
        ) -> torch.Tensor:
        """Physical (x, y, t) -> scaled (x_hat, y_hat, t_hat) in [a, b].

        Args:
            X (torch.Tensor): Physical coordinates (N, 3).

        Returns:
            X_hat (torch.Tensor): Scaled coordinates (N, 3).
        """
        return self.a + self.s * X

    def forward(
            self,
            X_hat: torch.Tensor,
        ) -> torch.Tensor:
        """Scaled coordinates -> scaled temperature rise theta.

        Args:
            X_hat (torch.Tensor): Scaled coordinates (N, 3).

        Returns:
            theta (torch.Tensor): Scaled temperature rise (N, 1).
        """
        theta = self.net(X_hat)
        if self.hard_ic:
            theta = (X_hat[:, 2:3] - self.a) / (self.b - self.a) * theta
        return theta

    @torch.no_grad()
    def temperature(
            self,
            X: torch.Tensor,
        ) -> torch.Tensor:
        """Temperature in K at physical coordinates (for evaluation).

        Args:
            X (torch.Tensor): Physical coordinates (N, 3).

        Returns:
            T (torch.Tensor): Temperature (N, 1) [K].
        """
        return self.T0 + self.T_scale * self(self.scale_inputs(X))
