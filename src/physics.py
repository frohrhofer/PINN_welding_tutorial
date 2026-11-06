# SPDX-FileCopyrightText: 2026 Franz Rohrhofer <frohrhofer@know-center.at>
# SPDX-License-Identifier: MIT
"""Physical problem: thin steel plate heated by a moving welding arc.

Because the plate is thin, the temperature is assumed to be uniform through
the thickness d and the problem is modelled in 2D:

    rho * c_p * dT/dt = k * (d2T/dx2 + d2T/dy2) + Q(x, y, t)

The arc delivers the effective power eta * P. Spread over a Gaussian spot of
width sigma and over the plate thickness d, this gives the volumetric heat
source [W/m^3]

    Q(x, y, t) = Q_0 * exp(-((x - x_s(t))^2 + (y - y_s)^2) / (2 sigma^2))

    Q_0 = eta * P / (2 pi sigma^2 d),    x_s(t) = x_0 + v t

whose integral over the plate volume is exactly eta * P.

Non-dimensional form
--------------------
The neural networks work with scaled inputs in [a, b] (default [-1, 1]) and a
scaled temperature rise theta:

    x_hat = a + s_x x,  y_hat = a + s_y y,  t_hat = a + s_t t,   T = T_0 + T_s theta

    s_x = (b - a) / L,  s_y = (b - a) / W,  s_t = (b - a) / t_end

With the chain rule (dT/dt = T_s s_t dtheta/dt_hat, d2T/dx2 = T_s s_x^2 d2theta/dx_hat2)
and division by rho c_p T_s s_t, the heat equation becomes

    theta_t = D_x theta_xx + D_y theta_yy + S q(x, y, t)

    D_x = alpha s_x^2 / s_t,  D_y = alpha s_y^2 / s_t,  S = Q_0 / (rho c_p T_s s_t)

where q = Q / Q_0 is the Gaussian shape of the source (peak value 1).
Initial condition:  theta = 0  at t_hat = a.
Boundary condition (outward normal n): -k dT/dn = h (T - T_inf) becomes

    n_x theta_x + Bi_x (theta - theta_inf) = 0     on x = 0, L
    n_y theta_y + Bi_y (theta - theta_inf) = 0     on y = 0, W

    Bi_x = h / (k s_x),  Bi_y = h / (k s_y),  theta_inf = (T_inf - T_0) / T_s
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NondimensionalCoefficients:
    """Coefficients of the non-dimensional problem (see module docstring)."""

    s_x: float          # chain-rule factor d x_hat / d x  [1/m]
    s_y: float          # chain-rule factor d y_hat / d y  [1/m]
    s_t: float          # chain-rule factor d t_hat / d t  [1/s]
    T_scale: float      # temperature scale T_s            [K]
    D_x: float          # diffusion coefficient in x_hat   [-]
    D_y: float          # diffusion coefficient in y_hat   [-]
    S: float            # source coefficient               [-]
    Bi_x: float         # Biot number at x = 0, L          [-]
    Bi_y: float         # Biot number at y = 0, W          [-]
    theta_inf: float    # scaled ambient temperature       [-]


@dataclass(frozen=True)
class WeldingProblem:
    """Problem parameters in SI units, named by their symbols."""

    # plate
    L: float            # length in x (welding direction)  [m]
    W: float            # width in y                        [m]
    d: float            # thickness                         [m]
    # material
    rho: float          # density                           [kg/m^3]
    cp: float           # specific heat capacity            [J/(kg K)]
    k: float            # thermal conductivity              [W/(m K)]
    T_melt: float       # approximate melting temperature   [K]
    # heat source
    P: float            # nominal arc power                 [W]
    eta: float          # arc efficiency                    [-]
    v: float            # travel speed                      [m/s]
    sigma: float        # Gaussian width                    [m]
    x0: float           # arc position at t = 0             [m]
    y_s: float          # weld centre line                  [m]
    # initial and boundary conditions
    T0: float           # initial temperature               [K]
    bc_type: str        # "convective" or "adiabatic"
    h: float            # heat-transfer coefficient         [W/(m^2 K)]
    T_inf: float        # ambient temperature               [K]
    # time
    t_end: float        # end time                          [s]

    def __post_init__(self):
        if self.bc_type not in ('convective', 'adiabatic'):
            raise ValueError(f"Unknown boundary condition type '{self.bc_type}'")

    @classmethod
    def from_config(
            cls,
            config: dict,
        ) -> 'WeldingProblem':
        """Creates the problem from the dictionary of config/problem.yaml.

        Args:
            config (dict): Problem configuration.

        Returns:
            problem (WeldingProblem): Problem instance.
        """
        plate, mat, src = config['plate'], config['material'], config['heat_source']
        ic, bc = config['initial_condition'], config['boundary_condition']
        return cls(
            L=plate['length'], W=plate['width'], d=plate['thickness'],
            rho=mat['density'], cp=mat['specific_heat'],
            k=mat['thermal_conductivity'], T_melt=mat['melting_temperature'],
            P=src['power'], eta=src['efficiency'], v=src['travel_speed'],
            sigma=src['gaussian_width'], x0=src['start_x'], y_s=src['path_y'],
            T0=ic['temperature'],
            bc_type=bc['type'], h=bc['heat_transfer_coefficient'],
            T_inf=bc['ambient_temperature'],
            t_end=config['time']['end'],
        )

    # ---------------- derived quantities ----------------

    @property
    def alpha(self) -> float:
        """Thermal diffusivity k / (rho c_p) [m^2/s]."""
        return self.k / (self.rho * self.cp)

    @property
    def h_eff(self) -> float:
        """Heat-transfer coefficient applied at the edges (0 if adiabatic) [W/(m^2 K)]."""
        return self.h if self.bc_type == 'convective' else 0.0

    @property
    def P_eff(self) -> float:
        """Effective arc power eta * P [W]."""
        return self.eta * self.P

    @property
    def Q0(self) -> float:
        """Peak volumetric heat source eta P / (2 pi sigma^2 d) [W/m^3]."""
        return self.P_eff / (2.0 * np.pi * self.sigma**2 * self.d)

    @property
    def heat_input(self) -> float:
        """Heat input per unit weld length eta P / v [J/m]."""
        return self.P_eff / self.v

    # ---------------- heat source ----------------

    def source_position(self, t):
        """x-coordinate of the arc centre x_s(t) = x_0 + v t [m]."""
        return self.x0 + self.v * t

    def heat_source_shape(self, x, y, t):
        """Gaussian shape q = Q / Q_0 of the heat source (peak value 1) [-].

        Works with NumPy arrays and PyTorch tensors alike.
        """
        r2 = (x - self.source_position(t))**2 + (y - self.y_s)**2
        z = -r2 / (2.0 * self.sigma**2)
        # torch tensors provide .exp(); np.exp would break autograd
        return z.exp() if hasattr(z, 'exp') else np.exp(z)

    def heat_source(self, x, y, t):
        """Volumetric heat source Q(x, y, t) = Q_0 q(x, y, t) [W/m^3]."""
        return self.Q0 * self.heat_source_shape(x, y, t)

    # ---------------- non-dimensional form ----------------

    def nondimensional(
            self,
            input_range: tuple[float, float],
            T_scale: float,
        ) -> NondimensionalCoefficients:
        """Coefficients of the non-dimensional problem.

        Args:
            input_range (tuple[float, float]): Range [a, b] of the scaled inputs.
            T_scale (float): Temperature scale T_s [K].

        Returns:
            coefficients (NondimensionalCoefficients): Chain-rule factors and
                coefficients of the scaled PDE, IC and BC.
        """
        a, b = input_range
        s_x, s_y, s_t = (b - a) / self.L, (b - a) / self.W, (b - a) / self.t_end
        return NondimensionalCoefficients(
            s_x=s_x, s_y=s_y, s_t=s_t, T_scale=T_scale,
            D_x=self.alpha * s_x**2 / s_t,
            D_y=self.alpha * s_y**2 / s_t,
            S=self.Q0 / (self.rho * self.cp * T_scale * s_t),
            Bi_x=self.h_eff / (self.k * s_x),
            Bi_y=self.h_eff / (self.k * s_y),
            theta_inf=(self.T_inf - self.T0) / T_scale,
        )

    def summary(self) -> str:
        """Human-readable overview of the problem."""
        bc = (f"convective, h = {self.h:.0f} W/(m^2 K), T_inf = {self.T_inf:.2f} K"
              if self.bc_type == 'convective' else 'adiabatic')
        return "\n".join([
            f"Plate         : {self.L*1e3:.0f} x {self.W*1e3:.0f} x {self.d*1e3:.1f} mm",
            f"Material      : rho = {self.rho:.0f} kg/m^3, c_p = {self.cp:.0f} J/(kg K), "
            f"k = {self.k:.1f} W/(m K)",
            f"Diffusivity   : alpha = {self.alpha:.3e} m^2/s",
            f"Arc           : P = {self.P:.0f} W, eta = {self.eta:.2f} -> eta P = {self.P_eff:.0f} W",
            f"Travel speed  : v = {self.v*1e3:.1f} mm/s, heat input = {self.heat_input/1e6:.3f} kJ/mm",
            f"Source        : sigma = {self.sigma*1e3:.1f} mm, Q_0 = {self.Q0:.3e} W/m^3",
            f"Arc path      : x = {self.x0*1e3:.0f} -> {self.source_position(self.t_end)*1e3:.0f} mm "
            f"at y = {self.y_s*1e3:.0f} mm",
            f"Initial cond. : T_0 = {self.T0:.2f} K",
            f"Boundary cond.: {bc}",
            f"Time          : 0 ... {self.t_end:.1f} s",
        ])
