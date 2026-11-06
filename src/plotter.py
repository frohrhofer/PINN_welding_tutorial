# SPDX-FileCopyrightText: 2026 Franz Rohrhofer <frohrhofer@know-center.at>
# SPDX-License-Identifier: MIT
"""Figures shared by all scenarios.

Colour conventions:
* temperature: sequential 'inferno', melting temperature as cyan contour
* absolute error: sequential 'Reds' starting at 0
* residuals / signed quantities: diverging 'RdBu_r', symmetric around 0
* lines: fixed categorical order, inactive loss terms dashed
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.pinn import PINN, LOSS_TERMS

COLORS = {
    'train': '#2a78d6', 'data': '#eb6834', 'pde': '#1baf7a',
    'ic': '#eda100', 'bc': '#e87ba4', 'test': '#008300',
}
REFERENCE_COLOR = '#222222'
PREDICTION_COLOR = '#2a78d6'
MEASUREMENT_COLOR = '#9a9a9a'

plt.rcParams.update({
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.8,
    'lines.linewidth': 1.5, 'font.size': 9,
})


class Plotter:
    """Creates and saves the figures of a trained PINN."""

    def __init__(
            self,
            pinn: PINN,
            path: str | Path | None = None,
            show: bool = False,
        ) -> None:
        """Initialize with a (trained) PINN instance.

        Args:
            pinn (PINN): Trained model incl. data and logs.
            path (str | Path, None): Output folder, default: PINN results folder.
            show (bool): Whether to show figures (e.g. in notebooks).
        """
        self.pinn = pinn
        self.data = pinn.data
        self.problem = pinn.problem
        self.path = Path(path or pinn.path)
        self.show = show

    def _finish(
            self,
            fig: plt.Figure,
            name: str,
        ) -> plt.Figure:
        """Saves the figure and shows or closes it."""
        self.path.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.path / f'{name}.png', dpi=150, bbox_inches='tight')
        if self.show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def _plate_axes(self, ax, title):
        ax.set(title=title, xlabel='x [mm]', ylabel='y [mm]', aspect='equal')
        ax.grid(False)

    def _sensors(self, ax):
        for s in self.data.sensors:
            ax.plot(s['x'] * 1e3, s['y'] * 1e3, 'o' if s['role'] == 'train' else 's',
                    ms=4, mfc='none', mec='white', mew=1)

    # ---------------- figures ----------------

    def learning_curves(self) -> plt.Figure:
        """Loss terms (left) and validation RMSE in K (right) vs. epoch."""
        log = self.pinn.log
        epochs = np.asarray(log['epoch'])
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)

        ax = axes[0]
        ax.plot(epochs, log['loss_train'], c=COLORS['train'], label='total (weighted)')
        for term in LOSS_TERMS:
            if not np.any(np.asarray(log[f'loss_{term}']) > 0):
                continue                    # e.g. IC loss with hard IC (identically 0)
            active = term in self.pinn.active
            ax.plot(epochs, log[f'loss_{term}'], c=COLORS[term], ls='-' if active else '--',
                    lw=1.5 if active else 1, label=term + ('' if active else ' (monitor)'))
        ax.plot(epochs, log['loss_test'], c=COLORS['test'], ls=':', label='test TCs (monitor)')
        ax.set(yscale='log', xlabel='epoch', ylabel='loss (non-dimensional)', title='Loss terms')
        ax.legend(frameon=False, fontsize=8)

        ax = axes[1]
        ax.plot(epochs, log['rmse_val'], c=REFERENCE_COLOR)
        ax.set(yscale='log', xlabel='epoch', ylabel='RMSE [K]',
               title='Validation error on reference grid')

        # switch from Adam to L-BFGS (afterwards: x = L-BFGS iterations)
        if self.pinn.config.get('lbfgs') and epochs.max() > self.pinn.training['N_epochs']:
            for ax in axes:
                ax.axvline(self.pinn.training['N_epochs'], c='#9a9a9a', lw=1, ls=':')
                ax.text(self.pinn.training['N_epochs'], 1, ' L-BFGS', transform=ax.get_xaxis_transform(),
                        va='top', fontsize=8, color='#6a6a6a')
        return self._finish(fig, 'learning_curves')

    def fields(
            self,
            times: tuple[float, ...] = (2.0, 6.0, 10.0),
        ) -> plt.Figure:
        """Reference, prediction and absolute error at selected times."""
        refs, preds = [], []
        for t in times:
            X_snap, T_ref = self.data.snapshot(t)
            refs.append(T_ref)
            preds.append(self.pinn.predict(X_snap).reshape(T_ref.shape))
        errors = [np.abs(p - r) for p, r in zip(preds, refs)]
        x_mm, y_mm = self.data.x * 1e3, self.data.y * 1e3
        T_min, T_max = self.problem.T0, max(np.max(refs), np.max(preds))
        err_max = max(e.max() for e in errors)

        fig, axes = plt.subplots(3, len(times), figsize=(4.6 * len(times), 7.5),
                                 constrained_layout=True)
        for k, t in enumerate(times):
            for row, (field, name) in enumerate([(refs[k], 'reference'), (preds[k], 'prediction')]):
                ax = axes[row, k]
                im_T = ax.pcolormesh(x_mm, y_mm, field, cmap='inferno', vmin=T_min, vmax=T_max)
                if field.max() > self.problem.T_melt:
                    ax.contour(x_mm, y_mm, field, [self.problem.T_melt], colors='cyan', linewidths=1)
                self._sensors(ax)
                self._plate_axes(ax, f'{name}, t = {t:.1f} s')
            ax = axes[2, k]
            im_e = ax.pcolormesh(x_mm, y_mm, errors[k], cmap='Reds', vmin=0, vmax=err_max)
            self._plate_axes(ax, f'|error|, t = {t:.1f} s (max {errors[k].max():.0f} K)')
        fig.colorbar(im_T, ax=axes[:2, :], label='T [K]', shrink=0.8)
        fig.colorbar(im_e, ax=axes[2, :], label='|T_pred - T_ref| [K]', shrink=0.8)
        return self._finish(fig, 'fields')

    def centerline(
            self,
            times: tuple[float, ...] = (2.0, 6.0, 10.0),
        ) -> plt.Figure:
        """Temperature along the weld line y = y_s at selected times."""
        j = int(np.argmin(np.abs(self.data.y - self.problem.y_s)))
        fig, axes = plt.subplots(1, len(times), figsize=(4.6 * len(times), 3.2),
                                 sharey=True, constrained_layout=True)
        for ax, t in zip(axes, times):
            X_snap, T_ref = self.data.snapshot(t)
            T_pred = self.pinn.predict(X_snap).reshape(T_ref.shape)
            ax.plot(self.data.x * 1e3, T_ref[j], c=REFERENCE_COLOR, ls='--', label='reference')
            ax.plot(self.data.x * 1e3, T_pred[j], c=PREDICTION_COLOR, label='prediction')
            ax.axhline(self.problem.T_melt, c='#9a9a9a', lw=1, ls=':', label='T_melt')
            ax.set(title=f'weld line y = {self.problem.y_s*1e3:.0f} mm, t = {t:.1f} s',
                   xlabel='x [mm]')
        axes[0].set_ylabel('T [K]')
        axes[0].legend(frameon=False)
        return self._finish(fig, 'centerline')

    def thermocouples(self) -> plt.Figure:
        """Measured, reference and predicted signal at every thermocouple."""
        names = self.data.sensor_names()
        n_cols = 5
        n_rows = int(np.ceil(len(names) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows),
                                 sharex=True, sharey=True, constrained_layout=True)
        for ax, s in zip(axes.flat, self.data.sensors):
            t, T_meas = self.data.thermocouple_signal(s['name'])
            _, T_ref = self.data.thermocouple_signal(s['name'], noise=False)
            X = np.stack([np.full_like(t, s['x']), np.full_like(t, s['y']), t], axis=1)
            ax.plot(t, T_meas, '.', ms=3, c=MEASUREMENT_COLOR, label='measured')
            ax.plot(t, T_ref, c=REFERENCE_COLOR, ls='--', lw=1, label='reference')
            ax.plot(t, self.pinn.predict(X), c=PREDICTION_COLOR, label='prediction')
            ax.set_title(f"{s['name']} ({s['role']}) at ({s['x']*1e3:.0f}, {s['y']*1e3:.0f}) mm")
        for ax in axes[-1]:
            ax.set_xlabel('t [s]')
        for ax in axes[:, 0]:
            ax.set_ylabel('T [K]')
        axes.flat[0].legend(frameon=False, fontsize=8)
        return self._finish(fig, 'thermocouples')

    def residuals(
            self,
            t: float = 6.0,
        ) -> plt.Figure:
        """Terms of the non-dimensional PDE of the network at time t."""
        X_snap, T_ref = self.data.snapshot(t)
        X = self.pinn._tensor(X_snap)
        terms = self.pinn.pde_residuals(X)
        c = self.pinn.coeff
        to_np = lambda a: a.detach().cpu().numpy().reshape(T_ref.shape)
        maps = {
            'theta_t': to_np(terms['theta_t']),
            'diffusion': to_np(c.D_x * terms['theta_xx'] + c.D_y * terms['theta_yy']),
            'source': to_np(terms['source']),
            'residual': to_np(terms['residual']),
        }
        x_mm, y_mm = self.data.x * 1e3, self.data.y * 1e3
        fig, axes = plt.subplots(1, 4, figsize=(19, 2.9), constrained_layout=True)
        for ax, (name, field) in zip(axes, maps.items()):
            v = np.abs(field).max()
            im = ax.pcolormesh(x_mm, y_mm, field, cmap='RdBu_r', vmin=-v, vmax=v)
            self._plate_axes(ax, f'{name}, t = {t:.1f} s')
            fig.colorbar(im, ax=ax, shrink=0.8)
        return self._finish(fig, 'residuals')

    def all(self) -> None:
        """Creates all figures."""
        self.learning_curves()
        self.fields()
        self.centerline()
        self.thermocouples()
        self.residuals()
