# SPDX-FileCopyrightText: 2026 Franz Rohrhofer <frohrhofer@know-center.at>
# SPDX-License-Identifier: MIT
"""Comparison figures of Section 2 of the paper, from the saved weights.

Creates in paper_figures/:
    thermocouples_comparison.{png,pdf}  thermocouple signals of all steps (Fig. 3)
    weldline_comparison.{png,pdf}       temperature along the weld line (Fig. 4)
    fields_comparison.{png,pdf}         temperature fields of all steps (Fig. 5)

Scenario k in config/ and results/ corresponds to Step k in the paper.

Usage (with the trained models in results/, e.g. those shipped with the code):
    python make_paper_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import load_config
from src.data_loader import DataLoader
from src.model import NeuralNet

STEPS = [
    ('Step 1: data', 'config/scenario_1_data.yaml', '#eb6834'),
    ('Step 2: data + PDE', 'config/scenario_2_data_pde.yaml', '#1baf7a'),
    ('Step 3: PDE + IC + BC', 'config/scenario_3_physics.yaml', '#2a78d6'),
]
# Each step has its own line style, so that the curves can be told apart in
# a black-and-white print as well. The reference is drawn as open grey
# circles (every n-th point only) in the background.
LINESTYLES = ['-', (0, (5, 2)), (0, (1, 1.2))]
LINEWIDTHS = [1.3, 1.3, 1.6]
REFERENCE_COLOR = '#222222'
REFERENCE_MARKERS = dict(ls='none', marker='o', ms=3.4, mfc='none',
                         mec='#8c8c8c', mew=0.8, zorder=1)
OUT = Path('paper_figures')

plt.rcParams.update({
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.8,
    'lines.linewidth': 1.4, 'font.size': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7.5, 'savefig.dpi': 300,
})


def load_models():
    """DataLoader (shared) and one trained network per step."""
    data, models = None, []
    for _, file, _ in STEPS:
        config = load_config(file)
        configs = config['configs']
        if data is None:
            data = DataLoader(load_config(configs['problem']),
                              load_config(configs['thermocouples']),
                              file=configs['reference'], seed=config['seed'])
        hard_ic = config.get('constraints', {}).get('hard_ic', False)
        model = NeuralNet(load_config(configs['network']), data.problem,
                          hard_ic=hard_ic, verbose=False)
        state = torch.load(Path('results') / config['version'] / 'model.pt',
                           map_location='cpu')
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return data, models


def predict(model, X, batch_size=65536):
    X = torch.tensor(X, dtype=torch.float32)
    return np.concatenate([model.temperature(X[i:i + batch_size]).numpy()
                           for i in range(0, len(X), batch_size)])


def snapshot(data, model, t):
    X, T_ref = data.snapshot(t)
    return T_ref, predict(model, X).reshape(T_ref.shape)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight')
    plt.close(fig)


def plate_axes(ax, show_x, show_y):
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 25, 50])
    ax.tick_params(labelbottom=show_x, labelleft=show_y, length=2, pad=1.5)
    if show_x:
        ax.set_xlabel('x [mm]', labelpad=1)
    if show_y:
        ax.set_ylabel('y [mm]', labelpad=1)


def sensors(ax, data):
    for s in data.sensors:
        ax.plot(s['x'] * 1e3, s['y'] * 1e3, 'o' if s['role'] == 'train' else 's',
                ms=3, mfc='none', mec='white', mew=0.8)


def fields_comparison(data, models, t_field=6.0, t_errors=(2.0, 10.0)):
    """Row 1: prediction at t_field; rows 2-3: absolute error at t_errors."""
    x_mm, y_mm = data.x * 1e3, data.y * 1e3
    T_m, T0 = data.problem.T_melt, data.problem.T0
    times = (t_field,) + tuple(t_errors)
    res = [[snapshot(data, m, t) for t in times] for m in models]   # [step][time]

    T_max = max(np.max(res[k][0][1]) for k in range(3))
    T_max = max(T_max, np.max(res[0][0][0]))
    err = [[np.abs(res[k][r][1] - res[k][r][0]) for r in (1, 2)] for k in range(3)]
    e_max_12 = max(e.max() for k in (0, 1) for e in err[k])
    e_max_3 = max(e.max() for e in err[2])

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 4.6), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.02, hspace=0.02)
    for k, (label, _, _) in enumerate(STEPS):
        T_ref, T_pred = res[k][0]
        ax = axes[0, k]
        im_T = ax.pcolormesh(x_mm, y_mm, T_pred, cmap='inferno', vmin=T0, vmax=T_max,
                             rasterized=True)
        if T_pred.max() > T_m:
            ax.contour(x_mm, y_mm, T_pred, [T_m], colors='cyan', linewidths=0.9)
        ax.contour(x_mm, y_mm, T_ref, [T_m], colors='white', linewidths=0.9,
                   linestyles='--')
        sensors(ax, data)
        ax.set_title(f'{label}\nprediction, t = {t_field:.0f} s', color='black')
        plate_axes(ax, False, k == 0)

        for r, t in enumerate(t_errors, start=1):
            e = err[k][r - 1]
            ax = axes[r, k]
            vmax = e_max_3 if k == 2 else e_max_12
            im = ax.pcolormesh(x_mm, y_mm, e, cmap='Reds', vmin=0, vmax=vmax,
                               rasterized=True)
            sensors_err(ax, data)
            ax.set_title(f'|error|, t = {t:.0f} s (max {e.max():.0f} K)')
            plate_axes(ax, r == 2, k == 0)
            if k == 2:
                im_e3 = im
            else:
                im_e12 = im

    fig.colorbar(im_T, ax=axes[0, :], label='T [K]', shrink=0.9, pad=0.01, aspect=12)
    fig.colorbar(im_e12, ax=axes[1:, 0:2], location='bottom', shrink=0.6,
                 label='|error| [K], Steps 1 and 2', pad=0.01, aspect=40)
    fig.colorbar(im_e3, ax=axes[1:, 2], location='bottom', shrink=0.9,
                 label='|error| [K], Step 3', pad=0.01, aspect=20)
    save(fig, 'fields_comparison')


def sensors_err(ax, data):
    for s in data.sensors:
        ax.plot(s['x'] * 1e3, s['y'] * 1e3, 'o' if s['role'] == 'train' else 's',
                ms=3, mfc='none', mec='#555555', mew=0.7)


def weldline_comparison(data, models, times=(2.0, 6.0, 10.0)):
    """Temperature along the weld line at selected times."""
    j = int(np.argmin(np.abs(data.y - data.problem.y_s)))
    x_mm = data.x * 1e3
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True,
                             constrained_layout=True)
    for ax, t in zip(axes, times):
        T_ref = data.snapshot(t)[1]
        ax.plot(x_mm[::3], T_ref[j, ::3], label='reference', **REFERENCE_MARKERS)
        for (label, _, color), ls, lw, m in zip(STEPS, LINESTYLES, LINEWIDTHS, models):
            ax.plot(x_mm, snapshot(data, m, t)[1][j], c=color, ls=ls, lw=lw,
                    label=label, zorder=3)
        ax.axhline(data.problem.T_melt, c='#9a9a9a', lw=0.9, ls=':', zorder=2)
        ax.set(title=f't = {t:.0f} s', xlabel='x [mm]')
    axes[0].set_ylabel('T [K] at y = 25 mm')
    axes[0].text(1, data.problem.T_melt + 30, r'$T_m$', fontsize=7, color='#6a6a6a')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=4, frameon=False,
               handlelength=2.5)
    save(fig, 'weldline_comparison')


def thermocouples_comparison(data, models):
    """Signals of all thermocouples: reference and all steps.

    Training sensors first, the two held-out test sensors in the last column
    (shaded) together with their RMSE per step.
    """
    order = [s for s in data.sensors if s['role'] == 'train'][:4] \
        + [s for s in data.sensors if s['role'] == 'test'][:1] \
        + [s for s in data.sensors if s['role'] == 'train'][4:] \
        + [s for s in data.sensors if s['role'] == 'test'][1:]
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.4), sharex=True, sharey=True,
                             constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.02, wspace=0.01, hspace=0.03)
    for ax, s in zip(axes.flat, order):
        t, T_ref = data.thermocouple_signal(s['name'], noise=False)
        X = np.stack([np.full_like(t, s['x']), np.full_like(t, s['y']), t], axis=1)
        ax.plot(t[::5], T_ref[::5], label='reference', **REFERENCE_MARKERS)
        lines = []
        for (label, _, color), ls, lw, m in zip(STEPS, LINESTYLES, LINEWIDTHS, models):
            T_pred = predict(m, X).ravel()
            ax.plot(t, T_pred, c=color, ls=ls, lw=lw, label=label, zorder=3)
            lines.append((color, np.sqrt(np.mean((T_pred - T_ref)**2))))
        test = s['role'] == 'test'
        ax.set_title(f"{s['name']} ({s['role']})",
                     fontweight='bold' if test else 'normal', pad=3)
        if test:
            # RMSE per step in a free corner: lower right if the signal has a
            # peak and cools down (TC9), upper left if it rises late (TC10)
            ax.set_facecolor('#f4f4f4')
            ends_high = np.argmax(T_ref) < len(T_ref) - 5
            x0, ha = (0.97, 'right') if ends_high else (0.05, 'left')
            ax.text(x0, 0.04 if ends_high else 0.93, 'RMSE [K]', transform=ax.transAxes,
                    ha=ha, va='bottom' if ends_high else 'top', fontsize=6.5,
                    color='#444444')
            for i, (color, rmse) in enumerate(lines):
                y = 0.04 + 0.11 * (3 - i) if ends_high else 0.93 - 0.11 * (i + 1)
                ax.text(x0, y, f'{rmse:.1f}', transform=ax.transAxes, ha=ha,
                        va='bottom' if ends_high else 'top', fontsize=6.5,
                        color=color, fontweight='bold')
        ax.set_xticks([0, 4, 8, 12])
    for ax in axes[-1]:
        ax.set_xlabel('t [s]')
    for ax in axes[:, 0]:
        ax.set_ylabel('T [K]')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=4, frameon=False,
               handlelength=2.5)
    save(fig, 'thermocouples_comparison')


if __name__ == '__main__':
    data, models = load_models()
    fields_comparison(data, models)
    weldline_comparison(data, models)
    thermocouples_comparison(data, models)
    print(f"Figures written to {OUT}/")
