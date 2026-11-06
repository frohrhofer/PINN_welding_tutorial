# Physics-Informed Neural Networks for Welding: A Tutorial

The tutorial builds a physics-informed neural network (PINN) step by step for a welding case study: the transient temperature field of a thin steel plate heated by a moving arc. A finite-difference reference solution serves as ground truth, so every step can be checked against the exact field.

| Step in the paper | Configuration | Training objective | Uses measurements |
| --- | --- | --- | --- |
| Step 1 (Section 2.2) | `config/scenario_1_data.yaml` | thermocouple data | yes |
| Step 2 (Section 2.3) | `config/scenario_2_data_pde.yaml` | data + heat equation | yes |
| Step 3 (Section 2.4) | `config/scenario_3_physics.yaml` | heat equation + initial and boundary conditions | no |

Scenario *k* in the code corresponds to Step *k* in the paper.

## Installation

Python 3.10 or newer is required. A CPU is sufficient; a GPU is used automatically if available.

```bash
git clone https://github.com/frohrhofer/PINN_welding_tutorial.git
cd PINN_welding_tutorial
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All commands below are run from the repository root.

## Quick start: reproduce the results with the trained models

The trained networks of the three steps are included in `results/`. Evaluating them takes a few seconds:

```bash
python main.py -c config/scenario_1_data.yaml --load
python main.py -c config/scenario_2_data_pde.yaml --load
python main.py -c config/scenario_3_physics.yaml --load
python make_paper_figures.py
```

This prints the error metrics of Table 1 and writes all figures of Section 2 and Appendix D:

| Expected output | RMSE [K] | Relative L2 error | Maximum error [K] |
| --- | --- | --- | --- |
| Step 1 | 136.8 | 0.446 | 1014 |
| Step 2 | 108.3 | 0.353 | 1208 |
| Step 3 | 1.7 | 0.006 | 30 |

## Training from scratch

```bash
python generate_reference_data.py --verify        # reference solution + grid convergence (Table B3)
python main.py -c config/scenario_1_data.yaml     # Step 1, about 10 s
python main.py -c config/scenario_2_data_pde.yaml # Step 2, about 5 min
python main.py -c config/scenario_3_physics.yaml  # Step 3, about 6 min
python make_paper_figures.py                      # Figures 3 to 5
```

Training times refer to a laptop CPU (AMD Ryzen 7 PRO 5850U). Training overwrites the shipped weights in `results/`. The reference solution is also included in `data/`, so the first command is optional; it reproduces the file exactly.

All runs use a fixed random seed. Results are reproducible on the same machine and software versions, but may differ slightly across hardware, operating systems or PyTorch versions, since floating-point operations are not guaranteed to be bitwise identical. The reference solution is reproduced exactly.

## Where to find the figures and tables of the paper

| Paper | Created by | File |
| --- | --- | --- |
| Table 1 | `main.py` (each step) | printed; `results/<scenario>/log.json`, key `evaluation` |
| Figure 3 (thermocouples) | `make_paper_figures.py` | `paper_figures/thermocouples_comparison.{png,pdf}` |
| Figure 4 (weld line) | `make_paper_figures.py` | `paper_figures/weldline_comparison.{png,pdf}` |
| Figure 5 (temperature fields) | `make_paper_figures.py` | `paper_figures/fields_comparison.{png,pdf}` |
| Table B3 (grid convergence) | `generate_reference_data.py --verify` | printed |
| Table C5 (training strategies) | `main.py` with `config/studies/table_c5_*.yaml` | `results/studies/...` |
| Figure D1 (learning curves, Step 1) | `main.py` | `results/scenario_1_data/learning_curves.png` |
| Figures D2 to D4 (fields per step) | `main.py` | `results/<scenario>/fields.png` |

`main.py` additionally writes the thermocouple signals, the temperature along the weld line and the terms of the heat equation for each step.

### Training-strategy study (Table C5) and longer training (Appendix D)

The configurations of the preliminary study on training strategies for Step 3 are in `config/studies/`, e.g.

```bash
python main.py -c config/studies/table_c5_6_hard_ic_lbfgs200_large.yaml
```

Rows 1 to 6 of Table C5 stop after a time budget of 300 s, so their results depend on the speed of the machine. `table_c5_7_baseline_adam20k.yaml` is the plain baseline without time limit (about 21 min), and `step1_adam20k.yaml` is Step 1 with 20 000 Adam epochs (Appendix D). The weights of these runs are not included.

## Repository structure

```
config/
  problem.yaml              plate, material, heat source, initial and boundary conditions
  thermocouples.yaml        sensor positions, sampling rate and noise
  network.yaml              network architecture and input/output scaling
  reference_solver.yaml     finite-difference solver settings
  scenario_*.yaml           Steps 1 to 3: loss weights, sampling and training settings
  studies/                  additional runs of Table C5 and Appendix D
data/
  reference_solution.npz    finite-difference reference solution (ground truth)
results/
  scenario_*/               trained weights (model.pt), training log (log.json), figures
src/
  physics.py                problem definition and non-dimensional coefficients
  reference_solver.py       explicit finite-difference (FTCS) reference solver
  data_loader.py            reference solution, synthetic thermocouples, sampling of training points
  model.py                  multilayer perceptron with input/output scaling and hard initial condition
  pinn.py                   loss terms, training (Adam and L-BFGS), evaluation
  plotter.py                figures per scenario
main.py                     trains (or loads) and evaluates one scenario
generate_reference_data.py  computes and verifies the reference solution
make_paper_figures.py       comparison figures of Section 2
```

## Changing the problem

All physical and numerical settings are in the YAML files in `config/`, so variations need no code changes. For example, a different travel speed or arc power is set in `config/problem.yaml`, a different sensor layout in `config/thermocouples.yaml`, and a different combination of loss terms in the scenario files (a weight of 0 switches a term off). After changing `problem.yaml`, regenerate the reference solution with `python generate_reference_data.py`; the data loader refuses a reference solution that was computed for a different problem.

## License

The code is released under the MIT License, see [LICENSE](LICENSE).

## Acknowledgements

This work was partially funded by the European Union's HORIZON Research and Innovation Programme under grant agreement No 101120657, project ENFIELD (European Lighthouse to Manifest Trustworthy and Green AI), and by the Austrian Climate Research Programme, project RegDTAlp. Know Center is a COMET competence center financed by the Austrian Federal Ministry of Innovation, Mobility and Infrastructure (BMIMI), the Austrian Federal Ministry of Economy, Energy and Tourism (BMWET), the Province of Styria, the Steirische Wirtschaftsförderungsgesellschaft m.b.H. (SFG), the Vienna Business Agency and the Standortagentur Tirol. The COMET programme is managed by the Austrian Research Promotion Agency FFG.
