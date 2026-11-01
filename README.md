# Circular Stop Optimization

Fixed-budget bus-stop placement on circular routes, with continuum optimization, feasible discrete layouts, and passenger reassignment.

The repository includes synthetic benchmarks and a separate timing-calibration analysis of published Orangeline observations. 

## Paper

[Read the paper](paper/Fixed_Budget_Stop_Design_on_Circular_Routes_Theory__Discrete_Benchmarks__and_Observational_Calibration.pdf) or see [the LaTeX source](paper/transit_final.tex).
To compile the paper with the included tables and figures:

```bash
cd paper
latexmk -pdf transit_final.tex
```

## Installation

Python 3.11 or newer:

```bash
python -m pip install -r requirements.txt
```

## Figures

Generate the seven figures from the included results:

```bash
python scripts/plot_figures.py
```

PNG and PDF outputs are saved under `figures/`.

## Reproduce analyses

```bash
python scripts/stop_design.py --test
python scripts/experiments.py --test
python scripts/observed_timing.py --test
python scripts/observed_timing.py
python scripts/experiments.py --case --out results
```

To recompute the 720-fit synthetic benchmark, run the following separately. This is more computationally intensive than regenerating the plots:

```bash
python scripts/experiments.py --suite --training-trips 300 --out results/sample_300
python scripts/experiments.py --suite --training-trips 3000 --out results
python scripts/experiments.py --suite --training-trips 30000 --out results/sample_30000
python scripts/plot_figures.py
```

## Repository structure

- `scripts/stop_design.py`: continuum solution, frozen-flow mixed-integer optimizer, and passenger-choice evaluator.
- `scripts/experiments.py`: finite-layout enumeration, phase selection, discrete search, sensitivity analyses, and synthetic benchmarks.
- `scripts/observed_timing.py`: training-only calibration and held-out evaluation of published timing observations.
- `scripts/plot_figures.py`: regenerates the seven figures from saved CSVs.
- `paper/`: manuscript PDF, editable LaTeX, and its input table.
- `figures/`: manuscript plots in PDF and PNG formats.
- `data/`: published aggregate measurements and source notes.
- `results/`: numerical outputs and saved synthetic experiment inputs.

The simulation experiments use generated demand and hypothetical siting constraints. No field trial, real-world stop-relocation savings, or superiority over existing transit-planning methods is claimed. The published observations are used only for out-of-period timing evaluation. See `data/README.md` for attribution and limitations.

## License

The MIT license applies to the original code. Published observations remain subject to the original source terms. The manuscript is included for reading and reproducibility.
