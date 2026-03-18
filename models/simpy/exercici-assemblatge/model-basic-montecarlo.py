#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monte Carlo simulation experiment for the assembly line model.

Runs multiple independent replications of the assembly model with
different random seeds, collects output statistics from each
replication, and computes confidence intervals using the t-distribution.

Usage
-----
Run directly::

    python experiment-montecarlo.py

Or import and call::

    from experiment_montecarlo import run_experiment
    results_df, ci_df = run_experiment(n_replications=10, sim_time=5000)
"""

import simpy
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Import the model from the same directory
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import sys

# Ensure the model module can be imported from the same directory
_model_path = Path(__file__).parent / "model-basic.py"
_spec = spec_from_file_location("model_basic", _model_path)
_model_module = module_from_spec(_spec)
_spec.loader.exec_module(_model_module)
AssemblyModel = _model_module.AssemblyModel


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Default experiment parameters
DEFAULT_N_REPLICATIONS = 10
DEFAULT_SIM_TIME = 800*60
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BASE_SEED = 1000


def run_single_replication(seed: int, sim_time: float) -> dict:
    """
    Run a single replication of the assembly model.

    Parameters
    ----------
    seed : int
        Random seed for this replication.
    sim_time : float
        Simulation end time in minutes.

    Returns
    -------
    dict
        Dictionary of output statistics from ``AssemblyModel.get_results()``.
    """
    env = simpy.Environment()
    model = AssemblyModel(env, seed=seed, verbose=False)
    model.run(until=sim_time)
    return model.get_results()


def compute_confidence_intervals(data: pd.DataFrame,
                                 confidence: float = 0.95) -> pd.DataFrame:
    """
    Compute confidence intervals for all output statistics.

    Uses the Student's t-distribution to compute intervals, which is
    appropriate for small sample sizes (n < 30) with unknown population
    variance.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame where each row is a replication and each column is
        an output statistic.
    confidence : float, optional
        Confidence level for the intervals (default is 0.95).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``'metric'``, ``'mean'``, ``'std'``,
        ``'n'``, ``'ci_lower'``, ``'ci_upper'``, ``'half_width'``,
        ``'rel_error_pct'``.
    """
    n = len(data)
    alpha = 1.0 - confidence
    t_critical = scipy_stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)

    records = []
    for col in data.columns:
        values = data[col].values
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # sample standard deviation
        se = std / np.sqrt(n)         # standard error
        half_width = t_critical * se
        ci_lower = mean - half_width
        ci_upper = mean + half_width
        rel_error = (half_width / mean * 100.0) if mean != 0 else 0.0

        records.append({
            'metric': col,
            'mean': mean,
            'std': std,
            'n': n,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'half_width': half_width,
            'rel_error_pct': rel_error,
        })

    return pd.DataFrame(records)


def run_experiment(n_replications: int = DEFAULT_N_REPLICATIONS,
                   sim_time: float = DEFAULT_SIM_TIME,
                   confidence: float = DEFAULT_CONFIDENCE_LEVEL,
                   base_seed: int = DEFAULT_BASE_SEED) -> tuple:
    """
    Run the full Monte Carlo experiment.

    Executes ``n_replications`` independent simulation runs with
    different random seeds, collects all output statistics, and
    computes confidence intervals.

    Parameters
    ----------
    n_replications : int, optional
        Number of independent replications (default is 10).
    sim_time : float, optional
        Simulation end time in minutes for each replication
        (default is 5000.0).
    confidence : float, optional
        Confidence level for the intervals (default is 0.95).
    base_seed : int, optional
        Starting seed; each replication uses ``base_seed + i``
        (default is 1000).

    Returns
    -------
    results_df : pd.DataFrame
        Raw results with one row per replication.
    ci_df : pd.DataFrame
        Confidence intervals with one row per metric.
    """
    print("=" * 80)
    print("MONTE CARLO SIMULATION EXPERIMENT")
    print("=" * 80)
    print(f"  Replications:      {n_replications}")
    print(f"  Simulation time:   {sim_time:.0f} min per replication")
    print(f"  Confidence level:  {confidence * 100:.0f}%")
    print(f"  Base seed:         {base_seed}")
    print("=" * 80)

    # Run all replications
    all_results = []
    for i in range(n_replications):
        seed = base_seed + i
        print(f"\n  Running replication {i + 1}/{n_replications} "
              f"(seed={seed})...", end="")
        result = run_single_replication(seed, sim_time)
        result['replication'] = i + 1
        result['seed'] = seed
        all_results.append(result)
        print(f" done. Products assembled: {result['finished_products']}")

    # Build DataFrame of raw results
    results_df = pd.DataFrame(all_results)

    # Move replication and seed to the front
    cols = ['replication', 'seed'] + [c for c in results_df.columns
                                       if c not in ('replication', 'seed')]
    results_df = results_df[cols]

    # Compute confidence intervals (exclude replication/seed columns)
    stat_cols = [c for c in results_df.columns
                 if c not in ('replication', 'seed')]
    ci_df = compute_confidence_intervals(results_df[stat_cols], confidence)

    return results_df, ci_df


def print_results(results_df: pd.DataFrame, ci_df: pd.DataFrame,
                  confidence: float = DEFAULT_CONFIDENCE_LEVEL) -> None:
    """
    Print formatted experiment results and confidence intervals.

    Parameters
    ----------
    results_df : pd.DataFrame
        Raw results with one row per replication.
    ci_df : pd.DataFrame
        Confidence intervals with one row per metric.
    confidence : float, optional
        Confidence level used (default is 0.95).
    """
    # --- Raw results per replication ---
    print("\n" + "=" * 80)
    print("RAW RESULTS PER REPLICATION")
    print("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(results_df.to_string(index=False))

    # --- Confidence intervals grouped by category ---
    pct = confidence * 100

    # Parts counters
    counter_metrics = [
        'covers_generated', 'covers_painted', 'covers_reworked',
        'boxes_received', 'boxes_unpacked', 'elements_ok',
        'elements_scrapped', 'finished_products',
    ]

    # Resource utilization
    util_metrics = ['util_paint', 'util_unpack', 'util_assembly']

    # Queue waiting times
    wait_metrics = ['wait_paint', 'wait_unpack', 'wait_assembly']

    # Queue lengths (station queues)
    qlen_station_metrics = ['qlen_paint', 'qlen_unpack', 'qlen_assembly']

    # Queue lengths (inter-stage queues)
    qlen_stage_metrics = [
        'qlen_sup_covers', 'qlen_inf_covers', 'qlen_int_elements',
    ]

    sections = [
        ("PARTS COUNTERS", counter_metrics),
        ("RESOURCE UTILIZATION", util_metrics),
        ("QUEUE AVERAGE WAITING TIMES (minutes)", wait_metrics),
        ("STATION QUEUE AVERAGE LENGTHS (entities)", qlen_station_metrics),
        ("INTER-STAGE QUEUE AVERAGE LENGTHS (entities)", qlen_stage_metrics),
    ]

    header = (f"  {'Metric':<25s} {'Mean':>10s} {'Std':>10s} "
              f"{'CI Lower':>10s} {'CI Upper':>10s} "
              f"{'Half-W':>10s} {'Rel.Err%':>9s}")
    separator = "  " + "-" * 76

    print("\n" + "=" * 80)
    print(f"CONFIDENCE INTERVALS ({pct:.0f}%)")
    print("=" * 80)

    for section_title, metric_list in sections:
        print(f"\n  {section_title}")
        print(separator)
        print(header)
        print(separator)

        for metric in metric_list:
            row = ci_df[ci_df['metric'] == metric].iloc[0]
            # Format utilization as percentage
            if metric.startswith('util_'):
                print(f"  {metric:<25s} {row['mean'] * 100:>9.2f}% "
                      f"{row['std'] * 100:>9.2f}% "
                      f"{row['ci_lower'] * 100:>9.2f}% "
                      f"{row['ci_upper'] * 100:>9.2f}% "
                      f"{row['half_width'] * 100:>9.2f}% "
                      f"{row['rel_error_pct']:>8.2f}%")
            else:
                print(f"  {metric:<25s} {row['mean']:>10.2f} "
                      f"{row['std']:>10.2f} "
                      f"{row['ci_lower']:>10.2f} "
                      f"{row['ci_upper']:>10.2f} "
                      f"{row['half_width']:>10.2f} "
                      f"{row['rel_error_pct']:>8.2f}%")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results_df, ci_df = run_experiment(
        n_replications=DEFAULT_N_REPLICATIONS,
        sim_time=DEFAULT_SIM_TIME,
        confidence=DEFAULT_CONFIDENCE_LEVEL,
        base_seed=DEFAULT_BASE_SEED,
    )
    print_results(results_df, ci_df, DEFAULT_CONFIDENCE_LEVEL)